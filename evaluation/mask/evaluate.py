# Evaluate interactive segmentation models using the NoC (Number of Clicks)
# metric on the DAVIS dataset.
#
# Supports three model families (auto-detected from ONNX input names):
#   - SAM custom-exported (sam21-small, sam21-tiny, sam21hq-large)
#   - SAM pre-converted ONNX (sam21-small-onnx)
#   - SegNext (segnext-b2hq)
#
# Usage:
#   python3 evaluation/mask/evaluate.py \
#       --encoder output/mask-object-segnext-b2hq/encoder.onnx \
#       --decoder output/mask-object-segnext-b2hq/decoder.onnx \
#       --dataset-path temp/DAVIS \
#       --limit 5

import argparse
import os
import sys
import time

import numpy as np
import onnxruntime as ort
from PIL import Image
from scipy import ndimage

MODEL_SIZE = 1024
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Model family detection
# ---------------------------------------------------------------------------

def detect_family(dec_session):
    """Detect model family from decoder ONNX input names."""
    input_names = {inp.name for inp in dec_session.get_inputs()}
    if "image_feats" in input_names:
        return "segnext"
    if "image_embeddings.0" in input_names:
        return "sam_onnx"
    if "image_embed" in input_names:
        return "sam_custom"
    raise ValueError(f"Unknown model family. Decoder inputs: {input_names}")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_image(image, family):
    """Resize to 1024x1024 and normalize for the model family."""
    image = image.resize((MODEL_SIZE, MODEL_SIZE), Image.LANCZOS)
    arr = np.array(image).astype(np.float32) / 255.0
    if family != "segnext":
        arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)
    return arr[np.newaxis]


# ---------------------------------------------------------------------------
# Encoder dispatch
# ---------------------------------------------------------------------------

def run_encoder(enc_session, input_tensor, family):
    """Run encoder and return a dict of named outputs."""
    if family == "sam_onnx":
        outputs = enc_session.run(None, {"pixel_values": input_tensor})
        return {"emb0": outputs[0], "emb1": outputs[1], "emb2": outputs[2]}
    elif family == "sam_custom":
        outputs = enc_session.run(None, {"image": input_tensor})
        return {"hrf0": outputs[0], "hrf1": outputs[1], "embed": outputs[2]}
    else:  # segnext
        (feats,) = enc_session.run(None, {"image": input_tensor})
        return {"feats": feats}


# ---------------------------------------------------------------------------
# Decoder dispatch
# ---------------------------------------------------------------------------

def run_decoder_sam_custom(dec_session, enc_out, point_coords, point_labels,
                           mask_input, has_mask_input):
    """Run SAM custom-exported decoder. Returns (mask_1024, low_res_mask_256)."""
    outputs = dec_session.run(None, {
        "image_embed": enc_out["embed"],
        "high_res_feats_0": enc_out["hrf0"],
        "high_res_feats_1": enc_out["hrf1"],
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
    })
    masks, iou_preds, low_res_masks = outputs
    best = int(np.argmax(iou_preds[0]))
    return masks[0, best], low_res_masks[0, best:best + 1]


def run_decoder_sam_onnx(dec_session, enc_out, point_coords, point_labels):
    """Run SAM pre-converted ONNX decoder. Returns mask_logits (H, W)."""
    # This interface uses [B, 1, N, 2] for points and [B, 1, N] int64 for labels
    n = point_coords.shape[1]
    input_points = point_coords.reshape(1, 1, n, 2)
    input_labels = point_labels.astype(np.int64).reshape(1, 1, n)
    input_boxes = np.zeros((1, 0, 4), dtype=np.float32)

    outputs = dec_session.run(None, {
        "image_embeddings.0": enc_out["emb0"],
        "image_embeddings.1": enc_out["emb1"],
        "image_embeddings.2": enc_out["emb2"],
        "input_points": input_points,
        "input_labels": input_labels,
        "input_boxes": input_boxes,
    })
    iou_scores, pred_masks, _ = outputs
    best = int(np.argmax(iou_scores[0, 0]))
    return pred_masks[0, 0, best]


def run_decoder_segnext(dec_session, enc_out, point_coords, point_labels,
                        prev_mask):
    """Run SegNext decoder. Returns mask_logits [1024, 1024]."""
    (mask,) = dec_session.run(None, {
        "image_feats": enc_out["feats"],
        "point_coords": point_coords,
        "point_labels": point_labels,
        "prev_mask": prev_mask,
    })
    return mask[0, 0]


# ---------------------------------------------------------------------------
# Click simulation
# ---------------------------------------------------------------------------

def compute_iou(pred, gt):
    """Compute IoU between two binary masks."""
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection) / float(union)


def simulate_click(pred_mask, gt_mask):
    """Find the next click point using the standard protocol.

    Places a click at the center of the largest error region.

    Args:
        pred_mask: (H, W) binary predicted mask
        gt_mask: (H, W) binary ground truth mask

    Returns:
        (row, col, label) where label is 1 for FN, 0 for FP.
        Returns None if masks are identical.
    """
    fn_region = gt_mask & ~pred_mask  # false negatives
    fp_region = pred_mask & ~gt_mask  # false positives

    fn_area = fn_region.sum()
    fp_area = fp_region.sum()

    if fn_area == 0 and fp_area == 0:
        return None

    # Pick the larger error type
    if fn_area >= fp_area:
        error_region = fn_region
        label = 1.0  # positive click
    else:
        error_region = fp_region
        label = 0.0  # negative click

    # Find largest connected component
    labeled, num_features = ndimage.label(error_region)
    if num_features == 0:
        return None

    component_sizes = ndimage.sum(error_region, labeled, range(1, num_features + 1))
    largest_idx = int(np.argmax(component_sizes)) + 1
    largest_component = (labeled == largest_idx)

    # Find center via distance transform (point farthest from boundary)
    dt = ndimage.distance_transform_edt(largest_component)
    center = np.unravel_index(np.argmax(dt), dt.shape)
    row, col = center

    return row, col, label


# ---------------------------------------------------------------------------
# DAVIS dataset
# ---------------------------------------------------------------------------

def load_davis(dataset_path):
    """Load DAVIS dataset: first frame of each sequence.

    Returns list of (image_path, gt_mask_path) tuples.
    """
    # DAVIS-2017-trainval-480p extracts to DAVIS/ with:
    #   JPEGImages/480p/<sequence>/00000.jpg
    #   Annotations/480p/<sequence>/00000.png
    img_root = os.path.join(dataset_path, "JPEGImages", "480p")
    ann_root = os.path.join(dataset_path, "Annotations", "480p")

    if not os.path.isdir(img_root):
        print(f"Error: DAVIS images not found at {img_root}")
        print("Download DAVIS-2017-trainval-480p.zip and extract to the dataset path.")
        sys.exit(1)

    pairs = []
    for seq in sorted(os.listdir(img_root)):
        seq_img_dir = os.path.join(img_root, seq)
        seq_ann_dir = os.path.join(ann_root, seq)
        if not os.path.isdir(seq_img_dir):
            continue

        # First frame
        img_path = os.path.join(seq_img_dir, "00000.jpg")
        ann_path = os.path.join(seq_ann_dir, "00000.png")
        if os.path.isfile(img_path) and os.path.isfile(ann_path):
            pairs.append((img_path, ann_path))

    if not pairs:
        print(f"Error: No image/annotation pairs found in {dataset_path}")
        sys.exit(1)

    return pairs


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate_sample(enc_session, dec_session, family, enc_out, gt_mask,
                    img_h, img_w, max_clicks, thresholds):
    """Evaluate one image with iterative click simulation.

    Args:
        enc_out: dict of encoder outputs
        gt_mask: (H, W) binary ground truth at original image resolution
        img_h, img_w: original image dimensions
        max_clicks: maximum number of clicks
        thresholds: list of IoU thresholds for NoC

    Returns:
        dict with:
            "ious": list of IoU after each click
            "nocs": dict of threshold -> number of clicks (or max_clicks)
    """
    # State
    all_coords = []  # list of (x_1024, y_1024)
    all_labels = []  # list of float labels

    # Mask feedback state
    mask_input_256 = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.array([0.0], dtype=np.float32)
    prev_mask_1024 = np.zeros((1, 1, MODEL_SIZE, MODEL_SIZE), dtype=np.float32)

    pred_binary = np.zeros_like(gt_mask, dtype=bool)
    ious = []
    max_threshold = max(thresholds)

    for click_idx in range(max_clicks):
        # Simulate next click at original image resolution
        click = simulate_click(pred_binary, gt_mask)
        if click is None:
            # Perfect match, fill remaining with current IoU
            current_iou = compute_iou(pred_binary, gt_mask)
            ious.append(current_iou)
            break

        row, col, label = click

        # Scale to 1024x1024 space: (x, y) = (col, row)
        x_1024 = col * MODEL_SIZE / img_w
        y_1024 = row * MODEL_SIZE / img_h
        all_coords.append([x_1024, y_1024])
        all_labels.append(label)

        point_coords = np.array([all_coords], dtype=np.float32)  # [1, N, 2]
        point_labels = np.array([all_labels], dtype=np.float32)  # [1, N]

        # Run decoder
        if family == "sam_custom":
            mask_logits, low_res = run_decoder_sam_custom(
                dec_session, enc_out, point_coords, point_labels,
                mask_input_256, has_mask_input,
            )
            # Update mask feedback
            mask_input_256 = low_res.reshape(1, 1, 256, 256)
            has_mask_input = np.array([1.0], dtype=np.float32)

        elif family == "sam_onnx":
            mask_logits = run_decoder_sam_onnx(
                dec_session, enc_out, point_coords, point_labels,
            )
            # No mask feedback for this interface

        else:  # segnext
            mask_logits = run_decoder_segnext(
                dec_session, enc_out, point_coords, point_labels,
                prev_mask_1024,
            )
            # Update mask feedback
            prev_mask_1024 = (mask_logits > 0).astype(np.float32).reshape(
                1, 1, MODEL_SIZE, MODEL_SIZE
            )

        # Resize logits to original resolution
        mask_img = Image.fromarray((mask_logits > 0).astype(np.uint8) * 255)
        mask_full = mask_img.resize((img_w, img_h), Image.LANCZOS)
        pred_binary = np.array(mask_full) > 127

        iou = compute_iou(pred_binary, gt_mask)
        ious.append(iou)

        if iou >= max_threshold:
            break

    # Compute NoC for each threshold
    nocs = {}
    for t in thresholds:
        noc = max_clicks  # default: didn't reach threshold
        for i, iou in enumerate(ious):
            if iou >= t:
                noc = i + 1
                break
        nocs[t] = noc

    return {"ious": ious, "nocs": nocs}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate mask models using NoC metric on DAVIS."
    )
    parser.add_argument("--encoder", required=True, help="Path to encoder.onnx")
    parser.add_argument("--decoder", required=True, help="Path to decoder.onnx")
    parser.add_argument("--dataset-path", required=True,
                        help="Path to extracted DAVIS dataset")
    parser.add_argument("--max-clicks", type=int, default=20,
                        help="Maximum clicks per image (default: 20)")
    parser.add_argument("--thresholds", type=str, default="0.85,0.90,0.95",
                        help="IoU thresholds for NoC (default: 0.85,0.90,0.95)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Evaluate only first N images (0 = all)")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    thresholds.sort()

    # Load models
    print(f"Loading encoder: {args.encoder}")
    enc_session = ort.InferenceSession(
        args.encoder, providers=["CPUExecutionProvider"]
    )
    print(f"Loading decoder: {args.decoder}")
    dec_session = ort.InferenceSession(
        args.decoder, providers=["CPUExecutionProvider"]
    )

    family = detect_family(dec_session)
    print(f"Model family: {family}")

    # Load dataset
    pairs = load_davis(args.dataset_path)
    if args.limit > 0:
        pairs = pairs[:args.limit]
    print(f"Dataset: DAVIS ({len(pairs)} images)")
    print(f"Max clicks: {args.max_clicks}")
    print(f"Thresholds: {thresholds}")
    print()

    # Evaluate
    all_nocs = {t: [] for t in thresholds}
    all_ious_at = {}  # click_idx -> list of ious
    t0 = time.perf_counter()

    for idx, (img_path, gt_path) in enumerate(pairs):
        seq_name = os.path.basename(os.path.dirname(img_path))

        # Load image
        image = Image.open(img_path).convert("RGB")
        img_w, img_h = image.size
        input_tensor = preprocess_image(image, family)

        # Load ground truth
        gt_img = Image.open(gt_path)
        gt_mask = np.array(gt_img) > 0
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, :, 0]

        # Run encoder once
        enc_out = run_encoder(enc_session, input_tensor, family)

        # Evaluate with iterative clicks
        result = evaluate_sample(
            enc_session, dec_session, family, enc_out, gt_mask,
            img_h, img_w, args.max_clicks, thresholds,
        )

        # Collect results
        for t in thresholds:
            all_nocs[t].append(result["nocs"][t])

        for i, iou in enumerate(result["ious"]):
            if i not in all_ious_at:
                all_ious_at[i] = []
            all_ious_at[i].append(iou)

        final_iou = result["ious"][-1] if result["ious"] else 0.0
        n_clicks = len(result["ious"])
        print(f"  [{idx + 1:3d}/{len(pairs)}] {seq_name:20s}  "
              f"clicks={n_clicks:2d}  IoU={final_iou:.4f}")

    elapsed = time.perf_counter() - t0
    print()
    print(f"Completed in {elapsed:.1f}s")
    print()

    # Report results
    print("Results:")
    for t in thresholds:
        noc_values = all_nocs[t]
        mean_noc = np.mean(noc_values)
        pct = int(t * 100)
        print(f"  NoC@{pct}: {mean_noc:.2f}")

    # Mean IoU at specific click counts
    for n in [1, 3, 5, 10]:
        if n - 1 in all_ious_at:
            vals = all_ious_at[n - 1]
            print(f"  Mean IoU@{n}: {np.mean(vals):.4f}")


if __name__ == "__main__":
    main()
