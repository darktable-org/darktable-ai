# Demo: run the SAM 2.1 ONNX model on an image with point prompts
# and save the source image with a red mask overlay.
#
# Usage:
#   python3 models/mask-object-sam21-small/demo.py \
#       --encoder output/mask-object-sam21-small/encoder.onnx \
#       --decoder output/mask-object-sam21-small/decoder.onnx \
#       --image samples/mask-object/example_03.jpg \
#       --point 0.28,0.65 \
#       --output output/mask-object-sam21-small-demo/example_03.png

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

MODEL_SIZE = 1024
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize to 1024x1024 and ImageNet-normalize."""
    image = image.resize((MODEL_SIZE, MODEL_SIZE), Image.LANCZOS)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    return arr[np.newaxis]


def make_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a red mask on the source image."""
    img_arr = np.array(image).astype(np.float32)
    red = np.array([255.0, 0.0, 0.0])
    mask_3d = mask[:, :, np.newaxis]
    img_arr = img_arr * (1 - mask_3d * alpha) + red * mask_3d * alpha
    return Image.fromarray(img_arr.clip(0, 255).astype(np.uint8))


def run_inference(encoder_path, decoder_path, image_path, output_path, points):
    """Run encoder+decoder inference with point prompts."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    parsed = [[float(v) for v in p.split(",")] for p in points]

    t0 = time.perf_counter()

    print(f"Loading encoder: {encoder_path}")
    enc_session = ort.InferenceSession(encoder_path, providers=["CPUExecutionProvider"])
    print(f"Loading decoder: {decoder_path}")
    dec_session = ort.InferenceSession(decoder_path, providers=["CPUExecutionProvider"])
    t_model = time.perf_counter()
    print(f"  Load models:   {t_model - t0:.3f}s")

    print(f"Loading image: {image_path}")
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    input_tensor = preprocess_image(image)
    t_image = time.perf_counter()
    print(f"  Original size: {orig_w}x{orig_h}")
    print(f"  Load image:    {t_image - t_model:.3f}s")

    # Run encoder
    # Outputs: high_res_feats_0 [1,32,256,256], high_res_feats_1 [1,64,128,128], image_embed [1,256,64,64]
    print("Running encoder...")
    enc_outputs = enc_session.run(None, {"image": input_tensor})
    high_res_feats_0, high_res_feats_1, image_embed = enc_outputs
    t_enc = time.perf_counter()
    print(f"  Encoder:       {t_enc - t_image:.3f}s")

    # Prepare point prompts (normalized coords -> 1024x1024 space)
    coords = [[px * MODEL_SIZE, py * MODEL_SIZE] for px, py in parsed]
    point_coords = np.array([coords], dtype=np.float32)
    point_labels = np.ones((1, len(coords)), dtype=np.float32)
    mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.array([0.0], dtype=np.float32)
    for i, (px, py) in enumerate(parsed):
        print(f"  Point {i+1} (1024): ({px * MODEL_SIZE:.0f}, {py * MODEL_SIZE:.0f})")

    # Run decoder
    print("Running decoder...")
    dec_outputs = dec_session.run(None, {
        "image_embed": image_embed,
        "high_res_feats_0": high_res_feats_0,
        "high_res_feats_1": high_res_feats_1,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "mask_input": mask_input,
        "has_mask_input": has_mask_input,
    })
    masks, iou_predictions, low_res_masks = dec_outputs
    t_dec = time.perf_counter()
    print(f"  Decoder:       {t_dec - t_enc:.3f}s")
    print(f"  Masks shape:   {masks.shape}")
    print(f"  IoU scores:    {iou_predictions[0]}")

    # masks shape: [B, 3, 1024, 1024] - select best by IoU
    best_idx = int(np.argmax(iou_predictions[0]))
    mask_logits = masks[0, best_idx]
    print(f"  Best mask:     #{best_idx} (IoU={iou_predictions[0, best_idx]:.4f})")

    # Resize mask to original image size
    mask_img = Image.fromarray((mask_logits > 0).astype(np.uint8) * 255)
    mask_full = mask_img.resize((orig_w, orig_h), Image.LANCZOS)
    mask_binary = (np.array(mask_full) > 127).astype(np.float32)

    # Save overlay
    result = make_overlay(image, mask_binary)
    result.save(output_path)
    t_total = time.perf_counter()
    print(f"Saved: {output_path}")
    print(f"  Total:         {t_total - t0:.3f}s")


def demo(encoder, decoder, image, output, **kwargs):
    """Entry point for programmatic demo."""
    points = kwargs.get("points", [])
    if not points and "point" in kwargs:
        points = [kwargs["point"]]
    run_inference(encoder, decoder, image, output, points=points)


def main():
    parser = argparse.ArgumentParser(description="SAM 2.1 ONNX segmentation demo.")
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--decoder", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--point", type=str, action="append", default=[],
                        help="Point prompt as x,y (normalized). Repeat for multiple.")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    demo(args.encoder, args.decoder, args.image, args.output, points=args.point)


if __name__ == "__main__":
    main()
