# Demo: run the SegNext ONNX model on an image with a single point prompt
# and save the source image with a red mask overlay.
#
# Usage:
#   python3 models/mask-object-segnext-b2hq/demo.py \
#       --encoder output/mask-object-segnext-b2hq/encoder.onnx \
#       --decoder output/mask-object-segnext-b2hq/decoder.onnx \
#       --image images/example_03.jpg \
#       --point 0.28,0.65 \
#       --output models/mask-object-segnext-b2hq/output/example_03.png

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

MODEL_SIZE = 1024


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize to 1024x1024 and convert to float32 [0, 1].

    Normalization is baked into the encoder ONNX model.
    """
    image = image.resize((MODEL_SIZE, MODEL_SIZE), Image.LANCZOS)
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)  # (3, H, W)
    return arr[np.newaxis]


def make_overlay(image: Image.Image, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Overlay a red mask on the source image."""
    img_arr = np.array(image).astype(np.float32)
    red = np.array([255.0, 0.0, 0.0])
    mask_3d = mask[:, :, np.newaxis]
    img_arr = img_arr * (1 - mask_3d * alpha) + red * mask_3d * alpha
    return Image.fromarray(img_arr.clip(0, 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="SegNext ONNX segmentation demo.")
    parser.add_argument("--encoder", type=str, required=True, help="Path to encoder.onnx")
    parser.add_argument("--decoder", type=str, required=True, help="Path to decoder.onnx")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--point", type=str, required=True,
                        help="Foreground point as normalized x,y (e.g. 0.28,0.65)")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    px, py = [float(v) for v in args.point.split(",")]

    t0 = time.perf_counter()

    print(f"Loading encoder: {args.encoder}")
    enc_session = ort.InferenceSession(args.encoder, providers=["CPUExecutionProvider"])
    print(f"Loading decoder: {args.decoder}")
    dec_session = ort.InferenceSession(args.decoder, providers=["CPUExecutionProvider"])
    t_model = time.perf_counter()
    print(f"  Load models:   {t_model - t0:.3f}s")

    print(f"Loading image: {args.image}")
    image = Image.open(args.image)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    input_tensor = preprocess_image(image)
    t_image = time.perf_counter()
    print(f"  Original size: {orig_w}x{orig_h}")
    print(f"  Load image:    {t_image - t_model:.3f}s")

    # Run encoder
    print("Running encoder...")
    (image_feats,) = enc_session.run(None, {"image": input_tensor})
    t_enc = time.perf_counter()
    print(f"  Encoder:       {t_enc - t_image:.3f}s")
    print(f"  Features:      {image_feats.shape}")

    # Prepare point prompt (normalized coords -> 1024x1024 space)
    point_coords = np.array([[[px * MODEL_SIZE, py * MODEL_SIZE]]], dtype=np.float32)
    point_labels = np.array([[1.0]], dtype=np.float32)
    prev_mask = np.zeros((1, 1, MODEL_SIZE, MODEL_SIZE), dtype=np.float32)
    print(f"  Point (1024):  ({point_coords[0,0,0]:.0f}, {point_coords[0,0,1]:.0f})")

    # Run decoder
    print("Running decoder...")
    (mask_logits,) = dec_session.run(None, {
        "image_feats": image_feats,
        "point_coords": point_coords,
        "point_labels": point_labels,
        "prev_mask": prev_mask,
    })
    t_dec = time.perf_counter()
    print(f"  Decoder:       {t_dec - t_enc:.3f}s")
    print(f"  Mask shape:    {mask_logits.shape}")

    # mask_logits shape: [1, 1, 1024, 1024]
    mask_1024 = mask_logits[0, 0]
    print(f"  Logit range:   [{mask_1024.min():.2f}, {mask_1024.max():.2f}]")

    # Resize mask to original image size
    mask_img = Image.fromarray((mask_1024 > 0).astype(np.uint8) * 255)
    mask_full = mask_img.resize((orig_w, orig_h), Image.LANCZOS)
    mask_binary = (np.array(mask_full) > 127).astype(np.float32)

    # Save overlay
    result = make_overlay(image, mask_binary)
    result.save(args.output)
    t_total = time.perf_counter()
    print(f"Saved: {args.output}")
    print(f"  Total:         {t_total - t0:.3f}s")


if __name__ == "__main__":
    main()
