"""Demo: compute image embedding and show top-5 zero-shot tags.

Saves a PNG with the original image and top-5 tag predictions overlaid.
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont, ImageOps

IMAGE_SIZE = 224


def preprocess(image):
    """Resize shortest edge to 224, center crop, normalize to [0, 1] BCHW."""
    w, h = image.size
    scale = IMAGE_SIZE / min(w, h)
    new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
    image = image.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - IMAGE_SIZE) // 2
    top = (new_h - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]  # BCHW
    return arr


def draw_tags(image, tags_scores):
    """Draw top tags on the image and return the annotated image."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    font_size = max(16, min(w, h) // 25)
    pad = max(4, font_size // 4)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except (OSError, IOError):
            font = ImageFont.load_default()

    y = pad * 2
    for tag, score in tags_scores:
        text = f"{tag}: {score:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.rectangle(
            [pad, y - pad // 2, pad * 2 + tw, y + th + pad // 2],
            fill=(0, 0, 0, 180),
        )
        draw.text((pad + pad // 2, y), text, fill=(255, 255, 255), font=font)
        y += th + pad * 2

    return image


def run_inference(model_path, image_path, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Load tags
    model_dir = os.path.dirname(model_path)
    tags_path = os.path.join(model_dir, "tags.json")
    if not os.path.isfile(tags_path):
        print(f"Warning: tags.json not found at {tags_path}")
        tags, tag_embeddings = [], None
    else:
        with open(tags_path) as f:
            data = json.load(f)
        tags = data["tags"]
        tag_embeddings = np.array(data["embeddings"], dtype=np.float32)

    t0 = time.perf_counter()

    # Load model
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    t_load = time.perf_counter()

    # Load and preprocess image
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    input_tensor = preprocess(image)
    t_pre = time.perf_counter()

    # Run inference
    (embedding,) = session.run(None, {"image": input_tensor})
    t_inf = time.perf_counter()

    print(f"  Image: {orig_w}x{orig_h}")
    print(f"  Embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")
    print(f"  Load: {t_load - t0:.3f}s  Preprocess: {t_pre - t_load:.3f}s  Inference: {t_inf - t_pre:.3f}s")

    # Zero-shot classification
    if tag_embeddings is not None:
        scores = (embedding @ tag_embeddings.T)[0]  # dot product = cosine sim
        top_idx = np.argsort(scores)[::-1][:5]
        tags_scores = [(tags[i], float(scores[i])) for i in top_idx]

        print("  Top-5 tags:")
        for tag, score in tags_scores:
            print(f"    {score:.3f}  {tag}")

        # Save annotated image
        annotated = image.copy().convert("RGBA")
        overlay = Image.new("RGBA", annotated.size, (0, 0, 0, 0))
        annotated = draw_tags(overlay, tags_scores)
        result = Image.alpha_composite(image.convert("RGBA"), annotated)
        result.convert("RGB").save(output_path)
    else:
        image.save(output_path)

    print(f"  Saved: {output_path}")
    print(f"  Total: {time.perf_counter() - t0:.3f}s")


def demo(model, image, output, **kwargs):
    """Entry point for programmatic demo."""
    run_inference(model, image, output)


def main():
    parser = argparse.ArgumentParser(description="OpenCLIP embedding demo")
    parser.add_argument("--model", required=True, help="Path to model.onnx")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    demo(args.model, args.image, args.output)


if __name__ == "__main__":
    main()
