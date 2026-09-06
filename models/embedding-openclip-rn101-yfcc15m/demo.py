"""Demo: compute the image embedding and write it out as JSON.

There is nothing to draw on an image for an embedding model, so the demo
writes the vector itself. The reported norm is the useful check: the export
bakes L2 normalisation into the graph, so it must come back as 1.0.
"""

import argparse
import json
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

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


def run_inference(model_path, image_path, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    t0 = time.perf_counter()

    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    t_load = time.perf_counter()

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    orig_w, orig_h = image.size
    input_tensor = preprocess(image)
    t_pre = time.perf_counter()

    (embedding,) = session.run(None, {"image": input_tensor})
    t_inf = time.perf_counter()

    vector = embedding[0].astype(np.float32)
    norm = float(np.linalg.norm(vector))

    print(f"  Image: {orig_w}x{orig_h}")
    print(f"  Embedding: dim={vector.shape[0]}, norm={norm:.4f}")
    print(f"  Load: {t_load - t0:.3f}s  Preprocess: {t_pre - t_load:.3f}s  Inference: {t_inf - t_pre:.3f}s")

    with open(output_path, "w") as f:
        json.dump(
            {
                "image": os.path.basename(image_path),
                "dim": int(vector.shape[0]),
                "norm": norm,
                "embedding": [float(v) for v in vector],
            },
            f,
            indent=2,
        )

    print(f"  Saved: {output_path}")
    print(f"  Total: {time.perf_counter() - t0:.3f}s")


def demo(model, image, output, **kwargs):
    """Entry point for programmatic demo."""
    run_inference(model, image, output)


def main():
    parser = argparse.ArgumentParser(description="OpenCLIP embedding demo")
    parser.add_argument("--model", required=True, help="Path to model.onnx")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    demo(args.model, args.image, args.output)


if __name__ == "__main__":
    main()
