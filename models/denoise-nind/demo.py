# Demo: run the NIND UNet denoiser ONNX model on an image.
#
# Usage:
#   python3 models/denoise-nind/demo.py \
#       --model output/denoise-nind/model.onnx \
#       --image images/example_01.jpg \
#       --output models/denoise-nind/output/example_01.png

import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


def main():
    parser = argparse.ArgumentParser(description="NIND UNet ONNX denoising demo.")
    parser.add_argument("--model", type=str, required=True, help="Path to model.onnx")
    parser.add_argument("--image", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output PNG path")
    parser.add_argument("--max-size", type=int, default=1024,
                        help="Downscale longest edge to this (0 = full resolution)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    t0 = time.perf_counter()

    print(f"Loading model: {args.model}")
    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    input_is_fp16 = model_input.type == "tensor(float16)"
    t_model = time.perf_counter()
    print(f"  Input name:    {input_name}")
    print(f"  FP16:          {input_is_fp16}")
    print(f"  Load model:    {t_model - t0:.3f}s")

    print(f"Loading image: {args.image}")
    image = Image.open(args.image)
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    t_image = time.perf_counter()
    print(f"  Original size: {image.size[0]}x{image.size[1]}")
    if args.max_size > 0:
        image.thumbnail((args.max_size, args.max_size), Image.LANCZOS)
        print(f"  Resized to:    {image.size[0]}x{image.size[1]}")
    print(f"  Load image:    {t_image - t_model:.3f}s")

    # Preprocess: RGB [0, 1], BCHW
    arr = np.array(image).astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    if input_is_fp16:
        arr = arr.astype(np.float16)

    print("Running inference...")
    [output] = session.run(None, {input_name: arr})
    t_infer = time.perf_counter()
    print(f"  Inference:     {t_infer - t_image:.3f}s")

    # Postprocess: BCHW -> HWC, clip, uint8
    output = output[0].astype(np.float32).transpose(1, 2, 0)
    output = np.clip(output, 0, 1)
    output = (output * 255).astype(np.uint8)

    Image.fromarray(output).save(args.output)
    print(f"Saved: {args.output}")
    print(f"  Total:         {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
