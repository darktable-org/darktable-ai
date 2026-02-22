# ------------------------------------------------------------------------
# Darktable AI Denoise - ONNX Demo
# ------------------------------------------------------------------------

import argparse
import sys
import os
import cv2
import numpy as np
import onnxruntime as ort
import time

def run_inference(model_path, input_path, output_path, max_size=1024):
    # Check inputs
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input image not found at {input_path}")

    start_time = time.time()
    print(f"Loading ONNX model: {model_path}")
    try:
        session = ort.InferenceSession(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
        
    model_input = session.get_inputs()[0]
    input_name = model_input.name
    input_is_fp16 = model_input.type == 'tensor(float16)'

    # 1. Read image
    print(f"Reading image: {input_path}")
    img = cv2.imread(input_path) # BGR
    if img is None:
        raise ValueError(f"Could not read image: {input_path}")
    h, w = img.shape[:2]
    print(f"  Original size: {w}x{h}")
    if max_size > 0 and max(h, w) > max_size:
        scale = max_size / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        print(f"  Resized to:    {img.shape[1]}x{img.shape[0]}")

    # 2. Preprocessing
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1] and float32
    img = img.astype(np.float32) / 255.0
    
    # HWC to CHW
    img = img.transpose(2, 0, 1)
    
    # Add batch dimension: BCHW
    img = img[None, :, :, :]

    # Cast to FP16 if model expects it
    if input_is_fp16:
        img = img.astype(np.float16)

    # 3. Run Inference
    print("Running inference...")
    try:
        output = session.run(None, {input_name: img})[0]
    except Exception as e:
        print(f"Inference failed: {e}")
        sys.exit(1)
        
    # 4. Postprocessing
    output = output[0].astype(np.float32) # Remove batch dimension: CHW
    
    # CHW to HWC
    output = output.transpose(1, 2, 0)
    
    # Clip to [0, 1]
    output = np.clip(output, 0, 1)
    
    # Float to uint8
    output = (output * 255.0).astype(np.uint8)
    
    # RGB to BGR
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    # Save result
    print(f"Saving result to: {output_path}")
    cv2.imwrite(output_path, output)
    print("Done.")
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.4f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Run NAFNet ONNX Demo')
    parser.add_argument('--model', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--output', type=str, required=True, help='Path to output image')
    parser.add_argument('--max-size', type=int, default=1024,
                        help='Downscale longest edge to this (0 = full resolution)')

    args = parser.parse_args()
    
    run_inference(args.model, args.image, args.output, max_size=args.max_size)

if __name__ == '__main__':
    main()
