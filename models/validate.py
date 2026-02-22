#!/usr/bin/env python3
"""Validate ONNX model output: check files exist, load, print I/O metadata."""

import argparse
import json
import os
import sys


def validate_onnx(path, label="model"):
    """Load an ONNX model and print its input/output metadata."""
    import onnxruntime as ort

    if not os.path.isfile(path):
        print(f"  FAIL: {label} not found: {path}")
        return False

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  {label}: {os.path.basename(path)} ({size_mb:.1f} MB)")

    try:
        session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"  FAIL: cannot load {label}: {e}")
        return False

    print("    Inputs:")
    for inp in session.get_inputs():
        print(f"      {inp.name}: {inp.shape} ({inp.type})")
    print("    Outputs:")
    for out in session.get_outputs():
        print(f"      {out.name}: {out.shape} ({out.type})")

    return True


def validate_config(path):
    """Validate config.json exists and has required fields."""
    if not os.path.isfile(path):
        print(f"  FAIL: config.json not found: {path}")
        return False

    with open(path) as f:
        config = json.load(f)

    required = ["id", "name", "description", "task", "backend", "version"]
    missing = [k for k in required if k not in config]
    if missing:
        print(f"  FAIL: config.json missing fields: {', '.join(missing)}")
        return False

    print(f"  config.json: OK (task={config['task']}, tiling={config.get('tiling', False)})")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate ONNX model output")
    parser.add_argument("--model-dir", required=True,
                        help="Path to output/<model_id>/")
    parser.add_argument("--model-type", default="single",
                        choices=["single", "split"],
                        help="'single' for model.onnx, 'split' for encoder+decoder")
    args = parser.parse_args()

    model_dir = args.model_dir
    print(f"Validating: {os.path.basename(model_dir)}")

    ok = True
    ok &= validate_config(os.path.join(model_dir, "config.json"))

    if args.model_type == "split":
        ok &= validate_onnx(os.path.join(model_dir, "encoder.onnx"), "encoder")
        ok &= validate_onnx(os.path.join(model_dir, "decoder.onnx"), "decoder")
    else:
        ok &= validate_onnx(os.path.join(model_dir, "model.onnx"), "model")

    if ok:
        print("  Result: PASS")
    else:
        print("  Result: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
