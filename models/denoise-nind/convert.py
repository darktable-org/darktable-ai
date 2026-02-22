"""Export NIND UNet denoiser to ONNX format.

Uses UNet from the cloned nind-denoise repository:
https://github.com/trougnouf/nind-denoise
"""

import argparse
import os
import sys

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "nind-denoise", "src"))

from nind_denoise.networks.ThirdPartyNets import UNet

try:
    import onnxconverter_common
    HAS_ONNX_CONVERTER = True
except ImportError:
    HAS_ONNX_CONVERTER = False


def load_model(checkpoint_path):
    model = UNet(n_channels=3, n_classes=3)
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def export_to_onnx(model, output_path, input_height=256, input_width=256,
                   dynamic_shapes=True, opset_version=11, fp16=False):
    dummy_input = torch.randn(1, 3, input_height, input_width)

    dynamic_axes = None
    if dynamic_shapes:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    print(f"Model exported to {output_path}")

    import onnx
    import onnxsim
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")

    print("Simplifying model...")
    onnx_model, ok = onnxsim.simplify(onnx_model)
    if ok:
        onnx.save(onnx_model, output_path)
        print("Model simplified successfully")
    else:
        print("Warning: simplification failed, using unsimplified model")

    if fp16:
        if not HAS_ONNX_CONVERTER:
            print("Warning: onnxconverter-common not installed. Skipping FP16 conversion.")
            return
        print("Converting to FP16...")
        from onnxconverter_common import float16
        fp16_model = float16.convert_float_to_float16(onnx_model)
        onnx.save(fp16_model, output_path)
        print(f"FP16 model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export NIND UNet to ONNX")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to .pt state dict checkpoint")
    parser.add_argument("--output", type=str, default="model.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--height", type=int, default=256,
                        help="Input height for tracing")
    parser.add_argument("--width", type=int, default=256,
                        help="Input width for tracing")
    parser.add_argument("--static", action="store_true",
                        help="Use static input shapes")
    parser.add_argument("--opset", type=int, default=11,
                        help="ONNX opset version")
    parser.add_argument("--fp16", action="store_true",
                        help="Export in half precision (FP16)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print("Loading NIND UNet model...")
    model = load_model(args.checkpoint)

    print("Exporting to ONNX...")
    export_to_onnx(
        model,
        args.output,
        input_height=args.height,
        input_width=args.width,
        dynamic_shapes=not args.static,
        opset_version=args.opset,
        fp16=args.fp16,
    )


if __name__ == "__main__":
    main()
