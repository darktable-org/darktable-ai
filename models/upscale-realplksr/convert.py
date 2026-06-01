"""Export RealPLKSR to ONNX via Spandrel.

Spandrel auto-detects the RealPLKSR variant from the checkpoint's
state_dict and builds the network without needing to clone neosr or
the original PLKSR training framework. For a one-shot ONNX export
this is significantly easier than wiring up the training-time scaffolding.

RealPLKSR is a pure-CNN architecture with no window-attention constraints,
so any input multiple of 4 works. We trace at the deployment dim directly
to keep the captured dims consistent with the static shape — same pattern
the DAT-2 export uses, applied here for simplicity.
"""

import argparse
import os

import torch

try:
    import onnxconverter_common
    HAS_ONNX_CONVERTER = True
except ImportError:
    HAS_ONNX_CONVERTER = False

from spandrel import ImageModelDescriptor, ModelLoader


def _patch_plkconv_for_clean_export():
    """Force PLKConv2d to use its split+cat forward path (which is its
    training-mode branch) regardless of mode. The eval-mode branch uses
    in-place indexed assignment (`x[:, :idx] = conv(...)`), which
    torch.onnx exports as ScatterND — ONNX Runtime then prints a
    "may not be deterministic if indices are duplicated" warning for
    each block at load time. The split+cat path is numerically
    identical for contiguous-from-0 channels and exports as plain
    Split+Conv+Concat, with no warning.
    """
    from spandrel.architectures.PLKSR.__arch.RealPLKSR import PLKConv2d

    def forward_export(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = torch.split(x, [self.idx, x.size(1) - self.idx], dim=1)
        x1 = self.conv(x1)
        return torch.cat([x1, x2], dim=1)

    PLKConv2d.forward = forward_export


def export_to_onnx(model, output_path, scale, height=256, width=256,
                   dynamic_shapes=True, opset_version=20, fp16=False):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    import onnx

    if dynamic_shapes:
        trace_h, trace_w = 64, 64
        dynamic_axes = {
            'input':  {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'},
        }
    else:
        trace_h, trace_w = height, width
        dynamic_axes = None

    dummy_input = torch.randn(1, 3, trace_h, trace_w)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    print(f"Exported: {output_path} (traced at {trace_h}x{trace_w})")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX verification passed.")

    if not dynamic_shapes:
        print(f"  Static dims baked: "
              f"{height}x{width} -> {height * scale}x{width * scale}")

    if fp16:
        if not HAS_ONNX_CONVERTER:
            print("Warning: onnxconverter-common not installed. Skipping FP16 conversion.")
            return
        print("Converting to FP16...")
        from onnxconverter_common import float16
        fp16_model = float16.convert_float_to_float16(onnx_model)
        onnx.save(fp16_model, output_path)
        print(f"FP16 model saved to {output_path}")


def convert(checkpoint, output, scale, height=256, width=256,
            dynamic_shapes=True, opset=20, fp16=False, static=False):
    """Entry point for programmatic conversion."""
    scale = int(scale)

    _patch_plkconv_for_clean_export()

    print(f"Loading RealPLKSR model via Spandrel: {checkpoint}")
    descriptor = ModelLoader().load_from_file(checkpoint)
    if not isinstance(descriptor, ImageModelDescriptor):
        raise TypeError(
            f"expected ImageModelDescriptor, got {type(descriptor).__name__}")
    if descriptor.scale != scale:
        raise ValueError(
            f"checkpoint scale={descriptor.scale} does not match requested scale={scale}")

    model = descriptor.model
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {descriptor.architecture.id}")
    print(f"  Scale:        x{descriptor.scale}")
    print(f"  Parameters:   {param_count:,}")

    print("Exporting to ONNX...")
    export_to_onnx(model, output, scale,
                   height=height, width=width,
                   dynamic_shapes=dynamic_shapes and not static,
                   opset_version=opset, fp16=fp16)


def main():
    parser = argparse.ArgumentParser(description='Export RealPLKSR to ONNX via Spandrel')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scale', type=int, required=True, choices=[2, 3, 4])
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--width', type=int, default=256)
    parser.add_argument('--opset', type=int, default=20)
    parser.add_argument('--fp16', action='store_true',
                        help='convert weights to FP16 after export (default: FP32)')
    parser.add_argument('--static', action='store_true',
                        help='bake input height/width into the graph '
                             '(disables dynamic shape axes)')
    args = parser.parse_args()

    convert(args.checkpoint, args.output, args.scale,
            height=args.height, width=args.width,
            opset=args.opset, fp16=args.fp16, static=args.static)


if __name__ == '__main__':
    main()
