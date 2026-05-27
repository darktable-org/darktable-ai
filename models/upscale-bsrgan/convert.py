"""Export BSRGAN RRDBNet to ONNX.

By default exports with dynamic spatial axes. Pass --static (or
`static: true` from model.yaml) to bake the input height/width into the
graph so JIT-compiling EPs (CoreML, MIGraphX) only pay the compile cost
once.
"""

import argparse
import functools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import onnxconverter_common
    HAS_ONNX_CONVERTER = True
except ImportError:
    HAS_ONNX_CONVERTER = False


# ---------------------------------------------------------------------------
# RRDBNet architecture from https://github.com/cszn/BSRGAN
# (inlined to avoid cloning the repo)
# ---------------------------------------------------------------------------

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super().__init__()
        self.RDB1 = ResidualDenseBlock(nf, gc)
        self.RDB2 = ResidualDenseBlock(nf, gc)
        self.RDB3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4):
        super().__init__()
        self.sf = sf
        RRDB_block_f = functools.partial(RRDB, nf=nf, gc=gc)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = nn.Sequential(*[RRDB_block_f() for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        if self.sf == 4:
            self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        if self.sf == 4:
            fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))
        return out


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def export_to_onnx(model, output_path, scale, height=256, width=256,
                   dynamic_shapes=True, opset_version=20, fp16=False):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    import onnx

    # Trace at a small dummy + dynamic_axes (cheap), then bake static
    # dims post-export if requested. Avoids OOM on BSRGAN's 23-RRDB
    # trace at the deployment dim on CI.
    trace_dim = 64
    dummy_input = torch.randn(1, 3, trace_dim, trace_dim)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input':  {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'},
        },
        verbose=False,
    )
    print(f"Exported: {output_path}")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX verification passed.")

    if not dynamic_shapes:
        from onnx.tools import update_model_dims
        from onnx import shape_inference
        onnx_model = update_model_dims.update_inputs_outputs_dims(
            onnx_model,
            {'input':  [1, 3, height, width]},
            {'output': [1, 3, height * scale, width * scale]})
        onnx_model = shape_inference.infer_shapes(onnx_model)
        onnx.save(onnx_model, output_path)
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

    print(f"Loading BSRGAN model (scale={scale})...")
    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=scale)
    # weights_only=False for consistency with the other converters; this
    # is a trusted checkpoint we ship. PyTorch 2.6 made weights_only=True
    # the default, which can reject checkpoints carrying non-tensor data.
    state_dict = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    print("Exporting to ONNX...")
    export_to_onnx(model, output, scale,
                   height=height, width=width,
                   dynamic_shapes=dynamic_shapes and not static,
                   opset_version=opset, fp16=fp16)


def main():
    parser = argparse.ArgumentParser(description='Export BSRGAN RRDBNet to ONNX')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scale', type=int, required=True, choices=[2, 4])
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
