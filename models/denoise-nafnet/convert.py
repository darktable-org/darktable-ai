"""Export NAFNet denoiser to ONNX.

By default exports with dynamic spatial axes. Pass --static (or
`static: true` from model.yaml) to bake the input height/width into the
graph so JIT-compiling EPs (CoreML, MIGraphX) only pay the compile cost
once.

Clone NAFNet first: git clone https://github.com/megvii-research/NAFNet
Then run: pip install -r requirements.txt && python setup.py develop --no_cuda_ext
"""

import argparse
import os
import sys
import yaml
from collections import OrderedDict
from unittest.mock import MagicMock

import torch
import torch.onnx

# Mock lzma if missing (common issue on some mac python builds)
try:
    import lzma
except ImportError:
    sys.modules['lzma'] = MagicMock()
    sys.modules['_lzma'] = MagicMock()

try:
    import onnxconverter_common
    HAS_ONNX_CONVERTER = True
except ImportError:
    HAS_ONNX_CONVERTER = False

from basicsr.models.archs.NAFNet_arch import NAFNet


def load_nafnet_model(config_path, checkpoint_path):
    """Load NAFNet model from config and checkpoint."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    network_g = config['network_g']
    model = NAFNet(
        img_channel=network_g.get('img_channel', 3),
        width=network_g.get('width', 64),
        middle_blk_num=network_g.get('middle_blk_num', 12),
        enc_blk_nums=network_g.get('enc_blk_nums', [2, 2, 4, 8]),
        dec_blk_nums=network_g.get('dec_blk_nums', [2, 2, 2, 2])
    )

    # weights_only=False: PyTorch 2.6 flipped this default to True, which
    # rejects the NAFNet checkpoint (it pickles non-tensor objects). The
    # checkpoint is a trusted file we ship, so full unpickling is safe.
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    else:
        state_dict = checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


def export_to_onnx(model, output_path, input_height=768, input_width=768,
                   dynamic_shapes=True, opset_version=20, fp16=False):
    """Export NAFNet model to ONNX format."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    dummy_input = torch.randn(1, 3, input_height, input_width)

    dynamic_axes = None
    if dynamic_shapes:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }

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
        verbose=False
    )

    print(f"Model exported to {output_path}")

    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("FP32 ONNX model verification passed!")

    if fp16:
        if not HAS_ONNX_CONVERTER:
            print("Warning: onnxconverter-common not installed. Skipping FP16 conversion.")
            return
        print("Converting to FP16...")
        from onnxconverter_common import float16
        fp16_model = float16.convert_float_to_float16(onnx_model)
        onnx.save(fp16_model, output_path)
        print(f"FP16 model saved to {output_path}")


def convert(config, checkpoint, output="nafnet.onnx", height=768, width=768,
            dynamic_shapes=True, opset=20, fp16=False, static=False):
    """Entry point for programmatic conversion."""
    print("Loading NAFNet model...")
    model = load_nafnet_model(config, checkpoint)

    print("Exporting to ONNX...")
    export_to_onnx(model, output, input_height=height, input_width=width,
                   dynamic_shapes=dynamic_shapes and not static,
                   opset_version=opset, fp16=fp16)


def main():
    parser = argparse.ArgumentParser(description='Export NAFNet to ONNX')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='nafnet.onnx')
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--width', type=int, default=768)
    parser.add_argument('--opset', type=int, default=20)
    parser.add_argument('--fp16', action='store_true',
                        help='convert weights to FP16 after export (default: FP32)')
    parser.add_argument('--static', action='store_true',
                        help='bake input height/width into the graph '
                             '(disables dynamic shape axes)')
    args = parser.parse_args()

    convert(args.config, args.checkpoint, args.output,
            height=args.height, width=args.width,
            opset=args.opset, fp16=args.fp16, static=args.static)


if __name__ == '__main__':
    main()
