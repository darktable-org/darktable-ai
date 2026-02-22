import os

import torch
import torch.onnx
import argparse
import yaml
from collections import OrderedDict

# Clone NAFNet first: git clone https://github.com/megvii-research/NAFNet
# Then run: pip install -r requirements.txt && python setup.py develop --no_cuda_ext

import sys
from unittest.mock import MagicMock

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
    """Load NAFNet model from config and checkpoint"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get network parameters from config
    network_g = config['network_g']
    
    # Create model
    model = NAFNet(
        img_channel=network_g.get('img_channel', 3),
        width=network_g.get('width', 64),
        middle_blk_num=network_g.get('middle_blk_num', 12),
        enc_blk_nums=network_g.get('enc_blk_nums', [2, 2, 4, 8]),
        dec_blk_nums=network_g.get('dec_blk_nums', [2, 2, 2, 2])
    )
    
    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    elif 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    return model


def export_to_onnx(model, output_path, input_height=256, input_width=256,
                   dynamic_shapes=True, opset_version=11, fp16=False):
    """Export NAFNet model to ONNX format"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, input_height, input_width)
    
    # Note: We export in FP32 first, then convert to FP16 if requested
    # because PyTorch CPU backend often doesn't support half precision ops
    
    # Define dynamic axes for variable input sizes
    if dynamic_shapes:
        dynamic_axes = {
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    else:
        dynamic_axes = None
    
    # Export to ONNX
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
    
    # Verify the exported model
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


def main():
    parser = argparse.ArgumentParser(description='Export NAFNet to ONNX')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--output', type=str, default='nafnet.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--height', type=int, default=256,
                        help='Input height for tracing')
    parser.add_argument('--width', type=int, default=256,
                        help='Input width for tracing')
    parser.add_argument('--static', action='store_true',
                        help='Use static input shapes')
    parser.add_argument('--opset', type=int, default=11,
                        help='ONNX opset version')
    parser.add_argument('--fp16', action='store_true',
                        help='Export in half precision (FP16)')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading NAFNet model...")
    model = load_nafnet_model(args.config, args.checkpoint)
    
    # Export to ONNX
    print("Exporting to ONNX...")
    export_to_onnx(
        model,
        args.output,
        input_height=args.height,
        input_width=args.width,
        dynamic_shapes=not args.static,
        opset_version=args.opset,
        fp16=args.fp16
    )


if __name__ == '__main__':
    main()