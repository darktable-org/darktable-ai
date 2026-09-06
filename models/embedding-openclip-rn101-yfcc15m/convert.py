"""Export the OpenCLIP RN101 (YFCC15M) image encoder to ONNX.

Produces:
  model.onnx  – image encoder with baked-in CLIP normalization + L2 norm

Why RN101 + YFCC15M specifically: YFCC15M is the only widely-available CLIP
training dataset where every image was opt-in licensed by its author
(Flickr Creative Commons). The CLIP family's other training corpora
(LAION, WebLI, WIT, DataComp) are all web scrapes without per-image
consent. For darktable's open-source compliance criteria this matters
more than the ImageNet zero-shot benchmark gap.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip


# ---------------------------------------------------------------------------
# ONNX wrapper
# ---------------------------------------------------------------------------

class ImageEncoderOnnx(nn.Module):
    """Wraps OpenCLIP visual encoder for ONNX export.

    Bakes in CLIP-specific normalization and L2 normalization so the
    caller only needs to provide [0, 1] float32 RGB input.

    Input:  image     – float32 [B, 3, 224, 224] in [0, 1]
    Output: embedding – float32 [B, 512] L2-normalized

    With fp16=True the visual backbone runs in FP16 (weights and
    activations) but the I/O boundary stays FP32: input cast happens
    after mean/std subtraction; output cast happens before L2 norm.
    """

    def __init__(self, model, fp16=False):
        super().__init__()
        self.visual = model.visual
        self.fp16 = fp16
        # OpenCLIP RN101 (yfcc15m) uses the OpenAI CLIP normalisation
        # constants – same as all other OpenCLIP variants
        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )
        if fp16:
            # cast the backbone (weights + buffers) to FP16. Mean/std
            # buffers stay FP32 so the input subtraction is done in FP32
            # before the downcast – avoids one round of precision loss.
            self.visual = self.visual.half()

    @torch.no_grad()
    def forward(self, image):
        x = (image - self.mean) / self.std
        if self.fp16:
            x = x.half()
        features = self.visual(x)
        if self.fp16:
            features = features.float()
        features = F.normalize(features, dim=-1)
        return features


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_image_encoder(model, output_path, opset, fp16=False):
    """Export image encoder to ONNX.

    For FP16 we cast the visual backbone to half() in PyTorch and let
    torch.onnx.export produce an FP16-internals graph directly. This is
    much faster than post-converting an FP32 ONNX file with
    onnxconverter-common (which scales badly on 130M-param graphs).
    The graph's IO stays FP32 because the casts happen inside the wrapper.
    """
    encoder = ImageEncoderOnnx(model, fp16=fp16)
    encoder.eval()

    # fixed seed so the ORT-vs-PyTorch diff print is reproducible across
    # runs and meaningful as a regression signal
    torch.manual_seed(0)
    dummy = torch.randn(1, 3, 224, 224)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    precision = "FP16" if fp16 else "FP32"
    print(f"Exporting image encoder ({precision} internals) to {output_path}...")
    # Legacy tracer rather than dynamo: ResNet has no attention ops to
    # confuse the tracer, and dynamo's BatchNorm handling crashes on
    # _native_batch_norm_legit_no_training (returns a tuple that the
    # ONNX type-inference path mishandles).
    torch.onnx.export(
        encoder,
        (dummy,),
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['image'],
        output_names=['embedding'],
        dynamic_axes={
            'image':     {0: 'batch'},
            'embedding': {0: 'batch'},
        },
        verbose=False,
    )

    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX checker passed.")

    # verify ONNX output matches PyTorch
    import onnxruntime as ort
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_out = session.run(None, {input_name: dummy.numpy()})[0]
    ref_out = encoder(dummy).numpy()
    diff = np.abs(ort_out - ref_out).max()
    print(f"ONNX vs PyTorch max diff ({precision}): {diff:.6f}")
    print(f"Output shape: {ort_out.shape}, norm: {np.linalg.norm(ort_out, axis=-1)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(output, opset=20, fp16=False):
    """Entry point for programmatic conversion."""
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print("Loading OpenCLIP RN101 (yfcc15m)...")
    # YFCC15M weights were trained with QuickGELU activation; the open_clip
    # factory defaults to standard GELU and only warns about the mismatch
    # – but the mismatch produces subtly wrong outputs. Force it on.
    model, _, preprocess = open_clip.create_model_and_transforms(
        "RN101", pretrained="yfcc15m", force_quick_gelu=True
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    export_image_encoder(model, output, opset, fp16=fp16)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Export OpenCLIP RN101 (YFCC15M) image encoder to ONNX"
    )
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=20, help="ONNX opset version")
    parser.add_argument("--fp16", action="store_true",
                        help="convert weights to FP16 after export (default: FP32)")
    args = parser.parse_args()

    convert(args.output, args.opset, fp16=args.fp16)


if __name__ == "__main__":
    main()
