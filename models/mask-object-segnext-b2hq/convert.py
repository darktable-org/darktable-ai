# Export SegNext encoder and decoder to separate ONNX files:
#   encoder.onnx  — image → normalize → ViT backbone → image features
#   decoder.onnx  — image_feats + clicks + prev_mask → mask
#
# The decoder includes an ONNX-friendly reimplementation of DistMaps
# (distance-based click encoding) to keep the interface consistent with
# SAM-style models: the caller passes raw point coordinates and labels.
#
# Based on SegNext (https://github.com/uncbiag/SegNext).

import argparse
import os
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add SegNext source to path for model loading
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "SegNext", "segnext"))

from isegm.utils.serialization import load_model

try:
    import onnxruntime

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

try:
    import onnxsim

    onnxsim_exists = True
except ImportError:
    onnxsim_exists = False

parser = argparse.ArgumentParser(
    description="Export SegNext encoder and decoder to separate ONNX files."
)
parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="Path to the SegNext model checkpoint (.pth).",
)
parser.add_argument(
    "--output-dir", type=str, required=True,
    help="Output directory for encoder.onnx and decoder.onnx.",
)
parser.add_argument(
    "--opset", type=int, default=17,
    help="ONNX opset version (default: 17).",
)


# ---------------------------------------------------------------------------
# Encoder wrapper
# ---------------------------------------------------------------------------

class SegNextEncoderOnnx(nn.Module):
    """Wraps the SegNext image encoder for ONNX export.

    Applies ImageNet normalization then runs the ViT backbone.

    Input:
        image: (1, 3, 1024, 1024) — RGB float32 in [0, 1]
    Output:
        image_feats: (1, 768, 64, 64)
    """

    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.register_buffer("norm_mean", model.normalization.mean.clone())
        self.register_buffer("norm_std", model.normalization.std.clone())

    @torch.no_grad()
    def forward(self, image):
        x = (image - self.norm_mean) / self.norm_std
        image_feats = self.backbone(x, keep_shape=True)
        return image_feats


# ---------------------------------------------------------------------------
# Decoder wrapper
# ---------------------------------------------------------------------------

class SegNextDecoderOnnx(nn.Module):
    """Wraps the SegNext decoder for ONNX export.

    Computes distance maps from click coordinates, encodes prompts,
    fuses with image features via self-attention, and produces a mask.

    Inputs:
        image_feats:  (1, 768, 64, 64) — from encoder
        point_coords: (1, N, 2) — click (x, y) in 1024x1024 coordinates
        point_labels: (1, N) — 1=foreground, 0=background, -1=padding
        prev_mask:    (1, 1, 1024, 1024) — previous mask (zeros if none)
    Output:
        mask: (1, 1, 1024, 1024) — segmentation logits
    """

    def __init__(self, model):
        super().__init__()
        self.visual_prompts_encoder = model.visual_prompts_encoder
        self.fusion_blocks = model.fusion_blocks
        self.neck = model.neck
        self.head = model.head

        self.target_length = model.target_length
        self.patch_size = model.visual_prompts_encoder.patch_size[0]
        self.embed_dim = model.backbone.embed_dim
        self.norm_radius = float(model.dist_maps.norm_radius)
        self.spatial_scale = float(model.dist_maps.spatial_scale)
        self.use_disks = model.dist_maps.use_disks

        # Pre-compute coordinate grids as constant buffers
        H = W = self.target_length
        rows = torch.arange(H, dtype=torch.float32)
        cols = torch.arange(W, dtype=torch.float32)
        grid_r, grid_c = torch.meshgrid(rows, cols, indexing="ij")
        self.register_buffer("grid_r", grid_r[None, None])  # [1, 1, H, W]
        self.register_buffer("grid_c", grid_c[None, None])

    @torch.no_grad()
    def forward(self, image_feats, point_coords, point_labels, prev_mask):
        # Distance maps from click coordinates
        dist_maps = self._compute_dist_maps(point_coords, point_labels)

        # Prompt encoding: prev_mask (1ch) + pos_dist (1ch) + neg_dist (1ch)
        prompt_mask = torch.cat([prev_mask, dist_maps], dim=1)
        prompt_feats = self.visual_prompts_encoder(prompt_mask)  # [B, N, C]

        B, N, C = prompt_feats.shape
        H = W = self.target_length // self.patch_size
        prompt_feats = prompt_feats.transpose(1, 2).contiguous().reshape(B, C, H, W)

        # Self-attention fusion
        fused = image_feats + prompt_feats
        B, C, H, W = fused.shape
        fused = fused.permute(0, 2, 3, 1).contiguous().reshape(B, H * W, C)
        fused = self.fusion_blocks(fused)
        fused = fused.transpose(1, 2).contiguous().reshape(B, C, H, W)

        # FPN neck + segmentation head
        pyramid = self.neck(fused)
        seg_logits = self.head(pyramid)

        # Upscale to full resolution
        seg_logits = F.interpolate(
            seg_logits, size=self.target_length,
            mode="bilinear", align_corners=True,
        )

        return seg_logits

    def _compute_dist_maps(self, point_coords, point_labels):
        """ONNX-friendly distance map computation.

        Reimplements DistMaps.get_coord_features() without in-place ops
        or boolean indexing.  Supports both disk mode (binary 0/1 maps)
        and continuous mode (tanh-normalized distance).

        Args:
            point_coords: [B, N, 2] — (x, y) = (col, row) at 1024 scale
            point_labels: [B, N] — 1=foreground, 0=background, -1=padding
        Returns:
            [B, 2, H, W] — (positive_dist, negative_dist)
        """
        # Map (x, y) → (row, col) for grid computation
        pr = point_coords[:, :, 1:2].unsqueeze(-1)  # row [B, N, 1, 1]
        pc = point_coords[:, :, 0:1].unsqueeze(-1)  # col [B, N, 1, 1]

        # Squared Euclidean distance from each point to every pixel
        dr = self.grid_r - pr
        dc = self.grid_c - pc
        if not self.use_disks:
            r = self.norm_radius * self.spatial_scale
            dr = dr / r
            dc = dc / r
        dist_sq = dr * dr + dc * dc  # [B, N, H, W]

        # Separate positive (label=1) and negative (label=0) via masking;
        # invalid/padding points (label=-1) get large distance so they
        # don't affect the per-group minimum.
        pos_mask = (point_labels == 1).float().unsqueeze(-1).unsqueeze(-1)
        neg_mask = (point_labels == 0).float().unsqueeze(-1).unsqueeze(-1)

        pos_dist = dist_sq + (1 - pos_mask) * 1e6
        neg_dist = dist_sq + (1 - neg_mask) * 1e6

        pos_dist = pos_dist.min(dim=1, keepdim=True)[0]  # [B, 1, H, W]
        neg_dist = neg_dist.min(dim=1, keepdim=True)[0]

        result = torch.cat([pos_dist, neg_dist], dim=1)  # [B, 2, H, W]

        if self.use_disks:
            # Binary disk: 1.0 within radius, 0.0 outside
            radius_sq = (self.norm_radius * self.spatial_scale) ** 2
            result = (result <= radius_sq).float()
        else:
            # Continuous: tanh-normalized distance
            result = torch.tanh(2 * torch.sqrt(result))

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_numpy(tensor):
    return tensor.cpu().numpy()


def load_segnext_model(checkpoint_path):
    """Load SegNext model from checkpoint."""
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model = load_model(state_dict["config"])
    model.load_state_dict(state_dict["state_dict"], strict=True)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model


def simplify_model(model_path):
    """Simplify ONNX model graph if onnxsim is available."""
    if not onnxsim_exists:
        print("  onnx-simplifier not installed, skipping.")
        return

    import onnx

    print("  Simplifying ONNX graph...")
    onnx_model = onnx.load(model_path)
    model_simp, check = onnxsim.simplify(onnx_model)
    if check:
        onnx.save(model_simp, model_path)
        print("  Simplification passed.")
    else:
        print("  Warning: simplified model failed validation, keeping original.")


def verify_model(model_path, dummy_inputs, name):
    """Verify exported ONNX model with ONNXRuntime."""
    if not onnxruntime_exists:
        return

    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(model_path, providers=providers)
    _ = ort_session.run(None, dummy_inputs)
    print(f"  {name} verified with ONNXRuntime.")


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def run_encoder_export(model, output, opset):
    """Export the image encoder to ONNX."""
    encoder = SegNextEncoderOnnx(model)
    encoder.eval()

    dummy_image = torch.randn(1, 3, 1024, 1024)
    image_feats = encoder(dummy_image)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting encoder to {output}...")
        torch.onnx.export(
            encoder,
            dummy_image,
            output,
            export_params=True,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["image_feats"],
            dynamo=False,
        )

    simplify_model(output)
    verify_model(output, {"image": to_numpy(dummy_image)}, "Encoder")

    return image_feats


def run_decoder_export(model, output, opset, image_feats):
    """Export the prompt decoder to ONNX."""
    decoder = SegNextDecoderOnnx(model)
    decoder.eval()

    num_points = 5
    dummy_inputs = {
        "image_feats": image_feats,
        "point_coords": torch.randint(0, 1024, (1, num_points, 2), dtype=torch.float),
        "point_labels": torch.ones(1, num_points, dtype=torch.float),
        "prev_mask": torch.zeros(1, 1, 1024, 1024, dtype=torch.float),
    }

    # Verify forward pass works before export
    _ = decoder(**dummy_inputs)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting decoder to {output}...")
        torch.onnx.export(
            decoder,
            tuple(dummy_inputs.values()),
            output,
            export_params=True,
            verbose=False,
            opset_version=opset,
            do_constant_folding=True,
            input_names=list(dummy_inputs.keys()),
            output_names=["mask"],
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )

    simplify_model(output)
    verify_model(
        output,
        {k: to_numpy(v) for k, v in dummy_inputs.items()},
        "Decoder",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    encoder_path = os.path.join(args.output_dir, "encoder.onnx")
    decoder_path = os.path.join(args.output_dir, "decoder.onnx")

    print(f"Loading model from {args.checkpoint}...")
    model = load_segnext_model(args.checkpoint)

    image_feats = run_encoder_export(model, encoder_path, args.opset)
    run_decoder_export(model, decoder_path, args.opset, image_feats)

    print("Done!")
