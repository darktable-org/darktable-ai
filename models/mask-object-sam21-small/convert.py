# Export SAM 2.1 image encoder and decoder (prompt encoder + mask decoder) to
# separate ONNX files:  encoder.onnx  and  decoder.onnx.
#
# Based on samexporter (https://github.com/vietanhdev/samexporter) with
# adaptations for SAM 2.1 and this repository's conventions.
#
# The encoder wrapper bakes conv_s0/conv_s1 from the mask decoder into the
# encoder so that high-res features are already projected before being passed
# to the decoder ONNX model.

import argparse
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from sam2.build_sam import build_sam2
from sam2.modeling.sam2_base import SAM2Base

try:
    import onnxruntime  # type: ignore

    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

try:
    import onnxsim

    onnxsim_exists = True
except ImportError:
    onnxsim_exists = False


class SAM2EncoderOnnxModel(nn.Module):
    """Wraps the SAM 2.1 image encoder for ONNX export.

    Runs the Hiera backbone + FPN neck, applies conv_s0/conv_s1 from the mask
    decoder to project high-res features, and adds no_mem_embed to the lowest
    resolution feature (image-only mode, no memory conditioning).

    Outputs three tensors:
        high_res_feats_0: (1, 32, 256, 256)
        high_res_feats_1: (1, 64, 128, 128)
        image_embed:      (1, 256, 64, 64)
    """

    def __init__(self, sam_model: SAM2Base):
        super().__init__()
        self.image_encoder = sam_model.image_encoder
        self.no_mem_embed = sam_model.no_mem_embed
        self.model = sam_model

    @torch.no_grad()
    def forward(self, image: torch.Tensor):
        backbone_out = self.image_encoder(image)

        # Apply high-res feature projections from mask decoder
        backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
            backbone_out["backbone_fpn"][0]
        )
        backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
            backbone_out["backbone_fpn"][1]
        )

        feature_maps = backbone_out["backbone_fpn"][
            -self.model.num_feature_levels :
        ]
        vision_pos_embeds = backbone_out["vision_pos_enc"][
            -self.model.num_feature_levels :
        ]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]

        # Flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        # Add no-memory embedding to lowest resolution feature (image-only mode)
        vision_feats[-1] = vision_feats[-1] + self.no_mem_embed

        # Reshape back to NxCxHxW
        feats = [
            feat.permute(1, 2, 0).reshape(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], feat_sizes[::-1])
        ][::-1]

        return feats[0], feats[1], feats[2]


class SAM2DecoderOnnxModel(nn.Module):
    """Wraps the SAM 2.1 prompt encoder + mask decoder for ONNX export.

    Re-implements prompt encoding inline (no control flow) to enable tracing.

    Inputs:
        image_embed:      (1, 256, 64, 64)
        high_res_feats_0: (1, 32, 256, 256)
        high_res_feats_1: (1, 64, 128, 128)
        point_coords:     (B, N, 2)
        point_labels:     (B, N)
        mask_input:       (B, 1, 256, 256)
        has_mask_input:   (1,)

    Outputs:
        masks:           (B, num_masks, 1024, 1024)
        iou_predictions: (B, num_masks)
        low_res_masks:   (B, num_masks, 256, 256)
    """

    def __init__(self, sam_model: SAM2Base, multimask_output: bool):
        super().__init__()
        self.mask_decoder = sam_model.sam_mask_decoder
        self.prompt_encoder = sam_model.sam_prompt_encoder
        self.model = sam_model
        self.img_size = sam_model.image_size
        self.multimask_output = multimask_output

    @torch.no_grad()
    def forward(
        self,
        image_embed: torch.Tensor,
        high_res_feats_0: torch.Tensor,
        high_res_feats_1: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        mask_input: torch.Tensor,
        has_mask_input: torch.Tensor,
    ):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        high_res_feats = [high_res_feats_0, high_res_feats_1]

        low_res_masks, iou_predictions, _, _ = self.mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
            repeat_image=False,
            high_res_features=high_res_feats,
        )

        # Upscale from 256x256 to full 1024x1024 resolution
        masks = F.interpolate(low_res_masks, size=(1024, 1024), mode="bilinear", align_corners=False)

        if self.multimask_output:
            masks = masks[:, 1:, :, :]
            iou_predictions = iou_predictions[:, 1:]
            low_res_masks = low_res_masks[:, 1:, :, :]
        else:
            masks, iou_predictions = (
                self.mask_decoder._dynamic_multimask_via_stability(
                    masks, iou_predictions
                )
            )

        masks = torch.clamp(masks, -32.0, 32.0)

        return masks, iou_predictions, low_res_masks

    def _embed_points(
        self, point_coords: torch.Tensor, point_labels: torch.Tensor
    ) -> torch.Tensor:
        point_coords = point_coords + 0.5

        # Append padding point
        padding_point = torch.zeros(
            (point_coords.shape[0], 1, 2), device=point_coords.device
        )
        padding_label = -torch.ones(
            (point_labels.shape[0], 1), device=point_labels.device
        )
        point_coords = torch.cat([point_coords, padding_point], dim=1)
        point_labels = torch.cat([point_labels, padding_label], dim=1)

        # Normalize to [0, 1]
        point_coords[:, :, 0] = point_coords[:, :, 0] / self.img_size
        point_coords[:, :, 1] = point_coords[:, :, 1] / self.img_size

        point_embedding = self.prompt_encoder.pe_layer._pe_encoding(
            point_coords
        )
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = (
            point_embedding
            + self.prompt_encoder.not_a_point_embed.weight
            * (point_labels == -1)
        )

        for i in range(self.prompt_encoder.num_point_embeddings):
            point_embedding = (
                point_embedding
                + self.prompt_encoder.point_embeddings[i].weight
                * (point_labels == i)
            )

        return point_embedding

    def _embed_masks(
        self, input_mask: torch.Tensor, has_mask_input: torch.Tensor
    ) -> torch.Tensor:
        mask_embedding = has_mask_input * self.prompt_encoder.mask_downscaling(
            input_mask
        )
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding


def to_numpy(tensor):
    return tensor.cpu().numpy()


def run_encoder_export(sam_model, output, opset):
    """Export the image encoder to ONNX."""
    encoder = SAM2EncoderOnnxModel(sam_model)
    encoder.eval()

    dummy_image = torch.randn(1, 3, 1024, 1024, dtype=torch.float)

    # Run forward pass to get real outputs for decoder export
    high_res_feats_0, high_res_feats_1, image_embed = encoder(dummy_image)

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
            output_names=["high_res_feats_0", "high_res_feats_1", "image_embed"],
            dynamo=False,
        )

    simplify_model(output)
    verify_model(output, {"image": to_numpy(dummy_image)}, "Encoder")

    return high_res_feats_0, high_res_feats_1, image_embed


def run_decoder_export(
    sam_model, output, opset,
    high_res_feats_0, high_res_feats_1, image_embed,
    multimask_output=True,
):
    """Export the prompt encoder + mask decoder to ONNX."""
    decoder = SAM2DecoderOnnxModel(sam_model, multimask_output=multimask_output)
    decoder.eval()

    embed_size = (
        sam_model.image_size // sam_model.backbone_stride,
        sam_model.image_size // sam_model.backbone_stride,
    )
    mask_input_size = [4 * x for x in embed_size]

    dummy_inputs = {
        "image_embed": image_embed,
        "high_res_feats_0": high_res_feats_0,
        "high_res_feats_1": high_res_feats_1,
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=1, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float),
    }

    # Verify forward pass works
    _ = decoder(**dummy_inputs)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output, "wb") as f:
            print(f"Exporting decoder to {output}...")
            torch.onnx.export(
                decoder,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=["masks", "iou_predictions", "low_res_masks"],
                dynamic_axes=dynamic_axes,
                dynamo=False,
            )

    simplify_model(output)
    verify_model(output, {k: to_numpy(v) for k, v in dummy_inputs.items()}, "Decoder")


def simplify_model(model_path):
    """Simplify ONNX model graph if onnxsim is available."""
    if not onnxsim_exists:
        print("Warning: onnx-simplifier not installed, skipping.")
        return

    import onnx

    print("Simplifying ONNX graph...")
    onnx_model = onnx.load(model_path)
    model_simp, check = onnxsim.simplify(onnx_model)
    if check:
        onnx.save(model_simp, model_path)
        print("Graph simplification passed.")
    else:
        print("Warning: simplified model failed validation, keeping original.")


def verify_model(model_path, dummy_inputs, name):
    """Verify exported ONNX model with ONNXRuntime."""
    if not onnxruntime_exists:
        return

    providers = ["CPUExecutionProvider"]
    ort_session = onnxruntime.InferenceSession(model_path, providers=providers)
    _ = ort_session.run(None, dummy_inputs)
    print(f"{name} has successfully been run with ONNXRuntime.")


def convert(model_cfg, checkpoint, output_dir, opset=17):
    """Entry point for programmatic conversion."""
    os.makedirs(output_dir, exist_ok=True)
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    decoder_path = os.path.join(output_dir, "decoder.onnx")

    print(f"Loading model ({model_cfg})...")
    sam_model = build_sam2(model_cfg, checkpoint, device="cpu")

    high_res_feats_0, high_res_feats_1, image_embed = run_encoder_export(
        sam_model,
        output=encoder_path,
        opset=opset,
    )

    run_decoder_export(
        sam_model,
        output=decoder_path,
        opset=opset,
        high_res_feats_0=high_res_feats_0,
        high_res_feats_1=high_res_feats_1,
        image_embed=image_embed,
    )

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Export SAM 2.1 encoder and decoder to separate ONNX files."
    )
    parser.add_argument("--model-cfg", type=str, required=True,
                        help="Hydra config file (e.g. configs/sam2.1/sam2.1_hiera_s.yaml).")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the SAM 2.1 model checkpoint (.pt).")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for encoder.onnx and decoder.onnx.")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version (default: 17).")
    args = parser.parse_args()

    convert(args.model_cfg, args.checkpoint, args.output_dir, opset=args.opset)


if __name__ == "__main__":
    main()
