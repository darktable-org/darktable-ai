"""Export OpenCLIP ViT-B-32 image encoder to ONNX and pre-compute tag embeddings.

Produces:
  model.onnx  — image encoder with baked-in CLIP normalization + L2 norm
  tags.json   — pre-computed text embeddings for 86 hierarchical photo tags

Tag vocabulary is defined in tags.md (human-readable reference).
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip

# ---------------------------------------------------------------------------
# Tag vocabulary for zero-shot classification
# Hierarchical tags using "|" separator, matching darktable's tag system.
# Each entry is (tag, CLIP prompt).  See tags.md for the human-readable list.
# ---------------------------------------------------------------------------

TAG_VOCAB = [
    # genre (12)
    ("genre|landscape", "landscape photography"),
    ("genre|portrait", "portrait photography"),
    ("genre|street", "street photography"),
    ("genre|wildlife", "wildlife photography"),
    ("genre|macro", "macro photography"),
    ("genre|architecture", "architecture photography"),
    ("genre|food", "food photography"),
    ("genre|sports", "sports photography"),
    ("genre|event", "event photography"),
    ("genre|abstract", "abstract photography"),
    ("genre|still life", "still life photography"),
    ("genre|aerial", "aerial photography"),
    # subject|people (6)
    ("subject|people|person", "a photo of a person"),
    ("subject|people|couple", "a photo of a couple"),
    ("subject|people|group", "a photo of a group of people"),
    ("subject|people|child", "a photo of a child"),
    ("subject|people|baby", "a photo of a baby"),
    ("subject|people|elderly person", "a photo of an elderly person"),
    # subject|animal (8)
    ("subject|animal|dog", "a photo of a dog"),
    ("subject|animal|cat", "a photo of a cat"),
    ("subject|animal|bird", "a photo of a bird"),
    ("subject|animal|horse", "a photo of a horse"),
    ("subject|animal|insect", "a photo of an insect"),
    ("subject|animal|fish", "a photo of a fish"),
    ("subject|animal|reptile", "a photo of a reptile"),
    ("subject|animal|wild animal", "a photo of a wild animal"),
    # subject|nature (10)
    ("subject|nature|flower", "a photo of a flower"),
    ("subject|nature|tree", "a photo of a tree"),
    ("subject|nature|mountain", "a photo of a mountain"),
    ("subject|nature|waterfall", "a photo of a waterfall"),
    ("subject|nature|river", "a photo of a river"),
    ("subject|nature|lake", "a photo of a lake"),
    ("subject|nature|ocean", "a photo of the ocean"),
    ("subject|nature|cloud", "a photo of clouds"),
    ("subject|nature|rock", "a photo of rocks"),
    ("subject|nature|field", "a photo of a field"),
    # subject|vehicle (5)
    ("subject|vehicle|car", "a photo of a car"),
    ("subject|vehicle|bicycle", "a photo of a bicycle"),
    ("subject|vehicle|boat", "a photo of a boat"),
    ("subject|vehicle|train", "a photo of a train"),
    ("subject|vehicle|airplane", "a photo of an airplane"),
    # subject|structure (5)
    ("subject|structure|building", "a photo of a building"),
    ("subject|structure|bridge", "a photo of a bridge"),
    ("subject|structure|tower", "a photo of a tower"),
    ("subject|structure|statue", "a photo of a statue"),
    ("subject|structure|ruin", "a photo of a ruin"),
    # setting (8)
    ("setting|indoor", "an indoor photograph"),
    ("setting|outdoor", "an outdoor photograph"),
    ("setting|urban", "a photo taken in an urban setting"),
    ("setting|rural", "a photo taken in a rural setting"),
    ("setting|beach", "a photo taken at a beach"),
    ("setting|forest", "a photo taken in a forest"),
    ("setting|desert", "a photo taken in a desert"),
    ("setting|studio", "a studio photograph"),
    # lighting (8)
    ("lighting|sunrise", "a photo taken at sunrise"),
    ("lighting|sunset", "a photo taken at sunset"),
    ("lighting|golden hour", "a photo taken during golden hour"),
    ("lighting|blue hour", "a photo taken during blue hour"),
    ("lighting|night", "a photo taken at night"),
    ("lighting|backlit", "a backlit photograph"),
    ("lighting|silhouette", "a silhouette photograph"),
    ("lighting|low light", "a low light photograph"),
    # technique (8)
    ("technique|black and white", "a black and white photograph"),
    ("technique|long exposure", "a long exposure photograph"),
    ("technique|bokeh", "a photograph with bokeh"),
    ("technique|panorama", "a panoramic photograph"),
    ("technique|close-up", "a close-up photograph"),
    ("technique|wide angle", "a wide angle photograph"),
    ("technique|motion blur", "a photograph with motion blur"),
    ("technique|reflection", "a photograph with reflections"),
    # mood (6)
    ("mood|dramatic", "a dramatic photograph"),
    ("mood|peaceful", "a peaceful photograph"),
    ("mood|moody", "a moody photograph"),
    ("mood|vibrant", "a vibrant photograph"),
    ("mood|minimal", "a minimalist photograph"),
    ("mood|chaotic", "a chaotic photograph"),
    # weather (6)
    ("weather|sunny", "a photo taken in sunny weather"),
    ("weather|cloudy", "a photo taken in cloudy weather"),
    ("weather|rainy", "a photo taken in rainy weather"),
    ("weather|snowy", "a photo taken in snowy weather"),
    ("weather|foggy", "a photo taken in foggy weather"),
    ("weather|stormy", "a photo taken in stormy weather"),
    # season (4)
    ("season|spring", "a photo taken in spring"),
    ("season|summer", "a photo taken in summer"),
    ("season|autumn", "a photo taken in autumn"),
    ("season|winter", "a photo taken in winter"),
]


# ---------------------------------------------------------------------------
# ONNX wrapper
# ---------------------------------------------------------------------------

class ImageEncoderOnnx(nn.Module):
    """Wraps OpenCLIP visual encoder for ONNX export.

    Bakes in CLIP-specific normalization and L2 normalization so the caller
    only needs to provide [0, 1] float32 RGB input.

    Input:  image     — float32 [B, 3, 224, 224] in [0, 1]
    Output: embedding — float32 [B, 512] L2-normalized
    """

    def __init__(self, model):
        super().__init__()
        self.visual = model.visual
        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1),
        )

    @torch.no_grad()
    def forward(self, image):
        x = (image - self.mean) / self.std
        features = self.visual(x)
        features = F.normalize(features, dim=-1)
        return features


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_image_encoder(model, output_path, opset):
    """Export image encoder to ONNX."""
    encoder = ImageEncoderOnnx(model)
    encoder.eval()

    dummy = torch.randn(1, 3, 224, 224)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    print(f"Exporting image encoder to {output_path}...")
    # Use dynamo exporter — the legacy tracer cannot handle the fused
    # aten::_native_multi_head_attention op used by torch >= 2.5.
    onnx_program = torch.onnx.export(
        encoder,
        (dummy,),
        dynamo=True,
    )
    onnx_program.save(output_path)

    # Consolidate external data into a single ONNX file.
    import onnx
    external_data_path = output_path + ".data"
    if os.path.isfile(external_data_path):
        print("Consolidating external data into single file...")
        onnx_model = onnx.load(output_path, load_external_data=True)
        onnx.save(onnx_model, output_path)
        os.remove(external_data_path)
    else:
        onnx_model = onnx.load(output_path)

    onnx.checker.check_model(onnx_model)
    print("ONNX checker passed.")

    # Verify ONNX output matches PyTorch
    import onnxruntime as ort
    session = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    ort_out = session.run(None, {input_name: dummy.numpy()})[0]
    ref_out = encoder(dummy).numpy()
    diff = np.abs(ort_out - ref_out).max()
    print(f"ONNX vs PyTorch max diff: {diff:.6f}")
    print(f"Output shape: {ort_out.shape}, norm: {np.linalg.norm(ort_out, axis=-1)}")


def generate_tags(model, tokenizer, output_path):
    """Pre-compute text embeddings for photo tags."""
    tags = [tag for tag, _ in TAG_VOCAB]
    prompts = [prompt for _, prompt in TAG_VOCAB]
    tokens = tokenizer(prompts)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = F.normalize(text_features, dim=-1)

    embeddings = text_features.cpu().numpy().tolist()

    data = {"tags": tags, "embeddings": embeddings}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(tags)} tag embeddings → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(output, tags_output, opset=17):
    """Entry point for programmatic conversion."""
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    print("Loading OpenCLIP ViT-B-32 (laion2b_s34b_b79k)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    export_image_encoder(model, output, opset)
    generate_tags(model, tokenizer, tags_output)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Export OpenCLIP ViT-B-32 image encoder to ONNX"
    )
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--tags-output", required=True, help="Output tags.json path")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    convert(args.output, args.tags_output, args.opset)


if __name__ == "__main__":
    main()
