"""Export OpenCLIP RN101 (YFCC15M) image encoder to ONNX and pre-compute tag embeddings.

Produces:
  model.onnx  – image encoder with baked-in CLIP normalization + L2 norm
  tags.json   – pre-computed text embeddings for 86 hierarchical photo tags

Tag vocabulary is defined in tags.md (human-readable reference).

Why RN101 + YFCC15M specifically: YFCC15M is the only widely-available CLIP
training dataset where every image was opt-in licensed by its author
(Flickr Creative Commons). The CLIP family's other training corpora
(LAION, WebLI, WIT, DataComp) are all web scrapes without per-image
consent. For darktable's open-source compliance criteria this matters
more than the ImageNet zero-shot benchmark gap.
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
# Tag vocabulary for zero-shot tag suggestion.
# Hierarchical names using "|" separator (darktable native).
# Each entry is (tag, CLIP prompt). See tags.md for the human-readable list.
# ---------------------------------------------------------------------------

# Each tag carries a *list* of prompts. Encoding ensembles them: encode
# each, average, re-normalize. This is the standard CLIP zero-shot recipe
# (Radford et al. 2021) and substantially de-biases each centroid from
# the quirks of any one phrasing.
#
# Prompts try to combine:
#   - one direct canonical form ("a photo of X")
#   - one or two descriptive forms with visual cues (helps high-frequency
#     classes like cat/dog stop matching every animal photo)
#   - context cues for setting/lighting/technique tags

TAG_VOCAB = [
    # --- genre (12) ---
    ("genre|landscape", [
        "a landscape photograph",
        "a scenic landscape photo",
        "a wide outdoor landscape view",
        "a nature landscape image",
    ]),
    ("genre|portrait", [
        "a portrait photograph",
        "a portrait of a person",
        "a close-up portrait photo",
        "a head-and-shoulders portrait",
    ]),
    ("genre|street", [
        "a street photography image",
        "a candid street photo",
        "a photo of city street life",
        "an unposed urban scene",
    ]),
    ("genre|wildlife", [
        "a wildlife photograph",
        "a photo of a wild animal in nature",
        "a nature wildlife image",
    ]),
    ("genre|macro", [
        "a macro photograph",
        "an extreme close-up photo",
        "a photo of a tiny subject magnified",
    ]),
    ("genre|architecture", [
        "an architecture photograph",
        "a photo of a building's architecture",
        "an architectural photograph of a structure",
    ]),
    ("genre|food", [
        "a food photograph",
        "a photo of a prepared meal or dish",
        "a culinary food photo",
    ]),
    ("genre|sports", [
        "a sports photograph",
        "a photo of athletes in action",
        "an action shot from a sporting event",
    ]),
    ("genre|event", [
        "an event photograph",
        "a photo of a wedding, party, or ceremony",
        "a photo from a public event",
    ]),
    ("genre|abstract", [
        "an abstract photograph",
        "an abstract image with no clear subject",
        "an abstract pattern photo",
    ]),
    ("genre|still life", [
        "a still life photograph",
        "an arranged still life photo of objects",
        "a tabletop still life image",
    ]),
    ("genre|aerial", [
        "an aerial photograph",
        "a top-down view from above",
        "a drone or aerial photo of the ground",
    ]),
    # --- subject|people (6) ---
    ("subject|people|person", [
        "a photograph of a person",
        "a single individual in a photo",
        "a person captured in an image",
    ]),
    ("subject|people|couple", [
        "a photograph of a couple",
        "two people together in a photo",
        "a romantic couple captured in a picture",
    ]),
    ("subject|people|group", [
        "a photograph of a group of people",
        "several people together in a photo",
        "a crowd or group photo",
    ]),
    ("subject|people|child", [
        "a photograph of a child",
        "a photo of a young kid",
        "a picture of a school-aged child",
    ]),
    ("subject|people|baby", [
        "a photograph of a baby",
        "a photo of an infant",
        "a picture of a newborn or toddler",
    ]),
    ("subject|people|elderly person", [
        "a photograph of an elderly person",
        "a photo of an older adult",
        "a picture of a senior citizen",
    ]),
    # --- subject|animal (8) ---
    ("subject|animal|dog", [
        "a photograph of a dog",
        "a close-up photo of a dog with fur and a snout",
        "a domestic dog or puppy",
        "a picture of a canine pet",
    ]),
    ("subject|animal|cat", [
        "a photograph of a cat",
        "a close-up photo of a cat with whiskers and pointed ears",
        "a domestic cat or kitten",
        "a picture of a feline pet",
    ]),
    ("subject|animal|bird", [
        "a photograph of a bird",
        "a close-up of a bird with feathers and a beak",
        "a wild or perched bird",
    ]),
    ("subject|animal|horse", [
        "a photograph of a horse",
        "a photo of a horse with mane and hooves",
        "an equine animal in a photo",
    ]),
    ("subject|animal|insect", [
        "a photograph of an insect",
        "a close-up of a bug with six legs and antennae",
        "a macro photo of an insect",
    ]),
    ("subject|animal|fish", [
        "a photograph of a fish",
        "a photo of a fish underwater with fins",
        "a picture of a marine or freshwater fish",
    ]),
    ("subject|animal|reptile", [
        "a photograph of a reptile",
        "a photo of a lizard, snake, or turtle",
        "a cold-blooded reptile with scales",
    ]),
    ("subject|animal|wild animal", [
        "a photograph of a wild animal",
        "a photo of an animal in its natural habitat",
        "a wildlife photo of an undomesticated creature",
    ]),
    # --- subject|nature (10) ---
    ("subject|nature|flower", [
        "a photograph of a flower",
        "a close-up of a flower with petals",
        "a bloom or blossom in a photo",
    ]),
    ("subject|nature|tree", [
        "a photograph of a tree",
        "a photo of a tree with leaves and branches",
        "a single prominent tree in a photo",
    ]),
    ("subject|nature|mountain", [
        "a photograph of a mountain",
        "a photo of a mountain peak or range",
        "a mountainous landscape",
    ]),
    ("subject|nature|waterfall", [
        "a photograph of a waterfall",
        "a photo of falling water down rocks",
        "a cascading waterfall in nature",
    ]),
    ("subject|nature|river", [
        "a photograph of a river",
        "a photo of a flowing river or stream",
        "a river winding through landscape",
    ]),
    ("subject|nature|lake", [
        "a photograph of a lake",
        "a photo of a calm lake or pond",
        "a still body of water in nature",
    ]),
    ("subject|nature|ocean", [
        "a photograph of the ocean",
        "a photo of the sea with waves",
        "an ocean or seascape image",
    ]),
    ("subject|nature|cloud", [
        "a photograph of clouds",
        "a sky filled with clouds",
        "a cloudscape photo",
    ]),
    ("subject|nature|rock", [
        "a photograph of rocks",
        "a photo of rocky terrain or boulders",
        "a stone or rock formation",
    ]),
    ("subject|nature|field", [
        "a photograph of a field",
        "a photo of an open grassy or agricultural field",
        "a meadow or pasture",
    ]),
    # --- subject|vehicle (5) ---
    ("subject|vehicle|car", [
        "a photograph of a car",
        "a photo of an automobile with wheels",
        "a picture of a parked or driving car",
    ]),
    ("subject|vehicle|bicycle", [
        "a photograph of a bicycle",
        "a photo of a bike with two wheels",
        "a cyclist or bicycle in a picture",
    ]),
    ("subject|vehicle|boat", [
        "a photograph of a boat",
        "a photo of a boat or ship on water",
        "a watercraft in a picture",
    ]),
    ("subject|vehicle|train", [
        "a photograph of a train",
        "a photo of a train on tracks",
        "a locomotive or rail vehicle",
    ]),
    ("subject|vehicle|airplane", [
        "a photograph of an airplane",
        "a photo of an aircraft in flight or on the ground",
        "a plane with wings and engines",
    ]),
    # --- subject|structure (5) ---
    ("subject|structure|building", [
        "a photograph of a building",
        "a photo of an architectural structure or house",
        "an exterior shot of a building",
    ]),
    ("subject|structure|bridge", [
        "a photograph of a bridge",
        "a photo of a bridge spanning water or a gap",
        "an architectural bridge structure",
    ]),
    ("subject|structure|tower", [
        "a photograph of a tower",
        "a photo of a tall vertical structure",
        "a tower rising into the sky",
    ]),
    ("subject|structure|statue", [
        "a photograph of a statue",
        "a photo of a sculpted figure or monument",
        "a sculpture in a public space",
    ]),
    ("subject|structure|ruin", [
        "a photograph of a ruin",
        "a photo of ancient or abandoned ruins",
        "a derelict historical structure",
    ]),
    # --- setting (8) ---
    ("setting|indoor", [
        "an indoor photograph",
        "a photo taken inside a building",
        "an interior scene image",
    ]),
    ("setting|outdoor", [
        "an outdoor photograph",
        "a photo taken outdoors",
        "an open-air outdoor scene",
    ]),
    ("setting|urban", [
        "an urban photograph",
        "a photo taken in a city or town",
        "an image of urban streets and buildings",
    ]),
    ("setting|rural", [
        "a rural photograph",
        "a photo taken in the countryside",
        "an image of rural farmland or villages",
    ]),
    ("setting|beach", [
        "a photograph at a beach",
        "a beach scene with sand and water",
        "a coastal photograph by the sea",
    ]),
    ("setting|forest", [
        "a photograph in a forest",
        "a photo of trees in a woodland",
        "a forest scene with dense trees",
    ]),
    ("setting|desert", [
        "a photograph in a desert",
        "a photo of a dry sandy desert landscape",
        "a barren desert scene",
    ]),
    ("setting|studio", [
        "a studio photograph",
        "a photo with controlled studio lighting and backdrop",
        "a posed studio image",
    ]),
    # --- lighting (8) ---
    ("lighting|sunrise", [
        "a photograph taken at sunrise",
        "an early morning photo with the sun rising",
        "a sunrise scene with warm sky colors",
    ]),
    ("lighting|sunset", [
        "a photograph taken at sunset",
        "an evening photo with the sun setting",
        "a sunset sky with warm orange colors",
    ]),
    ("lighting|golden hour", [
        "a photograph during golden hour",
        "a photo with warm low-angle golden sunlight",
        "an image bathed in golden afternoon light",
    ]),
    ("lighting|blue hour", [
        "a photograph during blue hour",
        "a twilight photo with a deep blue sky",
        "a dusk or pre-dawn image with cool blue tones",
    ]),
    ("lighting|night", [
        "a photograph taken at night",
        "a dark nighttime image",
        "a photo in low light after dark",
    ]),
    ("lighting|backlit", [
        "a backlit photograph",
        "a photo with the light source behind the subject",
        "a subject lit from behind with rim light",
    ]),
    ("lighting|silhouette", [
        "a silhouette photograph",
        "a dark subject outlined against a bright background",
        "a backlit silhouette of a figure or object",
    ]),
    ("lighting|low light", [
        "a low light photograph",
        "a dim photo with little available light",
        "an image taken in poor lighting conditions",
    ]),
    # --- technique (8) ---
    ("technique|black and white", [
        "a black and white photograph",
        "a monochrome grayscale image",
        "a desaturated B&W photo",
    ]),
    ("technique|long exposure", [
        "a long exposure photograph",
        "a photo with motion blurred by a slow shutter",
        "a smooth water or sky long exposure image",
    ]),
    ("technique|bokeh", [
        "a photograph with bokeh",
        "a photo with a blurred out-of-focus background",
        "an image showing creamy bokeh circles",
    ]),
    ("technique|panorama", [
        "a panoramic photograph",
        "a very wide aspect ratio panorama image",
        "a stitched panoramic scene",
    ]),
    ("technique|close-up", [
        "a close-up photograph",
        "a tight close-up of a subject",
        "a near-distance detailed photo",
    ]),
    ("technique|wide angle", [
        "a wide angle photograph",
        "a photo taken with a wide field of view",
        "an expansive wide-lens image",
    ]),
    ("technique|motion blur", [
        "a photograph with motion blur",
        "a photo where moving subjects are blurred",
        "an image with intentional motion streaks",
    ]),
    ("technique|reflection", [
        "a photograph with reflections",
        "a photo showing reflective surfaces like water or glass",
        "an image with mirrored reflections",
    ]),
    # --- mood (6) ---
    ("mood|dramatic", [
        "a dramatic photograph",
        "an image with intense contrast and powerful mood",
        "a striking dramatic scene",
    ]),
    ("mood|peaceful", [
        "a peaceful photograph",
        "a calm tranquil scene",
        "a serene quiet image",
    ]),
    ("mood|moody", [
        "a moody photograph",
        "a dark atmospheric image",
        "a brooding or melancholy photo",
    ]),
    ("mood|vibrant", [
        "a vibrant photograph",
        "a colorful saturated image",
        "an energetic high-color photo",
    ]),
    ("mood|minimal", [
        "a minimalist photograph",
        "an image with very few elements and negative space",
        "a clean simple minimalist composition",
    ]),
    ("mood|chaotic", [
        "a chaotic photograph",
        "a busy crowded disorderly scene",
        "an image with many overlapping elements",
    ]),
    # --- weather (6) ---
    ("weather|sunny", [
        "a photograph in sunny weather",
        "a bright clear-sky photo with sunshine",
        "an image taken on a sunny day",
    ]),
    ("weather|cloudy", [
        "a photograph in cloudy weather",
        "a photo under an overcast cloudy sky",
        "an image with a gray cloudy sky",
    ]),
    ("weather|rainy", [
        "a photograph in rainy weather",
        "a photo with visible rain or wet surfaces",
        "a rainy day image",
    ]),
    ("weather|snowy", [
        "a photograph in snowy weather",
        "a photo with snow covering the ground",
        "a snowy winter scene",
    ]),
    ("weather|foggy", [
        "a photograph in foggy weather",
        "a misty foggy scene with reduced visibility",
        "an image obscured by fog or mist",
    ]),
    ("weather|stormy", [
        "a photograph in stormy weather",
        "a photo of a storm with dramatic clouds",
        "an image during severe weather",
    ]),
    # --- season (4) ---
    ("season|spring", [
        "a photograph taken in spring",
        "a springtime scene with fresh green growth and flowers",
        "an image with spring bloom",
    ]),
    ("season|summer", [
        "a photograph taken in summer",
        "a summer scene with lush greenery and warm light",
        "an image from the summer season",
    ]),
    ("season|autumn", [
        "a photograph taken in autumn",
        "an autumn scene with red, orange, and yellow leaves",
        "a fall foliage image",
    ]),
    ("season|winter", [
        "a photograph taken in winter",
        "a winter scene with bare trees or snow",
        "a cold-weather image from the winter season",
    ]),
]


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


def generate_tags(model, tokenizer, output_path):
    """Pre-compute text embeddings for photo tags using prompt ensembling.

    For each tag:
      1. encode all of its prompt variants
      2. average the L2-normalized embeddings
      3. re-normalize the average to unit length

    No centering — centering shifts the cosine similarity range so much
    that the threshold becomes meaningless, and the top-K filter on the
    consumer side is a better way to limit over-eager tags.
    """
    tags = [tag for tag, _ in TAG_VOCAB]

    centroids = []
    with torch.no_grad():
        for _tag, prompts in TAG_VOCAB:
            tokens = tokenizer(prompts)
            feats = model.encode_text(tokens)
            feats = F.normalize(feats, dim=-1)        # normalize each prompt
            centroid = feats.mean(dim=0)               # average
            centroid = F.normalize(centroid, dim=-1)   # renormalize
            centroids.append(centroid.cpu().numpy())

    embeddings = [c.tolist() for c in centroids]

    data = {"tags": tags, "embeddings": embeddings}
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {len(tags)} tag centroids → {output_path}")
    print(f"  (ensembled from {sum(len(p) for _, p in TAG_VOCAB)} prompts)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def convert(output, tags_output, opset=20, fp16=False):
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

    tokenizer = open_clip.get_tokenizer("RN101")

    export_image_encoder(model, output, opset, fp16=fp16)
    generate_tags(model, tokenizer, tags_output)

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Export OpenCLIP RN101 (YFCC15M) image encoder to ONNX"
    )
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--tags-output", required=True, help="Output tags.json path")
    parser.add_argument("--opset", type=int, default=20, help="ONNX opset version")
    parser.add_argument("--fp16", action="store_true",
                        help="convert weights to FP16 after export (default: FP32)")
    args = parser.parse_args()

    convert(args.output, args.tags_output, args.opset, fp16=args.fp16)


if __name__ == "__main__":
    main()
