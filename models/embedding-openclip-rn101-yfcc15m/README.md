# OpenCLIP RN101 (YFCC15M)

ResNet-101 image embedder from the OpenCLIP project, trained on YFCC15M –
15 million Flickr photos with Creative Commons licences. Produces
512-dimensional embeddings used in darktable for centroid-based tag
suggestion and image-similarity search.

## Source

- Architecture & weights: [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip), `RN101 / yfcc15m`
- License: MIT
- Paper: [Reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143)
- Training data: [YFCC15M](https://www.researchgate.net/publication/270275998_The_New_Data_and_New_Challenges_in_Multimedia_Research_-_YFCC100M_The_New_Data_and_New_Challenges_in_Multimedia_Research) (Thomee et al.) – subset of YFCC100M curated by Radford et al. for CLIP training

## Why this model

YFCC15M is the only widely-available CLIP training dataset where every image
was opt-in licensed by its author (Flickr Creative Commons). Other CLIP-family
training corpora (LAION, WebLI, OpenAI's WIT-400M, DataComp, MetaCLIP) are
all web scrapes without per-image consent. For darktable's open-source
compliance criteria – *"Models trained on undisclosed or scraped personal
data without consent are not accepted"* – YFCC15M is the right fit.

The trade-off: ~31% ImageNet zero-shot top-1, vs ~67% for LAION-trained
ViT-B-32 or ~76% for SigLIP 2. For darktable's actual use case (photo-domain
centroid matching against a user's library) the gap is narrower than the
benchmark suggests, because YFCC's training distribution is photo-aligned
rather than the everything-on-the-web mix that boosts LAION's ImageNet score.

## How it's used in darktable

Two workflows, both image-to-image:

- **Per-user tag taxonomy.** Average image embeddings per user-applied tag
  → centroid in 512-dim space. Suggest the same tag on new images whose
  embedding is close to that centroid.
- **Image similarity.** The same embeddings answer "find more like this"
  with no text involved.

The package ships the image encoder only. There is no default tag
vocabulary: a centroid built from a user's own photos describes what that
user means by a tag far better than a generic label ever did.

## Architecture

| Property | Value |
| -------- | ----- |
| Visual backbone | Modified ResNet-101 (CLIP variant: anti-aliased pooling, attention pool head) |
| Input | 224×224 RGB, normalised with OpenAI CLIP stats (mean 0.485 / 0.458 / 0.408, std 0.269 / 0.261 / 0.276) |
| Embedding dim | 512 |
| Parameters | ~130M |

## ONNX Assets

| File         | Purpose                                                                  | Size          |
|--------------|--------------------------------------------------------------------------|---------------|
| `model.onnx` | image → 512-dim L2-normalised embedding (mean/std subtraction baked in)  | ~60 MB (FP16) |

The convert wrapper bakes mean/std subtraction and L2 normalisation into the
ONNX graph so the caller only needs to feed `[0, 1]` RGB pixels.

The visual backbone runs in FP16; the graph's input and output stay
FP32 because the casts happen inside the export wrapper (after mean/std
subtraction on the input, before L2 normalisation on the output).
Callers do not need to add any cast logic.

## Notes

- **No text encoder in the package.** Only the image encoder is exported, and
  none is run at convert time. Free-text search would require shipping the
  text encoder separately – deferred until that feature lands in darktable.
- **Multilingual upgrade path is clean.** A future multilingual text encoder
  distilled to match this YFCC15M-aligned space could be added as an optional
  sidecar package without invalidating users' indexed image embeddings.
- **No re-indexing required for text-encoder additions.** As long as the
  vision encoder stays unchanged, image embeddings remain valid forever.

## Selection Criteria

| Property | Value |
| -------- | ----- |
| Model license | MIT |
| OSAID v1.0 | Open Source AI |
| MOF | Class I (Open Science) |
| Training data license | Creative Commons (mixed: BY, BY-SA, BY-NC, BY-ND, BY-NC-SA, BY-NC-ND, public-domain) |
| Training data provenance | YFCC100M (Yahoo Flickr Creative Commons 100M), subset of 15M curated by Radford et al. for CLIP training. All images uploaded to Flickr by their authors under permissive licences. |
| Training code | Open: [open_clip](https://github.com/mlfoundations/open_clip) MIT |
| Known limitations | ResNet-101 backbone reaches modest benchmark scores compared to web-scrape-trained ViT models; the trade-off is deliberate to honour darktable's consent-based training-data criterion. |
| Published research | [Reproducible scaling laws for contrastive language-image learning](https://arxiv.org/abs/2212.07143) |
| Inference | Local only, no cloud dependencies |
| Scope | Image feature extraction for tagging and similarity (no image synthesis or modification) |
| Reproducibility | Full: `open_clip.create_model_and_transforms("RN101", pretrained="yfcc15m")` reproduces the weights; convert.py regenerates the ONNX export and tag centroids deterministically. |
