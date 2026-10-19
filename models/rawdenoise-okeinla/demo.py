"""Run the shared RawNIND Bayer demo with Okeinla weights."""

import argparse
from pathlib import Path

from darktable_ai.convert import _import_script


def demo(model_dir, image, output, **kwargs):
    import rawpy

    with rawpy.imread(image) as raw:
        if raw.raw_pattern is None or raw.raw_pattern.shape != (2, 2):
            print(f"Skipping non-Bayer RAW: {image}")
            return
    shared = _import_script(Path(__file__).resolve().parents[1] / "rawdenoise-nind/demo.py")
    shared.demo(model_dir, image, output, variant="bayer", **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    demo(args.model_dir, args.image, args.output)


if __name__ == "__main__":
    main()
