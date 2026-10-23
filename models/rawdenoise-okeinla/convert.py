"""Export the frozen Okeinla 1.0 checkpoint to FP32 ONNX."""

import argparse
import hashlib
import importlib.util
import shutil
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "vendor" / "okeinla"
CHECKPOINT_SHA256 = "93f90d2d20edef1035291adebcfe8833a7778a30b8395610ccf19d475ea362fc"


def load_model(checkpoint):
    with open(checkpoint, "rb") as handle:
        if hashlib.file_digest(handle, "sha256").hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint does not match Okeinla 1.0")
    # Load only the architecture; the research package imports training dependencies.
    spec = importlib.util.spec_from_file_location(
        "okeinla_nafnet", SOURCE / "src/rawnind/models/nafnet.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.NAFNet(in_channels=4, funit=32)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    return model.eval()


def convert(checkpoint, output):
    import numpy as np
    import onnx
    import onnxruntime as ort

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model = load_model(checkpoint)
    sample = torch.rand((1, 4, 512, 512), generator=torch.Generator().manual_seed(2026))
    with torch.inference_mode():
        expected = model(sample).numpy()
        torch.onnx.export(
            model, sample, str(output), input_names=["input"], output_names=["output"],
            opset_version=20, dynamo=False,
        )
    onnx.checker.check_model(str(output))
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    actual = session.run(None, {"input": sample.numpy()})[0]
    if actual.shape != (1, 3, 1024, 1024) or not np.isfinite(actual).all():
        raise ValueError("Invalid Okeinla output")
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-5)
    shutil.copyfile(SOURCE / "LICENSE", output.parent / "LICENSE")
    shutil.copyfile(SOURCE / "licenses/NAFNet-MIT.txt", output.parent / "NAFNet-MIT.txt")
    print("Okeinla ONNX validation and PyTorch parity passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
