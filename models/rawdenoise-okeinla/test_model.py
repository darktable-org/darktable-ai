"""Small safety checks: python -m unittest discover -s models/rawdenoise-okeinla."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from darktable_ai.convert import _import_script


class OkeinlaTests(unittest.TestCase):
    def test_bayer_patterns_keep_spatial_neighbours(self):
        import numpy as np

        shared = _import_script(Path(__file__).resolve().parents[1] / "rawdenoise-nind/demo.py")
        mosaic = np.arange(48).reshape(6, 8)
        for pattern, y, x in (
            ([[0, 1], [3, 2]], 0, 0),  # RGGB
            ([[2, 3], [1, 0]], 1, 1),  # BGGR
            ([[1, 0], [2, 3]], 0, 1),  # GRBG
            ([[3, 2], [0, 1]], 1, 0),  # GBRG
        ):
            with self.subTest(pattern=pattern):
                cropped = mosaic[y:6-y, x:8-x]
                expected = np.stack([cropped[::2, ::2], cropped[::2, 1::2],
                                     cropped[1::2, ::2], cropped[1::2, 1::2]])
                np.testing.assert_array_equal(shared.pack_rggb(mosaic, pattern, "RGBG"), expected)

    def test_rejects_invalid_bayer_inputs(self):
        import numpy as np

        shared = _import_script(Path(__file__).resolve().parents[1] / "rawdenoise-nind/demo.py")
        for mosaic, pattern in (
            (np.zeros((3, 4)), [[0, 1], [3, 2]]),
            (np.zeros((2, 2)), [[2, 3], [1, 0]]),
            (np.zeros((4, 4)), [[0, 1], [1, 0]]),
            (np.zeros((4, 4)), [[0, 4], [1, 2]]),
            (np.zeros((4, 4)), [[0, 1, 2]]),
        ):
            with self.subTest(pattern=pattern, shape=mosaic.shape):
                with self.assertRaises(ValueError):
                    shared.pack_rggb(mosaic, pattern, "RGBG")

    def test_rejects_wrong_checkpoint(self):
        converter = _import_script(Path(__file__).with_name("convert.py"))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "wrong.pt"
            checkpoint.write_bytes(b"not the release checkpoint")
            with self.assertRaisesRegex(ValueError, "does not match"):
                converter.load_model(checkpoint)

    def test_skips_non_bayer(self):
        demo = _import_script(Path(__file__).with_name("demo.py"))
        with patch("rawpy.imread") as read, patch.object(demo, "_import_script") as shared:
            read.return_value.__enter__.return_value = SimpleNamespace(raw_pattern=None)
            demo.demo("unused", "xtrans.raf", "unused.tif")
            shared.assert_not_called()


if __name__ == "__main__":
    unittest.main()
