import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

sys.modules.setdefault("librosa", types.ModuleType("librosa"))

import snn_coderun  # noqa: E402


class TestSnnCoderunDatasetCache(unittest.TestCase):
    def test_load_dataset_bundle_saves_cache_for_raw_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_dir = tmp_path / "raw"
            source_dir.mkdir()
            cache_path = tmp_path / "data" / "google_speech_commands_mels.npz"
            X = np.zeros((2, 128, 32), dtype=np.float32)
            y = np.asarray([0, 1], dtype=np.int64)
            label_map = {"no": 0, "yes": 1}

            with (
                mock.patch.object(snn_coderun, "default_cache_path", return_value=cache_path),
                mock.patch.object(snn_coderun, "create_dataset", return_value=(X, y, label_map)),
            ):
                loaded_X, loaded_y, loaded_labels, metadata, dataset_kind = (
                    snn_coderun.load_dataset_bundle(source_dir)
                )

            self.assertEqual(dataset_kind, "raw")
            self.assertTrue(cache_path.exists())
            self.assertEqual(metadata["cache_path"], str(cache_path))
            np.testing.assert_array_equal(loaded_X, X)
            np.testing.assert_array_equal(loaded_y, y)
            self.assertEqual(loaded_labels, label_map)


if __name__ == "__main__":
    unittest.main()
