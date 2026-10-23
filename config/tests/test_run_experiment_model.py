import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

import run_experiment_model as runner  # noqa: E402


class TestRunExperimentModel(unittest.TestCase):
    def _make_run_dir(self, tmp: str) -> Path:
        run_dir = Path(tmp) / "exp"
        rows = [
            {
                "attempt_index": 1,
                "attempt_id": "attempt_0001",
                "status": "success",
                "best_test_accuracy": 0.80,
                "param_count": 1_000_000,
                "checkpoint_metadata": {
                    "model_name": "FirstModel",
                    "dataset_path": str(Path(tmp) / "dataset.npz"),
                },
            },
            {
                "attempt_index": 2,
                "attempt_id": "attempt_0002",
                "status": "success",
                "best_test_accuracy": 0.91,
                "param_count": 2_000_000,
                "checkpoint_metadata": {"model_name": "BestModel"},
            },
        ]
        run_dir.mkdir(parents=True)
        (run_dir / "history.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
        (run_dir / "run_manifest.json").write_text(
            json.dumps(
                {
                    "epochs": 7,
                    "timesteps": 20,
                    "batch_size": 64,
                    "lr": 0.002,
                    "num_workers": 2,
                    "base_seed": 42,
                }
            ),
            encoding="utf-8",
        )
        for attempt_id in ("attempt_0001", "attempt_0002"):
            attempt_dir = run_dir / "attempts" / attempt_id
            attempt_dir.mkdir(parents=True)
            (attempt_dir / "config.json").write_text("{}", encoding="utf-8")
            (attempt_dir / "model.py").write_text("# model", encoding="utf-8")
        return run_dir

    def test_select_best_rank_and_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._make_run_dir(tmp)
            attempts = runner.rank_attempts(run_dir, runner.read_jsonl(run_dir / "history.jsonl"))

            self.assertEqual(runner.select_attempt(attempts, "best").attempt_id, "attempt_0002")
            self.assertEqual(runner.select_attempt(attempts, "rank:1").attempt_id, "attempt_0002")
            self.assertEqual(runner.select_attempt(attempts, "1").attempt_id, "attempt_0001")
            self.assertEqual(runner.select_attempt(attempts, "BestModel").attempt_id, "attempt_0002")

    def test_build_coderun_command_uses_manifest_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = self._make_run_dir(tmp)
            attempts = runner.rank_attempts(run_dir, runner.read_jsonl(run_dir / "history.jsonl"))
            manifest = runner.load_manifest(run_dir)
            attempt = runner.select_attempt(attempts, "attempt_0001")
            args = Namespace(
                data_path=None,
                output=None,
                epochs=None,
                timesteps=None,
                batch_size=None,
                lr=None,
                num_workers=None,
                seed=None,
                device="cpu",
                model_kwargs=None,
                coderun_path=runner.DEFAULT_CODERUN_PATH,
            )

            cmd = runner.build_coderun_command(
                args=args,
                attempt=attempt,
                manifest=manifest,
                model_path=attempt.model_path,
            )

            self.assertIn("--epochs", cmd)
            self.assertIn("7", cmd)
            self.assertIn("--device", cmd)
            self.assertIn("cpu", cmd)
            self.assertIn(str(Path(tmp) / "dataset.npz"), cmd)
            self.assertIn(str((attempt.attempt_dir / "checkpoint_rerun.pt").resolve()), cmd)


if __name__ == "__main__":
    unittest.main()
