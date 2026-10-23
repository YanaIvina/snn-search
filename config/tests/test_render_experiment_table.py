import json
import sys
import tempfile
import unittest
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

import render_experiment_table as render  # noqa: E402


class TestRenderExperimentTable(unittest.TestCase):
    def test_render_markdown_and_html(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            history_path = run_dir / "history.jsonl"
            rows = [
                {
                    "attempt_index": 1,
                    "attempt_id": "attempt_0001",
                    "status": "success",
                    "started_at": "2026-04-17T08:00:00+00:00",
                    "ended_at": "2026-04-17T08:10:00+00:00",
                    "best_test_accuracy": 0.91,
                    "test_accuracy": 0.90,
                    "param_count": 5_000_000,
                    "proposal_retries_used": 1,
                    "config_signature": "res=5, edges=14",
                    "rationale": "Тестовая архитектура",
                    "checkpoint_metadata": {"model_name": "TestModel"},
                    "config_path": str(run_dir / "attempts" / "attempt_0001" / "config.json"),
                },
                {
                    "attempt_index": 2,
                    "attempt_id": "attempt_0002",
                    "status": "success",
                    "started_at": "2026-04-17T08:10:00+00:00",
                    "ended_at": "2026-04-17T08:20:00+00:00",
                    "best_test_accuracy": 0.89,
                    "test_accuracy": 0.88,
                    "param_count": 4_000_000,
                    "proposal_retries_used": 2,
                    "config_signature": "res=4, edges=12",
                    "rationale": "Вторая архитектура",
                    "checkpoint_metadata": {"model_name": "OtherModel"},
                    "config_path": str(run_dir / "attempts" / "attempt_0002" / "config.json"),
                },
            ]
            history_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
            for attempt_id in ("attempt_0001", "attempt_0002"):
                attempt_dir = run_dir / "attempts" / attempt_id
                attempt_dir.mkdir(parents=True, exist_ok=True)
                (attempt_dir / "config.json").write_text("{}", encoding="utf-8")
                (attempt_dir / "attempt_summary.json").write_text("{}", encoding="utf-8")

            loaded = render._read_jsonl(history_path)
            md = render.render_markdown(run_dir, loaded)
            html = render.render_html(run_dir, loaded)

            self.assertIn("Результаты эксперимента", md)
            self.assertIn("attempt_0001", md)
            self.assertIn("0.9100", md)
            self.assertIn(f"<title>Результаты {run_dir.name}</title>", html)
            self.assertIn("Лучшая попытка", html)
            self.assertIn("attempt_0001", html)
            self.assertIn("class='best'", html)

    def test_render_uses_local_attempt_dir_for_copied_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_run_dir = tmp_path / "exp3"
            copied_run_dir = tmp_path / "exp3_light"
            source_attempt_dir = source_run_dir / "attempts" / "attempt_0001"
            copied_attempt_dir = copied_run_dir / "attempts" / "attempt_0001"
            source_attempt_dir.mkdir(parents=True, exist_ok=True)
            copied_attempt_dir.mkdir(parents=True, exist_ok=True)

            (source_attempt_dir / "config.json").write_text("{}", encoding="utf-8")
            (copied_attempt_dir / "config.json").write_text("{}", encoding="utf-8")
            (copied_attempt_dir / "attempt_summary.json").write_text("{}", encoding="utf-8")
            (copied_attempt_dir / "diagram.png").write_bytes(b"png")

            rows = [
                {
                    "attempt_index": 1,
                    "attempt_id": "attempt_0001",
                    "status": "success",
                    "started_at": "2026-04-17T08:00:00+00:00",
                    "ended_at": "2026-04-17T08:10:00+00:00",
                    "best_test_accuracy": 0.91,
                    "test_accuracy": 0.90,
                    "param_count": 5_000_000,
                    "proposal_retries_used": 1,
                    "config_signature": "res=5, edges=14",
                    "rationale": "Тестовая архитектура",
                    "checkpoint_metadata": {"model_name": "TestModel"},
                    "config_path": str(source_attempt_dir / "config.json"),
                }
            ]
            history_path = copied_run_dir / "history.jsonl"
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history_path.write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )

            loaded = render._read_jsonl(history_path)
            md = render.render_markdown(copied_run_dir, loaded)
            html = render.render_html(copied_run_dir, loaded)

            self.assertIn("[attempt_0001](attempts/attempt_0001/attempt_summary.json)", md)
            self.assertIn("href='attempts/attempt_0001/config.json'", html)
            self.assertIn("href='attempts/attempt_0001/diagram.png'", html)


if __name__ == "__main__":
    unittest.main()
