import copy
import contextlib
import io
import os
import sys
import tempfile
import unittest
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1]
if str(CONFIG_DIR) not in sys.path:
    sys.path.insert(0, str(CONFIG_DIR))

import snn_search_agent as agent  # noqa: E402


class TestSnnSearchAgent(unittest.TestCase):
    def test_build_prompts_require_russian_output(self) -> None:
        seed_config = agent._load_json(CONFIG_DIR / "examples" / "config_1to3to1_notebook.json")
        system_prompt, user_prompt = agent.build_prompts(
            seed_config=seed_config,
            context_text="предыдущая попытка: 0.51",
            feedback=[],
        )
        self.assertIn("Отвечай только валидным JSON без markdown.", system_prompt)
        self.assertIn("Все текстовые поля ответа должны быть на русском языке.", system_prompt)
        self.assertIn("`rationale` и `reasoning_summary` должны быть написаны по-русски.", user_prompt)
        self.assertIn("Ответ должен быть только JSON, без текста вне JSON.", user_prompt)
        self.assertIn("максимальную test accuracy на текущем датасете", user_prompt)
        self.assertIn("баланс между exploitation и exploration", user_prompt)
        self.assertIn("Анти-повтор и требования к новизне", user_prompt)
        self.assertIn("Новый конфиг обязан отличаться минимум по двум структурным осям", user_prompt)

    def test_build_context_marks_repeated_configs_and_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            memory = agent.ExperimentMemory(run_dir)
            for idx in range(1, 6):
                memory.append(
                    {
                        "attempt_index": idx,
                        "attempt_id": f"attempt_{idx:04d}",
                        "status": "success",
                        "best_test_accuracy": 0.5 + idx * 0.01,
                        "config_signature": "res=5, res_neurons=1536, edges=13, recurrent=5, cnn=on",
                        "config_sha256": "abc123",
                    }
                )

            context = memory.build_context(max_attempts=2, max_chars=10_000)

            self.assertIn("Repeated exact configs to avoid", context)
            self.assertIn("sha=abc123", context)
            self.assertIn("Exhausted/repeated signatures to avoid", context)
            self.assertIn("count=5", context)

    def test_extract_first_json_object_from_fenced_block(self) -> None:
        raw = """
        Here is the next candidate:
        ```json
        {"rationale":"short", "reasoning_summary":"explicit summary", "config":{"model_name":"X","nodes":[],"edges":[]}}
        ```
        """
        payload = agent.extract_first_json_object(raw)
        self.assertIn("config", payload)
        self.assertEqual(payload["rationale"], "short")

    def test_parse_proposal_plain_config(self) -> None:
        raw = '{"model_name":"A","nodes":[{"id":"inp","type":"input","size":4096},{"id":"r","type":"reservoir","size":256},{"id":"out","type":"output","size":30}],"edges":[{"from":"inp","to":"r"},{"from":"r","to":"r"},{"from":"r","to":"out"}]}'
        proposal = agent.parse_proposal(raw, retry_index=1)
        self.assertIsInstance(proposal.config, dict)
        self.assertTrue(proposal.rationale)
        self.assertEqual(proposal.reasoning_summary, proposal.rationale)

    def test_parse_proposal_preserves_reasoning_summary(self) -> None:
        raw = '{"rationale":"short rationale","reasoning_summary":"short explicit reasoning","config":{"model_name":"A","nodes":[{"id":"inp","type":"input","size":4096},{"id":"r","type":"reservoir","size":256},{"id":"out","type":"output","size":30}],"edges":[{"from":"inp","to":"r"},{"from":"r","to":"r"},{"from":"r","to":"out"}]}}'
        proposal = agent.parse_proposal(raw, retry_index=1)
        self.assertEqual(proposal.reasoning_summary, "short explicit reasoning")

    def test_find_seen_config_duplicate_rejects_exact_config(self) -> None:
        cfg = agent._load_json(CONFIG_DIR / "examples" / "config_1to3to1_notebook.json")
        history = [
            {
                "attempt_id": "attempt_0001",
                "best_test_accuracy": 0.77,
                "config_sha256": agent._json_sha256(cfg),
            }
        ]

        error = agent._find_seen_config_duplicate(cfg, history=history, run_dir=CONFIG_DIR)

        self.assertIsNotNone(error)
        self.assertIn("Duplicate config", error or "")
        self.assertIn("attempt_0001", error or "")

    def test_find_seen_config_duplicate_rejects_same_structure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            cfg = agent._load_json(CONFIG_DIR / "examples" / "config_1to3to1_notebook.json")
            attempt_dir = run_dir / "attempts" / "attempt_0001"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            agent._write_json(attempt_dir / "config.json", cfg)

            candidate = copy.deepcopy(cfg)
            candidate["model_name"] = "RenamedSameSkeleton"
            for edge in candidate["edges"]:
                if isinstance(edge, dict):
                    edge["weight_init"] = {"type": "normal", "mean": 0.0, "std": 0.02}
                    break
            history = [
                {
                    "attempt_id": "attempt_0001",
                    "best_test_accuracy": 0.77,
                    "config_path": str(attempt_dir / "config.json"),
                    "config_sha256": agent._json_sha256(cfg),
                }
            ]

            error = agent._find_seen_config_duplicate(candidate, history=history, run_dir=run_dir)

            self.assertIsNotNone(error)
            self.assertIn("Duplicate architecture structure", error or "")
            self.assertIn("weight initialization is not enough", error or "")

    def test_load_env_file_reads_export_and_quotes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                'OPENAI_API_KEY="sk-test"\nexport OPENAI_BASE_URL=https://example.test/v1\n',
                encoding="utf-8",
            )
            old_api = os.environ.pop("OPENAI_API_KEY", None)
            old_base = os.environ.pop("OPENAI_BASE_URL", None)
            try:
                loaded = agent.load_env_file(env_path, override=False)
                self.assertEqual(loaded["OPENAI_API_KEY"], "sk-test")
                self.assertEqual(os.environ["OPENAI_API_KEY"], "sk-test")
                self.assertEqual(os.environ["OPENAI_BASE_URL"], "https://example.test/v1")
            finally:
                if old_api is not None:
                    os.environ["OPENAI_API_KEY"] = old_api
                else:
                    os.environ.pop("OPENAI_API_KEY", None)
                if old_base is not None:
                    os.environ["OPENAI_BASE_URL"] = old_base
                else:
                    os.environ.pop("OPENAI_BASE_URL", None)

    def test_validate_config_accepts_example(self) -> None:
        cfg = agent._load_json(CONFIG_DIR / "examples" / "config_1to3to1_notebook.json")
        errors = agent.validate_config(cfg)
        self.assertEqual(errors, [])
        pcount = agent.estimate_param_count(cfg)
        self.assertLessEqual(pcount["total"], agent.MAX_MODEL_PARAMETERS)

    def test_run_command_streams_and_saves_logs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            stdout_path = tmpdir / "stdout.log"
            stderr_path = tmpdir / "stderr.log"
            cmd = [
                sys.executable,
                "-c",
                "import sys; print('hello'); print('warn', file=sys.stderr)",
            ]
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                result = agent.run_command(
                    cmd,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    timeout_seconds=5,
                )
            self.assertEqual(result.returncode, 0)
            self.assertIn("hello", stdout_capture.getvalue())
            self.assertIn("warn", stderr_capture.getvalue())
            self.assertIn("hello", stdout_path.read_text(encoding="utf-8"))
            self.assertIn("warn", stderr_path.read_text(encoding="utf-8"))

    def test_persist_attempt_record_writes_summary_and_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            attempt_dir = run_dir / "attempts" / "attempt_0001"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            memory = agent.ExperimentMemory(run_dir)
            record = {
                "run_id": "run_test",
                "attempt_index": 1,
                "attempt_id": "attempt_0001",
                "status": "success",
                "best_test_accuracy": 0.5,
            }
            saved = agent._persist_attempt_record(
                memory=memory,
                attempt_dir=attempt_dir,
                record=record,
            )
            self.assertEqual(saved, record)
            self.assertEqual(memory.load(), [record])
            summary = agent._load_json(attempt_dir / "attempt_summary.json")
            self.assertEqual(summary["attempt_id"], "attempt_0001")

    def test_validate_config_rejects_output_source_edge(self) -> None:
        cfg = agent._load_json(CONFIG_DIR / "examples" / "config_1to3to1_notebook.json")
        bad = copy.deepcopy(cfg)
        bad["edges"].append({"from": "out", "to": "res1"})
        errors = agent.validate_config(bad)
        joined = " | ".join(errors)
        self.assertIn("edges from output nodes are unsupported", joined)

    def test_validate_config_rejects_param_limit_exceeded(self) -> None:
        cfg = agent._load_json(CONFIG_DIR / "examples" / "config_1to3to1_notebook.json")
        bad = copy.deepcopy(cfg)
        for node in bad["nodes"]:
            if node["type"] == "reservoir":
                node["size"] = 4000
        errors = agent.validate_config(bad)
        joined = " | ".join(errors)
        self.assertIn("Model exceeds parameter limit", joined)

    def test_memory_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            memory = agent.ExperimentMemory(run_dir)
            memory.append(
                {
                    "attempt_index": 1,
                    "attempt_id": "attempt_0001",
                    "status": "success",
                    "best_test_accuracy": 0.55,
                    "rationale": "wider middle reservoirs",
                    "config_signature": "res=5",
                }
            )
            memory.append(
                {
                    "attempt_index": 2,
                    "attempt_id": "attempt_0002",
                    "status": "failed_train",
                    "error_message": "coderun exit=1",
                    "rationale": "bad topology",
                    "config_signature": "res=7",
                }
            )
            rows = memory.search("failed", limit=10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["attempt_id"], "attempt_0002")


if __name__ == "__main__":
    unittest.main()
