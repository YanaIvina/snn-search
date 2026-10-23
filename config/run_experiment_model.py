#!/usr/bin/env python3
"""
Select and run a model from a saved SNN search experiment.

Examples:
    python config/run_experiment_model.py --list --top 10
    python config/run_experiment_model.py best --dry-run
    python config/run_experiment_model.py 28 --epochs 1 --device cpu
    python config/run_experiment_model.py rank:1 --run-dir config/search_runs/exp3
    python config/run_experiment_model.py DualEncoderCrossMidHubSNN --dry-run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "search_runs" / "exp3_light"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "google_speech_commands_mels.npz"
DEFAULT_CODEGEN_PATH = SCRIPT_DIR / "snn_codegen.py"
DEFAULT_CODERUN_PATH = SCRIPT_DIR / "snn_coderun.py"


@dataclass(frozen=True)
class Attempt:
    row: dict[str, Any]
    run_dir: Path
    rank: int | None

    @property
    def attempt_id(self) -> str:
        return str(self.row.get("attempt_id", "")).strip()

    @property
    def attempt_index(self) -> int:
        try:
            return int(self.row.get("attempt_index", 0))
        except (TypeError, ValueError):
            return 0

    @property
    def model_name(self) -> str:
        metadata = self.row.get("checkpoint_metadata")
        if isinstance(metadata, dict) and metadata.get("model_name"):
            return str(metadata["model_name"])
        return ""

    @property
    def best_test_accuracy(self) -> float | None:
        return _to_float(self.row.get("best_test_accuracy"))

    @property
    def attempt_dir(self) -> Path:
        if self.attempt_id:
            local_dir = (self.run_dir / "attempts" / self.attempt_id).resolve()
            if local_dir.exists():
                return local_dir
        config_path = self.row.get("config_path")
        if isinstance(config_path, str) and config_path:
            return Path(config_path).expanduser().resolve().parent
        return (self.run_dir / "attempts" / self.attempt_id).resolve()

    @property
    def config_path(self) -> Path:
        local_config = self.attempt_dir / "config.json"
        if local_config.exists():
            return local_config
        config_path = self.row.get("config_path")
        if isinstance(config_path, str) and config_path:
            return Path(config_path).expanduser().resolve()
        return local_config

    @property
    def model_path(self) -> Path:
        local_model = self.attempt_dir / "model.py"
        local_config = self.attempt_dir / "config.json"
        if local_model.exists() or local_config.exists():
            return local_model
        model_path = self.row.get("model_path")
        if isinstance(model_path, str) and model_path:
            return Path(model_path).expanduser().resolve()
        return local_model

    @property
    def default_output_path(self) -> Path:
        return self.attempt_dir / "checkpoint_rerun.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select an attempt from config/search_runs and run it with snn_coderun.py."
    )
    parser.add_argument(
        "selector",
        nargs="?",
        default="best",
        help=(
            "Attempt selector: best, rank:N, attempt_0028, 28, or model name "
            "(default: best)."
        ),
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"Experiment directory with history.jsonl (default: {DEFAULT_RUN_DIR}).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List matching attempts instead of running a model.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="Rows to print for --list, sorted by best accuracy (default: 15).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected attempt and command without executing it.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Dataset path passed to snn_coderun.py. Defaults to attempt metadata or local cache.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override rate-coding timesteps.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--device", type=str, default=None, help="Device override: cpu, cuda, cuda:0, ...")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output checkpoint path. Defaults to <attempt_dir>/checkpoint_rerun.pt.",
    )
    parser.add_argument(
        "--model-kwargs",
        type=str,
        default=None,
        help="Optional JSON/Python dict forwarded to snn_coderun.py.",
    )
    parser.add_argument(
        "--codegen-path",
        type=Path,
        default=DEFAULT_CODEGEN_PATH,
        help="Path to snn_codegen.py for regenerating missing model.py files.",
    )
    parser.add_argument(
        "--coderun-path",
        type=Path,
        default=DEFAULT_CODERUN_PATH,
        help="Path to snn_coderun.py.",
    )
    parser.add_argument(
        "--no-codegen",
        action="store_true",
        help="Fail if model.py is missing instead of regenerating it from config.json.",
    )
    return parser.parse_args()


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"history.jsonl not found: {path}")
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            item = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSONL at {path}:{line_no}: {exc}") from exc
        if isinstance(item, dict):
            rows.append(item)
    return rows


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest if isinstance(manifest, dict) else {}


def rank_attempts(run_dir: Path, rows: list[dict[str, Any]]) -> list[Attempt]:
    successes = [
        row for row in rows
        if row.get("status") == "success" and _to_float(row.get("best_test_accuracy")) is not None
    ]
    by_score = sorted(
        successes,
        key=lambda row: _to_float(row.get("best_test_accuracy")) or -1.0,
        reverse=True,
    )
    rank_by_id = {str(row.get("attempt_id", "")): idx + 1 for idx, row in enumerate(by_score)}
    attempts = [Attempt(row=row, run_dir=run_dir, rank=rank_by_id.get(str(row.get("attempt_id", "")))) for row in rows]
    return sorted(attempts, key=lambda attempt: attempt.attempt_index)


def select_attempt(attempts: list[Attempt], selector: str) -> Attempt:
    normalized = selector.strip()
    lowered = normalized.lower()
    if not attempts:
        raise ValueError("No attempts found in history.")

    successful = [attempt for attempt in attempts if attempt.best_test_accuracy is not None]
    if lowered in {"best", "top"}:
        if not successful:
            raise ValueError("No successful attempts with best_test_accuracy found.")
        return max(successful, key=lambda attempt: attempt.best_test_accuracy or -1.0)

    if lowered.startswith("rank:"):
        rank_text = lowered.split(":", 1)[1]
        rank = int(rank_text)
        matches = [attempt for attempt in attempts if attempt.rank == rank]
        if matches:
            return matches[0]
        raise ValueError(f"No attempt with rank {rank}.")

    if normalized.isdigit():
        attempt_id = f"attempt_{int(normalized):04d}"
        matches = [
            attempt for attempt in attempts
            if attempt.attempt_id == attempt_id or attempt.attempt_index == int(normalized)
        ]
        if matches:
            return matches[0]

    if lowered.startswith("attempt_"):
        matches = [attempt for attempt in attempts if attempt.attempt_id.lower() == lowered]
        if matches:
            return matches[0]

    name_matches = [
        attempt for attempt in attempts
        if attempt.model_name.lower() == lowered
    ]
    if len(name_matches) == 1:
        return name_matches[0]
    if len(name_matches) > 1:
        ids = ", ".join(attempt.attempt_id for attempt in name_matches)
        raise ValueError(f"Model name is ambiguous, matches: {ids}")

    partial_matches = [
        attempt for attempt in attempts
        if lowered in attempt.model_name.lower()
    ]
    if len(partial_matches) == 1:
        return partial_matches[0]
    if len(partial_matches) > 1:
        ids = ", ".join(f"{attempt.attempt_id}:{attempt.model_name}" for attempt in partial_matches)
        raise ValueError(f"Selector is ambiguous, matches: {ids}")

    raise ValueError(f"No attempt matches selector: {selector}")


def print_attempts(attempts: list[Attempt], top: int) -> None:
    rows = sorted(
        [attempt for attempt in attempts if attempt.best_test_accuracy is not None],
        key=lambda attempt: attempt.best_test_accuracy or -1.0,
        reverse=True,
    )
    print(f"{'Rank':>4}  {'Attempt':<12}  {'Best acc':>8}  {'Params':>8}  Model")
    print("-" * 80)
    for attempt in rows[: max(top, 0)]:
        params = attempt.row.get("param_count")
        params_text = f"{float(params) / 1_000_000:.2f}M" if isinstance(params, (int, float)) else "n/a"
        acc = attempt.best_test_accuracy
        acc_text = f"{acc:.4f}" if acc is not None else "n/a"
        print(
            f"{attempt.rank or 0:>4}  {attempt.attempt_id:<12}  "
            f"{acc_text:>8}  {params_text:>8}  {attempt.model_name}"
        )


def manifest_default(manifest: dict[str, Any], key: str, fallback: Any = None) -> Any:
    value = manifest.get(key)
    return fallback if value is None else value


def metadata_default(attempt: Attempt, key: str, fallback: Any = None) -> Any:
    metadata = attempt.row.get("checkpoint_metadata")
    if isinstance(metadata, dict) and metadata.get(key) is not None:
        return metadata[key]
    return fallback


def generate_model_if_needed(attempt: Attempt, codegen_path: Path, dry_run: bool, no_codegen: bool) -> Path:
    model_path = attempt.model_path
    if model_path.exists():
        return model_path
    if no_codegen:
        raise FileNotFoundError(f"model.py not found for {attempt.attempt_id}: {model_path}")

    config_path = attempt.config_path
    if not config_path.exists():
        raise FileNotFoundError(
            f"Cannot regenerate model.py because config.json is missing: {config_path}"
        )

    cmd = [
        sys.executable,
        str(codegen_path.expanduser().resolve()),
        str(config_path),
        str(model_path),
    ]
    if dry_run:
        print("Codegen command:")
        print(" ".join(cmd))
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return model_path


def build_coderun_command(
    args: argparse.Namespace,
    attempt: Attempt,
    manifest: dict[str, Any],
    model_path: Path,
) -> list[str]:
    data_path = args.data_path
    if data_path is None:
        metadata_path = metadata_default(attempt, "dataset_path")
        data_path = Path(metadata_path) if metadata_path else DEFAULT_DATA_PATH

    output_path = args.output or attempt.default_output_path

    epochs = args.epochs if args.epochs is not None else manifest_default(manifest, "epochs")
    timesteps = args.timesteps if args.timesteps is not None else manifest_default(manifest, "timesteps")
    batch_size = args.batch_size if args.batch_size is not None else manifest_default(manifest, "batch_size")
    lr = args.lr if args.lr is not None else manifest_default(manifest, "lr")
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else manifest_default(manifest, "num_workers")
    )
    seed = args.seed if args.seed is not None else manifest_default(manifest, "base_seed")

    cmd = [
        sys.executable,
        str(args.coderun_path.expanduser().resolve()),
        str(model_path),
        "--data-path",
        str(data_path.expanduser().resolve()),
        "--output",
        str(output_path.expanduser().resolve()),
    ]
    optional_args = [
        ("--epochs", epochs),
        ("--timesteps", timesteps),
        ("--batch-size", batch_size),
        ("--lr", lr),
        ("--num-workers", num_workers),
        ("--seed", seed),
        ("--device", args.device),
        ("--model-kwargs", args.model_kwargs),
    ]
    for flag, value in optional_args:
        if value is not None:
            cmd.extend([flag, str(value)])
    return cmd


def print_selection(attempt: Attempt, model_path: Path) -> None:
    acc = attempt.best_test_accuracy
    acc_text = f"{acc:.4f}" if acc is not None else "n/a"
    print(f"Selected: {attempt.attempt_id}")
    print(f"Rank: {attempt.rank or 'n/a'}")
    print(f"Model: {attempt.model_name or 'n/a'}")
    print(f"Best accuracy: {acc_text}")
    print(f"Attempt dir: {attempt.attempt_dir}")
    print(f"Model path: {model_path}")


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    history_rows = read_jsonl(run_dir / "history.jsonl")
    manifest = load_manifest(run_dir)
    attempts = rank_attempts(run_dir, history_rows)

    if args.list:
        print_attempts(attempts, args.top)
        return 0

    attempt = select_attempt(attempts, args.selector)
    model_path = generate_model_if_needed(
        attempt=attempt,
        codegen_path=args.codegen_path,
        dry_run=args.dry_run,
        no_codegen=args.no_codegen,
    )
    cmd = build_coderun_command(args, attempt, manifest, model_path)
    print_selection(attempt, model_path)
    print("Run command:")
    print(" ".join(cmd))
    if args.dry_run:
        return 0
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
