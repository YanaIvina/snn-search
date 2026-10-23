from __future__ import annotations

import argparse
import datetime as dt
import html
import json
from pathlib import Path
from statistics import mean
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = SCRIPT_DIR / "search_runs" / "exp2"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"history file not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(json.loads(stripped))
    return rows


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_iso(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return dt.datetime.fromisoformat(value)
    except ValueError:
        return None


def _duration_minutes(started_at: Any, ended_at: Any) -> float | None:
    start = _parse_iso(started_at)
    end = _parse_iso(ended_at)
    if start is None or end is None:
        return None
    return (end - start).total_seconds() / 60.0


def _shorten(text: Any, max_len: int = 110) -> str:
    if not isinstance(text, str):
        return ""
    normalized = " ".join(text.split())
    if len(normalized) <= max_len:
        return normalized
    return normalized[: max_len - 3] + "..."


def _attempt_dir(run_dir: Path, row: dict[str, Any]) -> Path:
    attempt_id = str(row.get("attempt_id", "")).strip()
    if attempt_id:
        local_attempt_dir = (run_dir / "attempts" / attempt_id).resolve()
        if local_attempt_dir.exists():
            return local_attempt_dir
    config_path = row.get("config_path")
    if isinstance(config_path, str) and config_path:
        return Path(config_path).resolve().parent
    return (run_dir / "attempts" / attempt_id).resolve()


def _relative_link(base_dir: Path, target: Path) -> str:
    if not target.exists():
        return ""
    try:
        return target.resolve().relative_to(base_dir.resolve()).as_posix()
    except ValueError:
        return ""


def build_view_rows(run_dir: Path, history_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    successes = [
        row for row in history_rows
        if row.get("status") == "success" and _to_float(row.get("best_test_accuracy")) is not None
    ]
    best_order = sorted(
        successes,
        key=lambda row: _to_float(row.get("best_test_accuracy")) or -1.0,
        reverse=True,
    )
    rank_map = {str(row.get("attempt_id")): idx + 1 for idx, row in enumerate(best_order)}

    view_rows: list[dict[str, Any]] = []
    for row in sorted(history_rows, key=lambda item: int(item.get("attempt_index", 0))):
        attempt_dir = _attempt_dir(run_dir, row)
        best_acc = _to_float(row.get("best_test_accuracy"))
        test_acc = _to_float(row.get("test_accuracy"))
        params = row.get("param_count")
        params_m = f"{float(params) / 1_000_000:.2f}M" if isinstance(params, (int, float)) else "n/a"
        duration = _duration_minutes(row.get("started_at"), row.get("ended_at"))
        duration_text = f"{duration:.1f}" if duration is not None else "n/a"
        checkpoint_meta = row.get("checkpoint_metadata", {})
        model_name = checkpoint_meta.get("model_name", "") if isinstance(checkpoint_meta, dict) else ""
        view_rows.append(
            {
                "attempt_index": int(row.get("attempt_index", 0)),
                "attempt_id": str(row.get("attempt_id", "")),
                "rank": rank_map.get(str(row.get("attempt_id")), ""),
                "status": str(row.get("status", "")),
                "best_test_accuracy": best_acc,
                "test_accuracy": test_acc,
                "param_count": params_m,
                "proposal_retries_used": row.get("proposal_retries_used", "n/a"),
                "duration_min": duration_text,
                "config_signature": str(row.get("config_signature", "")),
                "model_name": model_name,
                "rationale": _shorten(row.get("rationale", "")),
                "attempt_dir": _relative_link(run_dir, attempt_dir),
                "diagram_path": _relative_link(run_dir, attempt_dir / "diagram.png"),
                "config_path": _relative_link(run_dir, attempt_dir / "config.json"),
                "summary_path": _relative_link(run_dir, attempt_dir / "attempt_summary.json"),
            }
        )
    return view_rows


def build_summary(history_rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [
        row for row in history_rows
        if row.get("status") == "success" and _to_float(row.get("best_test_accuracy")) is not None
    ]
    best_row = None
    if successes:
        best_row = max(successes, key=lambda row: _to_float(row.get("best_test_accuracy")) or -1.0)
    best_scores = [_to_float(row.get("best_test_accuracy")) for row in successes]
    best_scores = [score for score in best_scores if score is not None]
    return {
        "total_attempts": len(history_rows),
        "successful_attempts": len(successes),
        "best_attempt_id": best_row.get("attempt_id", "") if best_row else "",
        "best_accuracy": _to_float(best_row.get("best_test_accuracy")) if best_row else None,
        "mean_best_accuracy": mean(best_scores) if best_scores else None,
    }


def render_markdown(run_dir: Path, history_rows: list[dict[str, Any]]) -> str:
    view_rows = build_view_rows(run_dir, history_rows)
    summary = build_summary(history_rows)
    best_acc = summary["best_accuracy"]
    mean_acc = summary["mean_best_accuracy"]
    lines = [
        "# Результаты эксперимента",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Всего попыток: **{summary['total_attempts']}**",
        f"- Успешных попыток: **{summary['successful_attempts']}**",
        f"- Лучшая попытка: **{summary['best_attempt_id'] or 'n/a'}**",
        f"- Лучшая `best_test_accuracy`: **{best_acc:.4f}**" if isinstance(best_acc, float) else "- Лучшая `best_test_accuracy`: **n/a**",
        f"- Средняя `best_test_accuracy`: **{mean_acc:.4f}**" if isinstance(mean_acc, float) else "- Средняя `best_test_accuracy`: **n/a**",
        "",
        "| # | Attempt | Rank | Status | Best acc | Test acc | Params | Retries | Time, min | Signature | Model | Rationale |",
        "|---:|---|---:|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in view_rows:
        best_text = f"{row['best_test_accuracy']:.4f}" if isinstance(row["best_test_accuracy"], float) else "n/a"
        test_text = f"{row['test_accuracy']:.4f}" if isinstance(row["test_accuracy"], float) else "n/a"
        attempt_cell = row["attempt_id"]
        if row["summary_path"]:
            attempt_cell = f"[{row['attempt_id']}]({row['summary_path']})"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["attempt_index"]),
                    attempt_cell,
                    str(row["rank"]),
                    row["status"],
                    best_text,
                    test_text,
                    row["param_count"],
                    str(row["proposal_retries_used"]),
                    row["duration_min"],
                    row["config_signature"].replace("|", "/"),
                    row["model_name"].replace("|", "/"),
                    row["rationale"].replace("|", "/"),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_html(run_dir: Path, history_rows: list[dict[str, Any]]) -> str:
    view_rows = build_view_rows(run_dir, history_rows)
    summary = build_summary(history_rows)
    best_attempt_id = summary["best_attempt_id"]
    best_acc = summary["best_accuracy"]
    mean_acc = summary["mean_best_accuracy"]
    experiment_name = run_dir.name or "experiment"

    parts = [
        "<!DOCTYPE html>",
        "<html lang='ru'>",
        "<head>",
        "<meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        f"<title>Результаты {html.escape(experiment_name)}</title>",
        "<style>",
        "body{font-family:Segoe UI,Arial,sans-serif;margin:24px;background:#f6f8fb;color:#1d2433;}",
        ".wrap{max-width:1800px;margin:0 auto;}",
        ".summary{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:18px 0 24px;}",
        ".card{background:#fff;border:1px solid #d8dee9;border-radius:12px;padding:14px 16px;box-shadow:0 1px 3px rgba(0,0,0,.04);}",
        ".label{font-size:12px;text-transform:uppercase;color:#6b7280;letter-spacing:.04em;}",
        ".value{font-size:24px;font-weight:700;margin-top:6px;}",
        "table{width:100%;border-collapse:collapse;background:#fff;border:1px solid #d8dee9;}",
        "th,td{padding:10px 12px;border-bottom:1px solid #e5e7eb;vertical-align:top;text-align:left;font-size:14px;}",
        "th{position:sticky;top:0;background:#eef2f7;z-index:1;}",
        "tr.best{background:#edf8ee;}",
        ".mono{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;}",
        ".ok{color:#146c2e;font-weight:600;}",
        ".bad{color:#9f1239;font-weight:600;}",
        ".small{font-size:12px;color:#6b7280;}",
        ".links a{margin-right:10px;white-space:nowrap;}",
        "a{color:#0f4c81;text-decoration:none;} a:hover{text-decoration:underline;}",
        "</style>",
        "</head>",
        "<body>",
        "<div class='wrap'>",
        f"<h1>Результаты {html.escape(experiment_name)}</h1>",
        f"<div class='small mono'>{html.escape(str(run_dir))}</div>",
        "<div class='summary'>",
        f"<div class='card'><div class='label'>Всего попыток</div><div class='value'>{summary['total_attempts']}</div></div>",
        f"<div class='card'><div class='label'>Успешных попыток</div><div class='value'>{summary['successful_attempts']}</div></div>",
        f"<div class='card'><div class='label'>Лучшая попытка</div><div class='value mono'>{html.escape(str(best_attempt_id or 'n/a'))}</div></div>",
        f"<div class='card'><div class='label'>Лучшая best_test_accuracy</div><div class='value'>{best_acc:.4f}</div></div>" if isinstance(best_acc, float) else "<div class='card'><div class='label'>Лучшая best_test_accuracy</div><div class='value'>n/a</div></div>",
        f"<div class='card'><div class='label'>Средняя best_test_accuracy</div><div class='value'>{mean_acc:.4f}</div></div>" if isinstance(mean_acc, float) else "<div class='card'><div class='label'>Средняя best_test_accuracy</div><div class='value'>n/a</div></div>",
        "</div>",
        "<table>",
        "<thead><tr>"
        "<th>#</th><th>Attempt</th><th>Rank</th><th>Status</th><th>Best acc</th><th>Test acc</th>"
        "<th>Params</th><th>Retries</th><th>Time, min</th><th>Signature</th><th>Model</th><th>Links</th><th>Rationale</th>"
        "</tr></thead>",
        "<tbody>",
    ]
    for row in view_rows:
        best_text = f"{row['best_test_accuracy']:.4f}" if isinstance(row["best_test_accuracy"], float) else "n/a"
        test_text = f"{row['test_accuracy']:.4f}" if isinstance(row["test_accuracy"], float) else "n/a"
        row_class = "best" if row["attempt_id"] == best_attempt_id else ""
        status_class = "ok" if row["status"] == "success" else "bad"
        links: list[str] = []
        if row["summary_path"]:
            links.append(f"<a href='{html.escape(row['summary_path'])}'>summary</a>")
        if row["config_path"]:
            links.append(f"<a href='{html.escape(row['config_path'])}'>config</a>")
        if row["diagram_path"]:
            links.append(f"<a href='{html.escape(row['diagram_path'])}'>diagram</a>")
        parts.append(
            "<tr class='{row_class}'>".format(row_class=row_class)
            + f"<td>{row['attempt_index']}</td>"
            + f"<td class='mono'>{html.escape(row['attempt_id'])}</td>"
            + f"<td>{html.escape(str(row['rank']))}</td>"
            + f"<td class='{status_class}'>{html.escape(row['status'])}</td>"
            + f"<td>{best_text}</td>"
            + f"<td>{test_text}</td>"
            + f"<td>{html.escape(row['param_count'])}</td>"
            + f"<td>{html.escape(str(row['proposal_retries_used']))}</td>"
            + f"<td>{html.escape(row['duration_min'])}</td>"
            + f"<td>{html.escape(row['config_signature'])}</td>"
            + f"<td>{html.escape(row['model_name'])}</td>"
            + f"<td class='links'>{''.join(links)}</td>"
            + f"<td>{html.escape(row['rationale'])}</td>"
            + "</tr>"
        )
    parts.extend(["</tbody>", "</table>", "</div>", "</body>", "</html>"])
    return "\n".join(parts) + "\n"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render experiment history as Markdown and HTML tables.")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR, help="Search run directory.")
    parser.add_argument("--history-path", type=Path, default=None, help="Override history.jsonl path.")
    parser.add_argument("--markdown-out", type=Path, default=None, help="Output Markdown file.")
    parser.add_argument("--html-out", type=Path, default=None, help="Output HTML file.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    run_dir = args.run_dir.resolve()
    history_path = args.history_path.resolve() if args.history_path else run_dir / "history.jsonl"
    markdown_out = args.markdown_out.resolve() if args.markdown_out else run_dir / "results_table.md"
    html_out = args.html_out.resolve() if args.html_out else run_dir / "results_table.html"

    history_rows = _read_jsonl(history_path)
    markdown_out.write_text(render_markdown(run_dir, history_rows), encoding="utf-8")
    html_out.write_text(render_html(run_dir, history_rows), encoding="utf-8")

    print(f"[render] run_dir={run_dir}")
    print(f"[render] history_rows={len(history_rows)}")
    print(f"[render] markdown={markdown_out}")
    print(f"[render] html={html_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
