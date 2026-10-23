#!/usr/bin/env python3
"""
Iterative architecture-search agent for SNN configs driven by snn_codegen + snn_coderun.

The agent loop is:
1. Read previous attempts from disk.
2. Build compact context.
3. Ask an LLM for the next JSON architecture.
4. Validate JSON against supported codegen/coderun constraints.
5. Save config.
6. Run snn_codegen.py -> generated model.py.
7. Run snn_coderun.py -> training checkpoint/metrics.
8. Persist full attempt artifacts and append to history.

Examples:
    # Run two offline debug attempts without calling an external LLM
    python config/snn_search_agent.py run \
      --provider mock \
      --attempts 2 \
      --run-dir config/search_runs/mock_debug \
      --data-path data/google_speech_commands_mels.npz

    # Requires .env in the repo root, for example:
    # OPENAI_API_KEY=sk-...
    python config/snn_search_agent.py run \
      --attempts 15 \
      --run-dir config/search_runs/exp1 \
      --data-path data/google_speech_commands_mels.npz \
      --save-diagram

    python config/snn_search_agent.py run --attempts 15 --run-dir config/search_runs/exp1 --data-path data/google_speech_commands_mels.npz --save-diagram


    # Inspect previous attempts
    python config/snn_search_agent.py history \
      --run-dir config/search_runs/exp1 \
      --query failed
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_CODEGEN_PATH = SCRIPT_DIR / "snn_codegen.py"
DEFAULT_CODERUN_PATH = SCRIPT_DIR / "snn_coderun.py"
DEFAULT_SEED_CONFIG = SCRIPT_DIR / "examples" / "config_1to3to1_notebook.json"
DEFAULT_RUN_DIR = SCRIPT_DIR / "search_runs" / "default"
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

EXPECTED_INPUT_SIZE = 4096
EXPECTED_OUTPUT_SIZE = 30
MAX_MODEL_PARAMETERS = 10_000_000

CLI_EXAMPLES = textwrap.dedent(
    """
    Examples:
      python config/snn_search_agent.py run --provider mock --attempts 2 \
        --run-dir config/search_runs/mock_debug \
        --data-path data/google_speech_commands_mels.npz

      # Requires .env in the repo root with OPENAI_API_KEY=...
      python config/snn_search_agent.py run --provider openai --llm-model gpt-4.1-mini \
        --attempts 5 --run-dir config/search_runs/exp1 \
        --data-path data/google_speech_commands_mels.npz

      python config/snn_search_agent.py history --run-dir config/search_runs/exp1 --query failed
    """
).strip()

NODE_TYPES = {"input", "reservoir", "output"}
TOP_LEVEL_KEYS = {
    "model_name",
    "nodes",
    "edges",
    "neuron_params",
    "surrogate_params",
    "conv_frontend",
}
NODE_KEYS = {"id", "type", "size", "tau_short", "tau_long", "fraction_long"}
EDGE_KEYS = {"from", "to", "bias", "weight_init"}
NEURON_KEYS = {"vth", "dt", "fraction_long", "tau_short", "tau_long"}
SURROGATE_KEYS = {"h", "l", "s"}
CONV_KEYS = {"enabled", "in_channels", "channels", "use_bn", "hw", "pool"}
WEIGHT_INIT_KEYS = {
    "type",
    "a",
    "low",
    "high",
    "mean",
    "std",
    "sparsity",
    "gain",
}
SUPPORTED_WEIGHT_INITS = {
    "default",
    "pytorch_default",
    "kaiming_uniform",
    "kaiming_normal",
    "xavier_uniform",
    "xavier_normal",
    "uniform",
    "normal",
    "sparse",
    "orthogonal",
    "zeros",
    "eye",
}

NODE_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
CLASS_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _now_utc_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def load_env_file(env_path: Path, override: bool = False) -> dict[str, str]:
    env_path = env_path.expanduser().resolve()
    if not env_path.exists():
        raise FileNotFoundError(f"Required .env file not found: {env_path}")
    if not env_path.is_file():
        raise ValueError(f"Expected .env file path, got: {env_path}")

    loaded: dict[str, str] = {}
    with env_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                raise ValueError(f"Malformed .env line {line_no} in {env_path}: missing `=`")
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if not key or not NODE_ID_RE.match(key):
                raise ValueError(
                    f"Malformed .env line {line_no} in {env_path}: invalid variable name `{key}`"
                )
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
                value = value[1:-1]
            loaded[key] = value
            if override or key not in os.environ:
                os.environ[key] = value
    return loaded


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON object expected in {path}, got {type(obj).__name__}.")
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{i}: {exc}") from exc
            if isinstance(obj, dict):
                items.append(obj)
    return items


def _json_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _config_signature(cfg: dict[str, Any]) -> str:
    nodes = cfg.get("nodes", [])
    edges = cfg.get("edges", [])
    n_res = sum(1 for n in nodes if isinstance(n, dict) and n.get("type") == "reservoir")
    total_res_neurons = sum(
        int(n.get("size", 0))
        for n in nodes
        if isinstance(n, dict) and n.get("type") == "reservoir" and isinstance(n.get("size"), int)
    )
    rec_edges = sum(
        1
        for e in edges
        if isinstance(e, dict) and e.get("from") == e.get("to")
    )
    use_cnn = bool(cfg.get("conv_frontend", {}).get("enabled", False))
    return (
        f"res={n_res}, res_neurons={total_res_neurons}, edges={len(edges)}, "
        f"recurrent={rec_edges}, cnn={'on' if use_cnn else 'off'}"
    )


def _architecture_fingerprint(cfg: dict[str, Any]) -> str:
    """Fingerprint the structural architecture, ignoring names and weight init details."""
    nodes = cfg.get("nodes", [])
    edges = cfg.get("edges", [])
    id_to_role: dict[str, str] = {}
    reservoir_sizes: list[Any] = []
    input_count = 0
    output_count = 0
    reservoir_count = 0

    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id", ""))
        node_type = node.get("type")
        if node_type == "input":
            input_count += 1
            role = "input" if input_count == 1 else f"input{input_count}"
        elif node_type == "output":
            output_count += 1
            role = "output" if output_count == 1 else f"output{output_count}"
        elif node_type == "reservoir":
            reservoir_count += 1
            role = f"res{reservoir_count}"
            size = node.get("size")
            reservoir_sizes.append(size if isinstance(size, int) else str(size))
        else:
            role = f"{node_type or 'unknown'}:{node_id}"
        if node_id:
            id_to_role[node_id] = role

    canonical_edges: list[str] = []
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = id_to_role.get(str(edge.get("from", "")), str(edge.get("from", "")))
        dst = id_to_role.get(str(edge.get("to", "")), str(edge.get("to", "")))
        canonical_edges.append(f"{src}->{dst}")

    payload = {
        "reservoir_sizes": reservoir_sizes,
        "edges": sorted(canonical_edges),
        "conv_frontend": cfg.get("conv_frontend", {}),
    }
    return _json_sha256(payload)


def _history_config_path(run_dir: Path, row: dict[str, Any]) -> Path | None:
    config_path = row.get("config_path")
    if isinstance(config_path, str) and config_path:
        path = Path(config_path)
        return path if path.is_absolute() else run_dir / path

    attempt_id = row.get("attempt_id")
    if isinstance(attempt_id, str) and attempt_id:
        return run_dir / "attempts" / attempt_id / "config.json"
    return None


def _load_history_config(run_dir: Path, row: dict[str, Any]) -> dict[str, Any] | None:
    path = _history_config_path(run_dir, row)
    if path is None or not path.exists():
        return None
    try:
        cfg = _load_json(path)
    except (OSError, ValueError):
        return None
    return cfg if isinstance(cfg, dict) else None


def _best_accuracy_text(rows: list[dict[str, Any]]) -> str:
    scores = [
        float(row["best_test_accuracy"])
        for row in rows
        if isinstance(row.get("best_test_accuracy"), (int, float))
    ]
    return f"{max(scores):.4f}" if scores else "n/a"


def _summarize_attempts(rows: list[dict[str, Any]], limit: int = 4) -> str:
    attempt_ids = [str(row.get("attempt_id", "n/a")) for row in rows]
    if len(attempt_ids) <= limit:
        return ", ".join(attempt_ids)
    return ", ".join(attempt_ids[:limit]) + f" ... {attempt_ids[-1]}"


def _find_seen_config_duplicate(
    candidate_config: dict[str, Any],
    *,
    history: list[dict[str, Any]],
    run_dir: Path,
) -> str | None:
    candidate_sha = _json_sha256(candidate_config)
    for row in history:
        if row.get("config_sha256") == candidate_sha:
            return (
                "Duplicate config: this exact JSON architecture was already tested "
                f"as {row.get('attempt_id', 'unknown')} with "
                f"best_test_accuracy={row.get('best_test_accuracy', 'n/a')}. "
                "Generate a structurally different config."
            )

    candidate_fingerprint = _architecture_fingerprint(candidate_config)
    for row in history:
        fingerprint = row.get("architecture_fingerprint")
        if not isinstance(fingerprint, str) or not fingerprint:
            previous_config = _load_history_config(run_dir, row)
            if previous_config is None:
                continue
            fingerprint = _architecture_fingerprint(previous_config)
        if fingerprint == candidate_fingerprint:
            return (
                "Duplicate architecture structure: this graph/size/CNN skeleton was already tested "
                f"as {row.get('attempt_id', 'unknown')} with "
                f"best_test_accuracy={row.get('best_test_accuracy', 'n/a')}. "
                "Changing only names, JSON order, rationale, or weight initialization is not enough; "
                "generate a structurally different config."
            )

    return None


def extract_first_json_object(text: str) -> dict[str, Any]:
    """
    Parse first JSON object from free-form LLM text.
    Handles plain JSON, fenced blocks, and prefixed/suffixed chatter.
    """
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty LLM response.")

    # 1) direct parse
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # 2) fenced json block
    for match in JSON_BLOCK_RE.finditer(text):
        block = match.group(1).strip()
        if not block:
            continue
        try:
            parsed = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    # 3) first balanced {...} in whole text
    start = text.find("{")
    while start != -1:
        depth = 0
        for idx in range(start, len(text)):
            ch = text[idx]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        parsed = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = text.find("{", start + 1)

    raise ValueError("Could not extract a JSON object from LLM response.")


@dataclass
class Proposal:
    config: dict[str, Any]
    rationale: str
    reasoning_summary: str
    raw_text: str
    retry_index: int


def parse_proposal(raw_text: str, retry_index: int) -> Proposal:
    payload = extract_first_json_object(raw_text)
    if "config" in payload:
        cfg = payload["config"]
        rationale = str(payload.get("rationale", "")).strip()
        reasoning_summary = str(payload.get("reasoning_summary", "")).strip()
    else:
        cfg = payload
        rationale = ""
        reasoning_summary = ""
    if not isinstance(cfg, dict):
        raise ValueError("Proposal must include object field `config` or be a config object itself.")
    if not rationale:
        rationale = "Модель не вернула rationale."
    if not reasoning_summary:
        reasoning_summary = rationale
    return Proposal(
        config=cfg,
        rationale=rationale,
        reasoning_summary=reasoning_summary,
        raw_text=raw_text,
        retry_index=retry_index,
    )


def _validate_weight_init(path: str, wi: Any, errors: list[str]) -> None:
    if not isinstance(wi, dict):
        errors.append(f"{path} must be an object.")
        return
    unknown = sorted(set(wi) - WEIGHT_INIT_KEYS)
    if unknown:
        errors.append(f"{path} has unsupported keys: {unknown}")

    wtype = wi.get("type")
    if not isinstance(wtype, str):
        errors.append(f"{path}.type must be a string.")
        return
    if wtype not in SUPPORTED_WEIGHT_INITS:
        errors.append(f"{path}.type `{wtype}` is unsupported by snn_codegen.py.")

    for key in ("a", "low", "high", "mean", "std", "sparsity", "gain"):
        if key in wi and not _is_number(wi[key]):
            errors.append(f"{path}.{key} must be numeric.")

    if wtype == "uniform":
        low = wi.get("low", -0.1)
        high = wi.get("high", 0.1)
        if _is_number(low) and _is_number(high) and low >= high:
            errors.append(f"{path}: `low` must be < `high` for uniform init.")
    if wtype == "normal":
        std = wi.get("std", 0.01)
        if _is_number(std) and std <= 0:
            errors.append(f"{path}: `std` must be > 0 for normal init.")
    if wtype == "sparse":
        sparsity = wi.get("sparsity", 0.9)
        if _is_number(sparsity) and not (0 <= float(sparsity) < 1):
            errors.append(f"{path}: `sparsity` must be in [0, 1).")


def _validate_tau_pair(path: str, value: Any, errors: list[str]) -> None:
    if not isinstance(value, list) or len(value) != 2:
        errors.append(f"{path} must be a list of two numbers.")
        return
    a, b = value
    if not _is_number(a) or not _is_number(b):
        errors.append(f"{path} entries must be numeric.")
        return
    if float(a) >= float(b):
        errors.append(f"{path}[0] must be < {path}[1].")


def estimate_param_count(cfg: dict[str, Any]) -> dict[str, int]:
    """
    Estimate trainable parameters for the model generated by the current snn_codegen.py.
    This includes:
    - Linear weights and optional biases
    - Reservoir tau_m parameters
    - Conv front-end conv weights/biases, batch norm affine params, and conv tau_m
    """
    nodes = cfg["nodes"]
    edges = cfg["edges"]
    node_map = {n["id"]: n for n in nodes}

    incoming: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    recurrent: dict[str, dict[str, Any]] = {}
    for e in edges:
        s = e["from"]
        d = e["to"]
        if s == d:
            recurrent[s] = e
        else:
            incoming.setdefault(d, []).append((s, e))

    cfe = cfg.get("conv_frontend", {})
    use_cnn = bool(cfe.get("enabled", False))
    conv_total = 0
    conv_tau = 0
    cnn_out_size = 0
    if use_cnn:
        cnn_in_ch = int(cfe.get("in_channels", 1))
        cnn_chs = cfe.get("channels", [32, 64, 128])
        cnn_use_bn = bool(cfe.get("use_bn", True))
        cnn_hw = cfe.get("hw", [64, 64])
        if isinstance(cnn_hw, int):
            cnn_hw = [cnn_hw, cnn_hw]
        h_cur, w_cur = int(cnn_hw[0]), int(cnn_hw[1])
        prev_ch = cnn_in_ch
        for ch in cnn_chs:
            ch = int(ch)
            conv_weight = ch * prev_ch * 3 * 3
            conv_bias = 0 if cnn_use_bn else ch
            bn_params = 2 * ch if cnn_use_bn else 0
            block_feat = ch * h_cur * w_cur
            conv_total += conv_weight + conv_bias + bn_params + block_feat
            conv_tau += block_feat
            prev_ch = ch
            h_cur //= 2
            w_cur //= 2
        cnn_out_size = prev_ch * h_cur * w_cur

    linear_total = 0
    linear_bias_total = 0
    reservoir_tau = sum(
        int(n.get("size", 0))
        for n in nodes
        if n["type"] == "reservoir"
    )

    def add_linear(src_sz: int, dst_sz: int, bias: bool) -> None:
        nonlocal linear_total, linear_bias_total
        linear_total += src_sz * dst_sz
        if bias:
            linear_bias_total += dst_sz

    for node in nodes:
        nid = node["id"]
        ntype = node["type"]
        if ntype == "output":
            in_e = incoming.get(nid, [])
            if len(in_e) == 1:
                src_id, e = in_e[0]
                add_linear(
                    int(node_map[src_id]["size"]),
                    int(node["size"]),
                    bool(e.get("bias", False)),
                )
            elif len(in_e) > 1:
                src_total = sum(int(node_map[src_id]["size"]) for src_id, _ in in_e)
                add_linear(src_total, int(node["size"]), bool(in_e[0][1].get("bias", False)))
        elif ntype == "reservoir":
            in_e = incoming.get(nid, [])
            res_srcs = [(s, e) for s, e in in_e if node_map[s]["type"] == "reservoir"]
            other_srcs = [(s, e) for s, e in in_e if node_map[s]["type"] != "reservoir"]
            for src_id, e in other_srcs:
                src_node = node_map[src_id]
                src_sz = cnn_out_size if (use_cnn and src_node["type"] == "input") else int(src_node["size"])
                add_linear(src_sz, int(node["size"]), bool(e.get("bias", False)))
            if len(res_srcs) > 1:
                merge_in = sum(int(node_map[s]["size"]) for s, _ in res_srcs)
                add_linear(merge_in, int(node["size"]), bool(res_srcs[0][1].get("bias", False)))
            elif len(res_srcs) == 1:
                src_id, e = res_srcs[0]
                add_linear(int(node_map[src_id]["size"]), int(node["size"]), bool(e.get("bias", False)))
            if nid in recurrent:
                e = recurrent[nid]
                size = int(node["size"])
                add_linear(size, size, bool(e.get("bias", False)))

    total = linear_total + linear_bias_total + reservoir_tau + conv_total
    return {
        "linear_weights": linear_total,
        "linear_biases": linear_bias_total,
        "reservoir_tau": reservoir_tau,
        "conv_frontend": conv_total,
        "conv_tau": conv_tau,
        "total": total,
    }


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """
    Strict validation aligned with actual constraints of current snn_codegen.py + snn_coderun.py.
    """
    errors: list[str] = []

    unknown_top = sorted(set(cfg) - TOP_LEVEL_KEYS)
    if unknown_top:
        errors.append(f"Unsupported top-level keys: {unknown_top}")

    model_name = cfg.get("model_name", "SNNModel")
    if not isinstance(model_name, str) or not CLASS_NAME_RE.match(model_name):
        errors.append("`model_name` must be a valid Python class identifier.")

    nodes = cfg.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        errors.append("`nodes` must be a non-empty list.")
        return errors

    node_ids: set[str] = set()
    node_map: dict[str, dict[str, Any]] = {}
    input_nodes: list[str] = []
    output_nodes: list[str] = []
    reservoir_nodes: list[str] = []

    for i, node in enumerate(nodes):
        path = f"nodes[{i}]"
        if not isinstance(node, dict):
            errors.append(f"{path} must be an object.")
            continue
        unknown = sorted(set(node) - NODE_KEYS)
        if unknown:
            errors.append(f"{path} has unsupported keys: {unknown}")

        nid = node.get("id")
        if not isinstance(nid, str) or not nid:
            errors.append(f"{path}.id must be a non-empty string.")
            continue
        if not NODE_ID_RE.match(nid):
            errors.append(f"{path}.id `{nid}` is not a valid Python identifier.")
            continue
        if nid in node_ids:
            errors.append(f"Duplicate node id: `{nid}`.")
            continue

        ntype = node.get("type")
        if ntype not in NODE_TYPES:
            errors.append(f"{path}.type must be one of {sorted(NODE_TYPES)}.")
            continue

        nsize = node.get("size")
        if not isinstance(nsize, int) or nsize <= 0:
            errors.append(f"{path}.size must be a positive integer.")
            continue

        if "fraction_long" in node:
            fl = node["fraction_long"]
            if not _is_number(fl) or not (0 <= float(fl) <= 1):
                errors.append(f"{path}.fraction_long must be in [0, 1].")
        if "tau_short" in node:
            _validate_tau_pair(f"{path}.tau_short", node["tau_short"], errors)
        if "tau_long" in node:
            _validate_tau_pair(f"{path}.tau_long", node["tau_long"], errors)

        node_ids.add(nid)
        node_map[nid] = node
        if ntype == "input":
            input_nodes.append(nid)
        elif ntype == "output":
            output_nodes.append(nid)
        elif ntype == "reservoir":
            reservoir_nodes.append(nid)

    if len(input_nodes) != 1:
        errors.append("Exactly one input node is required.")
    if len(output_nodes) != 1:
        errors.append("Exactly one output node is required.")
    if not reservoir_nodes:
        errors.append("At least one reservoir node is required.")

    if input_nodes:
        in_size = node_map[input_nodes[0]].get("size")
        if in_size != EXPECTED_INPUT_SIZE:
            errors.append(
                f"Input node size must be {EXPECTED_INPUT_SIZE} for current snn_coderun dataset."
            )
    if output_nodes:
        out_size = node_map[output_nodes[0]].get("size")
        if out_size != EXPECTED_OUTPUT_SIZE:
            errors.append(
                f"Output node size must be {EXPECTED_OUTPUT_SIZE} for current label space."
            )

    edges = cfg.get("edges")
    if not isinstance(edges, list) or not edges:
        errors.append("`edges` must be a non-empty list.")
        return errors

    out_incoming = 0
    seen_edge_pairs: set[tuple[str, str]] = set()
    adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}
    for i, edge in enumerate(edges):
        path = f"edges[{i}]"
        if not isinstance(edge, dict):
            errors.append(f"{path} must be an object.")
            continue
        unknown = sorted(set(edge) - EDGE_KEYS)
        if unknown:
            errors.append(f"{path} has unsupported keys: {unknown}")

        src = edge.get("from")
        dst = edge.get("to")
        if not isinstance(src, str) or src not in node_map:
            errors.append(f"{path}.from must reference an existing node id.")
            continue
        if not isinstance(dst, str) or dst not in node_map:
            errors.append(f"{path}.to must reference an existing node id.")
            continue

        pair = (src, dst)
        if pair in seen_edge_pairs:
            errors.append(f"{path}: duplicate edge {src} -> {dst}.")
        seen_edge_pairs.add(pair)
        adjacency.setdefault(src, []).append(dst)

        src_type = node_map[src]["type"]
        dst_type = node_map[dst]["type"]

        if src_type == "output":
            errors.append(f"{path}: edges from output nodes are unsupported.")
        if dst_type == "input":
            errors.append(f"{path}: edges into input nodes are unsupported.")
        if src_type == "input" and dst_type != "reservoir":
            errors.append(f"{path}: input nodes may connect only to reservoir nodes.")
        if src_type == "reservoir" and dst_type not in {"reservoir", "output"}:
            errors.append(f"{path}: reservoir nodes may connect only to reservoir/output.")
        if src == dst and src_type != "reservoir":
            errors.append(f"{path}: self-loop is allowed only on reservoir nodes.")
        if dst_type == "output":
            out_incoming += 1
            if src_type != "reservoir":
                errors.append(f"{path}: output node must receive edges only from reservoirs.")

        if "bias" in edge and not isinstance(edge["bias"], bool):
            errors.append(f"{path}.bias must be boolean if provided.")
        if "weight_init" in edge:
            _validate_weight_init(f"{path}.weight_init", edge["weight_init"], errors)

    if out_incoming == 0:
        errors.append("Output node must have at least one incoming edge.")

    # Reachability: output should depend on input through directed graph.
    if input_nodes and output_nodes:
        src_input = input_nodes[0]
        dst_output = output_nodes[0]
        queue = [src_input]
        visited = {src_input}
        while queue:
            cur = queue.pop(0)
            for nxt in adjacency.get(cur, []):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        if dst_output not in visited:
            errors.append("No directed path from input to output.")

    neuron = cfg.get("neuron_params", {})
    if neuron is not None:
        if not isinstance(neuron, dict):
            errors.append("`neuron_params` must be an object if provided.")
        else:
            unknown = sorted(set(neuron) - NEURON_KEYS)
            if unknown:
                errors.append(f"`neuron_params` has unsupported keys: {unknown}")
            if "vth" in neuron and (not _is_number(neuron["vth"]) or float(neuron["vth"]) <= 0):
                errors.append("`neuron_params.vth` must be > 0.")
            if "dt" in neuron and (not _is_number(neuron["dt"]) or float(neuron["dt"]) <= 0):
                errors.append("`neuron_params.dt` must be > 0.")
            if "fraction_long" in neuron:
                fl = neuron["fraction_long"]
                if not _is_number(fl) or not (0 <= float(fl) <= 1):
                    errors.append("`neuron_params.fraction_long` must be in [0, 1].")
            if "tau_short" in neuron:
                _validate_tau_pair("neuron_params.tau_short", neuron["tau_short"], errors)
            if "tau_long" in neuron:
                _validate_tau_pair("neuron_params.tau_long", neuron["tau_long"], errors)

    surrogate = cfg.get("surrogate_params", {})
    if surrogate is not None:
        if not isinstance(surrogate, dict):
            errors.append("`surrogate_params` must be an object if provided.")
        else:
            unknown = sorted(set(surrogate) - SURROGATE_KEYS)
            if unknown:
                errors.append(f"`surrogate_params` has unsupported keys: {unknown}")
            for key in ("h", "l", "s"):
                if key in surrogate and not _is_number(surrogate[key]):
                    errors.append(f"`surrogate_params.{key}` must be numeric.")
            if "l" in surrogate and _is_number(surrogate["l"]) and float(surrogate["l"]) <= 0:
                errors.append("`surrogate_params.l` must be > 0.")
            if "s" in surrogate and _is_number(surrogate["s"]) and float(surrogate["s"]) <= 0:
                errors.append("`surrogate_params.s` must be > 0.")

    conv = cfg.get("conv_frontend", {})
    if conv is not None:
        if not isinstance(conv, dict):
            errors.append("`conv_frontend` must be an object if provided.")
        else:
            unknown = sorted(set(conv) - CONV_KEYS)
            if unknown:
                errors.append(f"`conv_frontend` has unsupported keys: {unknown}")
            enabled = bool(conv.get("enabled", False))
            if "enabled" in conv and not isinstance(conv["enabled"], bool):
                errors.append("`conv_frontend.enabled` must be boolean.")
            if "use_bn" in conv and not isinstance(conv["use_bn"], bool):
                errors.append("`conv_frontend.use_bn` must be boolean.")
            if "pool" in conv and conv["pool"] not in {"avg", "max"}:
                errors.append("`conv_frontend.pool` must be `avg` or `max`.")
            if "in_channels" in conv:
                in_ch = conv["in_channels"]
                if not isinstance(in_ch, int) or in_ch <= 0:
                    errors.append("`conv_frontend.in_channels` must be positive integer.")
            channels = conv.get("channels", [32, 64, 128])
            if not isinstance(channels, list) or not channels:
                errors.append("`conv_frontend.channels` must be a non-empty list.")
            else:
                for i, ch in enumerate(channels):
                    if not isinstance(ch, int) or ch <= 0:
                        errors.append(f"`conv_frontend.channels[{i}]` must be positive integer.")
            hw = conv.get("hw", [64, 64])
            if isinstance(hw, int):
                hw = [hw, hw]
            if not (isinstance(hw, list) and len(hw) == 2 and all(isinstance(v, int) and v > 0 for v in hw)):
                errors.append("`conv_frontend.hw` must be positive int or [H, W].")
            else:
                if enabled:
                    n_blocks = len(channels) if isinstance(channels, list) else 0
                    h_out = hw[0] // (2 ** max(n_blocks, 0))
                    w_out = hw[1] // (2 ** max(n_blocks, 0))
                    if h_out <= 0 or w_out <= 0:
                        errors.append("`conv_frontend.hw` is too small for number of pooling blocks.")
                    if input_nodes:
                        expected = int(conv.get("in_channels", 1)) * hw[0] * hw[1]
                        actual = node_map[input_nodes[0]]["size"]
                        if expected != actual:
                            errors.append(
                                "Input size must equal in_channels * H * W when conv_frontend is enabled."
                            )

    if not errors:
        pcount = estimate_param_count(cfg)
        if pcount["total"] > MAX_MODEL_PARAMETERS:
            errors.append(
                f"Model exceeds parameter limit: {pcount['total']:,} > {MAX_MODEL_PARAMETERS:,}."
            )

    return errors


class ExperimentMemory:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.history_path = run_dir / "history.jsonl"
        self.attempts_dir = run_dir / "attempts"

    def load(self) -> list[dict[str, Any]]:
        return _read_jsonl(self.history_path)

    def append(self, record: dict[str, Any]) -> None:
        _append_jsonl(self.history_path, record)

    def next_attempt_index(self) -> int:
        history = self.load()
        if not history:
            return 1
        max_idx = max(int(item.get("attempt_index", 0)) for item in history)
        return max_idx + 1

    def build_context(self, max_attempts: int = 8, max_chars: int = 4000) -> str:
        history = self.load()
        if not history:
            return "No previous attempts yet."

        successes = [
            h for h in history
            if h.get("status") == "success"
            and isinstance(h.get("best_test_accuracy"), (int, float))
        ]
        top = sorted(successes, key=lambda x: float(x.get("best_test_accuracy", -1.0)), reverse=True)[:3]
        recent = history if max_attempts <= 0 else history[-max_attempts:]
        failures = [h for h in recent if h.get("status") != "success"][-3:]

        repeated_by_sha: dict[str, list[dict[str, Any]]] = {}
        repeated_by_signature: dict[str, list[dict[str, Any]]] = {}
        for item in history:
            sha = item.get("config_sha256")
            if isinstance(sha, str) and sha:
                repeated_by_sha.setdefault(sha, []).append(item)
            signature = item.get("config_signature")
            if isinstance(signature, str) and signature:
                repeated_by_signature.setdefault(signature, []).append(item)

        exact_repeats = sorted(
            (
                (sha, items)
                for sha, items in repeated_by_sha.items()
                if len(items) >= 2
            ),
            key=lambda pair: len(pair[1]),
            reverse=True,
        )[:5]
        signature_repeats = sorted(
            (
                (signature, items)
                for signature, items in repeated_by_signature.items()
                if len(items) >= 5
            ),
            key=lambda pair: len(pair[1]),
            reverse=True,
        )[:5]

        lines: list[str] = []
        lines.append(f"Total attempts so far: {len(history)}")
        if top:
            lines.append("Top successful attempts:")
            for item in top:
                sha = str(item.get("config_sha256", ""))
                lines.append(
                    f"- #{item.get('attempt_index')} {item.get('attempt_id')} "
                    f"best_acc={float(item.get('best_test_accuracy')):.4f} "
                    f"| {item.get('config_signature', '')} "
                    f"| sha={sha[:12] if sha else 'n/a'}"
                )
        if exact_repeats:
            lines.append("Repeated exact configs to avoid:")
            for sha, items in exact_repeats:
                lines.append(
                    f"- sha={sha[:12]} count={len(items)} best={_best_accuracy_text(items)} "
                    f"attempts={_summarize_attempts(items)}"
                )
        if signature_repeats:
            lines.append("Exhausted/repeated signatures to avoid unless structurally changed:")
            for signature, items in signature_repeats:
                lines.append(
                    f"- count={len(items)} best={_best_accuracy_text(items)} "
                    f"| {signature} | attempts={_summarize_attempts(items)}"
                )
        lines.append("Most recent attempts:")
        for item in recent:
            best = item.get("best_test_accuracy")
            best_text = f"{float(best):.4f}" if isinstance(best, (int, float)) else "n/a"
            lines.append(
                f"- #{item.get('attempt_index')} {item.get('attempt_id')} "
                f"status={item.get('status')} best_acc={best_text} "
                f"reasoning={str(item.get('reasoning_summary', item.get('rationale', '')))[:160]}"
            )
        if failures:
            lines.append("Recent failure diagnostics:")
            for item in failures:
                lines.append(
                    f"- #{item.get('attempt_index')} {item.get('attempt_id')}: "
                    f"{str(item.get('error_message', ''))[:220]}"
                )

        context = "\n".join(lines)
        if len(context) > max_chars:
            context = context[: max_chars - 3] + "..."
        return context

    def search(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        query = query.lower().strip()
        items = self.load()
        if not query:
            return items[-limit:]
        result: list[dict[str, Any]] = []
        for item in items:
            blob = " ".join(
                str(item.get(k, ""))
                for k in (
                    "attempt_id",
                    "status",
                    "rationale",
                    "reasoning_summary",
                    "error_message",
                    "config_signature",
                )
            ).lower()
            if query in blob:
                result.append(item)
        return result[-limit:]


class ProposalProvider:
    def propose(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        attempt_index: int,
    ) -> str:
        raise NotImplementedError


class OpenAIProposalProvider(ProposalProvider):
    def __init__(
        self,
        *,
        model: str,
        api_key: str | None,
        base_url: str | None,
        temperature: float,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.temperature = temperature

    def propose(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        attempt_index: int,
    ) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for provider=openai.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Provider `openai` requires the `openai` Python package."
            ) from exc

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Prefer Responses API; fallback to Chat Completions for broader client compatibility.
        try:
            response = client.responses.create(
                model=self.model,
                input=messages,
                temperature=self.temperature,
            )
            text = getattr(response, "output_text", "") or ""
            if text.strip():
                return text
        except Exception:
            pass

        chat = client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=self.temperature,
        )
        content = chat.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [chunk.get("text", "") for chunk in content if isinstance(chunk, dict)]
            return "\n".join(parts)
        raise RuntimeError("OpenAI response did not contain text content.")


class MockProposalProvider(ProposalProvider):
    """
    Offline fallback provider for deterministic local testing.
    Produces small mutations around the seed config.
    """

    def __init__(self, seed_config: dict[str, Any], rng_seed: int = 42) -> None:
        self.seed_config = copy.deepcopy(seed_config)
        self.rng = random.Random(rng_seed)

    def propose(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        attempt_index: int,
    ) -> str:
        cfg = copy.deepcopy(self.seed_config)
        for node in cfg.get("nodes", []):
            if node.get("type") != "reservoir":
                continue
            base = int(node["size"])
            delta = self.rng.choice([-128, -64, 0, 64, 128])
            node["size"] = max(64, min(2048, base + delta))
            if self.rng.random() < 0.35:
                node["fraction_long"] = round(max(0.0, min(1.0, 0.5 + self.rng.uniform(-0.3, 0.3))), 3)

        for edge in cfg.get("edges", []):
            if self.rng.random() < 0.3:
                edge["weight_init"] = {"type": self.rng.choice(["kaiming_uniform", "xavier_uniform"])}
            if "bias" in edge:
                edge["bias"] = False

        rationale = (
            f"Mock proposal #{attempt_index}: немного изменены размеры резервуаров "
            "и типы инициализации, чтобы проверить компромисс между емкостью и устойчивостью."
        )
        reasoning_summary = (
            "Изменены размеры reservoir-узлов и часть инициализаций весов, чтобы проверить, "
            "даст ли дополнительная емкость прирост качества без потери устойчивости обучения."
        )
        return json.dumps(
            {
                "rationale": rationale,
                "reasoning_summary": reasoning_summary,
                "config": cfg,
            },
            ensure_ascii=False,
            indent=2,
        )


def build_prompts(
    *,
    seed_config: dict[str, Any],
    context_text: str,
    feedback: list[str],
) -> tuple[str, str]:
    system_prompt = textwrap.dedent(
        """
        Ты ассистент по поиску архитектур для SNN с reservoir-узлами.
        Отвечай только валидным JSON без markdown.
        Все текстовые поля ответа должны быть на русском языке.
        """
    ).strip()

    feedback_block = "\n".join(f"- {msg}" for msg in feedback) if feedback else "- нет"
    user_prompt = textwrap.dedent(
        f"""
        Предложи следующий JSON-конфиг архитектуры для текущего codebase.

        Требуемая схема ответа:
        {{
          "rationale": "короткое понятное объяснение",
          "reasoning_summary": "короткая явная сводка рассуждений, без скрытой цепочки мыслей",
          "config": <JSON compatible with snn_codegen.py>
        }}

        Жесткие ограничения:
        - Используй только поддерживаемые top-level поля: {sorted(TOP_LEVEL_KEYS)}.
        - Должен быть ровно один input-узел и ровно один output-узел.
        - Размер входа должен быть {EXPECTED_INPUT_SIZE}, размер выхода должен быть {EXPECTED_OUTPUT_SIZE}.
        - Каждый внутренний узел должен иметь тип "reservoir".
        - В сгенерированной модели должно быть не более {MAX_MODEL_PARAMETERS:,} обучаемых параметров.
        - Разрешенные типы узлов: input/reservoir/output.
        - Разрешенные направления ребер: input->reservoir, reservoir->reservoir (включая self-loop), reservoir->output.
        - Нельзя использовать неподдерживаемые поля.
        - `weight_init.type` должен быть одним из: {sorted(SUPPORTED_WEIGHT_INITS)}.
        - Ответ должен быть только JSON, без текста вне JSON.
        - `rationale` и `reasoning_summary` должны быть написаны по-русски.
        - `reasoning_summary` должен быть кратким и явным. Не выводи скрытую chain-of-thought.

        Цель поиска:
        - Найти архитектуру, которая покажет максимальную test accuracy на текущем датасете при совместимости с `snn_coderun.py`.
        - Соблюдай баланс между exploitation и exploration:
          используй сильные идеи из лучших прошлых попыток, но не ограничивайся только малыми вариациями одной и той же схемы.
        - Если недавняя архитектура уже сильная, можно наследовать только отдельные принципы, но не полный каркас. После 3-5 попыток без улучшения та же сигнатура считается исчерпанной.

        Анти-повтор и требования к новизне:
        - Запрещено предлагать конфиг, который уже был проверен ранее или является его почти точной копией.
        - Повтором считается архитектура с тем же числом reservoir-узлов, тем же вектором размеров reservoir, тем же набором ребер `from->to`, тем же набором self-loop и тем же `conv_frontend`.
        - Не считай новой архитектурой изменения только в `rationale`, `reasoning_summary`, именах узлов, `model_name`, порядке JSON-полей, незначительных `weight_init`-вариациях или замене 1-2 init-типов при той же топологии.
        - Новый конфиг обязан отличаться минимум по двум структурным осям: число reservoir-узлов; распределение размеров reservoir-узлов; topology/adjacency graph ребер; состав skip-связей; схема readout в output; набор recurrent/self-loop связей; параметры CNN frontend.
        - Если одна и та же сигнатура уже часто встречалась в истории и не улучшила лучший результат, считай этот регион поиска исчерпанным и выбери другую топологическую гипотезу.
        - Можно наследовать идею сильной попытки, но нельзя воспроизводить ее каркас без структурной новизны.
        - Выбери одну основную стратегию exploration для этой попытки: `topology_explore`, `size_explore`, `recurrence_explore`, `readout_explore` или `cnn_explore`.
        - Не выбирай `init_only_explore`, если последние попытки уже повторяли тот же каркас.
        - В `rationale` явно укажи ближайшую прошлую схему, от которой ты отталкиваешься, и минимум два структурных отличия новой конфигурации.

        Seed-конфиг:
        {json.dumps(seed_config, ensure_ascii=False)}

        Краткая память о предыдущих попытках:
        {context_text}

        Замечания валидации из предыдущих retry внутри этой же попытки:
        {feedback_block}
        """
    ).strip()
    return system_prompt, user_prompt


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    command: list[str]


def run_command(
    command: list[str],
    *,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int | None = None,
) -> CommandResult:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    def tee_stream(
        stream: Any,
        console: Any,
        file_handle: Any,
        chunks: list[str],
    ) -> None:
        try:
            for line in iter(stream.readline, ""):
                chunks.append(line)
                console.write(line)
                console.flush()
                file_handle.write(line)
                file_handle.flush()
        finally:
            stream.close()

    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    timed_out = False

    with stdout_path.open("w", encoding="utf-8") as stdout_file, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_file:
        proc = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        assert proc.stdout is not None
        assert proc.stderr is not None

        stdout_thread = threading.Thread(
            target=tee_stream,
            args=(proc.stdout, sys.stdout, stdout_file, stdout_chunks),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=tee_stream,
            args=(proc.stderr, sys.stderr, stderr_file, stderr_chunks),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        try:
            returncode = proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.kill()
            returncode = 124
            proc.wait()

        stdout_thread.join()
        stderr_thread.join()

    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    return CommandResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        command=command,
    )


def _to_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def read_checkpoint_metrics(checkpoint_path: Path) -> tuple[float | None, float | None, dict[str, Any]]:
    """
    Returns (test_accuracy, best_test_accuracy, metadata_subset).
    """
    if not checkpoint_path.exists():
        return None, None, {}

    try:
        import torch
    except ImportError:
        return None, None, {"warning": "torch is not installed; checkpoint metadata not read."}

    try:
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
    except Exception as exc:
        return None, None, {"warning": f"checkpoint read failed: {exc}"}

    if not isinstance(checkpoint, dict):
        return None, None, {"warning": "checkpoint payload is not a dict."}

    test_acc = _to_float_or_none(checkpoint.get("test_accuracy"))
    best_acc = _to_float_or_none(checkpoint.get("best_test_accuracy"))
    if best_acc is None:
        history = checkpoint.get("history")
        if isinstance(history, dict) and isinstance(history.get("test_acc"), list) and history["test_acc"]:
            tail = [x for x in history["test_acc"] if isinstance(x, (int, float))]
            if tail:
                best_acc = float(max(tail))
                if test_acc is None:
                    test_acc = float(tail[-1])
    subset = {
        "model_name": checkpoint.get("model_name"),
        "timesteps": checkpoint.get("timesteps"),
        "epochs": checkpoint.get("epochs"),
        "learning_rate": checkpoint.get("learning_rate"),
        "dataset_path": checkpoint.get("dataset_path"),
        "dataset_kind": checkpoint.get("dataset_kind"),
    }
    return test_acc, best_acc, subset


def _make_attempt_id(index: int) -> str:
    return f"attempt_{index:04d}"


def _persist_attempt_record(
    *,
    memory: ExperimentMemory,
    attempt_dir: Path,
    record: dict[str, Any],
) -> dict[str, Any]:
    _write_json(attempt_dir / "attempt_summary.json", record)
    memory.append(record)
    return record


def _format_error_lines(lines: list[str], max_items: int = 14) -> str:
    if not lines:
        return "Unknown error"
    clipped = lines[:max_items]
    if len(lines) > max_items:
        clipped.append(f"... and {len(lines) - max_items} more")
    return "; ".join(clipped)


def _shorten(text: str, max_len: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def execute_attempt(
    *,
    attempt_index: int,
    run_id: str,
    run_dir: Path,
    memory: ExperimentMemory,
    provider: ProposalProvider,
    seed_config: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    attempt_id = _make_attempt_id(attempt_index)
    attempt_dir = run_dir / "attempts" / attempt_id
    attempt_dir.mkdir(parents=True, exist_ok=True)
    print(f"[attempt {attempt_id}] stage=setup dir={attempt_dir}")

    started_at = _now_utc_iso()
    context_text = memory.build_context(max_attempts=args.context_attempts, max_chars=args.max_context_chars)
    context_path = attempt_dir / "context.txt"
    context_path.write_text(context_text + "\n", encoding="utf-8")
    print(
        f"[attempt {attempt_id}] stage=context history_items={len(memory.load())} "
        f"context_chars={len(context_text)}"
    )

    history = memory.load()
    feedback: list[str] = []
    proposal: Proposal | None = None
    proposal_errors: list[str] = []
    prompt_cache: list[dict[str, str]] = []

    for retry in range(1, args.proposal_retries + 1):
        print(
            f"[attempt {attempt_id}] stage=proposal retry={retry}/{args.proposal_retries}"
        )
        system_prompt, user_prompt = build_prompts(
            seed_config=seed_config,
            context_text=context_text,
            feedback=feedback,
        )
        prompt_cache.append({"system": system_prompt, "user": user_prompt})
        (attempt_dir / f"prompt_retry{retry}.txt").write_text(
            f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}\n",
            encoding="utf-8",
        )

        raw_text = provider.propose(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            attempt_index=attempt_index,
        )
        (attempt_dir / f"proposal_raw_retry{retry}.txt").write_text(raw_text, encoding="utf-8")

        try:
            candidate = parse_proposal(raw_text, retry_index=retry)
        except Exception as exc:
            proposal_errors = [f"Proposal parsing error: {exc}"]
            print(
                f"[attempt {attempt_id}] stage=proposal_parse status=failed "
                f"retry={retry} error={_shorten(str(exc))}"
            )
            feedback = proposal_errors
            (attempt_dir / f"proposal_validation_retry{retry}.txt").write_text(
                "\n".join(proposal_errors) + "\n",
                encoding="utf-8",
            )
            continue

        proposal_errors = validate_config(candidate.config)
        if proposal_errors:
            print(
                f"[attempt {attempt_id}] stage=proposal_validate status=failed "
                f"retry={retry} errors={len(proposal_errors)} first={_shorten(proposal_errors[0])}"
            )
            feedback = proposal_errors
            (attempt_dir / f"proposal_validation_retry{retry}.txt").write_text(
                "\n".join(proposal_errors) + "\n",
                encoding="utf-8",
            )
            continue

        duplicate_error = _find_seen_config_duplicate(
            candidate.config,
            history=history,
            run_dir=run_dir,
        )
        if duplicate_error:
            proposal_errors = [duplicate_error]
            print(
                f"[attempt {attempt_id}] stage=proposal_deduplicate status=failed "
                f"retry={retry} error={_shorten(duplicate_error)}"
            )
            feedback = proposal_errors
            (attempt_dir / f"proposal_validation_retry{retry}.txt").write_text(
                "\n".join(proposal_errors) + "\n",
                encoding="utf-8",
            )
            continue

        proposal = candidate
        print(
            f"[attempt {attempt_id}] stage=proposal_validate status=ok "
            f"retry={retry} rationale={_shorten(candidate.rationale, 120)}"
        )
        break

    if proposal is None:
        ended_at = _now_utc_iso()
        record = {
            "run_id": run_id,
            "attempt_index": attempt_index,
            "attempt_id": attempt_id,
            "status": "failed_proposal",
            "started_at": started_at,
            "ended_at": ended_at,
            "provider": args.provider,
            "model": args.llm_model,
            "rationale": "",
            "reasoning_summary": "",
            "config_signature": "",
            "config_sha256": "",
            "architecture_fingerprint": "",
            "param_count": None,
            "config_path": "",
            "model_path": "",
            "checkpoint_path": "",
            "test_accuracy": None,
            "best_test_accuracy": None,
            "error_message": _format_error_lines(proposal_errors or ["Proposal generation failed."]),
            "proposal_retries_used": args.proposal_retries,
        }
        print(
            f"[attempt {attempt_id}] stage=finish status=failed_proposal "
            f"error={_shorten(record['error_message'])}"
        )
        return _persist_attempt_record(memory=memory, attempt_dir=attempt_dir, record=record)

    config_path = attempt_dir / "config.json"
    _write_json(config_path, proposal.config)
    (attempt_dir / "reasoning_summary.txt").write_text(
        proposal.reasoning_summary.strip() + "\n",
        encoding="utf-8",
    )
    config_signature = _config_signature(proposal.config)
    config_sha = _json_sha256(proposal.config)
    architecture_fingerprint = _architecture_fingerprint(proposal.config)
    param_count = estimate_param_count(proposal.config)
    print(
        f"[attempt {attempt_id}] stage=config status=saved signature={config_signature} "
        f"params={param_count['total']:,}"
    )

    model_path = attempt_dir / "model.py"
    codegen_stdout = attempt_dir / "codegen_stdout.log"
    codegen_stderr = attempt_dir / "codegen_stderr.log"
    codegen_cmd = [
        sys.executable,
        str(args.codegen_path),
        str(config_path),
        str(model_path),
    ]
    if args.save_diagram:
        diagram_path = attempt_dir / "diagram.png"
        codegen_cmd.append(str(diagram_path))

    print(
        f"[attempt {attempt_id}] stage=codegen status=started output={model_path.name}"
    )
    codegen_res = run_command(
        codegen_cmd,
        stdout_path=codegen_stdout,
        stderr_path=codegen_stderr,
        timeout_seconds=args.timeout_seconds,
    )
    if codegen_res.returncode != 0:
        ended_at = _now_utc_iso()
        record = {
            "run_id": run_id,
            "attempt_index": attempt_index,
            "attempt_id": attempt_id,
            "status": "failed_codegen",
            "started_at": started_at,
            "ended_at": ended_at,
            "provider": args.provider,
            "model": args.llm_model,
            "rationale": proposal.rationale,
            "reasoning_summary": proposal.reasoning_summary,
            "config_signature": config_signature,
            "config_sha256": config_sha,
            "architecture_fingerprint": architecture_fingerprint,
            "param_count": param_count["total"],
            "config_path": str(config_path),
            "model_path": str(model_path),
            "checkpoint_path": "",
            "test_accuracy": None,
            "best_test_accuracy": None,
            "error_message": f"codegen exit={codegen_res.returncode}",
            "proposal_retries_used": proposal.retry_index,
        }
        print(
            f"[attempt {attempt_id}] stage=codegen status=failed "
            f"exit={codegen_res.returncode}"
        )
        return _persist_attempt_record(memory=memory, attempt_dir=attempt_dir, record=record)
    print(f"[attempt {attempt_id}] stage=codegen status=ok")

    checkpoint_path = attempt_dir / "checkpoint.pt"
    coderun_cmd = [
        sys.executable,
        str(args.coderun_path),
        str(model_path),
        "--epochs",
        str(args.epochs),
        "--timesteps",
        str(args.timesteps),
        "--batch-size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--num-workers",
        str(args.num_workers),
        "--seed",
        str(args.base_seed + attempt_index),
        "--output",
        str(checkpoint_path),
    ]
    if args.data_path is not None:
        coderun_cmd.extend(["--data-path", str(args.data_path.resolve())])
    if args.device:
        coderun_cmd.extend(["--device", args.device])

    print(
        f"[attempt {attempt_id}] stage=train status=started "
        f"epochs={args.epochs} timesteps={args.timesteps} seed={args.base_seed + attempt_index}"
    )
    coderun_stdout = attempt_dir / "coderun_stdout.log"
    coderun_stderr = attempt_dir / "coderun_stderr.log"
    coderun_res = run_command(
        coderun_cmd,
        stdout_path=coderun_stdout,
        stderr_path=coderun_stderr,
        timeout_seconds=args.timeout_seconds,
    )

    print(
        f"[attempt {attempt_id}] stage=train status={'ok' if coderun_res.returncode == 0 else 'failed'} "
        f"exit={coderun_res.returncode}"
    )
    print(f"[attempt {attempt_id}] stage=checkpoint status=reading")
    test_acc, best_acc, ckpt_meta = read_checkpoint_metrics(checkpoint_path)

    ended_at = _now_utc_iso()
    status = "success" if coderun_res.returncode == 0 else "failed_train"
    error_message = ""
    if coderun_res.returncode != 0:
        error_message = (
            "training timed out"
            if coderun_res.timed_out
            else f"coderun exit={coderun_res.returncode}"
        )

    record = {
        "run_id": run_id,
        "attempt_index": attempt_index,
        "attempt_id": attempt_id,
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "provider": args.provider,
        "model": args.llm_model,
        "rationale": proposal.rationale,
        "reasoning_summary": proposal.reasoning_summary,
        "config_signature": config_signature,
        "config_sha256": config_sha,
        "architecture_fingerprint": architecture_fingerprint,
        "param_count": param_count["total"],
        "config_path": str(config_path),
        "model_path": str(model_path),
        "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else "",
        "test_accuracy": test_acc,
        "best_test_accuracy": best_acc,
        "checkpoint_metadata": ckpt_meta,
        "error_message": error_message,
        "proposal_retries_used": proposal.retry_index,
    }
    best_text = f"{best_acc:.4f}" if isinstance(best_acc, float) else "n/a"
    print(
        f"[attempt {attempt_id}] stage=finish status={status} "
        f"best_test_accuracy={best_text}"
    )
    return _persist_attempt_record(memory=memory, attempt_dir=attempt_dir, record=record)


def _print_history_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("No attempts found.")
        return
    for row in rows:
        best = row.get("best_test_accuracy")
        best_text = f"{float(best):.4f}" if isinstance(best, (int, float)) else "n/a"
        print(
            f"#{row.get('attempt_index'):>4} {row.get('attempt_id')} "
            f"status={row.get('status')} best={best_text} "
            f"sig={row.get('config_signature', '')}"
        )


def _create_provider(args: argparse.Namespace, seed_config: dict[str, Any]) -> ProposalProvider:
    if args.provider == "mock":
        return MockProposalProvider(seed_config=seed_config, rng_seed=args.base_seed)
    if args.provider == "openai":
        return OpenAIProposalProvider(
            model=args.llm_model,
            api_key=args.api_key,
            base_url=args.base_url,
            temperature=args.temperature,
        )
    raise ValueError(f"Unsupported provider: {args.provider}")


def _ensure_manifest(run_dir: Path, args: argparse.Namespace, seed_config_path: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        return _load_json(manifest_path)

    run_id = dt.datetime.now(dt.timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    manifest = {
        "run_id": run_id,
        "created_at": _now_utc_iso(),
        "run_dir": str(run_dir),
        "env_file": str(args.env_file.resolve()),
        "seed_config_path": str(seed_config_path),
        "codegen_path": str(args.codegen_path),
        "coderun_path": str(args.coderun_path),
        "provider": args.provider,
        "llm_model": args.llm_model,
        "epochs": args.epochs,
        "timesteps": args.timesteps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "num_workers": args.num_workers,
        "base_seed": args.base_seed,
    }
    _write_json(manifest_path, manifest)
    return manifest


def run_search(args: argparse.Namespace) -> int:
    loaded_env = load_env_file(args.env_file)
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "attempts").mkdir(parents=True, exist_ok=True)

    seed_config = _load_json(args.seed_config)
    seed_errors = validate_config(seed_config)
    if seed_errors:
        raise ValueError(
            "Seed config is invalid for current pipeline: " + _format_error_lines(seed_errors)
        )

    manifest = _ensure_manifest(run_dir, args, args.seed_config.resolve())
    run_id = manifest.get("run_id", dt.datetime.now(dt.timezone.utc).strftime("run_%Y%m%dT%H%M%SZ"))

    memory = ExperimentMemory(run_dir)
    provider = _create_provider(args, seed_config=seed_config)

    start_idx = memory.next_attempt_index()
    completed: list[dict[str, Any]] = []

    print(f"[search] run_dir={run_dir}")
    print(f"[search] run_id={run_id}")
    print(f"[search] env_file={args.env_file.resolve()} loaded_keys={len(loaded_env)}")
    print(f"[search] starting attempt index: {start_idx}")
    print(f"[search] provider={args.provider}, model={args.llm_model}")
    print(f"[search] planned attempts: {args.attempts}")

    for offset in range(args.attempts):
        idx = start_idx + offset
        print(f"[search] attempt {idx} started")
        record = execute_attempt(
            attempt_index=idx,
            run_id=run_id,
            run_dir=run_dir,
            memory=memory,
            provider=provider,
            seed_config=seed_config,
            args=args,
        )
        completed.append(record)

        best = record.get("best_test_accuracy")
        best_text = f"{float(best):.4f}" if isinstance(best, (int, float)) else "n/a"
        print(
            f"[search] attempt {idx} status={record.get('status')} "
            f"best_test_accuracy={best_text}"
        )

    best_success = [
        item for item in completed
        if item.get("status") == "success" and isinstance(item.get("best_test_accuracy"), (int, float))
    ]
    if best_success:
        champion = max(best_success, key=lambda x: float(x["best_test_accuracy"]))
        print(
            f"[search] best in this run slice: {champion['attempt_id']} "
            f"best_test_accuracy={float(champion['best_test_accuracy']):.4f}"
        )
    else:
        print("[search] no successful training attempt in this run slice.")
    return 0


def show_history(args: argparse.Namespace) -> int:
    memory = ExperimentMemory(args.run_dir.resolve())
    rows = memory.search(query=args.query, limit=args.limit)
    _print_history_rows(rows)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Iterative architecture-search agent around snn_codegen.py + snn_coderun.py.",
        epilog=CLI_EXAMPLES,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")

    run = subparsers.add_parser("run", help="Run search attempts.")
    run.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    run.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)
    run.add_argument("--attempts", type=int, default=3)
    run.add_argument("--seed-config", type=Path, default=DEFAULT_SEED_CONFIG)
    run.add_argument("--codegen-path", type=Path, default=DEFAULT_CODEGEN_PATH)
    run.add_argument("--coderun-path", type=Path, default=DEFAULT_CODERUN_PATH)
    run.add_argument("--provider", choices=["openai", "mock"], default="openai")
    run.add_argument("--llm-model", type=str, default="anthropic/claude-opus-4.7")
    run.add_argument("--api-key", type=str, default=None)
    run.add_argument("--base-url", type=str, default="https://polza.ai/api/v1")
    run.add_argument("--temperature", type=float, default=0.2)
    run.add_argument("--proposal-retries", type=int, default=3)
    run.add_argument("--context-attempts", type=int, default=-1)
    run.add_argument("--max-context-chars", type=int, default=10e6)
    run.add_argument("--epochs", type=int, default=7)
    run.add_argument("--timesteps", type=int, default=20)
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--lr", type=float, default=2e-3)
    run.add_argument("--num-workers", type=int, default=2)
    run.add_argument("--data-path", type=Path, default=None)
    run.add_argument("--device", type=str, default=None)
    run.add_argument("--base-seed", type=int, default=42)
    run.add_argument("--timeout-seconds", type=int, default=None)
    run.add_argument("--save-diagram", action="store_true")

    hist = subparsers.add_parser("history", help="Search previous attempts.")
    hist.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    hist.add_argument("--query", type=str, default="")
    hist.add_argument("--limit", type=int, default=20)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    # Default command is `run` when omitted.
    if args.command is None:
        args = parser.parse_args(["run", *(argv or sys.argv[1:])])

    if args.command == "run":
        return run_search(args)
    if args.command == "history":
        return show_history(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
