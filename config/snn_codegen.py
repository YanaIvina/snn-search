#!/usr/bin/env python3
"""
SNN Model Code Generator  +  Graph Visualizer
==============================================
Reads a JSON config that describes a Spiking Neural Network graph,
emits a ready-to-use PyTorch module AND (optionally) saves a PNG
architecture diagram using NetworkX + Matplotlib.

Usage:
    python snn_codegen.py config.json                       # print code to stdout
    python snn_codegen.py config.json model.py              # save code
    python snn_codegen.py config.json model.py arch.png     # save code + diagram
    python snn_codegen.py config.json --visualize arch.png  # diagram only
"""

import json
import math
import sys
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers (codegen)
# ──────────────────────────────────────────────────────────────────────────────

def _topo_sort(node_ids: List[str], edges: List[Dict]) -> List[str]:
    """
    Kahn's topological sort.
    Self-loops AND back-edges that would create cycles are silently skipped
    so that Dense / fully-connected configs still produce a valid ordering.
    """
    # Keep only edges that go "forward" according to their declaration order
    # by doing one pass and discarding edges that introduce cycles.
    in_deg = {n: 0 for n in node_ids}
    adj: Dict[str, List[str]] = defaultdict(list)
    for e in edges:
        s, d = e["from"], e["to"]
        if s == d:
            continue           # self-loop – handled separately
        adj[s].append(d)
        in_deg[d] += 1
    q = deque(n for n in node_ids if in_deg[n] == 0)
    order: List[str] = []
    visited = set()
    while q:
        n = q.popleft()
        if n in visited:
            continue
        visited.add(n)
        order.append(n)
        for nb in adj[n]:
            in_deg[nb] -= 1
            if in_deg[nb] == 0:
                q.append(nb)
    # append any remaining nodes (cycle survivors) in original order
    for n in node_ids:
        if n not in visited:
            order.append(n)
    return order


def _init_code(cfg: Optional[Dict], expr: str) -> str:
    if not cfg:
        return ""
    t = cfg.get("type", "default")
    if t in ("default", "pytorch_default"):
        return ""
    elif t == "kaiming_uniform":
        return f"nn.init.kaiming_uniform_({expr}, a={cfg.get('a', 0)})"
    elif t == "kaiming_normal":
        return f"nn.init.kaiming_normal_({expr})"
    elif t == "xavier_uniform":
        return f"nn.init.xavier_uniform_({expr})"
    elif t == "xavier_normal":
        return f"nn.init.xavier_normal_({expr})"
    elif t == "uniform":
        return f"nn.init.uniform_({expr}, a={cfg.get('low',-0.1)}, b={cfg.get('high',0.1)})"
    elif t == "normal":
        return f"nn.init.normal_({expr}, mean={cfg.get('mean',0.0)}, std={cfg.get('std',0.01)})"
    elif t == "sparse":
        return (f"nn.init.sparse_({expr}, "
                f"sparsity={cfg.get('sparsity',0.9)}, std={cfg.get('std',0.01)})")
    elif t == "orthogonal":
        return f"nn.init.orthogonal_({expr}, gain={cfg.get('gain',1.0)})"
    elif t == "zeros":
        return f"nn.init.zeros_({expr})"
    elif t == "eye":
        return f"nn.init.eye_({expr})"
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Graph visualizer
# ──────────────────────────────────────────────────────────────────────────────

# Colour palette (dark background style)
_BG        = "#0f1117"
_C_INP     = "#4f98a3"   # teal   – input
_C_INP_CNN = "#e040fb"   # фиолетовый – input с CNN front-end
_C_RES = 	 "#4caf50"  # green  – reservoir
_C_OUT     = "#da7101"   # orange – output
_C_FF      = "#e8af34"   # gold   – feed-forward edge
_C_REC     = "#FFFFFF"   # purple – recurrent self-loop
_C_CROSS   = "#5591c7"   # blue   – cross-reservoir edge
_C_SKIP    = "#dd6974"   # red    – skip connection (src skips ≥1 res layer)
_WHITE     = "#e8e8e8"
_MUTED     = "#8a8fa8"


def _node_color(node_type: str, use_cnn: bool = False) -> str:
    return {
        "input":     _C_INP_CNN if use_cnn else _C_INP,
        "reservoir": _C_RES,
        "output":    _C_OUT,
    }.get(node_type, _C_RES)


def _node_size(node: Dict, default: int = 3000) -> int:
    """Scale circle area with reservoir size (clamped to readable range)."""
    s = node.get("size", 512)
    return int(max(1800, min(7000, s * 3.5)))


def _layout(nodes: List[Dict], edges: List[Dict]) -> Dict[str, Tuple[float, float]]:
    """
    Compute (x, y) positions via topological layering.
    Nodes in the same topo-layer share an x-column; y is spread evenly.
    """
    node_ids = [n["id"] for n in nodes]
    topo = _topo_sort(node_ids, edges)

    # assign layers (longest path)
    layer: Dict[str, int] = {n: 0 for n in node_ids}
    for nid in topo:
        for e in edges:
            if e["from"] == nid and e["to"] != nid:
                layer[e["to"]] = max(layer[e["to"]], layer[nid] + 1)

    # group by layer
    by_layer: Dict[int, List[str]] = defaultdict(list)
    for nid in topo:
        by_layer[layer[nid]].append(nid)

    max_layer = max(by_layer)
    pos: Dict[str, Tuple[float, float]] = {}
    for lyr, members in by_layer.items():
        x = lyr / max(max_layer, 1)
        for i, nid in enumerate(members):
            n_in_layer = len(members)
            y = (i + 1) / (n_in_layer + 1)
            pos[nid] = (x, y)
    return pos


def _classify_edge(src: str, dst: str, node_map: Dict,
                   topo: List[str]) -> str:
    """
    Classify a feed-forward edge into one of three visual groups:
      'ff'    – consecutive layers in topo order
      'cross' – same layer or between reservoirs (not strictly consecutive)
      'skip'  – long-range skip (src is ≥2 topo positions before dst)
    """
    if src == dst:
        return "rec"
    si = topo.index(src)
    di = topo.index(dst)
    gap = di - si
    if gap == 1:
        return "ff"
    elif gap >= 2:
        return "skip"
    else:
        return "cross"



def _count_params(cfg: Dict) -> Dict[str, int]:
    """
    Count trainable parameters by group:
      w_in     – input  → reservoir  (Linear weights)
      w_hidden – reservoir → reservoir  (ff + lateral + recurrent self-loops)
      w_out    – reservoir / input → output  (Linear weights)
      tau_m    – one per neuron in every reservoir  (BimodalLIF)
    Returns dict with keys: w_in, w_hidden, w_out, tau_m, total
    """
    nodes    = cfg["nodes"]
    edges    = cfg["edges"]
    node_map = {n["id"]: n for n in nodes}

    # resolve output-node concat size
    out_nodes = [n for n in nodes if n["type"] == "output"]
    out_id    = out_nodes[0]["id"] if out_nodes else None

    # sources feeding the output node
    out_srcs = [e["from"] for e in edges
                if e["to"] == out_id and e["from"] != out_id]

    # Если есть CNN front-end — реальный размер выхода отличается от inp_size в JSON
    cfe = cfg.get("conv_frontend", {})
    if cfe.get("enabled", False):
        cnn_chs = cfe.get("channels", [32, 64, 128])
        cnn_hw  = cfe.get("hw", [64, 64])
        if isinstance(cnn_hw, int):
            cnn_hw = [cnn_hw, cnn_hw]
        nb = len(cnn_chs)
        inp_effective_size = cnn_chs[-1] * (cnn_hw[0] // 2**nb) * (cnn_hw[1] // 2**nb)
    else:
        inp_effective_size = None   # будем брать из JSON как обычно

    w_in = w_hidden = w_out = tau_m = 0

    for e in edges:
        s, d   = e["from"], e["to"]
        s_node = node_map[s]
        d_node = node_map[d]
        s_sz   = s_node.get("size", 0)
        d_sz   = d_node.get("size", 0)

        s_type = s_node["type"]
        d_type = d_node["type"]

        if d_type == "output":
            # handled after loop (may be concat)
            continue
        elif s_type == "input" and d_type == "reservoir":
            effective_s = inp_effective_size if inp_effective_size is not None else s_sz
            w_in += effective_s * d_sz
        elif s_type == "reservoir" and d_type == "reservoir":
            # self-loop OR cross/ff between reservoirs
            w_hidden += s_sz * d_sz

    # output projection
    if out_id and out_srcs:
        concat_in = sum(node_map[s].get("size", 0) for s in out_srcs)
        w_out = concat_in * node_map[out_id].get("size", 0)

    # τ_m parameters: one per neuron per reservoir
    tau_m = sum(n.get("size", 0) for n in nodes if n["type"] == "reservoir")

    total = w_in + w_hidden + w_out + tau_m
    return dict(w_in=w_in, w_hidden=w_hidden, w_out=w_out, tau_m=tau_m, total=total)

def visualize_model(cfg: Dict, output_path: str) -> None:
    """
    Draw the SNN graph and save to *output_path* (PNG/SVG/PDF).

    Node colour  → type (input / reservoir / output)
    Node size    → reservoir neuron count
    Gold solid   → feed-forward connections
    Purple dashed→ recurrent self-loops (drawn as real loops ON the node)
    Red dashed   → skip connections (source skips ≥1 layer)
    Blue dotted  → cross-reservoir lateral connections
    """
    try:
        import networkx as nx
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError as exc:
        print(f"[visualize] Missing library: {exc}. "
              "Install with: pip install networkx matplotlib")
        return

    nodes    = cfg["nodes"]
    edges    = cfg["edges"]
    name     = cfg.get("model_name", "SNNModel")
    node_map = {n["id"]: n for n in nodes}
    node_ids = [n["id"] for n in nodes]
    topo     = _topo_sort(node_ids, edges)
    pos      = _layout(nodes, edges)

    # ── categorise edges ─────────────────────────────────────────────────
    ff_edges, skip_edges, cross_edges, rec_nodes = [], [], [], []

    # layer index for each node (from pos x-coordinate)
    xs       = sorted(set(round(pos[n][0], 4) for n in node_ids))
    x_to_lyr = {x: i for i, x in enumerate(xs)}
    def lyr(nid): return x_to_lyr[round(pos[nid][0], 4)]

    for e in edges:
        s, d = e["from"], e["to"]
        if s == d:
            rec_nodes.append(s)
            continue
        gap = lyr(d) - lyr(s)
        if gap < 0:
            cross_edges.append((s, d))   # backward lateral
        elif gap == 1:
            ff_edges.append((s, d))
        elif gap >= 2:
            skip_edges.append((s, d))    # true skip
        else:
            cross_edges.append((s, d))   # same-layer lateral

    # ── build graph  (NO self-loops – drawn separately via connectionstyle) ─
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    for s, d in ff_edges + skip_edges + cross_edges:
        G.add_edge(s, d)
    # add self-loops so nx knows about them for the self-loop draw call
    for nid in rec_nodes:
        G.add_edge(nid, nid)

    # ── figure ───────────────────────────────────────────────────────────
    n_layers = len(xs)
    fig_w    = max(11, n_layers * 2.8)
    n_nodes  = len(nodes)
    fig_h    = max(7, n_nodes * 0.95)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor(_BG)
    fig.subplots_adjust(bottom=0.28)   # set early so ax.get_position() is accurate
    ax.set_facecolor(_BG)
    ax.axis("off")
    # expand limits so arcs/loops are never clipped
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.15, 1.20)

    # ── node sizes ───────────────────────────────────────────────────────
    node_sizes  = [_node_size(node_map[n]) for n in G.nodes()]
    use_cnn = cfg.get("conv_frontend", {}).get("enabled", False)
    node_colors = [
    _node_color(node_map[n]["type"], use_cnn=(node_map[n]["type"] == "input" and use_cnn))
    for n in G.nodes()
]

    # ── draw helper ──────────────────────────────────────────────────────
    def draw_edges(elist, color, style, width=2.3, rad=0.0):
        if not elist:
            return
        nx.draw_networkx_edges(
            G, pos,
            edgelist=elist,
            edge_color=color,
            style=style,
            alpha=0.88,
            width=width,
            arrows=True,
            arrowsize=18,
            connectionstyle=f"arc3,rad={rad}",
            node_size=node_sizes,
            ax=ax,
        )

    # feed-forward (straight)
    draw_edges(ff_edges,   _C_FF,    "solid",  rad=0.0)
    # skip connections – arc UPWARD (negative rad = convex upward)
    draw_edges(skip_edges, _C_SKIP,  "dashed", rad=-0.28, width=2.0)
    # cross/lateral – gentle curve
    draw_edges(cross_edges, _C_CROSS, "dotted", rad=0.18, width=1.8)

    # ── recurrent self-loops — drawn as small FancyArrowPatch loops ─────
    import numpy as np
    from matplotlib.patches import FancyArrowPatch

    # Algebraic data→axes-fraction conversion using axis limits.
    # Does NOT depend on display rendering, so it is safe before fig.canvas.draw().
    def data_to_axes(x, y):
        xl = ax.get_xlim()
        yl = ax.get_ylim()
        return (x - xl[0]) / (xl[1] - xl[0]), (y - yl[0]) / (yl[1] - yl[0])

    # Force layout so ax.get_position() reflects subplots_adjust settings.
    fig.canvas.draw()
    _ax_pos   = ax.get_position()
    _ax_w_pts = _ax_pos.width  * fig.get_figwidth()  * 72
    _ax_h_pts = _ax_pos.height * fig.get_figheight() * 72

    for nid in rec_nodes:
        dx, dy = data_to_axes(*pos[nid])
        r_pts  = np.sqrt(_node_size(node_map[nid])) / 2  # scatter marker radius, pts
        r_ax_x = r_pts / _ax_w_pts   # node radius in x-axes-fraction
        r_ax_y = r_pts / _ax_h_pts   # node radius in y-axes-fraction
        # ── Mathematically correct self-loop geometry ─────────────────────
        # Anchor points: left/right edge of node, slightly above centre.
        # Peak: arc3 Bezier peak_offset_ax = RAD * r_ax_y / 2
        # (proof: chord is horizontal → perpendicular is vertical;
        #  chord_display = 2*gap_ax * ax_w_px; h_display = RAD * chord_display;
        #  peak_offset_ax = h_display/2 / ax_h_px = RAD * gap_ax * ax_w_px/ax_h_px
        #                 = RAD * r_ax_x * (_ax_w_pts/_ax_h_pts)
        #                 = RAD * r_ax_y      since gap = r_ax_x and r_ax_x*W=r_ax_y*H=r_pts)
        # → RAD = desired_peak_above_anchor / r_ax_y
        #
        # Anchors at: y = node_centre ± r_ax_y * 0.35  (just above/below centre)
        # Peak at:    y = node_centre ± r_ax_y * 1.3   (30% above top of node)
        # peak_offset above anchor = 1.3 - 0.35 = 0.95 r_ax_y → RAD ≈ 2.0

        gap  = r_ax_x          # horizontal: anchor at node left/right edge
        lift = r_ax_y * 0.35   # vertical: anchor slightly above centre (inside node)
        RAD  = 1.3             # ← tune to move peak up/down (2.0 = 1 node-r above anchor)

        # direction: upward unless a same-column neighbour sits directly above
        same_col = [n for n in node_ids
                    if n != nid and abs(pos[n][0] - pos[nid][0]) < 0.01]
        above = [pos[n][1] for n in same_col if pos[n][1] > pos[nid][1]]
        sign  = -1 if above else 1   # +1 = loop up, -1 = loop down

        fp = FancyArrowPatch(
            posA=(dx - gap, dy + sign * lift),
            posB=(dx + gap, dy + sign * lift),
            connectionstyle=f"arc3,rad={sign * RAD}",
            arrowstyle="-|>",
            mutation_scale=14,
            color=_C_REC,
            linewidth=2.0,
            linestyle=(0, (5, 3)),
            transform=ax.transAxes,
            zorder=5,
        )
        ax.add_patch(fp)

    # ── draw nodes ───────────────────────────────────────────────────────
    nx.draw_networkx_nodes(
        G, pos,
        node_color=node_colors,
        node_size=node_sizes,
        ax=ax,
        linewidths=2,
        edgecolors=_WHITE,
        alpha=0.95,
    )

    # ── node labels ──────────────────────────────────────────────────────
    # labels = {}
    # for n in nodes:
    #     nid  = n["id"]
    #     sz   = n.get("size", "")
    #     # fl   = n.get("fraction_long",
    #     #              cfg.get("neuron_params", {}).get("fraction_long", 0.5))
    #     if n["type"] == "reservoir":
    #         labels[nid] = f"{nid}\nN={sz}"
    #     else:
    #         labels[nid] = f"{nid}\nN={sz}"
    labels = {}
    # Если CNN включён — у inp реальный размер = выход CNN
    cfe = cfg.get("conv_frontend", {})
    if cfe.get("enabled", False):
        cnn_chs = cfe.get("channels", [32, 64, 128])
        cnn_hw  = cfe.get("hw", [64, 64])
        if isinstance(cnn_hw, int):
            cnn_hw = [cnn_hw, cnn_hw]
        nb = len(cnn_chs)
        cnn_out = cnn_chs[-1] * (cnn_hw[0] // 2**nb) * (cnn_hw[1] // 2**nb)
    else:
        cnn_out = None

    for n in nodes:
        nid = n["id"]
        sz  = cnn_out if (cnn_out is not None and n["type"] == "input") else n.get("size", "")
        if n["type"] == "reservoir":
            labels[nid] = f"{nid}\nN={sz}\n"
        else:
            labels[nid] = f"{nid}\nN={sz}"

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=8,
        font_color=_WHITE,
        font_weight="bold",
        ax=ax,
    )

    # ── edge weight-init labels (only init type abbreviation) ─────────────
    # Show only once per unique (src_type, dst_type) pair to reduce clutter
    shown_pairs = set()
    edge_label_map = {}
    for e in edges:
        s, d = e["from"], e["to"]
        if s == d:
            continue
        key = (node_map[s]["type"], node_map[d]["type"])
        if key in shown_pairs:
            continue
        # wi = e.get("weight_init", {})
        # if wi and wi.get("type", "default") != "default":
        #     short = wi["type"][:6]
        #     edge_label_map[(s, d)] = short
        #     shown_pairs.add(key)

    if edge_label_map:
        nx.draw_networkx_edge_labels(
            G, pos,
            edge_labels=edge_label_map,
            font_size=7,
            font_color=_MUTED,
            ax=ax,
            bbox=dict(boxstyle="round,pad=0.15", fc=_BG, ec="none", alpha=0.75),
        )

    # ── legend ───────────────────────────────────────────────────────────
    use_cnn   = cfg.get("conv_frontend", {}).get("enabled", False)
    inp_label = "Input + CNN front-end" if use_cnn else "Input"
    inp_color = _C_INP_CNN if use_cnn else _C_INP

    legend_handles = [
        mpatches.Patch(color=inp_color, label=inp_label),   # ← теперь внутри списка
        mpatches.Patch(color=_C_RES,   label="Reservoir (BimodalLIF)"),
        mpatches.Patch(color=_C_OUT,   label="Output (Linear)"),
        mpatches.Patch(color=_C_FF,    label="Feed-forward W"),
        mpatches.Patch(color=_C_REC,   label="Recurrent W_rec (self-loop)"),
        mpatches.Patch(color=_C_SKIP,  label="Skip connection"),
        mpatches.Patch(color=_C_CROSS, label="Cross-reservoir"),
    ]
    # ax.set_ylim(-0.35, 1.20)
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        ncol=1,
        frameon=True,
        facecolor="#1a1d27",
        edgecolor="#2a2d3e",
        labelcolor=_WHITE,
        fontsize=8,
        bbox_to_anchor=(1.0, -0.02),
        bbox_transform=ax.transAxes,
    )

    # ── title ─────────────────────────────────────────────────────────────
    res_count   = sum(1 for n in nodes if n["type"] == "reservoir")
    total_nrn   = sum(n.get("size", 0) for n in nodes if n["type"] == "reservoir")
    ax.set_title(
        f"{name}\n"
        f"{res_count} reservoir(s)   |   total neurons ≈ {total_nrn:,}",
        color=_WHITE, fontsize=12, pad=10, fontweight="bold",
    )

    # ── parameter count info box ──────────────────────────────────────────
    pcount = _count_params(cfg)

    def _fmt(n: int) -> str:
        if n >= 1_000_000: return f"{n/1_000_000:.2f}M"
        if n >= 1_000:     return f"{n/1_000:.1f}K"
        return str(n)

    box_text = "── Trainable parameters ──\n"
    for label, val in [
        ("W_in  (inp → res)",      pcount["w_in"]),
        ("W_hidden (res ↔ res)",   pcount["w_hidden"]),
        ("W_out (res → out)",      pcount["w_out"]),
        ("τ_m  (LIF time consts)", pcount["tau_m"]),
    ]:
        pct = 100 * val / pcount["total"] if pcount["total"] else 0
        box_text += f"  {label:<26s}  {_fmt(val):>7s}  ({pct:4.1f}%)\n"
    box_text += f"  {'TOTAL':<26s}  {_fmt(pcount['total']):>7s}"

    ax.text(
        0.5, -0.08,
        box_text,
        transform=ax.transAxes,
        ha="center", va="top",
        fontsize=8.5, family="monospace",
        color=_WHITE,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#1a1d27",
                  edgecolor="#2a2d3e", linewidth=1),
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close(fig)
    print(f"[visualize] Saved: {output_path}")



# ──────────────────────────────────────────────────────────────────────────────
# Main code generator (unchanged logic, kept intact)
# ──────────────────────────────────────────────────────────────────────────────
def generate_model_code(cfg: Dict) -> str:
    nodes    = cfg["nodes"]
    edges    = cfg["edges"]
    name     = cfg.get("model_name", "SNNModel")
    node_map = {n["id"]: n for n in nodes}
    node_ids = [n["id"] for n in nodes]

    inp_nodes = [n for n in nodes if n["type"] == "input"]
    res_nodes = [n for n in nodes if n["type"] == "reservoir"]
    out_nodes = [n for n in nodes if n["type"] == "output"]
    assert len(inp_nodes) >= 1, "Need at least one input node"
    assert len(out_nodes) == 1, "Need exactly one output node"
    out_node = out_nodes[0]

    incoming:  Dict[str, List] = defaultdict(list)
    recurrent: Dict[str, Dict] = {}
    for e in edges:
        s, d = e["from"], e["to"]
        if s == d:
            recurrent[s] = e
        else:
            incoming[d].append((s, e))

    topo = _topo_sort(node_ids, edges)

    np_ = cfg.get("neuron_params", {})
    vth       = np_.get("vth", 1.0)
    dt        = np_.get("dt",  1.0)
    frac_long = np_.get("fraction_long", 0.5)
    tau_short = tuple(np_.get("tau_short", [0,  20]))
    tau_long  = tuple(np_.get("tau_long",  [80, 100]))
    surr      = cfg.get("surrogate_params", {})
    s_h, s_l, s_s = surr.get("h", 0.15), surr.get("l", 0.5), surr.get("s", 6)

    # ── conv_frontend config ───────────────────────────────────────────────────
    cfe         = cfg.get("conv_frontend", {})
    use_cnn     = cfe.get("enabled", False)
    cnn_in_ch   = cfe.get("in_channels", 1)
    cnn_chs     = cfe.get("channels", [32, 64, 128])
    cnn_use_bn  = cfe.get("use_bn", True)
    cnn_hw      = cfe.get("hw", [64, 64])
    cnn_pool    = cfe.get("pool", "avg")   # "avg" | "max"
    if isinstance(cnn_hw, int):
        cnn_hw = [cnn_hw, cnn_hw]
    n_blocks    = len(cnn_chs)
    # output spatial size after N × Pool2d(2)
    cnn_oh = cnn_hw[0] // (2 ** n_blocks)
    cnn_ow = cnn_hw[1] // (2 ** n_blocks)
    cnn_out_size = cnn_chs[-1] * cnn_oh * cnn_ow

    L = []

    # ── imports ────────────────────────────────────────────────────────────────
    L += [
        "# Auto-generated – do not edit by hand.",
        f"# Model: {name}",
        "",
        "import numpy as np",
        "import torch",
        "import torch.nn as nn",
        "from scipy.stats import truncnorm",
        "",
    ]

    # ── MultiGaussianSurrogate ─────────────────────────────────────────────────
    L += [
        "",
        "class MultiGaussianSurrogate(torch.autograd.Function):",
        '    """Multi-Gaussian surrogate gradient for backprop through spikes."""',
        "",
        "    @staticmethod",
        "    def forward(ctx, input, h, l, s):",
        "        ctx.save_for_backward(input)",
        "        ctx.h, ctx.l, ctx.s = h, l, s",
        "        return (input > 0).float()",
        "",
        "    @staticmethod",
        "    def backward(ctx, grad_output):",
        "        input,  = ctx.saved_tensors",
        "        h = ctx.h",
        "        l = ctx.l",
        "        s = ctx.s",
        "",
        "        term1 = (1+h) * torch.exp(-(input/l)**2 / 2) / (np.sqrt(2*np.pi)*l)",
        "        term2 = h * torch.exp(-((input-l)/(s*l))**2 / 2) / (np.sqrt(2*np.pi)*(s*l))",
        "        term3 = h * torch.exp(-((input+l)/(s*l))**2 / 2) / (np.sqrt(2*np.pi)*(s*l))",
        "        grad = term1 - term2 - term3",
        "        return grad_output * grad, None, None, None",
        "",
    ]

    # ── BimodalLIFLayer ────────────────────────────────────────────────────────
    L += [
        "",
        "class BimodalLIFLayer(nn.Module):",
        '    """LIF neuron layer with trainable, bimodally distributed time constants."""',
        "",
        "    def __init__(self, size, tau_short=(0, 20), tau_long=(80, 100),",
        "                 fraction_long=0.5, vth=1.0, dt=1.0):",
        "        super().__init__()",
        "        a, b = (0 - 50) / 15, (100 - 50) / 15",
        "        tau_m = truncnorm.rvs(a, b, loc=50, scale=15,",
        "                              size=size).astype(np.float32)",
        "        self.tau_m = nn.Parameter(torch.from_numpy(tau_m), requires_grad=True)",
        "        self.vth, self.dt, self.size = vth, dt, size",
        "",
        "    def forward(self, x: torch.Tensor, v: torch.Tensor,",
        "                z: torch.Tensor) -> torch.Tensor:",
        "        rho   = torch.exp(-self.dt / self.tau_m.to(x.device))",
        "        return rho * v + (1 - rho) * x - self.vth * z",
        "",
        "    def reset(self, batch_size: int, device='cpu') -> torch.Tensor:",
        "        return torch.zeros(batch_size, self.size, device=device)",
        "",
    ]

    # ── SpikingConvBlock + SpikingConvFrontEnd (only if CNN enabled) ───────────
    if use_cnn:
        pool_cls_str = 'nn.AvgPool2d' if cnn_pool == 'avg' else 'nn.MaxPool2d'
        L += [
            '',
            'class SpikingConvBlock(nn.Module):',
            '',
            '    def __init__(self, cin, cout, hw, usebn=True, dt=1.0, vth=1.0,',
            '                 fraction_long=0.5):',
            '        super().__init__()',
            '        h, w = hw if isinstance(hw, (tuple, list)) else (hw, hw)',
            '        self.conv      = nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=not usebn)',
            '        self.bn        = nn.BatchNorm2d(cout) if usebn else nn.Identity()',
            f'        self.pool      = {pool_cls_str}(2)',
            '        self.vth       = vth',
            '        self.feat_size = cout * h * w',
            '        self.lif       = BimodalLIFLayer(',
            '            size=self.feat_size, fraction_long=fraction_long, dt=dt, vth=vth,',
            '        )',
            '',
            '    def init_state(self, batch_size, device, dtype=torch.float32):',
            '        return {',
            "            'v': torch.zeros(batch_size, self.feat_size, device=device, dtype=dtype),",
            "            'z': torch.zeros(batch_size, self.feat_size, device=device, dtype=dtype),",
            '        }',
            '',
            '    def forward_step(self, x, state, spike_params):',
            '        hs, ls, ss = spike_params',
            '        y      = self.bn(self.conv(x))',
            '        y_flat = y.flatten(1)',
            "        v_new  = self.lif(y_flat, state['v'], state['z'])",
            '        z_new  = MultiGaussianSurrogate.apply(v_new - self.vth, hs, ls, ss)',
            '        y_out  = self.pool(z_new.view_as(y))',
            "        state['v'], state['z'] = v_new, z_new",
            '        return y_out, state',
        ]

        # Build SpikingConvFrontEnd with concrete channel/size values from config
        block_inits = []
        h_cur, w_cur = cnn_hw[0], cnn_hw[1]
        prev_ch = cnn_in_ch
        for ch in cnn_chs:
            bn_str = str(cnn_use_bn)
            block_inits.append(
                f"            SpikingConvBlock({prev_ch}, {ch}, {h_cur}, {w_cur}, "
                f"use_bn={bn_str}, dt=dt, vth=vth, fraction_long=fraction_long),"
            )
            prev_ch = ch
            h_cur //= 2
            w_cur  //= 2

        ch_tuple = ', '.join(str(c) for c in cnn_chs)
        L += [
            '',
            'class SpikingConvFrontEnd(nn.Module):',
            '',
            f'    def __init__(self, inch={cnn_in_ch}, conv_channels=({ch_tuple}),',
            f'                 usebn={cnn_use_bn}, hw=({cnn_hw[0]}, {cnn_hw[1]}),',
            '                 dt=1.0, vth=1.0, fraction_long=0.5,',
            f'                 spike_params=({s_h}, {s_l}, {s_s})):',
            '        super().__init__()',
            '        h, w = hw if isinstance(hw, (tuple, list)) else (hw, hw)',
            '        self.spike_params = spike_params',
            '        self.blocks = nn.ModuleList()',
            '        prev = inch',
            '        for ch in conv_channels:',
            '            self.blocks.append(',
            '                SpikingConvBlock(prev, ch, (h, w), usebn, dt, vth, fraction_long)',
            '            )',
            '            prev, h, w = ch, h // 2, w // 2',
            '        self.feat_dim = prev * h * w',
            '',
            '    def init_state(self, batch_size, device, dtype=torch.float32):',
            '        return [blk.init_state(batch_size, device, dtype)',
            '                for blk in self.blocks]',
            '',
            '    def forward_step(self, x, state, spike_params):',
            '        for i, blk in enumerate(self.blocks):',
            '            x, state[i] = blk.forward_step(x, state[i], spike_params)',
            '        return x, state',
            '',
            '    def forward(self, x):',
            "        B = x.shape[0]",
            "        state = self.init_state(B, x.device, x.dtype)",
            "        out, _ = self.forward_step(x, state, self.spike_params)",
            "        return out",
        ]

    # ── Main model class ───────────────────────────────────────────────────────
    L += [
        "",
        f"class {name}(nn.Module):",
        '    """',
        f"    Nodes  : {', '.join(node_ids)}",
        f"    Topology: {' -> '.join(n['id'] for n in nodes if n['type'] != 'input')}",
        '    """',
        "",
    ]

    # __init__ signature
    size_params = [f"{n['id']}_size: int" for n in nodes
                   if n["type"] in ("input", "reservoir", "output")]
    extra_params = [
        f"fraction_long: float = {frac_long}",
        f"vth: float = {vth}",
        f"dt: float = {dt}",
    ]
    all_params = size_params + extra_params

    L.append("    def __init__(")
    L.append("            self,")
    for i, p in enumerate(all_params):
        comma = "," if i < len(all_params) - 1 else ""
        L.append(f"            {p}{comma}")
    L.append("    ):")
    L.append("        super().__init__()")
    L.append("        self.vth, self.dt = vth, dt")
    L.append("")

    for n in nodes:
        if n["type"] in ("input", "reservoir", "output"):
            L.append(f"        self.{n['id']}_size = {n['id']}_size")
    L.append("")

    # CNN front-end instantiation (if enabled)
    if use_cnn:
        L += [
            f'        self.conv_frontend = SpikingConvFrontEnd(',
            f'            inch={cnn_in_ch}, conv_channels=({ch_tuple}),',
            f'            usebn={cnn_use_bn}, hw=({cnn_hw[0]}, {cnn_hw[1]}),',
            f'            dt=dt, vth=vth, fraction_long=fraction_long,',
            f'            spike_params=({s_h}, {s_l}, {s_s}),',
            f'        )',
            f'        self.cnn_out_size = self.conv_frontend.feat_dim',
        ]

    # LIF layers
    L.append("        # ── LIF neuron layers ──────────────────────────────")
    for n in res_nodes:
        ts = tuple(n.get("tau_short", list(tau_short)))
        tl = tuple(n.get("tau_long",  list(tau_long)))
        fl = n.get("fraction_long", "fraction_long")
        L.append(f"        self.lif_{n['id']} = BimodalLIFLayer(")
        L.append(f"            {n['id']}_size, tau_short={ts}, tau_long={tl},")
        L.append(f"            fraction_long={fl}, vth=vth, dt=dt)")
    L.append("")

    # Weight matrices
    L.append("        # ── Synaptic weight matrices ───────────────────────")

    def _add_linear(lname, src_sz, dst_sz, bias, wi):
        L.append(f"        self.{lname} = nn.Linear({src_sz}, {dst_sz}, bias={bias})")
        ic = _init_code(wi, f"self.{lname}.weight")
        if ic:
            L.append(f"        {ic}")

    for n in nodes:
        nid = n["id"]
        if n["type"] == "output":
            in_e = incoming[nid]
            if len(in_e) == 1:
                src_id, e = in_e[0]
                _add_linear(f"w_{src_id}_to_{nid}",
                            f"{src_id}_size", f"{nid}_size",
                            e.get("bias", False), e.get("weight_init"))
            elif len(in_e) > 1:
                src_sz = " + ".join(f"{s}_size" for s, _ in in_e)
                e0 = in_e[0][1]
                L.append(f"        # Output proj: [{', '.join(s for s,_ in in_e)}] -> {nid}")
                _add_linear("wout", src_sz, f"{nid}_size",
                            e0.get("bias", False), e0.get("weight_init"))
        elif n["type"] == "reservoir":
            # Split incoming edges by source type
            res_srcs   = [(s, e) for s, e in incoming[nid]
                          if node_map[s]["type"] == "reservoir"]
            other_srcs = [(s, e) for s, e in incoming[nid]
                          if node_map[s]["type"] != "reservoir"]
            # Non-reservoir sources (input projections) — individual Linear
            for src_id, e in other_srcs:
                src_nd = node_map[src_id]
                if use_cnn and src_nd["type"] == "input":
                    src_sz_str = "self.cnn_out_size"
                else:
                    src_sz_str = f"{src_id}_size"
                _add_linear(f"w_{src_id}_to_{nid}",
                            src_sz_str, f"{nid}_size",
                            e.get("bias", False), e.get("weight_init"))
            # Reservoir sources — CONCAT merge (like notebook) if >1
            if len(res_srcs) > 1:
                merge_in = " + ".join(f"{s}_size" for s, _ in res_srcs)
                e0 = res_srcs[0][1]
                L.append(f"        # Concat-merge: [{', '.join(s for s,_ in res_srcs)}] -> {nid}")
                _add_linear(f"w_merge_{nid}",
                            merge_in, f"{nid}_size",
                            e0.get("bias", False), e0.get("weight_init"))
            elif len(res_srcs) == 1:
                src_id, e = res_srcs[0]
                _add_linear(f"w_{src_id}_to_{nid}",
                            f"{src_id}_size", f"{nid}_size",
                            e.get("bias", False), e.get("weight_init"))
            if nid in recurrent:
                e = recurrent[nid]
                _add_linear(f"w_rec_{nid}",
                            f"{nid}_size", f"{nid}_size",
                            e.get("bias", False), e.get("weight_init"))
    L.append("")

    # count_parameters
    L += [
        "    def count_parameters(self) -> int:",
        "        total = sum(p.numel() for p in self.parameters() if p.requires_grad)",
        "        print(f'Trainable parameters: {total:,}')",
        "        for name, p in self.named_parameters():",
        "            if p.requires_grad:",
        "                print(f'  {name:<50s} {p.numel():>12,}')",
        "        return total",
        "",
    ]

    # spike helper
    L += [
        "    def spike(self, v: torch.Tensor) -> torch.Tensor:",
        f"        return MultiGaussianSurrogate.apply(v - self.vth, {s_h}, {s_l}, {s_s})",
        "",
    ]

    in_node = inp_nodes[0]

    # forward
    L += [
        "    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:",
        '        """',
        f"        Args:  x_seq – (T, B, {in_node['id']}_size)",
        f"        Returns: (T, B, {out_node['id']}_size)",
        '        """',
        "        T, B, _ = x_seq.shape",
        "        device  = x_seq.device",
        "",
    ]

    # CNN projection (outside time loop — single frame from mean)
    if use_cnn:
        H, W = cnn_hw[0], cnn_hw[1]
        L += [
            "        # ── CNN front-end: encode mean frame once (outside time loop) ──",
            f"        x_img  = x_seq.mean(dim=0).view(B, {cnn_in_ch}, {H}, {W})",
            "        cnn_feat = self.conv_frontend(x_img).flatten(1)  # B, cnn_out_size",
            "",
        ]
        # inp→res1 projection using CNN output (expand over T)
        # We'll build a projected x_proj per input→res edge
        # Collect all edges from input nodes
        inp_ids = {n["id"] for n in inp_nodes}
        res_inp_edges = []
        for n in res_nodes:
            for src_id, e in incoming[n["id"]]:
                if src_id in inp_ids:
                    res_inp_edges.append((src_id, n["id"], e))
        # Pre-project before time loop
        projected = {}
        for src_id, dst_id, e in res_inp_edges:
            lname = f"w_{src_id}_to_{dst_id}"
            var   = f"x_proj_{src_id}_to_{dst_id}"
            L.append(f"        {var} = self.{lname}(cnn_feat)"
                     f".unsqueeze(0).expand(T, -1, -1)  # (T, B, {dst_id}_size)")
            projected[(src_id, dst_id)] = var
        L.append("")

    # init membrane potentials
    L.append("        # initialise membrane potentials and spike trains")
    for n in res_nodes:
        L.append(f"        v_{n['id']} = torch.zeros(B, self.{n['id']}_size, device=device)")
        L.append(f"        z_{n['id']} = torch.zeros(B, self.{n['id']}_size, device=device)")
    L.append("")
    L.append("        outputs = []")
    L.append("        for t in range(T):")
    if not use_cnn:
        L.append(f"            x = x_seq[t]  # (B, {in_node['id']}_size)")
        L.append("")

    for node_id in topo:
        nd = node_map[node_id]
        if nd["type"] in ("input", "output"):
            continue
        in_e  = incoming[node_id]
        # Split by source type (mirrors __init__ logic)
        res_srcs   = [(s, e) for s, e in in_e
                      if node_map[s]["type"] == "reservoir"]
        other_srcs = [(s, e) for s, e in in_e
                      if node_map[s]["type"] != "reservoir"]
        parts = []
        # Non-reservoir sources (input projections)
        for src_id, _ in other_srcs:
            src_nd = node_map[src_id]
            if use_cnn and src_nd["type"] == "input":
                var = projected.get((src_id, node_id))
                parts.append(f"{var}[t]")
            else:
                src_expr = "x" if src_nd["type"] == "input" else f"z_{src_id}"
                parts.append(f"self.w_{src_id}_to_{node_id}({src_expr})")
        # Reservoir sources — concat-merge if >1, else single projection
        if len(res_srcs) > 1:
            cat_vars = ", ".join(f"z_{s}" for s, _ in res_srcs)
            parts.append(f"self.w_merge_{node_id}(torch.cat([{cat_vars}], dim=-1))")
        elif len(res_srcs) == 1:
            src_id = res_srcs[0][0]
            parts.append(f"self.w_{src_id}_to_{node_id}(z_{src_id})")
        # Recurrent self-loop
        if node_id in recurrent:
            parts.append(f"self.w_rec_{node_id}(z_{node_id})")
        if not parts:
            parts = [f"torch.zeros(B, self.{node_id}_size, device=device)"]
        cur = " + ".join(parts)
        L.append(f"            # reservoir: {node_id}")
        L.append(f"            v_{node_id} = self.lif_{node_id}(")
        L.append(f"                {cur},")
        L.append(f"                v_{node_id}, z_{node_id})")
        L.append(f"            z_{node_id} = self.spike(v_{node_id})")
        L.append("")

    out_nid  = out_node["id"]
    in_e_out = incoming[out_nid]
    L.append("            # output projection")
    if len(in_e_out) == 1:
        src_id, _ = in_e_out[0]
        lname    = f"w_{src_id}_to_{out_nid}"
        src_nd   = node_map[src_id]
        src_expr = "x" if src_nd["type"] == "input" else f"z_{src_id}"
        L.append(f"            out = self.{lname}({src_expr})")
    else:
        cat_parts = []
        for src_id, _ in in_e_out:
            src_nd = node_map[src_id]
            cat_parts.append("x" if src_nd["type"] == "input" else f"z_{src_id}")
        L.append(f"            out = self.wout(torch.cat([{', '.join(cat_parts)}], dim=-1))")

    L += [
        "            outputs.append(out)",
        "",
        "        return torch.stack(outputs, dim=0)  # (T, B, output_size)",
    ]

    # ── usage example comment ─────────────────────────────────────────────
    sizes = {n["id"]: n.get("size", 0) for n in nodes}
    size_args = ", ".join(f"{n['id']}_size={sizes[n['id']]}"
                          for n in nodes if n["type"] in ("input", "reservoir", "output"))

    L += [
        "",
        "",
        "# ── Example usage ────────────────────────────────────────────────────",
        "# import torch",
        f"# model = {name}({size_args})",
        "# model = model.to('cuda' if torch.cuda.is_available() else 'cpu')",
        "# model.count_parameters()",
        "#",
        "# T, B = 20, 64   # timesteps, batch size",
        f"# x = torch.zeros(T, B, {sizes[inp_nodes[0]['id']]})  # (T, B, inp_size)",
        "# out = model(x)   # → (T, B, out_size)",
        "# pred = out.mean(dim=0).argmax(dim=-1)  # majority vote over time",
    ]

    return "\n".join(L) + "\n"


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _is_image(path: str) -> bool:
    return any(path.endswith(ext) for ext in (".png", ".svg", ".pdf", ".jpg"))


if __name__ == "__main__":


    if len(sys.argv) < 2:
        print("Usage: python snn_codegen.py <name>")
        print("  e.g. python snn_codegen.py binary_tree")
        sys.exit(1)

    # Если передано одно слово — разворачиваем в пути автоматически
    if len(sys.argv) == 2 and not sys.argv[1].endswith(".json"):
        name = sys.argv[1]
        cfg_path = f"examples/config_{name}.json"
        code_path = f"results/res_{name}.py"
        viz_path = f"gallery/{name}.png"
    else:
        # Старый формат: явные пути
        cfg_path = sys.argv[1]
        code_path = sys.argv[2] if len(sys.argv) > 2 else None
        viz_path = sys.argv[3] if len(sys.argv) > 3 else None

    if cfg_path is None:
        print("Error: no config file specified.")
        sys.exit(1)

    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)

    # generate code
    if code_path or viz_path is None:   # default: print or save code
        code = generate_model_code(cfg)
        if code_path:
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)
            print(f"[codegen] Generated: {code_path}")
        else:
            print(code)

    # visualize
    if viz_path:
        visualize_model(cfg, viz_path)
