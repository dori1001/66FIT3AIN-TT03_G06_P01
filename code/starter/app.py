"""Final Milestone Interactive Application — Project 01 (Smart Campus Route Planner, HANU).

Features:
- 1-Click Standard Scenarios (S1-S5 from data/scenarios.json) & Stress Scenarios (S6-S8)
- Dynamic blocked-edge maintenance simulator & wheelchair accessibility toggle
- Side-by-Side Dual-Map Route Comparison (Route A vs. Route B) with delta metrics
- Turn-by-Turn Navigation Panel (distance, travel time, floor transition, crowd level)
- Live Execution Log Breakdown (Nodes Expanded, Generated, Edge Checks, Duplicates, Cycles)
- Cached Final A* vs. Uncached Midterm A* toggle
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from graph import Graph, load_graph, load_scenarios  # noqa: E402
import student_core  # noqa: E402
import profile_search_logs as psl  # noqa: E402

st.set_page_config(page_title="Smart Campus Route Planner — HANU", layout="wide")
st.title("Smart Campus Route Planner — Hanoi University (HANU)")
st.markdown(
    "Interactive AI wayfinding system implementing **BFS**, **Greedy Best-First Search (GBFS)**, "
    "and **Optimal A*** (with cached admissible scaling) from scratch."
)

graph = load_graph()
scenarios_payload = load_scenarios()
scenarios_list = scenarios_payload.get("scenarios", [])
scenarios_by_id: Dict[str, dict] = {s["id"]: s for s in scenarios_list}

# Stress Scenarios (S6-S8) for Final Milestone Defence
STRESS_SCENARIOS: Dict[str, dict] = {
    "S6_stress_multi_corridor": {
        "id": "S6_stress_multi_corridor",
        "class": "stress",
        "start": "N_GATE",
        "end": "N_D7_101",
        "constraints": {
            "blocked_edges": [
                ["W_CROSS_EAST", "W_SOUTH_STADIUM"],
                ["W_SOUTH_STADIUM", "N_GREEN_SQUARE"],
            ],
            "accessible_only": False,
        },
        "description": (
            "Stress Case S6 (Multi-Corridor Outage): Both the main South Stadium spine and "
            "the Green Square bypass are closed simultaneously, forcing dynamic recalculation "
            "via the Northern Stadium perimeter or Library corridor."
        ),
    },
    "S7_stress_combined_outage_accessible": {
        "id": "S7_stress_combined_outage_accessible",
        "class": "stress",
        "start": "N_GATE",
        "end": "N_C_201",
        "constraints": {
            "blocked_edges": [
                ["N_D1_ENTRANCE", "N_D1_101"],
                ["W_CROSS_EAST", "W_SOUTH_STADIUM"],
            ],
            "accessible_only": True,
        },
        "description": (
            "Stress Case S7 (Combined Maintenance + Wheelchair): Building D1 corridor is blocked "
            "while wheelchair mode is enabled, forcing an accessible detour around Building D2/D3 "
            "to reach the Building C elevator."
        ),
    },
    "S8_stress_complete_isolation": {
        "id": "S8_stress_complete_isolation",
        "class": "stress-failure",
        "start": "N_GATE",
        "end": "N_D7_101",
        "constraints": {
            "blocked_edges": [
                ["N_D7_ENTRANCE", "N_D7_101"],
            ],
            "accessible_only": False,
        },
        "description": (
            "Stress Case S8 (Complete Destination Isolation): The sole corridor into D7.101 "
            "is closed for maintenance. The system must exhaust all reachable nodes and report "
            "No Path Found gracefully without crashing."
        ),
    },
}

all_presets: Dict[str, dict] = {**scenarios_by_id, **STRESS_SCENARIOS}
node_ids = sorted(graph.node_ids)

HEURISTICS = {
    "euclidean": student_core.euclidean_heuristic,
    "manhattan": student_core.manhattan_heuristic,
}


def _coord_to_px(x: float, y: float, width: int = 1200, height: int = 900) -> Tuple[float, float]:
    min_x, max_x = 0.0, 620.0
    min_y, max_y = 0.0, 470.0
    margin_x, margin_y = 90.0, 80.0
    map_w = width - 2 * margin_x
    map_h = height - 2 * margin_y
    scale = min(map_w / (max_x - min_x), map_h / (max_y - min_y))
    px = margin_x + (x - min_x) * scale
    py = height - margin_y - (y - min_y) * scale
    return px, py


def _render_route_map(
    map_path: Path,
    graph: Graph,
    path: Optional[List[str]],
    start_id: str,
    end_id: str,
    blocked_edges: Optional[List[List[str]]] = None,
    route_color: str = "#D62728",
    title: str = "Route Overlay",
):
    bg_img = Image.open(map_path)
    width, height = bg_img.size
    fig, ax = plt.subplots(figsize=(8.5, 6.4), dpi=130)
    ax.imshow(bg_img)

    if blocked_edges:
        for idx, pair in enumerate(blocked_edges):
            if len(pair) == 2 and pair[0] in graph.node_ids and pair[1] in graph.node_ids:
                u, v = pair[0], pair[1]
                p1 = _coord_to_px(graph.get_node(u).x, graph.get_node(u).y, width, height)
                p2 = _coord_to_px(graph.get_node(v).x, graph.get_node(v).y, width, height)
                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="black", linewidth=4.0, linestyle=":")
                mx, my = (p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0
                ax.scatter(
                    [mx], [my], marker="X", s=200, color="#FF1744", zorder=10,
                    label="Blocked Edge" if idx == 0 else None,
                )

    if path:
        pts = [_coord_to_px(graph.get_node(nid).x, graph.get_node(nid).y, width, height) for nid in path]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, linewidth=4.5, color=route_color, alpha=0.95, zorder=8, label="Computed Route")
        ax.scatter(xs, ys, s=55, color=route_color, edgecolors="white", linewidth=1.3, zorder=9)

    sx, sy = _coord_to_px(graph.get_node(start_id).x, graph.get_node(start_id).y, width, height)
    gx, gy = _coord_to_px(graph.get_node(end_id).x, graph.get_node(end_id).y, width, height)
    ax.scatter([sx], [sy], s=170, color="#00C853", edgecolors="black", linewidth=1.8, marker="o", zorder=11, label=f"Start ({start_id})")
    ax.scatter([gx], [gy], s=220, color="#FF1744", edgecolors="black", linewidth=1.8, marker="*", zorder=11, label=f"Goal ({end_id})")

    ax.set_title(title, fontsize=11.5, fontweight="bold")
    ax.axis("off")
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.92)
    plt.tight_layout()
    return fig


def _run_search(
    graph: Graph,
    start_id: str,
    end_id: str,
    algorithm: str,
    heuristic_name: str,
    edge_weight: str,
    blocked_edges: Optional[List[List[str]]],
    accessible_only: bool,
    use_scale_cache: bool,
) -> dict:
    h_fn = HEURISTICS[heuristic_name]
    t0 = time.perf_counter()
    if algorithm == "bfs":
        path, exp = student_core.bfs(graph, start_id, end_id, blocked_edges=blocked_edges, accessible_only=accessible_only)
    elif algorithm == "greedy":
        path, exp = student_core.greedy(graph, start_id, end_id, h_fn, blocked_edges=blocked_edges, accessible_only=accessible_only)
    else:
        path, exp = student_core.astar(
            graph, start_id, end_id, h_fn,
            edge_weight=edge_weight,
            blocked_edges=blocked_edges,
            accessible_only=accessible_only,
            use_scale_cache=use_scale_cache,
        )
    runtime_ms = (time.perf_counter() - t0) * 1000.0

    # Non-intrusive trace log
    if algorithm == "bfs":
        log = psl.trace_bfs(graph, start_id, end_id, blocked_edges=blocked_edges, accessible_only=accessible_only)
    elif algorithm == "greedy":
        log = psl.trace_greedy(graph, start_id, end_id, h_fn, blocked_edges=blocked_edges, accessible_only=accessible_only)
    else:
        log = psl.trace_astar(graph, start_id, end_id, h_fn, edge_weight=edge_weight, blocked_edges=blocked_edges, accessible_only=accessible_only, use_scale_cache=use_scale_cache)

    dist = student_core.path_cost(graph, path, "distance", blocked_edges, accessible_only) if path else None
    ttime = student_core.path_cost(graph, path, "travel_time", blocked_edges, accessible_only) if path else None

    return {
        "path": path,
        "expanded": exp,
        "runtime_ms": runtime_ms,
        "distance": dist,
        "travel_time": ttime,
        "hops": (len(path) - 1) if path else None,
        "log": log,
    }


# ---------------------------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1-Click Scenarios & Stress Simulator")
    preset_options = ["Custom"] + list(all_presets.keys())
    selected_preset = st.selectbox(
        "Select preset (S1–S5 Standard, S6–S8 Stress):",
        preset_options,
        index=1,
    )

    if selected_preset != "Custom":
        active_sc = all_presets[selected_preset]
        default_start = active_sc["start"]
        default_end = active_sc["end"]
        sc_cons = active_sc.get("constraints", {})
        default_accessible = bool(sc_cons.get("accessible_only", False))
        default_blocked_list = sc_cons.get("blocked_edges", [])
        st.info(f"**{active_sc['id']} ({active_sc['class']})**: {active_sc['description']}")
    else:
        default_start = "N_GATE"
        default_end = "N_D7_101"
        default_accessible = False
        default_blocked_list = []

    st.header("Route Endpoints & Constraints")
    start_id = st.selectbox(
        "Start landmark",
        node_ids,
        index=node_ids.index(default_start) if default_start in node_ids else 0,
        format_func=lambda nid: f"{nid} — {graph.get_node(nid).name}",
        key=f"start_{selected_preset}",
    )
    end_id = st.selectbox(
        "Destination landmark",
        node_ids,
        index=node_ids.index(default_end) if default_end in node_ids else len(node_ids) - 1,
        format_func=lambda nid: f"{nid} — {graph.get_node(nid).name}",
        key=f"end_{selected_preset}",
    )
    accessible_only = st.checkbox(
        "Accessible-only mode (wheelchair / no stairs)",
        value=default_accessible,
        key=f"acc_{selected_preset}",
    )
    use_scale_cache = st.checkbox("Enable Final alpha_w cache (A* optimization)", value=True)

    all_undirected_edges = []
    seen_pairs = set()
    for u in graph.node_ids:
        for e in graph.neighbors(u):
            pk = tuple(sorted([e.source, e.target]))
            if pk not in seen_pairs:
                seen_pairs.add(pk)
                all_undirected_edges.append(f"{pk[0]} -- {pk[1]}")
    all_undirected_edges.sort()

    default_blocked_labels = []
    for pair in default_blocked_list:
        if len(pair) == 2:
            pk = tuple(sorted([pair[0], pair[1]]))
            lbl = f"{pk[0]} -- {pk[1]}"
            if lbl in all_undirected_edges:
                default_blocked_labels.append(lbl)

    selected_blocked_labels = st.multiselect(
        "Dynamically blocked corridors (maintenance):",
        options=all_undirected_edges,
        default=default_blocked_labels,
        key=f"blk_{selected_preset}",
    )
    blocked_edges = [lbl.split(" -- ") for lbl in selected_blocked_labels] or None

    st.header("Side-by-Side Comparison Config")
    col_sa, col_sb = st.columns(2)
    with col_sa:
        st.caption("**Route A (Primary)**")
        alg_a = st.selectbox("Alg A", ["astar", "bfs", "greedy"], index=0, key="alg_a")
        heur_a = st.selectbox("Heur A", ["euclidean", "manhattan"], index=0, key="heur_a")
        wt_a = st.selectbox("Weight A", ["distance", "travel_time"], index=0, key="wt_a")
    with col_sb:
        st.caption("**Route B (Compare)**")
        alg_b = st.selectbox("Alg B", ["bfs", "greedy", "astar"], index=0, key="alg_b")
        heur_b = st.selectbox("Heur B", ["euclidean", "manhattan"], index=0, key="heur_b")
        wt_b = st.selectbox("Weight B", ["distance", "travel_time"], index=0, key="wt_b")

    st.header("Graph Summary")
    summary = graph.summary()
    st.metric("Nodes", summary["num_nodes"])
    st.metric("Undirected Edges", summary["num_edges"])

# ---------------------------------------------------------------------------
# Execute Route A & Route B
# ---------------------------------------------------------------------------
res_a = _run_search(graph, start_id, end_id, alg_a, heur_a, wt_a, blocked_edges, accessible_only, use_scale_cache)
res_b = _run_search(graph, start_id, end_id, alg_b, heur_b, wt_b, blocked_edges, accessible_only, use_scale_cache)

label_a = f"Route A: {alg_a.upper()}" + (f" ({heur_a}, {wt_a})" if alg_a == "astar" else (f" ({heur_a})" if alg_a == "greedy" else " (hops)"))
label_b = f"Route B: {alg_b.upper()}" + (f" ({heur_b}, {wt_b})" if alg_b == "astar" else (f" ({heur_b})" if alg_b == "greedy" else " (hops)"))

map_path = PROJECT_ROOT / "data" / "campus_map.png"

# Top Side-by-Side Route Comparison View
st.subheader("Side-by-Side Route Comparison")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown(f"#### {label_a}")
    if res_a["path"] is None:
        st.error(f"No valid route found! (Expanded: {res_a['expanded']} | Edge Checks: {res_a['log']['edge_checks']} | Runtime: {res_a['runtime_ms']:.3f} ms)")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hops", res_a["hops"])
        c2.metric("Expanded", res_a["expanded"])
        c3.metric("Distance (m)", f"{res_a['distance']:.1f}")
        c4.metric("Time (s)", f"{res_a['travel_time']:.2f}")
        st.caption(f"Edge Checks: **{res_a['log']['edge_checks']}** | Duplicates: **{res_a['log']['duplicate_total']}** | Re-expansions: **{res_a['log']['re_expansions']}** | Runtime: **{res_a['runtime_ms']:.4f} ms**")
    if map_path.exists():
        fig_a = _render_route_map(map_path, graph, res_a["path"], start_id, end_id, blocked_edges, route_color="#D62728", title=label_a)
        st.pyplot(fig_a, use_container_width=True)
        plt.close(fig_a)

with col_right:
    st.markdown(f"#### {label_b}")
    if res_b["path"] is None:
        st.error(f"No valid route found! (Expanded: {res_b['expanded']} | Edge Checks: {res_b['log']['edge_checks']} | Runtime: {res_b['runtime_ms']:.3f} ms)")
    else:
        d_dist = (res_a["distance"] - res_b["distance"]) if (res_a["distance"] is not None and res_b["distance"] is not None) else None
        d_exp = res_a["expanded"] - res_b["expanded"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Hops", res_b["hops"])
        c2.metric("Expanded", res_b["expanded"], delta=f"{-d_exp:+d} vs A" if d_exp != 0 else "0", delta_color="inverse")
        c3.metric("Distance (m)", f"{res_b['distance']:.1f}", delta=f"{-d_dist:+.1f} m vs A" if d_dist is not None and abs(d_dist) > 1e-6 else "0.0 m", delta_color="inverse")
        c4.metric("Time (s)", f"{res_b['travel_time']:.2f}")
        st.caption(f"Edge Checks: **{res_b['log']['edge_checks']}** | Duplicates: **{res_b['log']['duplicate_total']}** | Re-expansions: **{res_b['log']['re_expansions']}** | Runtime: **{res_b['runtime_ms']:.4f} ms**")
    if map_path.exists():
        fig_b = _render_route_map(map_path, graph, res_b["path"], start_id, end_id, blocked_edges, route_color="#1F77B4", title=label_b)
        st.pyplot(fig_b, use_container_width=True)
        plt.close(fig_b)

# Turn-by-Turn Navigation for Route A
if res_a["path"] is not None:
    with st.expander("Turn-by-Turn Directions (Route A)", expanded=True):
        st.success(" → ".join(f"**{nid}** ({graph.get_node(nid).name})" for nid in res_a["path"]))
        steps_list = student_core.describe_route_steps(
            graph, res_a["path"],
            edge_weight=wt_a if alg_a == "astar" else "distance",
            blocked_edges=blocked_edges,
            accessible_only=accessible_only,
        )
        if steps_list:
            st.dataframe(pd.DataFrame(steps_list), use_container_width=True, hide_index=True)
        else:
            st.info("Start and destination are the same landmark (0 hops, 0.0 m).")

# Full 7-Configuration Live Benchmark Table
st.subheader("Live All-Algorithm Benchmark & Execution Log Breakdown")
bench_configs = [
    ("BFS (Unweighted)", "bfs", "euclidean", "distance"),
    ("GBFS (Euclidean)", "greedy", "euclidean", "distance"),
    ("GBFS (Manhattan)", "greedy", "manhattan", "distance"),
    ("A* (Euclidean, distance)", "astar", "euclidean", "distance"),
    ("A* (Manhattan, distance)", "astar", "manhattan", "distance"),
    ("A* (Euclidean, travel_time)", "astar", "euclidean", "travel_time"),
    ("A* (Manhattan, travel_time)", "astar", "manhattan", "travel_time"),
]
bench_rows = []
for lbl, a_k, h_k, w_k in bench_configs:
    r = _run_search(graph, start_id, end_id, a_k, h_k, w_k, blocked_edges, accessible_only, use_scale_cache)
    lg = r["log"]
    bench_rows.append({
        "Configuration": lbl,
        "Found": "Yes" if r["path"] is not None else "No (None)",
        "Hops": r["hops"],
        "Nodes Expanded": r["expanded"],
        "Generated": lg["generated"],
        "Edge Checks": lg["edge_checks"],
        "Duplicates (Cycle)": f"{lg['duplicate_total']} ({lg['cycle_rejections']})",
        "Peak Frontier": lg["peak_frontier"],
        "Distance (m)": round(r["distance"], 1) if r["distance"] is not None else None,
        "Time (s)": round(r["travel_time"], 2) if r["travel_time"] is not None else None,
        "Runtime (ms)": round(r["runtime_ms"], 4),
    })
st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True)
