"""Stress, boundary, caching, and ground-truth optimality tests for Final Milestone."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
STARTER_DIR = PROJECT_DIR / "starter"

sys.path.insert(0, str(STARTER_DIR))
import graph as graph_module  # noqa: E402


def _load_core():
    spec = importlib.util.spec_from_file_location("p01_final_core", STARTER_DIR / "student_core.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["p01_final_core"] = mod
    spec.loader.exec_module(mod)
    return mod


student_core = _load_core()


def test_stress_s6_multi_corridor_outage():
    g = graph_module.load_graph(DATA_DIR)
    blocked = [
        ["W_CROSS_EAST", "W_SOUTH_STADIUM"],
        ["W_SOUTH_STADIUM", "N_GREEN_SQUARE"],
    ]
    path, exp = student_core.astar(
        g, "N_GATE", "N_D7_101", student_core.euclidean_heuristic,
        edge_weight="distance", blocked_edges=blocked,
    )
    assert path is not None and path[0] == "N_GATE" and path[-1] == "N_D7_101"
    assert exp > 0
    for u, v in zip(path, path[1:]):
        assert {u, v} != {"W_CROSS_EAST", "W_SOUTH_STADIUM"}
        assert {u, v} != {"W_SOUTH_STADIUM", "N_GREEN_SQUARE"}


def test_stress_s7_combined_maintenance_and_accessibility():
    g = graph_module.load_graph(DATA_DIR)
    blocked = [
        ["N_D1_ENTRANCE", "N_D1_101"],
        ["W_CROSS_EAST", "W_SOUTH_STADIUM"],
    ]
    p_euc, _ = student_core.astar(
        g, "N_GATE", "N_C_201", student_core.euclidean_heuristic,
        edge_weight="distance", blocked_edges=blocked, accessible_only=True,
    )
    p_man, _ = student_core.astar(
        g, "N_GATE", "N_C_201", student_core.manhattan_heuristic,
        edge_weight="distance", blocked_edges=blocked, accessible_only=True,
    )
    assert p_euc is not None and p_man is not None
    c_euc = student_core.path_cost(g, p_euc, "distance", blocked, True)
    c_man = student_core.path_cost(g, p_man, "distance", blocked, True)
    assert c_euc == pytest.approx(c_man)


def test_stress_s8_complete_destination_isolation_no_crash():
    g = graph_module.load_graph(DATA_DIR)
    blocked = [["N_D7_ENTRANCE", "N_D7_101"]]
    for fn in [
        lambda: student_core.bfs(g, "N_GATE", "N_D7_101", blocked_edges=blocked),
        lambda: student_core.greedy(g, "N_GATE", "N_D7_101", student_core.euclidean_heuristic, blocked_edges=blocked),
        lambda: student_core.astar(g, "N_GATE", "N_D7_101", student_core.euclidean_heuristic, blocked_edges=blocked),
    ]:
        path, exp = fn()
        assert path is None
        assert exp == 53


def test_boundary_invalid_and_trivial_endpoints():
    g = graph_module.load_graph(DATA_DIR)
    assert student_core.astar(g, "UNKNOWN_A", "N_GATE", student_core.euclidean_heuristic) == (None, 0)
    assert student_core.astar(g, "N_GATE", "N_GATE", student_core.euclidean_heuristic) == (["N_GATE"], 0)
    with pytest.raises(ValueError):
        student_core.astar(g, "N_GATE", "N_D7_101", student_core.euclidean_heuristic, edge_weight="invalid_weight")


def test_cached_final_matches_uncached_midterm_and_dijkstra_all_pairs():
    g = graph_module.load_graph(DATA_DIR)
    zero_h = lambda gr, a, b: 0.0
    student_core.clear_cost_scale_cache()
    # Check representative subset + all pairs for distance optimality
    for w in ("distance", "travel_time"):
        for u in g.node_ids:
            for v in g.node_ids:
                p_dijk, _ = student_core.astar(g, u, v, zero_h, edge_weight=w, scale_heuristic=False)
                p_mid, _ = student_core.astar(g, u, v, student_core.euclidean_heuristic, edge_weight=w, use_scale_cache=False)
                p_fin, _ = student_core.astar(g, u, v, student_core.euclidean_heuristic, edge_weight=w, use_scale_cache=True)
                c_dijk = student_core.path_cost(g, p_dijk, w)
                c_mid = student_core.path_cost(g, p_mid, w)
                c_fin = student_core.path_cost(g, p_fin, w)
                assert c_mid == pytest.approx(c_dijk)
                assert c_fin == pytest.approx(c_dijk)
