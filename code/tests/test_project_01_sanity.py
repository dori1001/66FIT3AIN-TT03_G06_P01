"""Sanity and functional tests for Project 01 (Smart Campus Route Planner)."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
STARTER_DIR = PROJECT_DIR / "starter"

sys.path.insert(0, str(STARTER_DIR))

import graph as graph_module  # noqa: E402


def _load_module(name: str, path: Path):
    """Load a module under a unique name so `starter/student_core.py`
    (which also exists in project-02) never collides via `sys.modules`
    when both projects' sanity tests run in the same pytest session."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


student_core = _load_module("project01_student_core", STARTER_DIR / "student_core.py")


# ---------------------------------------------------------------------------
# Data files
# ---------------------------------------------------------------------------


def test_data_files_exist():
    for name in ["nodes.csv", "edges.csv", "scenarios.json", "campus_map.png", "SHA256SUMS", "README.md"]:
        assert (DATA_DIR / name).exists(), f"missing data/{name}"


def test_nodes_csv_schema_and_size():
    with open(DATA_DIR / "nodes.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == [
            "node_id",
            "name",
            "x",
            "y",
            "floor",
            "building",
            "node_type",
        ]
        rows = list(reader)
    assert 30 <= len(rows) <= 60, f"expected 30-60 nodes, got {len(rows)}"
    ids = [r["node_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate node_id values"


def test_edges_csv_schema_and_size():
    with open(DATA_DIR / "edges.csv", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == [
            "source",
            "target",
            "distance",
            "travel_time",
            "crowd_level",
            "stairs",
            "indoor",
            "accessible",
        ]
        rows = list(reader)
    assert 50 <= len(rows) <= 120, f"expected 50-120 edges, got {len(rows)}"


def test_edges_reference_known_nodes():
    with open(DATA_DIR / "nodes.csv", newline="", encoding="utf-8") as fh:
        node_ids = {row["node_id"] for row in csv.DictReader(fh)}
    with open(DATA_DIR / "edges.csv", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            assert row["source"] in node_ids
            assert row["target"] in node_ids


def test_scenarios_json_covers_required_classes():
    with open(DATA_DIR / "scenarios.json", encoding="utf-8") as fh:
        data = json.load(fh)
    scenarios = data["scenarios"]
    ids = {s["id"] for s in scenarios}
    for required_id in [
        "S1_normal_route",
        "S2_blocked_edge",
        "S3_accessibility_mode",
        "S4_heuristic_comparison",
        "S5_accessibility_no_path",
    ]:
        assert required_id in ids, f"missing scenario id {required_id}"

    normal = next(s for s in scenarios if s["id"] == "S1_normal_route")
    blocked = next(s for s in scenarios if s["id"] == "S2_blocked_edge")
    accessible = next(s for s in scenarios if s["id"] == "S3_accessibility_mode")
    comparison = next(s for s in scenarios if s["id"] == "S4_heuristic_comparison")

    assert "blocked_edges" in blocked["constraints"]
    assert accessible["constraints"].get("accessible_only") is True
    assert len(comparison.get("heuristics", [])) >= 2
    assert normal["start"] and normal["end"]


def test_sha256sums_match_committed_data():
    import hashlib

    expected = {}
    with open(DATA_DIR / "SHA256SUMS", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            digest, name = line.split("  ", 1)
            expected[name] = digest

    for name in ["nodes.csv", "edges.csv", "scenarios.json"]:
        actual = hashlib.sha256((DATA_DIR / name).read_bytes()).hexdigest()
        assert actual == expected[name], f"{name} does not match SHA256SUMS"


# ---------------------------------------------------------------------------
# Starter graph loader (no search allowed here)
# ---------------------------------------------------------------------------


def test_graph_loads_and_has_neighbors():
    g = graph_module.load_graph(DATA_DIR)
    summary = g.summary()
    assert 30 <= summary["num_nodes"] <= 60
    assert 50 <= summary["num_edges"] <= 120

    any_node = g.node_ids[0]
    neighbors = g.neighbors(any_node)
    assert isinstance(neighbors, list)
    for edge in neighbors:
        assert edge.source == any_node or edge.target == any_node


def test_graph_unknown_node_raises_keyerror():
    g = graph_module.load_graph(DATA_DIR)
    with pytest.raises(KeyError):
        g.get_node("N999_does_not_exist")


def test_graph_module_has_no_search_functions():
    forbidden = {"bfs", "greedy", "astar", "dfs"}
    defined = {name for name in dir(graph_module) if not name.startswith("_")}
    leaked = forbidden & defined
    assert not leaked, f"starter/graph.py must not define search functions: {leaked}"


# ---------------------------------------------------------------------------
# Student AI Core & Scenarios
# ---------------------------------------------------------------------------


def test_student_core_algorithms_and_scenarios():
    g = graph_module.load_graph(DATA_DIR)
    start, goal = "N_GATE", "N_D7_101"

    # Heuristics
    assert student_core.euclidean_heuristic(g, start, goal) > 0
    assert student_core.manhattan_heuristic(g, start, goal) > 0

    # BFS & Greedy
    bfs_path, bfs_exp = student_core.bfs(g, start, goal)
    assert bfs_path is not None and bfs_path[0] == start and bfs_path[-1] == goal
    assert bfs_exp > 0

    greedy_path, greedy_exp = student_core.greedy(g, start, goal, student_core.euclidean_heuristic)
    assert greedy_path is not None and greedy_path[0] == start and greedy_path[-1] == goal
    assert greedy_exp > 0

    # S1: Normal route
    s1_path, s1_exp = student_core.astar(g, start, goal, student_core.euclidean_heuristic)
    assert s1_path is not None and s1_path[0] == start and s1_path[-1] == goal
    assert s1_exp > 0

    # S2: Blocked edge
    blocked = [["W_CROSS_EAST", "W_SOUTH_STADIUM"]]
    s2_path, _ = student_core.astar(
        g, start, goal, student_core.euclidean_heuristic, blocked_edges=blocked
    )
    assert s2_path is not None
    for a, b in zip(s2_path, s2_path[1:]):
        assert {a, b} != {"W_CROSS_EAST", "W_SOUTH_STADIUM"}

    # S3: Accessibility mode
    s3_path, _ = student_core.astar(
        g, "N_GATE", "N_C_201", student_core.euclidean_heuristic, accessible_only=True
    )
    assert s3_path is not None and s3_path[-1] == "N_C_201"

    # S4: Heuristic comparison
    s4_euclid, _ = student_core.astar(
        g, "N_A1_PARKING", "N_D6_101", student_core.euclidean_heuristic
    )
    s4_manh, _ = student_core.astar(
        g, "N_A1_PARKING", "N_D6_101", student_core.manhattan_heuristic
    )
    assert s4_euclid is not None and s4_manh is not None
    assert student_core.path_cost(g, s4_euclid) == pytest.approx(
        student_core.path_cost(g, s4_manh)
    )

    # S5: Accessibility no path
    s5_path, s5_exp = student_core.astar(
        g, "N_GATE", "N_C_ATTIC", student_core.euclidean_heuristic, accessible_only=True
    )
    assert s5_path is None and s5_exp > 0


def test_no_pygame_dependency():
    requirements = (PROJECT_DIR / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "pygame" not in requirements, "project 01 must not require pygame"


def test_heuristics_with_blocked_and_accessibility_constraints():
    """Verify A* with both Euclidean and Manhattan heuristics under combined
    blocked_edges and accessible_only constraints (Priority 2)."""
    g = graph_module.load_graph(DATA_DIR)
    blocked = [["W_CROSS_EAST", "W_SOUTH_STADIUM"]]

    # Both heuristics under blocked_edges + accessible_only must find the exact same optimal cost
    p_euc_d, exp_euc = student_core.astar(
        g, "N_GATE", "N_D7_101", student_core.euclidean_heuristic,
        edge_weight="distance", blocked_edges=blocked, accessible_only=True
    )
    p_man_d, exp_man = student_core.astar(
        g, "N_GATE", "N_D7_101", student_core.manhattan_heuristic,
        edge_weight="distance", blocked_edges=blocked, accessible_only=True
    )
    assert p_euc_d is not None and p_man_d is not None
    c_euc = student_core.path_cost(g, p_euc_d, "distance", blocked_edges=blocked, accessible_only=True)
    c_man = student_core.path_cost(g, p_man_d, "distance", blocked_edges=blocked, accessible_only=True)
    assert c_euc == pytest.approx(c_man)
    assert exp_euc <= exp_man  # Euclidean dominance

    # Accessibility constraint on S3 with Manhattan vs Euclidean
    p_s3_euc, _ = student_core.astar(
        g, "N_GATE", "N_C_201", student_core.euclidean_heuristic,
        edge_weight="travel_time", accessible_only=True
    )
    p_s3_man, _ = student_core.astar(
        g, "N_GATE", "N_C_201", student_core.manhattan_heuristic,
        edge_weight="travel_time", accessible_only=True
    )
    assert student_core.path_cost(g, p_s3_euc, "travel_time", accessible_only=True) == pytest.approx(
        student_core.path_cost(g, p_s3_man, "travel_time", accessible_only=True)
    )
