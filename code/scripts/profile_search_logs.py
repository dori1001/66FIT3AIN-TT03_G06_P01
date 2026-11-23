#!/usr/bin/env python3
"""Non-intrusive Search Execution Log Profiler for Project 01 (HANU Route Planner).

Instruments `bfs`, `greedy`, and `astar` with the EXACT same logic and edge-case
handling (`_valid_endpoints`, `start == goal`) as `starter/student_core.py` to measure:
- Nodes_Expanded: nodes popped and expanded
- Generated_Nodes: total nodes pushed to frontier (including start=1 when search runs)
- Edge_Checks: total incident edges examined in `for edge in graph.neighbors(current)`
- Filter_Rejections: edges rejected by `edge_ok(edge)` (blocked or inaccessible)
- Duplicate_Total: Cycle_Rejections + Visited_Frontier_Duplicates + Stale_Heap_Pops
- Re_Expansions: nodes expanded more than once (0 when heuristic is consistent)
- Peak_Frontier_Size: maximum simultaneous entries in queue/heap
"""
from __future__ import annotations

import heapq
import math
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Set

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
STARTER_DIR = PROJECT_DIR / "starter"

sys.path.insert(0, str(STARTER_DIR))
import graph as graph_module  # noqa: E402
import student_core  # noqa: E402


def _get_ancestors(parent_map: Dict[str, Optional[str]], node: str) -> Set[str]:
    anc: Set[str] = set()
    curr: Optional[str] = node
    while curr is not None:
        anc.add(curr)
        curr = parent_map.get(curr)
    return anc


def _empty_result(path: Optional[List[str]], expanded: int = 0) -> dict:
    return {
        "path": path,
        "expanded": expanded,
        "generated": 0,
        "edge_checks": 0,
        "filter_rejections": 0,
        "duplicate_total": 0,
        "cycle_rejections": 0,
        "other_duplicates": 0,
        "stale_pops": 0,
        "re_expansions": 0,
        "peak_frontier": 0,
        "steps": [],
    }


def trace_bfs(graph, start: str, goal: str, blocked_edges=None, accessible_only: bool = False) -> dict:
    if not student_core._valid_endpoints(graph, start, goal):
        return _empty_result(None, 0)
    if start == goal:
        return _empty_result([start], student_core.START_EQUALS_GOAL_EXPANDED)

    edge_ok = student_core.make_edge_filter(blocked_edges, accessible_only)
    queue = deque([start])
    parent: Dict[str, Optional[str]] = {start: None}

    expanded = 0
    generated = 1
    edge_checks = 0
    filter_rejections = 0
    cycle_rejections = 0
    other_duplicates = 0
    re_expansions = 0
    expanded_set: Set[str] = set()
    peak_frontier = len(queue)
    steps = []

    while queue:
        peak_frontier = max(peak_frontier, len(queue))
        current = queue.popleft()
        if current in expanded_set:
            re_expansions += 1
        expanded_set.add(current)
        expanded += 1

        ancestors = _get_ancestors(parent, current)
        pushed = []

        if current == goal:
            steps.append({
                "step": expanded,
                "expanded_node": current,
                "metric_info": f"hop={len(ancestors)-1}",
                "generated_neighbors": "GOAL REACHED",
                "frontier_size_after": len(queue),
            })
            return {
                "path": student_core._build_path(parent, goal),
                "expanded": expanded,
                "generated": generated,
                "edge_checks": edge_checks,
                "filter_rejections": filter_rejections,
                "duplicate_total": cycle_rejections + other_duplicates,
                "cycle_rejections": cycle_rejections,
                "other_duplicates": other_duplicates,
                "stale_pops": 0,
                "re_expansions": re_expansions,
                "peak_frontier": peak_frontier,
                "steps": steps,
            }

        for edge in graph.neighbors(current):
            edge_checks += 1
            nbr = edge.target
            if nbr in parent:
                if nbr in ancestors:
                    cycle_rejections += 1
                else:
                    other_duplicates += 1
                continue
            if not edge_ok(edge):
                filter_rejections += 1
                continue
            parent[nbr] = current
            queue.append(nbr)
            generated += 1
            pushed.append(nbr)
            peak_frontier = max(peak_frontier, len(queue))

        steps.append({
            "step": expanded,
            "expanded_node": current,
            "metric_info": f"hop={len(ancestors)-1}",
            "generated_neighbors": ", ".join(pushed) if pushed else "(none)",
            "frontier_size_after": len(queue),
        })

    return {
        "path": None,
        "expanded": expanded,
        "generated": generated,
        "edge_checks": edge_checks,
        "filter_rejections": filter_rejections,
        "duplicate_total": cycle_rejections + other_duplicates,
        "cycle_rejections": cycle_rejections,
        "other_duplicates": other_duplicates,
        "stale_pops": 0,
        "re_expansions": re_expansions,
        "peak_frontier": peak_frontier,
        "steps": steps,
    }


def trace_greedy(graph, start: str, goal: str, heuristic, blocked_edges=None, accessible_only: bool = False) -> dict:
    if not student_core._valid_endpoints(graph, start, goal):
        return _empty_result(None, 0)
    if start == goal:
        return _empty_result([start], student_core.START_EQUALS_GOAL_EXPANDED)

    edge_ok = student_core.make_edge_filter(blocked_edges, accessible_only)
    counter = 0
    pq = [(heuristic(graph, start, goal), counter, start, None)]
    parent: Dict[str, Optional[str]] = {}

    expanded = 0
    generated = 1
    edge_checks = 0
    filter_rejections = 0
    cycle_rejections = 0
    other_duplicates = 0
    stale_pops = 0
    re_expansions = 0
    expanded_set: Set[str] = set()
    peak_frontier = len(pq)
    steps = []

    while pq:
        peak_frontier = max(peak_frontier, len(pq))
        h_val, _, current, par = heapq.heappop(pq)
        if current in parent:
            stale_pops += 1
            continue
        parent[current] = par
        if current in expanded_set:
            re_expansions += 1
        expanded_set.add(current)
        expanded += 1

        ancestors = _get_ancestors(parent, current)
        pushed = []

        if current == goal:
            steps.append({
                "step": expanded,
                "expanded_node": current,
                "metric_info": f"h={h_val:.1f}",
                "generated_neighbors": "GOAL REACHED",
                "frontier_size_after": len(pq),
            })
            return {
                "path": student_core._build_path(parent, goal),
                "expanded": expanded,
                "generated": generated,
                "edge_checks": edge_checks,
                "filter_rejections": filter_rejections,
                "duplicate_total": cycle_rejections + other_duplicates + stale_pops,
                "cycle_rejections": cycle_rejections,
                "other_duplicates": other_duplicates,
                "stale_pops": stale_pops,
                "re_expansions": re_expansions,
                "peak_frontier": peak_frontier,
                "steps": steps,
            }

        for edge in graph.neighbors(current):
            edge_checks += 1
            nbr = edge.target
            if nbr in parent:
                if nbr in ancestors:
                    cycle_rejections += 1
                else:
                    other_duplicates += 1
                continue
            if not edge_ok(edge):
                filter_rejections += 1
                continue
            counter += 1
            h_nbr = heuristic(graph, nbr, goal)
            heapq.heappush(pq, (h_nbr, counter, nbr, current))
            generated += 1
            pushed.append(f"{nbr}(h={h_nbr:.1f})")
            peak_frontier = max(peak_frontier, len(pq))

        steps.append({
            "step": expanded,
            "expanded_node": current,
            "metric_info": f"h={h_val:.1f}",
            "generated_neighbors": ", ".join(pushed) if pushed else "(none)",
            "frontier_size_after": len(pq),
        })

    return {
        "path": None,
        "expanded": expanded,
        "generated": generated,
        "edge_checks": edge_checks,
        "filter_rejections": filter_rejections,
        "duplicate_total": cycle_rejections + other_duplicates + stale_pops,
        "cycle_rejections": cycle_rejections,
        "other_duplicates": other_duplicates,
        "stale_pops": stale_pops,
        "re_expansions": re_expansions,
        "peak_frontier": peak_frontier,
        "steps": steps,
    }


def trace_astar(graph, start: str, goal: str, heuristic, edge_weight: str = "distance",
                blocked_edges=None, accessible_only: bool = False, scale_heuristic: bool = True,
                use_scale_cache: bool = True) -> dict:
    if edge_weight not in student_core._VALID_WEIGHTS:
        raise ValueError("edge_weight must be 'distance' or 'travel_time'")
    if not student_core._valid_endpoints(graph, start, goal):
        return _empty_result(None, 0)
    if start == goal:
        return _empty_result([start], student_core.START_EQUALS_GOAL_EXPANDED)

    edge_ok = student_core.make_edge_filter(blocked_edges, accessible_only)
    if scale_heuristic:
        factor = student_core._ADMISSIBILITY_FACTOR.get(heuristic, 1.0)
        if use_scale_cache:
            base_scale = student_core._get_cached_cost_scale(
                graph, edge_weight, edge_ok, blocked_edges, accessible_only
            )
        else:
            base_scale = student_core._cost_scale(graph, edge_weight, edge_ok)
        scale = base_scale * factor
        h = lambda n: scale * heuristic(graph, n, goal)  # noqa: E731
    else:
        h = lambda n: heuristic(graph, n, goal)  # noqa: E731

    g_score: Dict[str, float] = {start: 0.0}
    parent: Dict[str, Optional[str]] = {start: None}
    counter = 0
    pq = [(h(start), counter, 0.0, start)]

    expanded = 0
    generated = 1
    edge_checks = 0
    filter_rejections = 0
    cycle_rejections = 0
    other_duplicates = 0
    stale_pops = 0
    re_expansions = 0
    expanded_set: Set[str] = set()
    peak_frontier = len(pq)
    steps = []

    while pq:
        peak_frontier = max(peak_frontier, len(pq))
        f_val, _, g_val, current = heapq.heappop(pq)
        if g_val > g_score.get(current, math.inf):
            stale_pops += 1
            continue
        if current in expanded_set:
            re_expansions += 1
        expanded_set.add(current)
        expanded += 1

        ancestors = _get_ancestors(parent, current)
        pushed = []

        if current == goal:
            steps.append({
                "step": expanded,
                "expanded_node": current,
                "metric_info": f"g={g_val:.1f}, f={f_val:.1f}",
                "generated_neighbors": "GOAL REACHED",
                "frontier_size_after": len(pq),
            })
            return {
                "path": student_core._build_path(parent, goal),
                "expanded": expanded,
                "generated": generated,
                "edge_checks": edge_checks,
                "filter_rejections": filter_rejections,
                "duplicate_total": cycle_rejections + other_duplicates + stale_pops,
                "cycle_rejections": cycle_rejections,
                "other_duplicates": other_duplicates,
                "stale_pops": stale_pops,
                "re_expansions": re_expansions,
                "peak_frontier": peak_frontier,
                "steps": steps,
            }

        for edge in graph.neighbors(current):
            edge_checks += 1
            if not edge_ok(edge):
                filter_rejections += 1
                continue
            nbr = edge.target
            tentative = g_val + getattr(edge, edge_weight)
            if tentative < g_score.get(nbr, math.inf):
                g_score[nbr] = tentative
                parent[nbr] = current
                counter += 1
                f_nbr = tentative + h(nbr)
                heapq.heappush(pq, (f_nbr, counter, tentative, nbr))
                generated += 1
                pushed.append(f"{nbr}(g={tentative:.1f},f={f_nbr:.1f})")
                peak_frontier = max(peak_frontier, len(pq))
            else:
                if nbr in ancestors:
                    cycle_rejections += 1
                else:
                    other_duplicates += 1

        steps.append({
            "step": expanded,
            "expanded_node": current,
            "metric_info": f"g={g_val:.1f}, f={f_val:.1f}",
            "generated_neighbors": ", ".join(pushed) if pushed else "(none)",
            "frontier_size_after": len(pq),
        })

    return {
        "path": None,
        "expanded": expanded,
        "generated": generated,
        "edge_checks": edge_checks,
        "filter_rejections": filter_rejections,
        "duplicate_total": cycle_rejections + other_duplicates + stale_pops,
        "cycle_rejections": cycle_rejections,
        "other_duplicates": other_duplicates,
        "stale_pops": stale_pops,
        "re_expansions": re_expansions,
        "peak_frontier": peak_frontier,
        "steps": steps,
    }
