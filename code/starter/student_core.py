"""AI core implementation for Project 01 (Smart Campus Route Planner — HANU).

Algorithms
----------
- bfs    - fewest hops (ignores weights; NOT shortest distance/time)
- greedy - greedy best-first search (fast, not optimal)
- astar  - optimal for the chosen edge weight, using an admissible & consistent
           scaled heuristic with O(1) cached cost-scale lookup in the Final version

All searches accept optional ``blocked_edges`` / ``accessible_only`` filters.
Every search returns ``(path_or_None, expanded_count)``; use ``path_cost`` to
get the total distance/time of a returned path.
"""

from __future__ import annotations

import heapq
import math
import sys
import weakref
from collections import deque
from pathlib import Path
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

try:
    from .graph import Graph, Edge
except ImportError:  # running as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from graph import Graph, Edge  # noqa: E402

PathResult = Tuple[Optional[List[str]], int]
Heuristic = Callable[[Graph, str, str], float]
EdgeFilter = Callable[[Edge], bool]

_VALID_WEIGHTS = ("distance", "travel_time")

# expanded_count reported when start == goal (no search loop runs).
START_EQUALS_GOAL_EXPANDED = 0

# Weak-key cache mapping Graph -> {(edge_weight, accessible_only, blocked_key): alpha_w}
# Eliminates the O(|E|) pre-scaling scan on repeated queries without leaking memory.
_COST_SCALE_CACHE: "weakref.WeakKeyDictionary[Graph, Dict[Tuple[str, bool, FrozenSet[FrozenSet[str]]], float]]" = (
    weakref.WeakKeyDictionary()
)


def clear_cost_scale_cache() -> None:
    """Clear all cached heuristic cost-scale multipliers (useful for benchmarking)."""
    _COST_SCALE_CACHE.clear()


# --------------------------------------------------------------------------- #
# Edge filtering
# --------------------------------------------------------------------------- #
def _normalize_blocked(
    blocked_edges: Optional[Iterable[Iterable[str]]] = None,
) -> FrozenSet[FrozenSet[str]]:
    if not blocked_edges:
        return frozenset()
    return frozenset(frozenset(pair) for pair in blocked_edges)


def make_edge_filter(
    blocked_edges: Optional[Iterable[Iterable[str]]] = None,
    accessible_only: bool = False,
) -> EdgeFilter:
    """Build a fast predicate ``edge -> allowed``.

    Blocked edges are treated as undirected. The blocked list is converted to a
    set once, so each check is O(1) instead of O(len(blocked_edges)).
    """
    blocked: Set[frozenset] = {frozenset(pair) for pair in (blocked_edges or [])}

    def allowed(edge: Edge) -> bool:
        if accessible_only and not edge.accessible:
            return False
        if blocked and frozenset((edge.source, edge.target)) in blocked:
            return False
        return True

    return allowed


def _build_path(parent: Dict[str, Optional[str]], goal: str) -> List[str]:
    path: List[str] = []
    current: Optional[str] = goal
    while current is not None:
        path.append(current)
        current = parent.get(current)
    return path[::-1]


def _valid_endpoints(graph: Graph, start: str, goal: str) -> bool:
    return start in graph.node_ids and goal in graph.node_ids


# --------------------------------------------------------------------------- #
# Heuristics
# --------------------------------------------------------------------------- #
def euclidean_heuristic(graph: Graph, node_id: str, goal_id: str) -> float:
    node, goal = graph.get_node(node_id), graph.get_node(goal_id)
    return math.hypot(node.x - goal.x, node.y - goal.y)


def manhattan_heuristic(graph: Graph, node_id: str, goal_id: str) -> float:
    node, goal = graph.get_node(node_id), graph.get_node(goal_id)
    return abs(node.x - goal.x) + abs(node.y - goal.y)


# Manhattan distance can be up to sqrt(2) x the straight-line distance, so it is
# divided by sqrt(2) before use in A* to keep it admissible on non-grid graphs.
_ADMISSIBILITY_FACTOR: Dict[Heuristic, float] = {
    euclidean_heuristic: 1.0,
    manhattan_heuristic: 1.0 / math.sqrt(2.0),
}


def _cost_scale(graph: Graph, edge_weight: str, edge_ok: EdgeFilter) -> float:
    """Smallest (edge cost / straight-line length) over all usable edges.

    Multiplying the Euclidean distance by this ratio gives a lower bound on the
    remaining cost for *any* weight type and *any* coordinate units, which makes
    the heuristic admissible (and consistent). Returns 0.0 if no usable edges.
    """
    best = math.inf
    for node_id in graph.node_ids:
        a = graph.get_node(node_id)
        for edge in graph.neighbors(node_id):
            if not edge_ok(edge):
                continue
            b = graph.get_node(edge.target)
            length = math.hypot(a.x - b.x, a.y - b.y)
            if length > 0:
                best = min(best, getattr(edge, edge_weight) / length)
    return 0.0 if best == math.inf else best


def _get_cached_cost_scale(
    graph: Graph,
    edge_weight: str,
    edge_ok: EdgeFilter,
    blocked_edges: Optional[Iterable[Iterable[str]]] = None,
    accessible_only: bool = False,
) -> float:
    """Return cached alpha_w for (graph, edge_weight, accessible_only, blocked_edges)."""
    key = (edge_weight, bool(accessible_only), _normalize_blocked(blocked_edges))
    graph_cache = _COST_SCALE_CACHE.get(graph)
    if graph_cache is None:
        graph_cache = {}
        _COST_SCALE_CACHE[graph] = graph_cache
    if key not in graph_cache:
        graph_cache[key] = _cost_scale(graph, edge_weight, edge_ok)
    return graph_cache[key]


# --------------------------------------------------------------------------- #
# Searches
# --------------------------------------------------------------------------- #
def bfs(
    graph: Graph,
    start: str,
    goal: str,
    blocked_edges: Optional[List[List[str]]] = None,
    accessible_only: bool = False,
) -> PathResult:
    """Breadth-first search: returns the path with the FEWEST HOPS."""
    if not _valid_endpoints(graph, start, goal):
        return None, 0
    if start == goal:
        return [start], START_EQUALS_GOAL_EXPANDED

    edge_ok = make_edge_filter(blocked_edges, accessible_only)
    queue = deque([start])
    parent: Dict[str, Optional[str]] = {start: None}
    expanded = 0

    while queue:
        current = queue.popleft()
        expanded += 1
        if current == goal:
            return _build_path(parent, goal), expanded
        for edge in graph.neighbors(current):
            nbr = edge.target
            if nbr not in parent and edge_ok(edge):
                parent[nbr] = current
                queue.append(nbr)
    return None, expanded


def greedy(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Heuristic,
    blocked_edges: Optional[List[List[str]]] = None,
    accessible_only: bool = False,
) -> PathResult:
    """Greedy best-first search: follows the heuristic only; not optimal."""
    if not _valid_endpoints(graph, start, goal):
        return None, 0
    if start == goal:
        return [start], START_EQUALS_GOAL_EXPANDED

    edge_ok = make_edge_filter(blocked_edges, accessible_only)
    counter = 0
    pq: List[Tuple[float, int, str, Optional[str]]] = [
        (heuristic(graph, start, goal), counter, start, None)
    ]
    parent: Dict[str, Optional[str]] = {}
    expanded = 0

    while pq:
        _, _, current, par = heapq.heappop(pq)
        if current in parent:
            continue
        parent[current] = par
        expanded += 1
        if current == goal:
            return _build_path(parent, goal), expanded
        for edge in graph.neighbors(current):
            nbr = edge.target
            if nbr not in parent and edge_ok(edge):
                counter += 1
                heapq.heappush(pq, (heuristic(graph, nbr, goal), counter, nbr, current))
    return None, expanded


def astar(
    graph: Graph,
    start: str,
    goal: str,
    heuristic: Heuristic,
    edge_weight: str = "distance",
    blocked_edges: Optional[List[List[str]]] = None,
    accessible_only: bool = False,
    scale_heuristic: bool = True,
    use_scale_cache: bool = True,
) -> PathResult:
    """A* search, optimal for ``edge_weight`` ("distance" or "travel_time").

    With ``scale_heuristic=True`` (default) the heuristic is scaled so it never
    overestimates the true remaining cost, whatever the weight type or units.
    With ``use_scale_cache=True`` (Final version), the O(|E|) cost-scale ratio
    is cached per (graph, edge_weight, accessible_only, blocked_edges).
    Pass ``use_scale_cache=False`` to reproduce the uncached Midterm runtime.
    """
    if edge_weight not in _VALID_WEIGHTS:
        raise ValueError("edge_weight must be 'distance' or 'travel_time'")
    if not _valid_endpoints(graph, start, goal):
        return None, 0
    if start == goal:
        return [start], START_EQUALS_GOAL_EXPANDED

    edge_ok = make_edge_filter(blocked_edges, accessible_only)

    if scale_heuristic:
        factor = _ADMISSIBILITY_FACTOR.get(heuristic, 1.0)
        if use_scale_cache:
            base_scale = _get_cached_cost_scale(
                graph, edge_weight, edge_ok, blocked_edges, accessible_only
            )
        else:
            base_scale = _cost_scale(graph, edge_weight, edge_ok)
        scale = base_scale * factor
        h = lambda n: scale * heuristic(graph, n, goal)  # noqa: E731
    else:
        h = lambda n: heuristic(graph, n, goal)  # noqa: E731

    g_score: Dict[str, float] = {start: 0.0}
    parent: Dict[str, Optional[str]] = {start: None}
    counter = 0
    pq: List[Tuple[float, int, float, str]] = [(h(start), counter, 0.0, start)]
    expanded = 0

    while pq:
        _, _, g, current = heapq.heappop(pq)
        if g > g_score.get(current, math.inf):
            continue  # stale entry: lazy deletion
        expanded += 1

        if current == goal:
            return _build_path(parent, goal), expanded

        for edge in graph.neighbors(current):
            if not edge_ok(edge):
                continue
            nbr = edge.target
            tentative = g + getattr(edge, edge_weight)
            if tentative < g_score.get(nbr, math.inf):
                g_score[nbr] = tentative
                parent[nbr] = current
                counter += 1
                heapq.heappush(pq, (tentative + h(nbr), counter, tentative, nbr))
    return None, expanded


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def path_cost(
    graph: Graph,
    path: Optional[List[str]],
    edge_weight: str = "distance",
    blocked_edges: Optional[List[List[str]]] = None,
    accessible_only: bool = False,
) -> float:
    """Total cost of ``path`` under ``edge_weight``.

    Uses the cheapest allowed edge between each consecutive pair. Returns
    ``math.inf`` if ``path`` is None or a step has no allowed edge.
    """
    if edge_weight not in _VALID_WEIGHTS:
        raise ValueError("edge_weight must be 'distance' or 'travel_time'")
    if path is None:
        return math.inf
    edge_ok = make_edge_filter(blocked_edges, accessible_only)
    total = 0.0
    for a, b in zip(path, path[1:]):
        costs = [
            getattr(e, edge_weight)
            for e in graph.neighbors(a)
            if e.target == b and edge_ok(e)
        ]
        if not costs:
            return math.inf
        total += min(costs)
    return total


def describe_route_steps(
    graph: Graph,
    path: Optional[List[str]],
    edge_weight: str = "distance",
    blocked_edges: Optional[List[List[str]]] = None,
    accessible_only: bool = False,
) -> List[Dict[str, object]]:
    """Return turn-by-turn segment metadata for a valid route."""
    if not path or len(path) < 2:
        return []
    edge_ok = make_edge_filter(blocked_edges, accessible_only)
    steps: List[Dict[str, object]] = []
    for idx, (a, b) in enumerate(zip(path, path[1:]), start=1):
        candidates = [e for e in graph.neighbors(a) if e.target == b and edge_ok(e)]
        if not candidates:
            break
        best_edge = min(candidates, key=lambda e: getattr(e, edge_weight))
        na, nb = graph.get_node(a), graph.get_node(b)
        mode = (
            "Stairs"
            if best_edge.stairs
            else (
                "Elevator / Ramp"
                if na.floor != nb.floor
                else ("Indoor" if best_edge.indoor else "Outdoor Walkway")
            )
        )
        steps.append(
            {
                "Step": idx,
                "From": f"{a} ({na.name})",
                "To": f"{b} ({nb.name})",
                "Floor": f"F{na.floor} -> F{nb.floor}",
                "Mode": mode,
                "Crowd": best_edge.crowd_level,
                "Distance (m)": round(best_edge.distance, 1),
                "Time (s)": round(best_edge.travel_time, 2),
                "Accessible": "Yes" if best_edge.accessible else "No",
            }
        )
    return steps
