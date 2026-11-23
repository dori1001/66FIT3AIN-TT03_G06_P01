# Smart Campus Route Planner — Hanoi University (HANU)

**Project 01 (P01)** — Artificial Intelligence Final Product (`66FIT3AIN-TT03_G06_P01`).

A fully offline AI campus wayfinding application implementing Breadth-First Search (BFS), Greedy Best-First Search (GBFS), and Optimal A* Search with cached admissible/consistent heuristic scaling from scratch in Python.

## 1. Environment & Installation

- **Python version:** Python 3.11+ (tested on Python 3.11 and 3.12)
- **Offline execution:** Runs 100% locally with zero external API or LLM calls.

```bash
python3.11 -m venv .venv
# Linux / macOS:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## 2. One-Command Application Launch

```bash
streamlit run starter/app.py
```

The Streamlit web application provides:
- **1-Click Standard Scenarios (S1–S5)** from `data/scenarios.json` and **1-Click Stress Scenarios (S6–S8)** (multi-corridor outage, combined maintenance + wheelchair accessibility, complete destination isolation).
- **Side-by-Side Dual-Map Comparison** allowing independent selection of Route A vs. Route B with real-time delta metrics.
- **Dynamic Maintenance Simulator** (`blocked_edges` multiselect) and **Accessible-Only Mode** (`accessible_only`).
- **Turn-by-Turn Directions** and **Live Execution Log Breakdown** (`Nodes Expanded`, `Generated Nodes`, `Edge Checks`, `Duplicates`, `Cycle Rejections`, `Re-expansions`).

## 3. Reproducing All Report Tables and Figures

Run the deterministic benchmark and figure generation script:

```bash
python scripts/run_experiments.py --seed 42
```

This command reproduces:
- `scripts/output/heuristic_audit.csv` — Exhaustive Admissibility (2,916 node pairs) and Consistency (8,100 / 7,992 / 7,668 / 7,560 directed edge-goal checks across 4 constraint regimes).
- `scripts/output/final_benchmark_results.csv` — Baseline (BFS, GBFS), Midterm Uncached A*, and Final Cached A* metrics across S1–S8 (`Mean ± Std` over $N=200$ runs, plus execution log counters).
- `scripts/output/fig1_architecture.png` through `fig6_route_overlays.png` — All figures embedded in the Final Report.

## 4. Automated Test Suite

```bash
pytest tests/ -v
```

Includes `tests/test_project_01_sanity.py` (12 data integrity, SHA-256, schema, and functional tests), `tests/test_stress_scenarios.py` (5 stress, boundary, caching, and 2,916 all-pairs Dijkstra ground-truth optimality tests), and `tests/test_app_smoke.py` (2 end-to-end headless Streamlit UI smoke tests covering all presets S1–S8, dual-map rendering, and start==goal edge cases — **19 tests total**).

## 5. Repository Structure

```text
code/
├── README.md
├── requirements.txt
├── data/               # 54 nodes, 75 undirected edges (150 arcs), scenarios.json, campus_map.png, SHA256SUMS
├── starter/            # graph.py, student_core.py (AI Core), app.py (Final Streamlit Product)
├── scripts/            # run_experiments.py, profile_search_logs.py, output/
└── tests/              # test_project_01_sanity.py, test_stress_scenarios.py, test_app_smoke.py
```

## 6. GitHub Repository

- GitHub URL: Not yet created (to be updated once the group publishes its GitHub repository).
