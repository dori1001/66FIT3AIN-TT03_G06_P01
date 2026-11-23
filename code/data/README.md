# HANU Campus Graph Dataset (`data/`)

Campus wayfinding graph dataset for **Hanoi University (HANU)** — Project 01 (Smart Campus Route Planner).

## Files

- **`nodes.csv`**: 54 campus nodes representing gates, building entrances, classrooms, laboratories, library, stadium, canteen, parking lots, and walkway intersections.
  - Schema: `node_id, name, x, y, floor, building, node_type`
- **`edges.csv`**: 75 undirected edges (150 directed arcs) connecting campus nodes, including parallel multi-edges (stairs vs. elevator) with distance, travel time, crowd level, stairs, indoor, and accessibility attributes.
  - Schema: `source, target, distance, travel_time, crowd_level, stairs, indoor, accessible`
- **`scenarios.json`**: Five standard evaluation scenarios (`S1_normal_route`, `S2_blocked_edge`, `S3_accessibility_mode`, `S4_heuristic_comparison`, `S5_accessibility_no_path`).
- **`campus_map.png`**: Visual map of HANU buildings, nodes, and campus corridors.
- **`SHA256SUMS`**: SHA-256 checksums verifying the integrity of `nodes.csv`, `edges.csv`, and `scenarios.json`.
