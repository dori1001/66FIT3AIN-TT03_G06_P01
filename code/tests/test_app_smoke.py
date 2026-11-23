"""End-to-end headless smoke tests for starter/app.py across all presets S1-S8 and edge cases."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_DIR / "starter" / "app.py"


class _DummyCtx:
    def __init__(self, st_mod):
        self._st = st_mod

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getattr__(self, name):
        return getattr(self._st, name)


def _make_mock_st(preset_index: int = 1, start_override=None, end_override=None, alg_a_idx: int = 0, alg_b_idx: int = 0):
    st = types.ModuleType("streamlit")
    st._figs = []
    st._dataframes = []
    st._errors = []
    st._infos = []
    st._keys = set()

    def _record_key(key):
        if key is not None:
            assert key not in st._keys, f"Duplicate Streamlit widget key: {key}"
            st._keys.add(key)

    st.set_page_config = lambda **kw: None
    st.title = lambda t: None
    st.markdown = lambda t: None
    st.header = lambda t: None
    st.subheader = lambda t: None
    st.caption = lambda t: None
    st.info = lambda t: st._infos.append(t)
    st.success = lambda t: None
    st.warning = lambda t: None
    st.error = lambda t: st._errors.append(t)
    st.metric = lambda label, value, delta=None, delta_color="normal": None

    def selectbox(label, options, index=0, format_func=None, key=None):
        _record_key(key)
        if format_func:
            for opt in options:
                _ = format_func(opt)
        if "Select preset" in label:
            return options[preset_index]
        if label == "Start landmark" and start_override is not None:
            return start_override
        if label == "Destination landmark" and end_override is not None:
            return end_override
        if key == "alg_a":
            return options[alg_a_idx]
        if key == "alg_b":
            return options[alg_b_idx]
        return options[index]

    def checkbox(label, value=False, key=None):
        _record_key(key)
        return value

    def multiselect(label, options, default=None, key=None):
        _record_key(key)
        return list(default or [])

    st.selectbox = selectbox
    st.checkbox = checkbox
    st.multiselect = multiselect
    st.columns = lambda n: [_DummyCtx(st) for _ in range(n)]
    st.expander = lambda label, expanded=False: _DummyCtx(st)
    st.pyplot = lambda fig, use_container_width=True: st._figs.append(fig)
    st.dataframe = lambda df, use_container_width=True, hide_index=True: st._dataframes.append(df)
    st.sidebar = _DummyCtx(st)
    return st


def _run_app_headless(preset_index: int = 1, start_override=None, end_override=None, alg_a_idx: int = 0, alg_b_idx: int = 0):
    orig_st = sys.modules.get("streamlit")
    mock_st = _make_mock_st(preset_index, start_override, end_override, alg_a_idx, alg_b_idx)
    sys.modules["streamlit"] = mock_st
    try:
        spec = importlib.util.spec_from_file_location(f"hanu_app_smoke_{preset_index}", APP_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod, mock_st
    finally:
        if orig_st is not None:
            sys.modules["streamlit"] = orig_st
        else:
            sys.modules.pop("streamlit", None)


def test_app_all_presets_render_dual_maps_and_tables():
    # Presets 0..8 = Custom, S1..S5, S6..S8
    for idx in range(9):
        mod, mock_st = _run_app_headless(preset_index=idx)
        # Both Route A and Route B maps must render cleanly on campus_map.png
        assert len(mock_st._figs) == 2, f"Preset {idx} failed to render 2 side-by-side maps"
        # No-path scenarios S5 (idx=5) and S8 (idx=8) must report 2 non-crashing st.error alerts
        if idx in (5, 8):
            assert len(mock_st._errors) == 2
            assert mod.res_a["path"] is None and mod.res_a["expanded"] == 53
            assert mod.res_b["path"] is None and mod.res_b["expanded"] == 53
            assert len(mock_st._dataframes) == 1  # 7-config live benchmark table
        else:
            assert len(mock_st._errors) == 0
            assert mod.res_a["path"] is not None
            assert mod.res_b["path"] is not None
            assert len(mock_st._dataframes) == 2  # Turn-by-turn + 7-config benchmark table
            assert mock_st._dataframes[-1].shape == (7, 11)
        # Ensure trace_log expanded count strictly matches student_core expanded count
        assert mod.res_a["expanded"] == mod.res_a["log"]["expanded"]
        assert mod.res_b["expanded"] == mod.res_b["log"]["expanded"]


def test_app_start_equals_goal_edge_case():
    mod, mock_st = _run_app_headless(preset_index=0, start_override="N_GATE", end_override="N_GATE")
    assert len(mock_st._figs) == 2
    assert len(mock_st._errors) == 0
    assert mod.res_a["path"] == ["N_GATE"] and mod.res_a["expanded"] == 0
    assert mod.res_a["log"]["expanded"] == 0
    assert any("same landmark" in msg for msg in mock_st._infos)
