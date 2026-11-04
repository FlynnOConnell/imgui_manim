import importlib.util

## the window's tests need the `window` extra; the models' tests need numpy alone
collect_ignore = [] if all(importlib.util.find_spec(name) for name in ("masknmf", "manim", "imgui_data_loader")) else ["window"]
