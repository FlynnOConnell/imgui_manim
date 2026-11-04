"""
Capture the Studio's real windows on demoMovie.tif through an offscreen canvas (as scripts/capture_docs.py does):
the viewer, the Set Metadata popup, the Run tab, and the ROI view with the suite2p run loaded and three cells grouped.
Writes PNGs and canvas_rects.json (where the image sits in each capture, in pixels) to animations/data/suite2p_demo/windows/.

    python capture_windows.py [plane_dir] [dir_out]
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from mbo_utilities.gui.run_gui import _create_image_widget
from mbo_utilities.reader import imread

filepath_movie = Path.home() / "data" / "demoMovie.tif"
dir_plane = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "data" / "demoMovie_studio_demo" / "zplane01_tp00001-02000"
dir_out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parents[2] / "animations" / "data" / "suite2p_demo" / "windows"
dir_out.mkdir(parents=True, exist_ok=True)
dir_frames = dir_out.parent
dir_frames.mkdir(parents=True, exist_ok=True)
SHOWN_STAT = (42, 30, 5)
rects = {}


def build(widget: str, size: tuple, fs: float | None = 10.0):
    array = imread(filepath_movie)
    if fs is not None:
        array.metadata["fs"] = fs
    iw = _create_image_widget(array, widget=widget, figure_kwargs_override={"canvas": "offscreen", "size": size})
    gui = next(g for g in iw.figure.imgui_windows.values() if g is not None)
    return iw, gui


def draw(iw, n: int) -> np.ndarray:
    frame = None
    for _ in range(n):
        frame = iw.figure.canvas.draw()
    return np.asarray(frame)[..., :3].copy()


def sequence(iw, name: str, n_frames: int, n_timepoints: int = 2000) -> None:
    """The window at n_frames time indices across the movie, as one uint8 (n, H, W, 3) npz in artifacts/data."""
    frames = []
    for t in np.linspace(0, n_timepoints - 1, n_frames).astype(int):
        iw.current_index = {"t": int(t)}
        frames.append(draw(iw, 2))
    iw.current_index = {"t": 0}
    np.savez_compressed(dir_frames / f"{name}_frames.npz", frames=np.stack(frames))
    print(name, "frames", len(frames), (dir_frames / f"{name}_frames.npz").stat().st_size // 1_000_000, "MB")


def save(iw, name: str, n: int = 8) -> None:
    Image.fromarray(draw(iw, n)).save(dir_out / f"{name}.png")
    subplot = iw.figure[0, 0]
    x0, y0 = subplot.map_world_to_screen((-0.5, -0.5, 0.0))
    x1, y1 = subplot.map_world_to_screen((79.5, 59.5, 0.0))
    rects[name] = {"x0": float(x0), "y0": float(y0), "x1": float(x1), "y1": float(y1), "width": int(iw.figure.canvas.get_logical_size()[0]), "height": int(iw.figure.canvas.get_logical_size()[1])}
    print(name, rects[name])


iw, gui = build("preview", (1000, 660), fs=None)
draw(iw, 5)
save(iw, "viewer")
sequence(iw, "viewer", 40)
gui._show_metadata_popup = True
save(iw, "set_metadata_missing")
iw.figure.canvas.close()

iw, gui = build("preview", (1000, 660))
draw(iw, 5)
gui._show_metadata_popup = True
save(iw, "set_metadata")
iw.figure.canvas.close()

iw, gui = build("preview", (1000, 900))
draw(iw, 5)
gui._force_run_tab = True
save(iw, "run_tab")
iw.figure.canvas.close()

iw, gui = build("manualroi", (1000, 760))
draw(iw, 5)
roi = gui.manual_roi
print("loaded run:", roi.load_run(dir_plane), "derived sets:", [(s.name, len(s)) for s in roi.derived])
for si in [i for i, s in enumerate(roi.derived) if "studio_demo" not in s.name][::-1]:
    roi.unload_set(si)  ## another run beside the data (the other instance's) is adopted too; the demo shows this topic's
roi.masks_by_peak = False
roi.refresh_overlay()
roi.focus_tab = True
draw(iw, 5)
save(iw, "cells_masks")
roi.masks_by_peak = True
roi.set_contours(True)
draw(iw, 5)
save(iw, "cells_contours")
roi.set_contours(False)
si = next(i for i, s in enumerate(roi.derived) if "studio_demo" in s.name)
roi.select_derived(si, SHOWN_STAT[0])
for k in SHOWN_STAT[1:]:
    roi.buffer_toggle(si, k)
roi.focus_tab = True
save(iw, "cells_traces", 12)
sequence(iw, "cells_traces", 100)
iw.figure.canvas.close()
(dir_out / "canvas_rects.json").write_text(json.dumps(rects, indent=2))
