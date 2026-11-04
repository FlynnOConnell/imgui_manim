# /// script
# requires-python = ">=3.11"
# dependencies = ["manim>=0.22", "numpy", "pillow<13"]
# ///
"""
The Studio from the command line to suite2p's cells and traces, in its own windows: `mbo ~/data/demoMovie.tif` opens
the viewer, Set Metadata (shift+M) gets the frame rate, the Run tab runs Suite2p, the ROIs tab shows every ROI the run
found and the Traces tab plots three grouped cells against the playing movie; the files written sit beside the data.
Every window is the real GUI captured offscreen on demoMovie.tif (scripts/studio/capture_windows.py); the step
timings come from the run itself (scripts/studio/run_suite2p.py, exported by scripts/studio/export_suite2p.py).

    uv run -p 3.12 animations/suite2p_demo.py --out <dir> [--stills-only] [--width 1920] [--gif-width 960]

Text only, no LaTeX. Needs ffmpeg on PATH. The windows and the run are stored in animations/data/suite2p_demo.
"""

import json
from pathlib import Path

import manim as mn
import numpy as np
from PIL import Image

import kit
from kit import ACCENT, DIM, EDGE, HIGHLIGHT, KEY, TEXT, TRUTH, Count, Play, at, chapter, color, label, put, replace, timeline

DIR_DATA = Path(__file__).resolve().parent / "data" / "suite2p_demo"

TITLE = "The Studio: from a raw movie to cells and their traces"
CHAPTERS = (("", 6.0), ("Set the frame rate", 5.0), ("Run Suite2p", 7.5), ("Cells", 6.0), ("Traces", 9.0), ("Files beside the data", 7.0))
UNTITLED = {0: "open"}
COMMAND = "$ mbo ~/data/demoMovie.tif"
STEPS = (("registration", "rigid"), ("detection", "sparsery"), ("extraction", "F, Fneu"), ("deconvolution", "spks"), ("classification", "iscell"))
FILES = (("zplane01_tp00001-02000/", "the suite2p plane: F, Fneu, spks, stat, iscell, ops, data.bin"),
         ("demoMovie.rois.zarr", "drawn ROIs, quick traces, loaded runs; autosaved"),
         ("mbo results zplane01_tp00001-02000", "one results zarr: traces, membership, summary images"))
RUN_BUTTON_PX = (726, 727, 946, 746)  ## x0, y0, x1, y1 of "Run Suite2p" in run_tab.png

PX = 0.0078  ## scene units per window pixel
WX, WY = -3.1, -0.25
X_RIGHT, X_MID = 1.3, 4.1
TERMINAL = (0.06, 0.06, 0.07)


def window(name: str) -> mn.ImageMobject:
    """One captured window, drawn at PX per pixel and centered where every window sits."""
    rgb = np.asarray(Image.open(DIR_DATA / "windows" / f"{name}.png").convert("RGB"))
    mobject = mn.ImageMobject(rgb)
    mobject.set_resampling_algorithm(mn.RESAMPLING_ALGORITHMS["bicubic"])
    mobject.height = rgb.shape[0] * PX
    return put(mobject, WX, WY)


def frames_of(name: str) -> list:
    return list(np.load(DIR_DATA / f"{name}_frames.npz")["frames"])


def scene_rect(px_rect: tuple, shape: tuple, c: tuple = KEY) -> mn.Rectangle:
    """A frame around a pixel rectangle of a window with the given (height, width)."""
    x0, y0, x1, y1 = px_rect
    h, w = shape
    center = at(WX + ((x0 + x1) / 2 - w / 2) * PX, WY - ((y0 + y1) / 2 - h / 2) * PX)
    return mn.Rectangle(width=(x1 - x0 + 8) * PX, height=(y1 - y0 + 8) * PX, color=color(c), stroke_width=2.5).move_to(center)


def mono(text: str, c: tuple = TEXT, font_size: float = 16) -> mn.Text:
    mobject = mn.Text(text, font="Liberation Mono", font_size=font_size, color=color(c))
    for glyph in mobject.get_family():
        if hasattr(glyph, "path_obj"):
            glyph.path_obj = None
    mobject.id_to_vgroup_dict = {}
    return mobject


def typed(n: float) -> mn.Text:
    """The command's first n characters; the bare prompt stays invisible so nothing shows before the terminal opens."""
    return mono(COMMAND[: int(n)]).set_opacity(0.0 if n <= 2 else 1.0)


def left(mobject: mn.Mobject, x: float, y: float) -> mn.Mobject:
    return mobject.move_to(at(x, y), aligned_edge=mn.LEFT)


def build(seconds: list) -> kit.Timeline:
    data = np.load(DIR_DATA / "suite2p_demo.npz", allow_pickle=True)
    timings = json.loads(str(data["timings"]))
    n_rois, n_cells = len(data["iscell"]), int(data["iscell"].sum())
    total = sum(timings[name] for name, _ in STEPS)
    print(f"cells: {n_cells} of {n_rois}; steps " + ", ".join(f"{n} {timings[n]:.2f} s" for n, _ in STEPS) + f"; total {total:.1f} s")

    title = put(label(TITLE, TEXT, 28), 0.0, 3.6)
    titles = [put(label(name, DIM, 24), X_MID, 3.1) if name else None for name, _ in CHAPTERS]

    terminal = put(mn.RoundedRectangle(corner_radius=0.08, width=5.0, height=0.5, fill_color=color(TERMINAL), fill_opacity=1.0, stroke_color=color(EDGE), stroke_width=1.0), -4.4, 2.95)
    command = left(mono(COMMAND), -6.7, 2.95)

    win_viewer, win_meta_missing, win_meta, win_run, win_cells, win_traces = [window(n) for n in ("viewer", "set_metadata_missing", "set_metadata", "run_tab", "cells_masks", "cells_traces")]
    frames_viewer, frames_traces = frames_of("viewer"), frames_of("cells_traces")
    key_meta = put(label("shift+M", KEY, 22), X_MID, 1.2)

    run_frame = scene_rect(RUN_BUTTON_PX, (900, 1000))
    step_names = [left(label(f"{name}   {how}", DIM, 16), X_RIGHT, 1.7 - 0.38 * k) for k, (name, how) in enumerate(STEPS)]
    step_times = [left(label(f"{timings[name]:.1f} s" if timings[name] >= 0.1 else f"{timings[name]:.2f} s", TEXT, 16), X_RIGHT + 3.6, 1.7 - 0.38 * k) for k, (name, _) in enumerate(STEPS)]
    dot = mn.Dot(at(X_RIGHT + 0.1, -0.6), radius=0.08, color=color(HIGHLIGHT))
    dot_done = mn.Dot(at(X_RIGHT + 0.1, -0.6), radius=0.08, color=color(TRUTH))
    elapsed = left(label("0.0 s", TEXT, 17), X_RIGHT + 0.35, -0.6)
    bar_full = put(mn.Rectangle(width=5.4, height=0.07, fill_color=color(ACCENT), fill_opacity=1.0, stroke_width=0), X_RIGHT + 2.7, -0.95)
    bar = bar_full.copy().stretch_to_fit_width(0.01).move_to(bar_full, aligned_edge=mn.LEFT)

    count = left(label("0 ROIs", KEY, 36), X_RIGHT, 1.4)
    count_notes = mn.VGroup(left(label(f"{n_cells} accepted by suite2p's classifier", DIM, 16), X_RIGHT, 0.85),
                            left(label("every ROI loads; curation happens in the ROIs tab", DIM, 16), X_RIGHT, 0.5))
    cursor_note = mn.VGroup(left(label("Click a cell to plot it; ctrl+click groups several", DIM, 16), X_RIGHT, 1.4),
                            left(label("The trace plot's cursor, the T slider and the image", DIM, 16), X_RIGHT, 0.95),
                            left(label("land on the same instant", DIM, 16), X_RIGHT, 0.65))
    files = []
    for k, (name, detail) in enumerate(FILES):
        files.append(mn.VGroup(left(mono(name, TEXT, 16), X_RIGHT, 1.6 - 1.25 * k), left(label(detail, DIM, 14), X_RIGHT, 1.28 - 1.25 * k)))

    run_time_steps = 3.8
    step_reveals, done = [], 0.0
    for (name, _), name_label, time_label in zip(STEPS, step_names, step_times):
        done += timings[name]
        step_reveals.append(mn.AnimationGroup(mn.Wait(run_time=max(run_time_steps * done / total, 0.05)),
                                              mn.AnimationGroup(mn.FadeIn(time_label), name_label.animate.set_color(color(TEXT)), run_time=0.3), lag_ratio=1.0))

    chapters = [
        chapter(seconds[0],
                mn.AnimationGroup(mn.FadeIn(title), mn.FadeIn(terminal), lag_ratio=0.3, run_time=0.6),
                Count(command, 2, len(COMMAND), typed, decimals=0, run_time=1.4, rate_func=mn.linear),
                mn.FadeIn(win_viewer, run_time=0.6),
                Play(win_viewer, frames_viewer, run_time=2.2, rate_func=mn.linear),
                lag_ratio=1.0),
        chapter(seconds[1],
                mn.AnimationGroup(mn.FadeOut(title), *replace(titles[0], titles[1]), mn.FadeIn(win_meta_missing), mn.FadeIn(key_meta), lag_ratio=0.2, run_time=1.2),
                mn.Wait(run_time=1.2),
                mn.FadeIn(win_meta, run_time=0.6),
                lag_ratio=1.0),
        chapter(seconds[2],
                mn.AnimationGroup(*replace(titles[1], titles[2]), mn.FadeOut(key_meta), mn.FadeOut(terminal), mn.FadeOut(command),
                                  mn.FadeOut(win_meta_missing), mn.FadeOut(win_meta), mn.FadeIn(win_run),
                                  mn.LaggedStart(*[mn.FadeIn(s) for s in step_names], lag_ratio=0.2), lag_ratio=0.05, run_time=1.2),
                mn.AnimationGroup(mn.Create(run_frame), mn.FadeIn(dot), mn.FadeIn(elapsed), mn.FadeIn(bar), run_time=0.4),
                mn.AnimationGroup(Count(elapsed, 0.0, total, lambda v: label(f"{v:.1f} s", TEXT, 17), run_time=run_time_steps, rate_func=mn.linear),
                                  mn.Transform(bar, bar_full, run_time=run_time_steps, rate_func=mn.linear), *step_reveals),
                mn.Transform(dot, dot_done, run_time=0.3),
                lag_ratio=1.0),
        chapter(seconds[3],
                mn.AnimationGroup(*replace(titles[2], titles[3]), *[mn.FadeOut(m) for m in (run_frame, dot, elapsed, bar, *step_names, *step_times)],
                                  mn.FadeOut(win_run), mn.FadeIn(win_cells), lag_ratio=0.02, run_time=1.0),
                mn.AnimationGroup(mn.AnimationGroup(mn.FadeIn(count, run_time=0.3), Count(count, 0, n_rois, lambda v: label(f"{int(v)} ROIs", KEY, 36), decimals=0, run_time=2.0, rate_func=mn.linear), lag_ratio=1.0),
                                  mn.LaggedStart(*[mn.FadeIn(n) for n in count_notes], lag_ratio=0.5, run_time=1.5), lag_ratio=0.0),  ## the FadeIn ends before the glyph count changes

                lag_ratio=1.0),
        chapter(seconds[4],
                mn.AnimationGroup(*replace(titles[3], titles[4]), mn.FadeOut(count), mn.FadeOut(count_notes), mn.FadeOut(win_cells), mn.FadeIn(win_traces), run_time=1.0),
                mn.AnimationGroup(Play(win_traces, frames_traces, run_time=5.5, rate_func=mn.linear),
                                  mn.LaggedStart(*[mn.FadeIn(n) for n in cursor_note], lag_ratio=0.4, run_time=1.5), lag_ratio=0.0),
                lag_ratio=1.0),
        chapter(seconds[5],
                mn.AnimationGroup(*replace(titles[4], titles[5]), mn.FadeOut(cursor_note), run_time=0.8),
                mn.LaggedStart(*[mn.FadeIn(row, shift=0.2 * mn.RIGHT) for row in files], lag_ratio=0.4, run_time=2.4),
                lag_ratio=1.0),
    ]
    return timeline([], chapters)


if __name__ == "__main__":
    kit.run("suite2p_demo", CHAPTERS, UNTITLED, build, __doc__)
