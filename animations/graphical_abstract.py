# /// script
# requires-python = ">=3.11"
# dependencies = ["imgui_ophys @ git+https://github.com/FlynnOConnell/imgui_ophys", "manim>=0.22", "numpy", "pillow<13", "tifffile", "matplotlib"]
# ///
"""
The Studio as one picture: the files it reads flow in from the left, the Studio in the middle plays a real movie
while its core features appear, the files it writes fan out to the right, and the pipelines it runs sit below, with
their results coming back in. Formats and features are mbo_utilities' own (`mbo formats`, docs/usage/gui_guide.md).

    uv run -p 3.12 animations/graphical_abstract.py --out <dir> [--stills-only] [--gif-width 720]

Text only, no LaTeX. Needs ffmpeg on PATH and ~/data/demoMovie.tif.
"""

from pathlib import Path

import manim as mn
import numpy as np
import tifffile
from matplotlib import colormaps

import kit
from kit import ACCENT, DIM, EDGE, KEY, TEXT, Play, at, chapter, color, label, put, timeline

FILEPATH_TIF = Path.home() / "data" / "demoMovie.tif"

TITLE = "Miller Brain Studio: view, convert and process imaging data"
CHAPTERS = (("", 5.0), ("", 7.0), ("", 4.5), ("", 5.5), ("", 4.0))
UNTITLED = {0: "reads", 1: "the-studio", 2: "writes", 3: "runs", 4: "all"}

INPUTS = (("OME-TIFF", "ScanImage, BigTIFF, ImageJ"), ("OME-Zarr", "zarr v3"), ("Suite2p", ".bin + ops.npy"),
          ("MaskNMF", "demixing results"), ("HDF5", ""), ("NumPy", "in memory or .npy"))
FEATURES = ("Scrub, project and bin frames", "Scan-phase correction", "Metadata: view, edit, save",
            "Z-stats: signal quality", "ROIs, traces and dF/F", "Notebooks beside the data")
OUTPUTS = (("TIFF", "BigTIFF, OME"), ("Zarr", "OME-NGFF, pyramids"), ("MP4", "movie"), ("Suite2p", ".bin"), ("HDF5", ""), ("NumPy", ".npy"))
PIPELINES = (("Suite2p", "cells and traces"), ("MaskNMF", "denoise, demix"), ("vnoiser", "voltage traces"), ("SLURM", "on the cluster"))

CARD = (0.17, 0.18, 0.21)
STUDIO = (0.13, 0.14, 0.16)
PIPE = (0.20, 0.30, 0.48)
X_IN, X_OUT = -5.5, 5.45
BOX_L, BOX_R, BOX_B, BOX_T = -3.0, 3.1, -2.0, 2.55
Y_PIPES = -3.25
FRAMES_PLAY = 150


def rgb_frames(n: int) -> list:
    movie = tifffile.imread(str(FILEPATH_TIF)).astype(np.float32)
    lo, hi = np.percentile(movie[::20], 0.5), np.percentile(movie[::20], 99.8)
    lut = (colormaps["gnuplot2"](np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
    picked = movie[np.linspace(0, movie.shape[0] - 1, n).astype(int)]
    index = np.clip((picked - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    return [lut[f] for f in index]


def card(name: str, detail: str, x: float, y: float, width: float, fill: tuple = CARD, c: tuple = TEXT) -> mn.VGroup:
    """A rounded box centered at (x, y) with the format's name and, under it, what it holds."""
    height = 0.62 if detail else 0.46
    rect = put(mn.RoundedRectangle(corner_radius=0.08, width=width, height=height, fill_color=color(fill), fill_opacity=1.0,
                                   stroke_color=color(EDGE), stroke_width=1.2), x, y)
    name_text = label(name, c, 18)
    if not detail:
        return mn.VGroup(rect, name_text.move_to(rect))
    detail_text = label(detail, DIM, 13)
    column = mn.VGroup(name_text, detail_text).arrange(mn.DOWN, buff=0.04).move_to(rect)
    return mn.VGroup(rect, column)


def arrow(start: np.ndarray, end: np.ndarray, c: tuple = DIM) -> mn.Arrow:
    return mn.Arrow(start, end, buff=0.08, stroke_width=2.2, max_tip_length_to_length_ratio=0.12, tip_length=0.16, color=color(c))


def word(text: str, x: float, y: float) -> mn.Text:
    return put(label(text, DIM, 15), x, y)


def build(seconds: list) -> kit.Timeline:
    frames = rgb_frames(FRAMES_PLAY)
    title = put(label(TITLE, TEXT, 28), 0.0, 3.55)

    y_in = np.linspace(2.2, -1.8, len(INPUTS))
    inputs = [card(name, detail, X_IN, y, 2.7) for (name, detail), y in zip(INPUTS, y_in)]
    arrows_in = [arrow(at(X_IN + 1.35, y), at(BOX_L, 0.9 + 0.3 * (y - y_in.mean()) / 2.0)) for y in y_in]
    reads = word("reads", (X_IN + 1.35 + BOX_L) / 2, y_in[0] + 0.55)

    studio = put(mn.RoundedRectangle(corner_radius=0.12, width=BOX_R - BOX_L, height=BOX_T - BOX_B, fill_color=color(STUDIO), fill_opacity=1.0,
                                     stroke_color=color(ACCENT), stroke_width=2.0), (BOX_L + BOX_R) / 2, (BOX_B + BOX_T) / 2)
    studio_name = label("Studio", ACCENT, 22).move_to(at(BOX_L + 0.3, BOX_T - 0.3), aligned_edge=mn.LEFT)
    command = label("mbo <path>", DIM, 15).move_to(at(BOX_R - 0.3, BOX_T - 0.3), aligned_edge=mn.RIGHT)
    panel = kit.image(frames[0], 1.9, BOX_L + 1.6, 1.0)
    track = put(mn.RoundedRectangle(corner_radius=0.05, width=2.5, height=0.14, fill_color=color(PIPE), fill_opacity=1.0, stroke_width=0), BOX_L + 1.6, -0.2)
    handle = put(mn.RoundedRectangle(corner_radius=0.05, width=0.3, height=0.14, fill_color=color((0.55, 0.70, 0.95)), fill_opacity=1.0, stroke_width=0), BOX_L + 0.5, -0.2)
    handle_end = handle.copy().move_to(at(BOX_L + 2.7, -0.2))
    features = [label(text, TEXT, 15).move_to(at(BOX_L + 3.0, 1.85 - 0.37 * k), aligned_edge=mn.LEFT) for k, text in enumerate(FEATURES)]
    save_run = label("Save as (s)      Run", TEXT, 15).move_to(at(BOX_L + 0.3, BOX_B + 0.75), aligned_edge=mn.LEFT)
    save_detail = label("a copy in any format below; the viewer keeps the original", DIM, 12).move_to(at(BOX_L + 0.3, BOX_B + 0.42), aligned_edge=mn.LEFT)

    y_out = np.linspace(2.2, -1.5, len(OUTPUTS))
    outputs = [card(name, detail, X_OUT, y, 2.4) for (name, detail), y in zip(OUTPUTS, y_out)]
    arrows_out = [arrow(at(BOX_R, 0.35 + 0.25 * (y - y_out.mean()) / 1.85), at(X_OUT - 1.2, y)) for y in y_out]
    writes = word("writes", (BOX_R + X_OUT - 1.2) / 2, y_out[0] + 0.55)

    x_pipes = np.linspace(-3.4, 3.4, len(PIPELINES))
    pipelines = [card(name, detail, x, Y_PIPES, 2.0, PIPE) for (name, detail), x in zip(PIPELINES, x_pipes)]
    arrow_run = arrow(at((BOX_L + BOX_R) / 2, BOX_B), at((BOX_L + BOX_R) / 2, Y_PIPES + 0.35))
    runs = word("runs", (BOX_L + BOX_R) / 2 + 0.45, (BOX_B + Y_PIPES + 0.35) / 2)
    back = mn.CurvedArrow(at(x_pipes[-1] + 1.05, Y_PIPES + 0.2), at(BOX_R + 0.1, BOX_B - 0.1), angle=-mn.TAU / 5, stroke_width=2.2, tip_length=0.16, color=color(KEY))
    back_name = label("results open in the Studio", KEY, 14).move_to(at(BOX_R + 0.45, (Y_PIPES + BOX_B) / 2 + 0.2), aligned_edge=mn.LEFT)

    chapters = [
        chapter(seconds[0],
                mn.AnimationGroup(mn.FadeIn(title), mn.FadeIn(studio), mn.FadeIn(studio_name), mn.FadeIn(command), mn.FadeIn(panel), lag_ratio=0.1, run_time=1.2),
                mn.AnimationGroup(mn.LaggedStart(*[mn.FadeIn(c, shift=0.3 * mn.RIGHT) for c in inputs], lag_ratio=0.15),
                                  mn.LaggedStart(*[mn.Create(a) for a in arrows_in], lag_ratio=0.15), mn.FadeIn(reads), lag_ratio=0.2, run_time=2.2),
                lag_ratio=1.0),
        chapter(seconds[1],
                mn.AnimationGroup(mn.FadeIn(track), mn.FadeIn(handle), run_time=0.4),
                mn.AnimationGroup(mn.Transform(handle, handle_end), Play(panel, frames[:FRAMES_PLAY // 2]),
                                  mn.LaggedStart(*[mn.FadeIn(f, shift=0.15 * mn.RIGHT) for f in features], lag_ratio=0.3),
                                  mn.AnimationGroup(mn.FadeIn(save_run), mn.FadeIn(save_detail), lag_ratio=0.5),
                                  run_time=4.5, rate_func=mn.linear),
                lag_ratio=1.0),
        chapter(seconds[2],
                mn.AnimationGroup(mn.LaggedStart(*[mn.Create(a) for a in arrows_out], lag_ratio=0.12),
                                  mn.LaggedStart(*[mn.FadeIn(c, shift=0.3 * mn.RIGHT) for c in outputs], lag_ratio=0.12), mn.FadeIn(writes),
                                  Play(panel, frames[FRAMES_PLAY // 2:FRAMES_PLAY * 3 // 4]), lag_ratio=0.15, run_time=2.5),
                lag_ratio=1.0),
        chapter(seconds[3],
                mn.AnimationGroup(mn.Create(arrow_run), mn.FadeIn(runs), run_time=0.6),
                mn.AnimationGroup(mn.LaggedStart(*[mn.FadeIn(c, shift=0.3 * mn.DOWN) for c in pipelines], lag_ratio=0.2),
                                  Play(panel, frames[FRAMES_PLAY * 3 // 4:]), run_time=1.8),
                mn.AnimationGroup(mn.Create(back), mn.FadeIn(back_name), lag_ratio=0.5, run_time=1.2),
                lag_ratio=1.0),
        chapter(seconds[4], Play(panel, frames, rate_func=mn.linear), active=1.0),
    ]
    return timeline([], chapters)


if __name__ == "__main__":
    kit.run("graphical_abstract", CHAPTERS, UNTITLED, build, __doc__)
