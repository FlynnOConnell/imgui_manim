# /// script
# requires-python = ">=3.11"
# dependencies = ["manim>=0.22", "numpy", "pillow<13"]
# ///
"""
masknmf's compression and its temporal denoiser as one animation: a movie cut into overlapping blocks, one block's SVD,
the roughness test against noise, the rebuilt movie, the blind-spot network, the blend, the second pass, and what a short
movie does to the training. Every number and trace comes from masknmf itself, through animations/compression_run.py and
its npz: the lesson's run.npz on the simulated movie, or --run for a recording, which has no true traces to compare with.
Drawn with kit.py.

    uv run -p 3.12 animations/compression.py --out <dir> [--run <npz>] [--stills-only] [--gif-width 720]
"""

import argparse
import functools
import sys
from dataclasses import dataclass
from pathlib import Path

import manim as mn
import numpy as np

import kit
from kit import (ACCENT, BAR, DIM, EDGE, HIGHLIGHT, KEY, TEXT, TRACE, TRUTH, Count, Play, at, axes_points, caption, chapter, color, diverging_rgb, grid, image, label, path,
                 plain_axes, put, replace, scale_bars, timeline)

M = kit.lesson_model("compression")

TITLE = "masknmf compression: a few smooth components per block"
CHAPTERS = (
    ("", 5.0),
    ("Overlapping blocks, 20 x 20 px", 4.5),
    ("Block SVD: Y ~ U V", 5.0),
    ("Candidates: Y ~ u1 v1 + u2 v2 + ... + u20 v20", 9.5),
    ("Roughness test: keep a component while it is smoother than noise", 9.0),
    ("Movie ~ noise SD x U V + mean", 6.5),
    ("Blind-spot network: predicts frame t from its neighbors", 7.0),
    ("Denoised = w x prediction + (1 - w) x trace", 5.5),
    ("Second pass: compress again, denoising every trace", 8.0),
    ("Crop the movie to 100 s, 80 x 80 px", 6.0),
    ("", 11.0),
)
UNTITLED = {0: "the-movie", 10: "results"}

N_ROWS = M.N_ROWS
ROW_YS = (1.95, 1.1, 0.25, -0.6)
ROW_SIDE = 0.72
ROW_TRACE_X, ROW_TRACE_WIDTH = -3.05, 4.4
ROW_TRACE_FRAMES = 900  # 30 s
X_MOVIE, Y_MOVIE, H_MOVIE = -4.5, 0.1, 3.8
Y_LOW, H_LOW, W_LOW, X_LOW = -2.25, 1.5, 8.0, -2.4  # the followed cell's trace under the rows and the panels
X_NUMBER = 2.5
PANEL_XS, Y_PANEL, H_PANEL = (-4.6, 0.0, 4.6), 1.0, 2.9
MORPH_STEPS = 12  # images per morph of a component's image
FOLD_SECONDS = (1.1, 0.45, 0.45, 0.45)  # column of U to image: slow for the first candidate, so that the others read at a glance
FLY_SECONDS = (0.8, 0.4, 0.4, 0.4)  # image to its row
X_UNFOLDED = 0.65
CELL_OVERLAP = -0.01  # a grid's gap: cells overlap, or they show seams and a column of U at 0.5 px per cell fades out


@dataclass
class RowScale:
    """What every row's trace shares: half its height in noise SDs, and the frame rate in Hz."""

    half_height: float
    fs: float


@dataclass
class Strip:
    """A log axis for roughness, from x0 to x1 in the scene, baseline at y."""

    x0: float
    x1: float
    y: float
    lo: float
    hi: float

    def x(self, value: float) -> float:
        return self.x0 + (self.x1 - self.x0) * float(np.log(value / self.lo) / np.log(self.hi / self.lo))


STRIP_TEMPORAL = Strip(1.0, 6.7, 1.25, 0.05, 3.2)
STRIP_SPATIAL = Strip(1.0, 6.7, -0.55, 0.3, 1.8)


class Track(mn.Animation):
    """A mobject moving to points[0] .. points[-1] in turn over the animation."""

    def __init__(self, mobject: mn.Mobject, points: np.ndarray, **kwargs) -> None:
        super().__init__(mobject, **kwargs)
        self.points = points

    def interpolate_mobject(self, alpha: float) -> None:
        self.mobject.move_to(self.points[min(int(self.rate_func(alpha) * len(self.points)), len(self.points) - 1)])


def blocks_label(v: float) -> mn.Text:
    return label(f"{int(v)} blocks", TEXT, 20)


def rank_label(v: float) -> mn.Text:
    return label(f"rank {int(v)}", KEY, 44)


def rank_label_small(v: float) -> mn.Text:
    return label(f"rank {int(v)}", KEY, 36)


def r_label(v: float) -> mn.Text:
    return label(f"r = {v:.2f}", KEY, 36)


def gray_frames(frames: np.ndarray) -> list:
    return [np.repeat(f[..., None], 3, axis=-1) for f in frames]


def component_rgb(u: np.ndarray) -> np.ndarray:
    return diverging_rgb(u, float(np.abs(u).max()), ACCENT)


def morph_frames(a: np.ndarray, b: np.ndarray) -> list:
    """Images stepping from a to b."""
    return [((1 - w) * a + w * b).astype(np.uint8) for w in np.linspace(0, 1, MORPH_STEPS)]


def decision_color(passed: bool, kept: bool) -> tuple:
    if passed and kept:
        return ACCENT
    return HIGHLIGHT if kept else DIM


def row_box(y: float, c: tuple, stroke_width: float = 2.0) -> mn.Rectangle:
    return put(mn.Rectangle(width=6.05, height=ROW_SIDE + 0.08, color=color(c), stroke_width=stroke_width), -3.72, y)


def row_axes(y: float, scale: RowScale) -> mn.Axes:
    return put(plain_axes(0, ROW_TRACE_FRAMES / scale.fs, -scale.half_height, scale.half_height, ROW_TRACE_WIDTH, ROW_SIDE, visible=False), ROW_TRACE_X, y)


def row_path(y: float, trace: np.ndarray, scale: RowScale) -> mn.VMobject:
    t = np.arange(ROW_TRACE_FRAMES) / scale.fs
    return path(row_axes(y, scale), t, trace[:ROW_TRACE_FRAMES], TRACE, 1.0, -scale.half_height, scale.half_height)


def row_time_bar(fs: float, seconds: float = 5.0) -> mn.VGroup:
    """A time bar for every row's trace, at the right end of the header line."""
    x1 = ROW_TRACE_X + ROW_TRACE_WIDTH / 2
    bar = mn.Line(at(x1 - ROW_TRACE_WIDTH * seconds * fs / ROW_TRACE_FRAMES, 2.5), at(x1, 2.5), color=color(BAR), stroke_width=3.0)
    return mn.VGroup(bar, label(f"{seconds:g} s", BAR, 16).next_to(bar, mn.LEFT, buff=0.1))


def matrix_column_box(matrix: mn.ImageMobject, k: int, n_columns: int) -> mn.Rectangle:
    """A box around column k of a matrix drawn as an image."""
    width = matrix.width / n_columns
    return mn.Rectangle(width=width, height=matrix.height, color=color(KEY), stroke_width=2.5).move_to(matrix.get_left() + at((k + 0.5) * width, 0))


def matrix_row_box(matrix: mn.ImageMobject, k: int, n_rows: int) -> mn.Rectangle:
    """A box around row k of a matrix drawn as an image."""
    height = matrix.height / n_rows
    return mn.Rectangle(width=matrix.width, height=height, color=color(KEY), stroke_width=2.5).move_to(matrix.get_top() - at(0, (k + 0.5) * height))


def flattened_onto(trace: mn.VMobject, box: mn.Rectangle) -> mn.VMobject:
    """The trace squashed flat along box, in KEY: the start of a Transform that lifts a row of V into a trace."""
    return trace.stretch_to_fit_height(0.02).stretch_to_fit_width(box.width).move_to(box).set_stroke(color=color(KEY), width=2.0)


def strip_line(s: Strip, ticks: tuple) -> mn.VGroup:
    """The baseline of a strip and its labeled ticks."""
    parts = [mn.Line(at(s.x0, s.y), at(s.x1, s.y), color=color(EDGE), stroke_width=1.5)]
    for tick in ticks:
        x = s.x(tick)
        parts.append(mn.Line(at(x, s.y), at(x, s.y - 0.08), color=color(BAR), stroke_width=1.5))
        parts.append(label(f"{tick:g}", BAR, 14).next_to(at(x, s.y - 0.08), mn.DOWN, buff=0.03))
    return mn.VGroup(*parts)


def noise_bars(s: Strip, values: np.ndarray, n_bins: int = 12, height: float = 0.6) -> mn.VGroup:
    """A histogram of noise scores on a strip; always n_bins bars, so that two of them transform into each other."""
    counts, edges = np.histogram(np.log(values), bins=n_bins)
    bars = mn.VGroup()
    for count, a, b in zip(counts, np.exp(edges[:-1]), np.exp(edges[1:])):
        h = max(height * count / counts.max(), 0.002)
        width = max(s.x(b) - s.x(a), 0.012)
        bars.add(mn.Rectangle(width=width, height=h, stroke_width=0, fill_color=color(DIM), fill_opacity=0.8).move_to(at(0.5 * (s.x(a) + s.x(b)), s.y + h / 2)))
    return bars


def cut_line(s: Strip, value: float, height: float = 0.85) -> mn.Line:
    return mn.Line(at(s.x(value), s.y), at(s.x(value), s.y + height), color=color(TEXT), stroke_width=2.0)


def score_dots(s: Strip, values: np.ndarray, colors: list) -> list:
    """One dot per candidate under the strip's labels, in row order from the top."""
    return [mn.Dot(at(s.x(float(v)), s.y + 0.14 + 0.13 * (N_ROWS - 1 - k)), radius=0.07, color=color(c)) for k, (v, c) in enumerate(zip(values, colors))]


def low_axes(duration: float, lo: float, hi: float) -> mn.Axes:
    return put(plain_axes(0, duration, lo, hi, W_LOW, H_LOW, visible=False), X_LOW, Y_LOW)


@dataclass
class Rows:
    """The candidates of one block in one pass: what chapters 4 and 8 draw."""

    numbers: list
    images: list
    traces: list
    boxes: list
    headers: mn.VGroup
    rest: mn.Text


def candidate_rows(spatial: np.ndarray, temporal: np.ndarray, kept: np.ndarray, scale: RowScale) -> Rows:
    numbers = [label(str(k + 1), DIM, 18).move_to(at(-6.5, y)) for k, y in enumerate(ROW_YS)]
    images = [image(component_rgb(spatial[:, :, k]), ROW_SIDE, -5.75, y) for k, y in enumerate(ROW_YS)]
    traces = [row_path(y, temporal[k], scale) for k, y in enumerate(ROW_YS)]
    boxes = [row_box(y, EDGE, 1.0) for y in ROW_YS]
    headers = mn.VGroup(label("image u", DIM, 16).move_to(at(-5.75, 2.5)), label("trace v", DIM, 16).move_to(at(-4.9, 2.5), aligned_edge=mn.LEFT), row_time_bar(scale.fs))
    rest = label(M.rest_text(kept=kept), DIM, 16).move_to(at(-6.72, -1.17), aligned_edge=mn.LEFT)
    return Rows(numbers, images, traces, boxes, headers, rest)


def build(seconds: list, filepath_run: Path) -> kit.Timeline:
    run = M.load_run(filepath_run)
    n = run["numbers"]
    simulated = n["simulated"]
    fs = n["fs"]
    height_fov, width_fov = n["shape"]
    table = M.table_rows(run) if simulated else M.table_rows_recording(run)
    titles = kit.titles_for(CHAPTERS)
    title = put(label(TITLE, TEXT, 30), 0.0, 3.6)

    raw_rgb, plain_rgb = gray_frames(run["raw_frames"]), gray_frames(run["plain_frames"])
    panel_y, panel_x, panel_side = n["panel"]
    panel_window = (slice(None), slice(panel_y, panel_y + panel_side), slice(panel_x, panel_x + panel_side))
    raw_panel, plain_panel = gray_frames(run["raw_frames"][panel_window]), gray_frames(run["plain_frames"][panel_window])
    difference = run["raw_frames"][panel_window].astype(np.int16) - run["plain_frames"][panel_window].astype(np.int16)
    residual_rgb = [diverging_rgb(d, 3.0 * float(difference.std()), TEXT) for d in difference]

    f0 = float(np.percentile(run["trace_raw"], 20))
    dff = {name: M.to_dff(run[name], f0) for name in ("trace_raw", "trace_plain", "trace_denoised", *(("trace_long",) if simulated else ()))}
    dff["short"] = M.to_dff(run["trace_short"][0], f0)
    reference = run["truth"] / f0 if simulated else dff["trace_raw"]
    reference_name, reference_color = ("true trace (simulated)", TRUTH) if simulated else ("raw trace", TRACE)
    t = np.arange(reference.size) / fs
    duration = reference.size / fs
    headroom = 1.05 if simulated else 1.35  ## a recording's transients reach the right end, where the dF/F bar sits
    lo, hi = float(np.percentile(dff["trace_raw"], 0.02)) * 1.1, float(dff["trace_raw"].max()) * headroom

    movie = image(raw_rgb[0], H_MOVIE, X_MOVIE, Y_MOVIE)
    movie_frame = kit.frame_of(movie)
    movie_name = label(n["movie_name"], DIM, 18).next_to(movie_frame, mn.DOWN, buff=0.12).align_to(movie_frame, mn.LEFT)
    scale = H_MOVIE / height_fov
    corner = movie.get_corner(mn.UL)
    ring = mn.Circle(radius=8 * scale, color=color(KEY), stroke_width=2.5).move_to(corner + at((n["center"][1] + 0.5) * scale, -(n["center"][0] + 0.5) * scale))

    axes_trace = put(plain_axes(0, duration, lo, hi, 8.2, 2.6, visible=False), 2.5, 0.55)
    trace_raw = path(axes_trace, t, dff["trace_raw"], TRACE, 1.2, lo, hi)
    bars_trace = scale_bars(axes_trace, duration, lo, hi, 10.0, 0.5)
    spikes_drawn = []
    if simulated:
        spike_times = t[run["spikes"] > 0]
        tick_y = lo - 0.1 * (hi - lo)
        ticks = mn.VGroup(*[mn.Line(*axes_points(axes_trace, np.array([s, s]), np.array([tick_y - 0.06 * (hi - lo), tick_y])), color=color(TRUTH), stroke_width=2.0) for s in spike_times])
        spikes_drawn = [ticks, label("Ground truth spikes (simulated)", TRUTH, 18).next_to(ticks, mn.DOWN, buff=0.08).align_to(axes_trace, mn.LEFT)]
    trace_name = label("Raw trace of the ringed cell", DIM, 18).next_to(axes_trace, mn.UP, buff=0.12).align_to(axes_trace, mn.LEFT)

    lattice = mn.VGroup(*[mn.Rectangle(width=M.BLOCK * scale, height=M.BLOCK * scale, color=color(BAR), stroke_width=1.0, stroke_opacity=0.7)
                          .move_to(corner + at((x + M.BLOCK / 2) * scale, -(y + M.BLOCK / 2) * scale)) for y in M.block_starts(height_fov) for x in M.block_starts(width_fov)])
    block_y, block_x = n["block"]
    block_box = mn.Rectangle(width=M.BLOCK * scale, height=M.BLOCK * scale, color=color(KEY), stroke_width=3.0).move_to(corner + at((block_x + M.BLOCK / 2) * scale, -(block_y + M.BLOCK / 2) * scale))
    count_blocks = blocks_label(0).next_to(movie_frame, mn.UP, buff=0.12).align_to(movie_frame, mn.LEFT)

    spatial_1, temporal_1, signs_1 = M.oriented(spatial=run["block_spatial_1"], temporal=run["block_temporal_1"])
    spatial_2, temporal_2, _ = M.oriented(spatial=run["block_spatial_2"], temporal=run["block_temporal_2"])
    n_candidates = spatial_1.shape[2]
    row_scale = RowScale(n["row_range"], fs)
    y_matrix = 0.45
    matrix_y = image(np.repeat(np.repeat(run["block_y"][:, :ROW_TRACE_FRAMES:3, None], 3, axis=-1), 2, axis=1), 2.2, -0.35, y_matrix)
    matrix_y_frame = kit.frame_of(matrix_y)
    u_flat = spatial_1.reshape(-1, n_candidates)
    matrix_u = image(np.repeat(diverging_rgb(u_flat / np.abs(u_flat).max(axis=0), 1.0, ACCENT), 10, axis=1), 2.2, 2.5, y_matrix)
    v_rgb = diverging_rgb(temporal_1[:, :ROW_TRACE_FRAMES:3], 3.0, TEXT)
    matrix_v = image(np.repeat(v_rgb, 5, axis=0), 1.1, 4.85, y_matrix + 0.55)
    approx = label("~", DIM, 34).move_to(at(1.68, y_matrix))
    names_matrices = mn.VGroup(label(f"Y: {M.BLOCK * M.BLOCK} pixels x {n['n_frames']} frames", DIM, 18).next_to(matrix_y_frame, mn.DOWN, buff=0.12),
                               label(f"U: {n_candidates} images", ACCENT, 18).next_to(matrix_u, mn.DOWN, buff=0.12),
                               label(f"V: {n_candidates} traces", DIM, 18).next_to(matrix_v, mn.DOWN, buff=0.12))
    box_fly = block_box.copy()
    box_u, box_v = matrix_column_box(matrix_u, 0, n_candidates), matrix_row_box(matrix_v, 0, n_candidates)
    name_candidate = label("Candidate 1: one image, one trace", KEY, 18).next_to(names_matrices, mn.DOWN, buff=0.3).align_to(matrix_u, mn.LEFT)

    rows_1 = candidate_rows(spatial=spatial_1, temporal=temporal_1, kept=run["kept_1"], scale=row_scale)
    traces_landed = rows_1.traces
    rows_1.traces = [flattened_onto(row_path(y, temporal_1[k], row_scale), matrix_row_box(matrix_v, k, n_candidates)) for k, y in enumerate(ROW_YS)]
    columns_u = [grid(component_rgb(spatial_1[:, :, k]).reshape(-1, 1, 3), matrix_u.width / n_candidates, matrix_column_box(matrix_u, k, n_candidates).get_x(), y_matrix,
                      gap=CELL_OVERLAP, cell_h=matrix_u.height / M.BLOCK**2) for k in range(N_ROWS)]
    ## NOTE: as tall as U, so each run of 20 pixels in the column sits at the height of its row in the image and only spreads sideways
    images_unfolded = [grid(component_rgb(spatial_1[:, :, k]), matrix_u.height / M.BLOCK, X_UNFOLDED, y_matrix, gap=CELL_OVERLAP) for k in range(N_ROWS)]
    images_folded = [grid(component_rgb(spatial_1[:, :, k]), ROW_SIDE / M.BLOCK, -5.75, y, gap=CELL_OVERLAP) for k, y in enumerate(ROW_YS)]
    rest_undecided = label(f"Candidates {N_ROWS + 1} to {n_candidates}", DIM, 16).move_to(rows_1.rest, aligned_edge=mn.LEFT)
    colors_1 = [decision_color(bool(p), bool(k)) for p, k in zip(run["passed_1"][:N_ROWS], run["kept_1"][:N_ROWS])]
    colors_2 = [decision_color(bool(p), bool(k)) for p, k in zip(run["passed_2"][:N_ROWS], run["kept_2"][:N_ROWS])]
    cut_spatial_1, cut_temporal_1 = n["thresholds_1"]
    cut_spatial_2, cut_temporal_2 = n["thresholds_2"]
    ticks_temporal, ticks_spatial = strip_line(STRIP_TEMPORAL, (0.1, 0.3, 1.0, 3.0)), strip_line(STRIP_SPATIAL, (0.5, 1.0))
    bars_temporal, bars_spatial = noise_bars(STRIP_TEMPORAL, run["noise_temporal_1"]), noise_bars(STRIP_SPATIAL, run["noise_spatial_1"])
    lines_cut = cut_line(STRIP_TEMPORAL, cut_temporal_1), cut_line(STRIP_SPATIAL, cut_spatial_1)
    name_temporal = label("Trace roughness", DIM, 18).move_to(at(STRIP_TEMPORAL.x0, STRIP_TEMPORAL.y + 1.25), aligned_edge=mn.LEFT)
    name_spatial = label("Image roughness", DIM, 18).move_to(at(STRIP_SPATIAL.x0, STRIP_SPATIAL.y + 1.0), aligned_edge=mn.LEFT)
    name_cut = label("cut: 5th percentile of 250 noise blocks", TEXT, 16).next_to(lines_cut[0].get_top(), mn.LEFT, buff=0.1)
    dots_temporal = score_dots(STRIP_TEMPORAL, run["temporal_stats_1"][:N_ROWS], colors_1)
    dots_spatial = score_dots(STRIP_SPATIAL, run["spatial_stats_1"][:N_ROWS], colors_1)
    legend = mn.VGroup(*[mn.VGroup(mn.Dot(radius=0.07, color=color(c)), label(text, DIM, 16)).arrange(mn.RIGHT, buff=0.1)
                         for c, text in ((ACCENT, "passed"), (HIGHLIGHT, "first failure: kept, then stop"), (DIM, "dropped"))]).arrange(mn.RIGHT, buff=0.35)
    legend.move_to(at(-2.7, -1.17), aligned_edge=mn.LEFT)
    rank = rank_label(0).move_to(at(X_NUMBER, -2.05), aligned_edge=mn.LEFT)
    rank_final = int(run["kept_1"].sum())

    panels = [image(frames[0], H_PANEL, x, Y_PANEL) for frames, x in zip((raw_panel, plain_panel, residual_rgb), PANEL_XS)]
    panel_frames = [kit.frame_of(p) for p in panels]
    panel_names = mn.VGroup(*[label(text, DIM, 18).next_to(f, mn.DOWN, buff=0.1) for text, f in zip(("Raw", "Compressed", "Residual"), panel_frames)])
    n_panel = 100
    axes_low = low_axes(duration, lo, hi)
    truth_low = path(axes_low, t, reference, reference_color, 1.4, lo, hi)
    plain_low = path(axes_low, t, dff["trace_plain"], ACCENT, 1.4, lo, hi)
    bars_low = scale_bars(axes_low, duration, lo, hi, 10.0, 0.08)
    name_low = mn.VGroup(label(reference_name, reference_color, 16), label("compressed", ACCENT, 16)).arrange(mn.RIGHT, buff=0.3).next_to(axes_low, mn.UP, buff=0.04).align_to(axes_low, mn.LEFT)
    stored = label(f"{100 * n['stored_1']:.1f}% of the movie's values", DIM, 18).next_to(rank, mn.DOWN, buff=0.14).align_to(rank, mn.LEFT)

    window = slice(*n["zoom_frames"])
    k_zoom = n["k_cell"]
    sign = float(signs_1[k_zoom])
    x_zoom = sign * run["blend_input"][k_zoom, window]
    prediction_zoom = sign * run["prediction"][k_zoom, window]
    blended_zoom = sign * run["blended"][k_zoom, window]
    weight_zoom = run["weight"][k_zoom, window]
    t_zoom = np.arange(x_zoom.size) / fs
    z_lo, z_hi = float(x_zoom.min()) * 1.05, float(x_zoom.max()) * 1.05
    axes_zoom = put(plain_axes(0, float(t_zoom[-1]), z_lo, z_hi, 12.4, 2.6, visible=False), 0.0, 0.85)
    trace_zoom = path(axes_zoom, t_zoom, x_zoom, TRACE, 1.3, z_lo, z_hi)
    path_prediction = path(axes_zoom, t_zoom, prediction_zoom, ACCENT, 2.5, z_lo, z_hi)
    target_blended = path(axes_zoom, t_zoom, blended_zoom, TEXT, 2.0, z_lo, z_hi)
    bars_zoom = scale_bars(axes_zoom, float(t_zoom[-1]), z_lo, z_hi, 1.0, 0.12, with_dff=False)
    radius_units = n["receptive_radius"] / fs * 12.4 / float(t_zoom[-1])
    height_zoom = 2.6
    x_left = axes_points(axes_zoom, np.array([0.0]), np.array([0.0]))[0, 0]
    x_step = 12.4 / float(t_zoom[-1]) / fs
    spot = mn.VGroup(mn.Rectangle(width=radius_units, height=height_zoom, stroke_width=0, fill_color=color(ACCENT), fill_opacity=0.28).move_to(at(-radius_units / 2 - 0.02, 0)),
                     mn.Rectangle(width=radius_units, height=height_zoom, stroke_width=0, fill_color=color(ACCENT), fill_opacity=0.28).move_to(at(radius_units / 2 + 0.02, 0)),
                     mn.Line(at(0, -height_zoom / 2), at(0, height_zoom / 2), color=color(KEY), stroke_width=2.5))
    spot.move_to(at(x_left, 0.85))
    centers_spot = np.array([at(x_left + i * x_step, 0.85) for i in range(x_zoom.size)])
    name_zoom = mn.VGroup(label("frame t: hidden", KEY, 18), label(f"+/- {n['receptive_radius']} frames: what the network reads", ACCENT, 18), label("prediction", ACCENT, 18)).arrange(mn.RIGHT, buff=0.5)
    name_zoom.next_to(axes_zoom, mn.UP, buff=0.1).align_to(axes_zoom, mn.LEFT)
    name_blend = mn.VGroup(label("trace", TRACE, 18), label("prediction", ACCENT, 18)).arrange(mn.RIGHT, buff=0.5).next_to(axes_zoom, mn.UP, buff=0.1).align_to(axes_zoom, mn.LEFT)
    name_denoised = label("denoised", TEXT, 18).next_to(axes_zoom, mn.UP, buff=0.1).align_to(axes_zoom, mn.LEFT)
    training = n["training"]
    name_training = label(f"Trained on pass 1's {n['rank_1']} traces, {training['steps']} steps", DIM, 18).move_to(at(0.0, -1.0))

    axes_w = put(plain_axes(0, float(t_zoom[-1]), 0.0, 1.0, 12.4, 1.0, visible=False), 0.0, -1.75)
    path_w = path(axes_w, t_zoom, weight_zoom, ACCENT, 2.0, 0.0, 1.0)
    w_floor = mn.Line(*axes_points(axes_w, np.array([0.0, float(t_zoom[-1])]), np.array([0.0, 0.0])), color=color(EDGE), stroke_width=1.0)
    name_w = label(f"w, weight on the prediction: {float(run['weight'][k_zoom].min()):.2f} to 1.00", ACCENT, 18).next_to(axes_w, mn.UP, buff=0.04).align_to(axes_w, mn.LEFT)
    ticks_w = mn.VGroup(*[label(text, BAR, 14).next_to(axes_points(axes_w, np.array([0.0]), np.array([y]))[0], mn.LEFT, buff=0.1) for text, y in (("1", 1.0), ("0", 0.0))])

    rows_2 = candidate_rows(spatial=spatial_1, temporal=temporal_1, kept=run["kept_2"], scale=row_scale)
    boxes_2 = [row_box(y, c) for y, c in zip(ROW_YS, colors_1)]
    ticks_temporal_2, ticks_spatial_2 = strip_line(STRIP_TEMPORAL, (0.1, 0.3, 1.0, 3.0)), strip_line(STRIP_SPATIAL, (0.5, 1.0))
    bars_temporal_2, bars_spatial_2 = noise_bars(STRIP_TEMPORAL, run["noise_temporal_1"]), noise_bars(STRIP_SPATIAL, run["noise_spatial_1"])
    lines_cut_2 = cut_line(STRIP_TEMPORAL, cut_temporal_1), cut_line(STRIP_SPATIAL, cut_spatial_1)
    names_strip_2 = mn.VGroup(name_temporal.copy(), name_spatial.copy())
    name_cut_2 = name_cut.copy()
    name_cut_target = name_cut.copy().next_to(cut_line(STRIP_TEMPORAL, cut_temporal_2).get_top(), mn.LEFT, buff=0.1)
    dots_temporal_2 = score_dots(STRIP_TEMPORAL, run["temporal_stats_1"][:N_ROWS], colors_1)
    dots_spatial_2 = score_dots(STRIP_SPATIAL, run["spatial_stats_1"][:N_ROWS], colors_1)
    legend_2 = legend.copy()
    truth_second = path(axes_low, t, reference, reference_color, 1.4, lo, hi)
    trace_second = path(axes_low, t, dff["trace_plain"], ACCENT, 1.4, lo, hi)
    denoised_low = path(axes_low, t, dff["trace_denoised"], TEXT, 1.4, lo, hi)
    bars_second = scale_bars(axes_low, duration, lo, hi, 10.0, 0.08)
    name_second = mn.VGroup(label(reference_name, reference_color, 16), label("ringed cell", TEXT, 16)).arrange(mn.RIGHT, buff=0.3).next_to(axes_low, mn.DOWN, buff=0.1).align_to(axes_low, mn.LEFT)
    rank_second = rank_label_small(n["rank_1"]).move_to(at(X_NUMBER, -1.72), aligned_edge=mn.LEFT)
    r_drawn, count_r_second, count_r_short = [], [], []
    if simulated:
        r_plain, r_denoised = n["r"]["plain"][n["cell"]], n["r"]["denoised"][n["cell"]]
        r_value = r_label(r_plain).move_to(at(X_NUMBER, -2.4), aligned_edge=mn.LEFT)
        r_drawn = [r_value, label("correlation with the true trace", DIM, 16).next_to(r_value, mn.DOWN, buff=0.1).align_to(r_value, mn.LEFT)]
        count_r_second = [Count(r_value, r_plain, r_denoised, r_label, decimals=2)]
        count_r_short = [Count(r_value, r_denoised, n["short"][0]["r"], r_label, decimals=2, run_time=2.4)]

    axes_big = put(plain_axes(0, duration, lo, hi, 12.4, 3.2, visible=False), 0.0, 0.7)
    denoised_big = path(axes_big, t, dff["trace_denoised"], TEXT, 1.6, lo, hi)
    short_big = path(axes_big, t, dff["short"], TEXT, 1.6, lo, hi)
    truth_big = path(axes_big, t, reference, reference_color, 1.6, lo, hi)
    bars_big = scale_bars(axes_big, duration, lo, hi, 10.0, 0.12)
    name_big = name_second.copy().next_to(axes_big, mn.DOWN, buff=0.14).align_to(axes_big, mn.LEFT)
    name_short = label(f"{n['short'][0]['steps']} training steps, not {training['steps']}: the same code, with fewer and shorter traces to train on", TEXT, 20).move_to(at(0.0, -1.55))

    captions = [None] * len(CHAPTERS)
    captions[2] = caption("Every pixel has its mean removed and is divided by its noise SD before the SVD")
    captions[3] = caption("Each column of U folds into a 20 x 20 px image; each row of V is a trace")
    captions[4] = caption("A component passes if its trace and its image are both smoother than the cut")
    if not simulated:
        captions[5] = caption(f"Panels: {panel_side} x {panel_side} px around the ringed cell", y=-3.72)
    captions[6] = caption("Noise at frame t is independent of its neighbors, so the network can only predict the signal")
    captions[7] = caption("w = noise variance / predicted variance; the noise variance is the 30th percentile of the predicted variance")
    captions[8] = caption("The 250 noise blocks go through the denoiser too, so the cut moves with them", y=-3.72)

    table_rows, table_header, table_foot = results_table(table=table, reference=reference, reference_name=reference_name, reference_color=reference_color, dff=dff, t=t, lo=lo, hi=hi, n=n)

    gone_ch2 = (trace_raw, bars_trace, *spikes_drawn, trace_name, lattice, count_blocks)
    gone_ch3 = (movie, movie_frame, movie_name, ring, block_box, matrix_y, matrix_y_frame, approx, names_matrices[0])
    gone_ch4 = (matrix_u, matrix_v, names_matrices[1], names_matrices[2], box_u, box_v)
    rows_all = [*rows_1.numbers, *rows_1.images, *rows_1.traces, *rows_1.boxes, rows_1.headers, rows_1.rest]
    gone_ch5 = (*rows_all, ticks_temporal, ticks_spatial, bars_temporal, bars_spatial, *lines_cut, name_temporal, name_spatial, name_cut, *dots_temporal, *dots_spatial, legend)
    gone_ch6 = (*panels, *panel_frames, panel_names, truth_low, plain_low, bars_low, name_low, stored, rank)
    gone_ch8 = (trace_zoom, bars_zoom, name_denoised, path_w, w_floor, name_w, ticks_w)
    rows_2_all = [*rows_2.numbers, *rows_2.images, *rows_2.traces, *boxes_2, rows_2.headers, rows_2.rest]
    gone_ch9 = (*rows_2_all, ticks_temporal_2, ticks_spatial_2, bars_temporal_2, bars_spatial_2, *lines_cut_2, names_strip_2, name_cut_2, *dots_temporal_2, *dots_spatial_2, legend_2,
                rank_second, bars_second)

    unpack = []
    for k in range(N_ROWS):
        lift = [mn.FadeIn(columns_u[k]), mn.FadeIn(rows_1.traces[k])]
        if k > 0:
            lift += [mn.Transform(box_u, matrix_column_box(matrix_u, k, n_candidates)), mn.Transform(box_v, matrix_row_box(matrix_v, k, n_candidates))]
        if k == 1:
            lift.append(mn.FadeOut(name_candidate))
        landed = [mn.FadeIn(rows_1.images[k]), mn.FadeOut(columns_u[k]), mn.FadeIn(rows_1.numbers[k]), mn.FadeIn(rows_1.boxes[k])]
        if k == 0:
            landed.append(mn.FadeIn(rows_1.headers))
        unpack += [mn.AnimationGroup(*lift, run_time=0.25),
                   mn.AnimationGroup(mn.AnimationGroup(mn.Transform(columns_u[k], images_unfolded[k], run_time=FOLD_SECONDS[k]), mn.Transform(columns_u[k], images_folded[k], run_time=FLY_SECONDS[k]), lag_ratio=1.0),
                                     mn.Transform(rows_1.traces[k], traces_landed[k], run_time=FOLD_SECONDS[k] + FLY_SECONDS[k])),
                   mn.AnimationGroup(*landed, run_time=0.2)]

    steps = []
    count_so_far = 0
    count_rest = []
    for k in range(N_ROWS):
        kept = bool(run["kept_1"][k])
        reveal = [mn.FadeIn(dots_temporal[k], scale=1.5), mn.FadeIn(dots_spatial[k], scale=1.5), mn.Transform(rows_1.boxes[k], row_box(ROW_YS[k], colors_1[k]))]
        if kept:
            reveal.append(Count(rank, count_so_far, count_so_far + 1, rank_label, decimals=0))
            count_so_far += 1
        steps.append(mn.AnimationGroup(*reveal, run_time=1.2))
    if rank_final > count_so_far:
        count_rest = [Count(rank, count_so_far, rank_final, rank_label, decimals=0)]

    second_pass = [mn.Transform(rows_2.traces[k], row_path(ROW_YS[k], temporal_2[k], row_scale)) for k in range(N_ROWS)]
    second_pass += [Play(rows_2.images[k], morph_frames(component_rgb(spatial_1[:, :, k]), component_rgb(spatial_2[:, :, k]))) for k in range(N_ROWS)]
    second_pass += [mn.Transform(boxes_2[k], row_box(ROW_YS[k], colors_2[k])) for k in range(N_ROWS)]
    second_pass += [mn.Transform(a, b) for a, b in zip(dots_temporal_2, score_dots(STRIP_TEMPORAL, run["temporal_stats_2"][:N_ROWS], colors_2))]
    second_pass += [mn.Transform(a, b) for a, b in zip(dots_spatial_2, score_dots(STRIP_SPATIAL, run["spatial_stats_2"][:N_ROWS], colors_2))]
    second_pass += [mn.Transform(bars_temporal_2, noise_bars(STRIP_TEMPORAL, run["noise_temporal_2"])), mn.Transform(bars_spatial_2, noise_bars(STRIP_SPATIAL, run["noise_spatial_2"])),
                    mn.Transform(lines_cut_2[0], cut_line(STRIP_TEMPORAL, cut_temporal_2)), mn.Transform(lines_cut_2[1], cut_line(STRIP_SPATIAL, cut_spatial_2)),
                    mn.Transform(name_cut_2, name_cut_target), mn.Transform(trace_second, denoised_low),
                    Count(rank_second, n["rank_1"], n["rank_2"], rank_label_small, decimals=0), *count_r_second]

    chapters = [
        chapter(seconds[0],
                mn.AnimationGroup(mn.FadeIn(movie, run_time=0.4), Play(movie, raw_rgb[:95], run_time=seconds[0] - 0.4, rate_func=mn.linear), lag_ratio=1.0),
                mn.FadeIn(title, run_time=0.6), mn.FadeIn(movie_frame, run_time=0.4), mn.FadeIn(movie_name, run_time=0.6),
                mn.AnimationGroup(mn.FadeIn(ring, run_time=0.6), mn.AnimationGroup(mn.Create(trace_raw), mn.FadeIn(bars_trace), mn.FadeIn(trace_name), *[mn.FadeIn(m) for m in spikes_drawn], run_time=2.6), lag_ratio=1.0),
                active=1.0),
        chapter(seconds[1],
                Play(movie, raw_rgb[95:185], run_time=seconds[1], rate_func=mn.linear),
                mn.AnimationGroup(mn.FadeOut(title, run_time=0.5), mn.FadeIn(titles[1], run_time=0.5), lag_ratio=1.0),
                mn.AnimationGroup(mn.Wait(run_time=0.8), mn.AnimationGroup(mn.FadeIn(count_blocks, run_time=0.2), run_time=0.2),
                                  mn.AnimationGroup(mn.FadeIn(lattice, lag_ratio=0.02, run_time=2.0),
                                                    Count(count_blocks, 0, n["n_blocks"], blocks_label, decimals=0, run_time=2.0)),
                                  mn.Create(block_box, run_time=0.5), lag_ratio=1.0),
                active=1.0),
        chapter(seconds[2],
                mn.AnimationGroup(*replace(titles[1], titles[2]), *replace(None, captions[2]), *[mn.FadeOut(m) for m in gone_ch2], run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(box_fly, run_time=0.1), mn.Transform(box_fly, matrix_y_frame.copy(), run_time=1.0), lag_ratio=1.0),
                mn.AnimationGroup(mn.FadeIn(matrix_y), mn.FadeIn(matrix_y_frame), mn.FadeOut(box_fly), mn.FadeIn(names_matrices[0]), run_time=0.7),
                mn.AnimationGroup(mn.FadeIn(approx), mn.FadeIn(matrix_u), mn.FadeIn(matrix_v), mn.FadeIn(names_matrices[1]), mn.FadeIn(names_matrices[2]), run_time=0.8), lag_ratio=1.0),
        chapter(seconds[3],
                mn.AnimationGroup(*replace(titles[2], titles[3]), *replace(captions[2], captions[3]), *[mn.FadeOut(m) for m in gone_ch3], run_time=0.7),
                mn.AnimationGroup(mn.Create(box_u), mn.Create(box_v), mn.FadeIn(name_candidate), run_time=0.7),
                *unpack,
                mn.FadeIn(rest_undecided, run_time=0.3), lag_ratio=1.0),
        chapter(seconds[4],
                mn.AnimationGroup(*replace(titles[3], titles[4]), *replace(captions[3], captions[4]), *[mn.FadeOut(m) for m in gone_ch4], run_time=0.8),
                mn.AnimationGroup(*[mn.FadeIn(m) for m in (ticks_temporal, ticks_spatial, bars_temporal, bars_spatial, *lines_cut, name_temporal, name_spatial, name_cut, legend, rank)], run_time=1.0),
                *steps,
                mn.AnimationGroup(*replace(rest_undecided, rows_1.rest), *count_rest, run_time=0.5), lag_ratio=1.0),
        chapter(seconds[5],
                mn.AnimationGroup(mn.AnimationGroup(*replace(titles[4], titles[5]), *replace(captions[4], captions[5]), *[mn.FadeOut(m) for m in gone_ch5], run_time=0.7),
                                  mn.AnimationGroup(*[mn.FadeIn(m) for m in (*panels, *panel_frames, panel_names, name_low, bars_low, stored)], mn.Create(truth_low), mn.Create(plain_low), run_time=0.8),
                                  mn.AnimationGroup(Play(panels[0], raw_panel[:n_panel], rate_func=mn.linear), Play(panels[1], plain_panel[:n_panel], rate_func=mn.linear),
                                                    Play(panels[2], residual_rgb[:n_panel], rate_func=mn.linear), run_time=seconds[5] - 1.5), lag_ratio=1.0),
                mn.AnimationGroup(mn.Wait(run_time=0.9), Count(rank, rank_final, n["rank_1"], rank_label, decimals=0, run_time=3.0), lag_ratio=1.0),
                active=1.0),
        chapter(seconds[6],
                mn.AnimationGroup(*replace(titles[5], titles[6]), *replace(captions[5], captions[6]), *[mn.FadeOut(m) for m in gone_ch6], run_time=0.7),
                mn.AnimationGroup(mn.Create(trace_zoom), mn.FadeIn(bars_zoom), run_time=0.7),
                mn.AnimationGroup(mn.FadeIn(spot), mn.FadeIn(name_zoom), mn.FadeIn(name_training), run_time=0.5),
                mn.AnimationGroup(Track(spot, centers_spot, rate_func=mn.linear), mn.Create(path_prediction, rate_func=mn.linear), run_time=4.0), lag_ratio=1.0),
        chapter(seconds[7],
                mn.AnimationGroup(*replace(titles[6], titles[7]), *replace(captions[6], captions[7]), *replace(name_zoom, name_blend), mn.FadeOut(spot), mn.FadeOut(name_training), run_time=0.7),
                mn.AnimationGroup(mn.FadeIn(w_floor), mn.FadeIn(ticks_w), mn.Create(path_w), mn.FadeIn(name_w), run_time=1.0),
                mn.AnimationGroup(mn.Transform(trace_zoom, target_blended), *replace(name_blend, name_denoised), mn.FadeOut(path_prediction), run_time=2.0), lag_ratio=1.0),
        chapter(seconds[8],
                mn.AnimationGroup(*replace(titles[7], titles[8]), *replace(captions[7], captions[8]), *[mn.FadeOut(m) for m in gone_ch8], run_time=0.7),
                mn.AnimationGroup(*[mn.FadeIn(m) for m in (*rows_2.numbers, *rows_2.images, *rows_2.traces, *boxes_2, rows_2.headers, rows_2.rest, ticks_temporal_2, ticks_spatial_2, bars_temporal_2,
                                                           bars_spatial_2, *lines_cut_2, names_strip_2, name_cut_2, *dots_temporal_2, *dots_spatial_2, legend_2, truth_second, trace_second, bars_second,
                                                           name_second, rank_second, *r_drawn)], run_time=0.9),
                mn.AnimationGroup(*second_pass, run_time=4.0), lag_ratio=1.0),
        chapter(seconds[9],
                mn.AnimationGroup(*replace(titles[8], titles[9]), *replace(captions[8], None), *[mn.FadeOut(m) for m in gone_ch9], run_time=1.0),
                mn.AnimationGroup(mn.Transform(trace_second, denoised_big), mn.Transform(truth_second, truth_big), mn.Transform(name_second, name_big), mn.FadeIn(bars_big), run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(name_short, run_time=0.5), mn.Transform(trace_second, short_big, run_time=2.4), *count_r_short),
                lag_ratio=1.0),
        chapter(seconds[10],
                mn.AnimationGroup(*replace(titles[9], None), *[mn.FadeOut(m) for m in (trace_second, truth_second, bars_big, name_second, name_short, *r_drawn)], run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(table_header), mn.FadeIn(table_foot), run_time=0.6),
                mn.LaggedStart(*table_rows, lag_ratio=0.4, run_time=5.0),
                lag_ratio=1.0),
    ]

    print(f"rank {n['rank_1']} then {n['rank_2']} over {n['n_blocks']} blocks; stored {100 * n['stored_1']:.1f}% then {100 * n['stored_2']:.1f}%; "
          f"cuts temporal {cut_temporal_1:.3f} then {cut_temporal_2:.3f}, spatial {cut_spatial_1:.3f} then {cut_spatial_2:.3f}; {training['steps']} steps; followed cell {n['cell']}, block {n['block']}")
    for row in table:
        print(row["name"], {key: np.round(value, 3) for key, value in row.items() if key not in ("name", "trace")})
    return timeline([], chapters)


def span(values: tuple) -> str:
    """One value, or lowest to highest when training seeds differ."""
    lo, hi = values
    return f"{lo:.2f}" if abs(hi - lo) < 0.005 else f"{lo:.2f} to {hi:.2f}"


def results_table(table: list, reference: np.ndarray, reference_name: str, reference_color: tuple, dff: dict, t: np.ndarray, lo: float, hi: float, n: dict) -> tuple:
    """One animation per row, drawing its reference trace and trace and fading in its name and numbers; the header and the footnote."""
    short_name = f"100 s, 80 x 80 px, denoised\n{n['short'][0]['steps']} steps"
    names = {"raw": "Raw movie", "plain": "Compressed", "shipped": f"Compressed + denoised\n{n['training']['steps']} steps", "short": short_name}
    keys = {"raw": "trace_raw", "plain": "trace_plain", "shipped": "trace_denoised", "long": "trace_long", "short": "short"}
    if n["simulated"]:
        names["long"] = f"10x learning rate, 3x epochs\n{n['spread']['long'][0]['steps']} steps, not the default"
        numbers = (("r", "r, ringed cell", KEY), ("r_median", "r, median cell", TEXT), ("gain_median", "gain, median cell", TEXT))
        note = "r: correlation with the true trace. Gain: amplitude relative to raw. Ranges: 5 training seeds, 3 for the crop."
    else:
        numbers = (("noise", "noise,\nringed cell", KEY), ("noise_median", "noise,\nmedian cell", TEXT), ("gain_median", "gain,\nmedian cell", TEXT))
        note = f"Noise: frame-to-frame SD. Gain: amplitude. Both relative to raw; {len(n['noise']['plain'])} cells; ranges over 3 training seeds."
    columns = (2.45, 4.1, 5.85)
    duration = dff["trace_raw"].size / n["fs"]
    y_top, pitch, x_trace, w_trace = 2.0, 1.04, -1.05, 4.9
    rows = []
    for k, row in enumerate(table):
        y = y_top - k * pitch
        axes = put(plain_axes(0, duration, lo, hi, w_trace, 0.86, visible=False), x_trace, y)
        faded = [label(names[row["name"]], DIM if row["name"] == "long" else TEXT, 17).move_to(at(-6.75, y), aligned_edge=mn.LEFT)]
        faded += [label(span(row[key]), c, 17).move_to(at(x, y)) for x, (key, _, c) in zip(columns, numbers)]
        if k == len(table) - 1:
            faded.append(scale_bars(axes, duration, lo, hi, 10.0, 0.08))
        rows.append(mn.AnimationGroup(mn.Create(path(axes, t, reference, reference_color, 1.0, lo, hi)), mn.Create(path(axes, t, dff[keys[row["name"]]], TEXT if k >= 2 else TRACE, 1.1, lo, hi)),
                                      *[mn.FadeIn(m) for m in faded]))
    header = mn.VGroup(*[label(name, DIM, 16).move_to(at(x, 2.72)) for x, (_, name, _) in zip(columns, numbers)], label("Ringed cell", DIM, 16).move_to(at(x_trace - w_trace / 2, 2.72), aligned_edge=mn.LEFT),
                       label(reference_name, reference_color, 16).move_to(at(x_trace + w_trace / 2, 2.72), aligned_edge=mn.RIGHT))
    return rows, header, caption(note, y=-3.6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run", type=Path, default=M.FILEPATH_RUN)
    args, sys.argv[1:] = parser.parse_known_args()
    kit.run("compression", CHAPTERS, UNTITLED, functools.partial(build, filepath_run=args.run), __doc__)
