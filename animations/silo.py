# /// script
# requires-python = ">=3.11"
# dependencies = ["manim>=0.22", "numpy", "pillow<13"]
# ///
"""
SILo as one animation: a simulated dendrite strip, one pixel's trace through the matched filter (its events' d'
climbing, as in the d' animation), the space-time maxima pooled into the activity image and its peaks, then
Z ~ H X + B fitted in GIAnT's two rounds (as in the NMF animation), lambda lowered from GIAnT's 0.9 to 0.3, and a
table of each synapse's trace against a plain ROI mean. Built from imgui_ophys's SILo model (lessons/silo/model.py);
drawn with kit.py.

    uv run -p 3.12 animations/silo.py --out <dir> [--stills-only] [--gif-width 720]
"""

from functools import partial

import manim as mn
import numpy as np

import kit
from kit import (DIM, KEY, PARTS, PINK, TEXT, TRACE, TRUTH, Count, Play, at, axes_points, caption, chapter, color,
                 dots_at, gray_rgb, image, label, path, plain_axes, put, replace, timeline, tinted_rgb)

M = kit.lesson_model("silo")

TITLE = "SILo: synapses found from their events, then their traces fitted"
LAMBDA_LOW = 0.3
CHAPTERS = (
    ("", 6.0),
    ("", 5.0),
    ("Matched filter: DoG in space, reversed exponential in time", 6.5),
    ("Maxima in x, y and t", 5.0),
    ("Activity image: each pixel's squared maxima, summed", 6.0),
    ("Sources: 3 x 3 maxima above median + 8 robust SDs", 5.0),
    ("Z ~ H X + B", 8.0),
    ("Events ≥ 0 under an L1 penalty, λ = 0.9, 2 rounds", 8.0),
    (f"Lower λ to {LAMBDA_LOW:g}", 6.0),
    ("", 10.0),
)
UNTITLED = {0: "the-strip", 1: "one-pixel", 9: "table"}
SOURCE = (PARTS[0], PARTS[1], PARTS[4], PINK)  # synapses 1 to 4, in the model's order
KINDS = ("spine", "shaft", "shaft", "spine")
EXAMPLE = 2  # the synapse whose traces the table shows
H_IMAGE, X_LEFT, X_RIGHT, Y_TOP = 2.0, -3.5, 3.5, 1.5  # the strip panels, scene units
X_ZOOM, Y_ZOOM, W_ZOOM, H_ZOOM = -2.2, -1.5, 8.4, 1.9
X_NUMBER, Y_NUMBER = 2.7, -0.9
Y_EQUATION, H_SMALL, ROW = -1.65, 0.8, 0.48  # the Z ~ H X + B row; footprint tiles and X traces stack ROW apart
X_Z, X_H, X_X, X_B, X_R, W_X = -5.6, -3.5, -0.5, 3.4, 5.0, 3.8
MOVIE_FRAMES = 100  # 8-frame means played in the first chapter, about 2 s slowed to 4
CROP = 11  # px, the patch each footprint is drawn on


class Blend(mn.Animation):
    """An image crossing from the pixels it has when the animation begins to rgb's ((h, w, 3) uint8, same shape)."""

    def __init__(self, mobject: mn.ImageMobject, rgb: np.ndarray, **kwargs) -> None:
        super().__init__(mobject, **kwargs)
        self.target = mn.ImageMobject(rgb).pixel_array.astype(float)

    def begin(self) -> None:
        self.source = self.mobject.pixel_array.astype(float)
        super().begin()

    def interpolate_mobject(self, alpha: float) -> None:
        a = self.rate_func(alpha)
        self.mobject.pixel_array = ((1 - a) * self.source + a * self.target).astype(np.uint8)


def clear_image(rgb: np.ndarray, height: float, x: float, y: float) -> mn.ImageMobject:
    """kit.image at opacity 0: a Transform source that the flight reveals."""
    mobject = image(rgb, height, x, y)
    mobject.pixel_array[..., 3] = 0
    return mobject


def pixel_point(r: float, c: float, shape: tuple, height: float, x: float, y: float) -> np.ndarray:
    """The scene point of pixel (r, c)'s center in an image of shape (rows, cols) drawn height tall at (x, y)."""
    cell = height / shape[0]
    return at(x + (c - (shape[1] - 1) / 2) * cell, y + ((shape[0] - 1) / 2 - r) * cell)


def rings(points: np.ndarray, radius: float, c: tuple, stroke_width: float = 2.0) -> mn.VGroup:
    return mn.VGroup(*[mn.Circle(radius=radius, color=color(c), stroke_width=stroke_width).move_to(pt) for pt in points])


def um_bar(panel: mn.Mobject, um: float, um_per_px: float, cell: float) -> mn.VGroup:
    """A scale bar of um under a panel's right end."""
    right = panel.get_corner(mn.DR) + at(0, -0.12)
    bar = mn.Line(right - at(um / um_per_px * cell, 0), right, color=color(kit.BAR), stroke_width=3.0)
    return mn.VGroup(bar, label(f"{um:g} µm", kit.BAR, 16).next_to(bar, mn.DOWN, buff=0.06))


def stacked(values: np.ndarray, lo: float, hi: float, x: float, y: float, width: float, height: float, c: tuple) -> tuple:
    """(path, axes) of one trace of values on [lo, hi], width x height centered at (x, y)."""
    axes = put(plain_axes(0, values.size - 1, lo, hi, width, height, visible=False), x, y)
    return path(axes, np.arange(values.size), values, c, 1.6, lo, hi), axes


def means8(x: np.ndarray) -> np.ndarray:
    """(K, T) to (K, T / 8): the 8-frame means the traces are drawn and scored on."""
    return M.average_frames(x.T).T


def number_label(text: str) -> mn.Text:
    return label(text, KEY, 44)


def d_label(d: float) -> mn.Text:
    return number_label(f"d' = {d:.1f}")


def r_small(r: float, c: tuple) -> mn.Text:
    return label(f"{r:.2f}", c, 20)


def in_sds(trace: np.ndarray) -> np.ndarray:
    """trace over its robust SD (1.4826 x the median absolute deviation), so differently filtered traces share a scale."""
    return trace / (1.4826 * np.median(np.abs(trace - np.median(trace))))


def r_with_truth(traces: np.ndarray, truth8: np.ndarray, order: np.ndarray) -> list:
    """Per synapse, r of its source's 8-frame means with its true trace."""
    return [M.correlation(traces[order[k]], truth8[k]) for k in range(len(order))]


def build(seconds: list) -> kit.Timeline:
    strip = M.simulate()
    det = M.detect(strip)
    fits = M.extract(strip, det.peaks)
    fit_low = M.extract(strip, det.peaks, lam=LAMBDA_LOW)[-1]
    order = M.match(det.peaks.astype(float), strip.centers)  # the fit's source per synapse
    if (order < 0).any():
        raise ValueError(f"detection missed a synapse: peaks {det.peaks.tolist()}, matched {order}")
    p = strip.params
    shape = (p.rows, p.cols)
    cell = H_IMAGE / p.rows
    um_per_px = 0.25
    n_ds = det.z.shape[0]
    t_ds = np.arange(n_ds) / strip.fs_ds
    r0, c0 = det.peaks[order[0]]
    onsets = strip.onsets(0)
    trace_z, trace_f = in_sds(det.z[:, r0, c0]), in_sds(det.filtered[:, r0, c0])
    d_raw = M.event_dprime(trace_z, onsets, strip.fs_ds)
    d_filtered = M.event_dprime(trace_f, onsets, strip.fs_ds)

    title = put(label(TITLE, TEXT, 30), 0.0, 3.6)
    titles = kit.titles_for(CHAPTERS)

    # the strip: the movie plays on the left, the mean image on the right, the synapses ringed; the unscanned margin is black
    averaged = np.where(strip.scanned, det.averaged, 0.0)
    onset_frames = np.flatnonzero(strip.events[0]) // M.DS
    candidates = [f for f in onset_frames if MOVIE_FRAMES <= f < n_ds - 5]
    last = int(max(candidates, key=lambda f: det.z[f, r0, c0]))
    hi_movie = float(np.percentile(averaged, 99.5))
    frames = [gray_rgb(averaged[f], 0, hi_movie) for f in range(last - MOVIE_FRAMES + 1, last + 1)]
    screen = image(frames[0], H_IMAGE, X_LEFT, Y_TOP)
    screen_edge = kit.frame_of(screen)
    mean_ds = averaged.mean(axis=0)
    hi_mean = float(mean_ds.max())
    mean_image = image(gray_rgb(mean_ds, 0, hi_mean), H_IMAGE, X_RIGHT, Y_TOP)
    mean_edge = kit.frame_of(mean_image)
    name_screen = label("Simulated dendrite, 8-frame means", DIM, 18).next_to(screen, mn.DOWN, buff=0.12).align_to(screen, mn.LEFT)
    name_mean = label("Mean image", DIM, 18).next_to(mean_image, mn.DOWN, buff=0.12).align_to(mean_image, mn.LEFT)
    truth_left = [pixel_point(r, c, shape, H_IMAGE, X_LEFT, Y_TOP) for r, c in strip.centers]
    truth_right = [pixel_point(r, c, shape, H_IMAGE, X_RIGHT, Y_TOP) for r, c in strip.centers]
    rings_left = rings(np.array(truth_left), 1.7 * cell, TRUTH)
    rings_right = rings(np.array(truth_right), 1.7 * cell, TRUTH)
    legend_truth = label("Synapses (simulated)", TRUTH, 18).next_to(mean_image, mn.DOWN, buff=0.12).align_to(mean_image, mn.RIGHT)
    bar_um = um_bar(screen, 2.0, um_per_px, cell)

    # one pixel: a box on it flies down into the zoom, and its trace unrolls out of it
    pixel = pixel_point(r0, c0, shape, H_IMAGE, X_LEFT, Y_TOP)
    box = mn.Square(side_length=1.6 * cell, color=color(KEY), stroke_width=2.0).move_to(pixel)
    box_flying = box.copy()
    z_lo = float(min(trace_z.min(), trace_f.min()))
    z_hi = float(max(trace_z.max(), trace_f.max()))
    z_lo, z_hi = z_lo - 0.05 * (z_hi - z_lo), z_hi + 0.05 * (z_hi - z_lo)
    axes_zoom = put(plain_axes(0, float(t_ds[-1]), z_lo, z_hi, W_ZOOM, H_ZOOM, visible=False), X_ZOOM, Y_ZOOM)
    frame_zoom = mn.Rectangle(width=W_ZOOM, height=H_ZOOM, color=color(DIM), stroke_width=1.5).move_to(axes_zoom)
    path_z = path(axes_zoom, t_ds, trace_z, TRACE, 1.3, z_lo, z_hi)
    path_f = path(axes_zoom, t_ds, trace_f, TRACE, 1.3, z_lo, z_hi)
    path_pixel = mn.VMobject(stroke_color=color(TRACE), stroke_width=1.3, stroke_opacity=0.0).set_points_as_corners(np.repeat(pixel[None], n_ds, axis=0))
    bars_zoom = mn.VGroup(kit.hbar(axes_zoom, float(t_ds[-1]) - 1.0, float(t_ds[-1]), z_lo, "1 s", 0.55),
                          *kit.vbar(axes_zoom, float(t_ds[-1]), z_hi - 5.0, z_hi, "5 SD"))
    tick_y = Y_ZOOM - H_ZOOM / 2 - 0.08
    ticks = mn.VGroup(*[mn.Line(axes_points(axes_zoom, np.array([s]), np.array([z_lo]))[0] * np.array([1, 0, 0]) + at(0, tick_y),
                                axes_points(axes_zoom, np.array([s]), np.array([z_lo]))[0] * np.array([1, 0, 0]) + at(0, tick_y - 0.12),
                                color=color(TRUTH), stroke_width=2.0) for s in onsets])
    legend_ticks = label("Events at this synapse (simulated)", TRUTH, 16).next_to(ticks, mn.DOWN, buff=0.06).align_to(frame_zoom, mn.LEFT)
    number = d_label(d_raw).move_to(at(X_NUMBER, Y_NUMBER), aligned_edge=mn.LEFT)

    # the matched filter: the left panel turns into the filtered frame; the kernels it uses
    filtered_frame = np.nan_to_num(det.filtered[last])
    filtered_rgb = gray_rgb(filtered_frame, 0, float(np.nanpercentile(det.filtered, 99.5)))
    name_filtered = label("Matched filter, same frame", DIM, 18).move_to(name_screen, aligned_edge=mn.LEFT)
    x_space = np.linspace(-8, 8, 161)
    sigma_big = M.SIGMA_PX * np.sqrt(26.0)  ## G_sigma, then G_5sigma of that: the wide Gaussian has sigma sqrt(1 + 25) sigma
    dog_profile = np.exp(-0.5 * (x_space / M.SIGMA_PX) ** 2) / M.SIGMA_PX - np.exp(-0.5 * (x_space / sigma_big) ** 2) / sigma_big
    dog_profile /= dog_profile.max()
    axes_space = put(plain_axes(-8, 8, -0.3, 1.0, 1.6, 0.7, visible=False), X_NUMBER + 0.9, -2.2)
    t_kernel = np.linspace(0, 6 * M.TAU_S, 61)
    axes_time = put(plain_axes(0, 6 * M.TAU_S, 0, 1.0, 1.6, 0.7, visible=False), X_NUMBER + 3.0, -2.2)
    kernels = mn.VGroup(path(axes_space, x_space, dog_profile, KEY, 2.0, -0.3, 1.0), path(axes_time, t_kernel, np.exp(-t_kernel / M.TAU_S), KEY, 2.0, 0, 1.0))
    kernel_names = mn.VGroup(label(f"space, {M.SIGMA_PX:g} px", DIM, 16).next_to(axes_space, mn.DOWN, buff=0.1),
                             label(f"time, {1000 * M.TAU_S:g} ms", DIM, 16).next_to(axes_time, mn.DOWN, buff=0.1))

    # maxima at this pixel, then pooled over every pixel into the activity image
    peaks_at = np.flatnonzero(det.maxima[:, r0, c0])
    dots_max = dots_at(axes_points(axes_zoom, t_ds[peaks_at], np.clip(trace_f[peaks_at], z_lo, z_hi)), KEY, 0.04)
    dots_home = dots_at(np.repeat(pixel[None], peaks_at.size, axis=0), KEY, 0.02, opacity=0.0)
    activity_hi = float(min(det.activity[r, c] for r, c in det.peaks))  # the weakest peak is white
    activity_rgb = gray_rgb(np.nan_to_num(det.activity), 0, activity_hi)
    name_activity = label(f"Activity image, {int(strip.scanned.sum())} pixels x {n_ds} frames", DIM, 18).move_to(name_screen, aligned_edge=mn.LEFT)

    # the peaks against the truth
    peak_points = np.array([pixel_point(r, c, shape, H_IMAGE, X_LEFT, Y_TOP) for r, c in det.peaks])
    circles = rings(peak_points, 1.2 * cell, KEY)
    rings_again = rings(np.array(truth_left), 1.7 * cell, TRUTH)
    found = int((order >= 0).sum())
    number_found = put(number_label(f"{found} of {len(strip.centers)} synapses"), 0.0, -1.2)
    caption_peaks = caption("The second-brightest pixel of each 3 x 3 clears the threshold too")

    # Z ~ H X + B on the fitted pixels, the footprints flying down from their peaks
    y_rows = [Y_EQUATION + (1.5 - k) * ROW for k in range(len(SOURCE))]
    z_frames = [gray_rgb(np.where(strip.scanned, strip.photons[f], 0.0), 0, 3.0) for f in range(MOVIE_FRAMES)]
    z_screen = image(z_frames[0], H_SMALL, X_Z, Y_EQUATION)
    z_edge = kit.frame_of(z_screen)
    every = [*fits, fit_low]
    tile_h = 0.42
    tiles = [image(footprint_rgb(fits[0], order[k], SOURCE[k], shape, det.peaks[order[k]]), tile_h, X_H, y_rows[k]) for k in range(len(SOURCE))]
    tile_flights = [clear_image(footprint_rgb(fits[0], order[k], SOURCE[k], shape, det.peaks[order[k]]), CROP * cell, *peak_points[order[k]][:2])
                    for k in range(len(SOURCE))]
    tile_edges = [kit.frame_of(t) for t in tiles]
    b_rgb = [gray_rgb(scatter(fit.B, fit.pixels, shape), 0, hi_mean) for fit in every]  # black off the fitted pixels
    b_target = image(b_rgb[0], H_SMALL, X_B, Y_EQUATION)
    b_edge = kit.frame_of(b_target)
    colored = mn.VGroup(*[mn.Circle(radius=1.2 * cell, color=color(SOURCE[k]), stroke_width=2.5).move_to(peak_points[order[k]]) for k in range(len(SOURCE))])
    x_sets = [means8(fits[0].X_ls)] + [means8(fit.X) for fit in every[1:]]
    x_lo = float(min(np.percentile(s, 0.5) for s in x_sets))
    x_hi = float(max(s.max() for s in x_sets))
    x_paths = [[stacked(s[order[k]], x_lo, x_hi, X_X, y_rows[k], W_X, 0.9 * ROW, SOURCE[k])[0] for k in range(len(SOURCE))] for s in x_sets]
    axes_x = stacked(x_sets[0][0], x_lo, x_hi, X_X, y_rows[-1], W_X, 0.9 * ROW, TRACE)[1]
    n8 = x_sets[0].shape[1]
    bars_x = mn.VGroup(kit.hbar(axes_x, n8 - 1 - strip.fs_ds, n8 - 1, x_lo, "1 s", 0.12))
    signs = mn.VGroup(label("~", DIM, 34).move_to(at(0.5 * (X_Z + 1.5 * H_SMALL + X_H - tile_h / 2), Y_EQUATION)),
                      label("x", DIM, 26).move_to(at(0.5 * (X_H + tile_h / 2 + X_X - W_X / 2), Y_EQUATION)),
                      label("+", DIM, 30).move_to(at(0.5 * (X_X + W_X / 2 + X_B - 1.5 * H_SMALL), Y_EQUATION)))
    y_names = Y_EQUATION + 2 * ROW + 0.2
    names = mn.VGroup(*[mn.VGroup(label(letter, KEY, 22), label(text, DIM, 16)).arrange(mn.RIGHT, buff=0.15, aligned_edge=mn.DOWN).move_to(at(x, y_names))
                        for letter, text, x in (("Z", "photons, 430 Hz", X_Z), ("H", "footprints", X_H),
                                                ("X", "events * exp(-t / 24 ms)", X_X), ("B", "background", X_B))])
    name_r = label("r, true trace", DIM, 16).move_to(at(X_R, y_names), aligned_edge=mn.LEFT)
    round_names = [label(text, DIM, 16).move_to(at(X_X - W_X / 2, Y_EQUATION - 2 * ROW - 0.25), aligned_edge=mn.LEFT)
                   for text in ("least squares", "round 1", "round 2", f"round 2, λ = {LAMBDA_LOW:g}")]
    truth8 = means8(strip.traces)
    r_sets = [r_with_truth(s, truth8, order) for s in x_sets]
    r_labels = [r_small(r_sets[0][k], SOURCE[k]).move_to(at(X_R, y_rows[k]), aligned_edge=mn.LEFT) for k in range(len(SOURCE))]

    # the table: each synapse's r for a plain ROI mean and the fit at both lambdas; one shaft synapse's three traces
    rois = means8(np.stack([M.roi_mean(strip, strip.centers[k], det.f0) for k in range(len(SOURCE))]))
    r_roi = [M.correlation(rois[k], truth8[k]) for k in range(len(SOURCE))]
    columns = (("ROI mean", r_roi), ("λ = 0.9", r_sets[2]), (f"λ = {LAMBDA_LOW:g}", r_sets[3]))
    x_columns, y_header, y_table = (-3.6, -2.2, -0.8), -0.75, [-1.25, -1.75, -2.25, -2.75]
    header = mn.VGroup(label("r with true trace (simulated)", DIM, 16).move_to(at(-6.9, -0.3), aligned_edge=mn.LEFT),
                       *[label(name, DIM, 16).move_to(at(x, y_header), aligned_edge=mn.LEFT) for (name, _), x in zip(columns, x_columns)])
    table_rows = [mn.VGroup(label(f"synapse {k + 1}, {KINDS[k]}", SOURCE[k], 18).move_to(at(-6.9, y), aligned_edge=mn.LEFT),
                            *[label(f"{values[k]:.2f}", TEXT, 20).move_to(at(x, y), aligned_edge=mn.LEFT) for (_, values), x in zip(columns, x_columns)])
                  for k, y in enumerate(y_table)]
    example = (rois[EXAMPLE], x_sets[2][order[EXAMPLE]], x_sets[3][order[EXAMPLE]])
    x_example, w_example, y_example = 4.3, 4.6, (-1.0, -1.75, -2.5)
    example_rows = []
    for (name, _), values, y in zip(columns, example, y_example):
        trace, axes_row = stacked(values, float(np.percentile(values, 0.5)), float(values.max()), x_example, y, w_example, 0.55,
                                  TRACE if name == "ROI mean" else SOURCE[EXAMPLE])
        example_rows.append(mn.VGroup(label(name, DIM, 16).move_to(at(0.75, y), aligned_edge=mn.LEFT), trace))
    example_name = label(f"synapse {EXAMPLE + 1}, each trace to its peak", SOURCE[EXAMPLE], 16).move_to(at(0.75, -0.3), aligned_edge=mn.LEFT)
    example_ticks = mn.VGroup(*[mn.Line(at(axes_row.c2p(s * strip.fs_ds, 0)[0], y_example[-1] - 0.36), at(axes_row.c2p(s * strip.fs_ds, 0)[0], y_example[-1] - 0.46),
                                        color=color(TRUTH), stroke_width=2.0) for s in strip.onsets(EXAMPLE)])
    legend_example = label("Events (simulated)", TRUTH, 16).move_to(at(0.75, y_example[-1] - 0.62), aligned_edge=mn.LEFT)
    bar_example = kit.hbar(axes_row, n8 - 1 - strip.fs_ds, n8 - 1, float(np.percentile(example[-1], 0.5)), "1 s", 0.55)

    print(f"strip {shape}, {p.duration:g} s at {p.fs:g} Hz, brightest 1% of pixels {np.percentile(strip.photons.mean(0), 99):.2f} photons / frame")
    print(f"peaks {det.peaks.tolist()}, matched {order.tolist()}, threshold {det.threshold:.4f}")
    print(f"d' at pixel ({r0}, {c0}): {d_raw:.2f} raw, {d_filtered:.2f} filtered; {peaks_at.size} maxima there")
    for name, rs in zip(("least squares", "round 1", "round 2", f"lambda {LAMBDA_LOW:g}", "ROI mean"), (*r_sets, r_roi)):
        print(f"r with truth, {name}: {np.round(rs, 2).tolist()}")

    captions = [None, caption("One pixel at a synapse, (F - F0) / noise SD"), None,
                caption("Each dot is above its 3 x 3 neighbours and the frames before and after")]
    zoom_parts = [path_pixel, bars_zoom, ticks, legend_ticks, box, kernels, kernel_names]
    equation = [z_screen, z_edge, b_edge, *tile_flights, *tile_edges, mean_image, signs, names, *x_paths[0], bars_x, round_names[3], name_r, *r_labels]
    chapters = [
        chapter(seconds[0],
                mn.AnimationGroup(mn.FadeIn(title), mn.FadeIn(screen), mn.FadeIn(screen_edge), mn.FadeIn(name_screen), mn.FadeIn(bar_um),
                                  mn.FadeIn(mean_image), mn.FadeIn(mean_edge), mn.FadeIn(name_mean), run_time=0.8),
                mn.AnimationGroup(Play(screen, frames, rate_func=mn.linear),
                                  mn.AnimationGroup(mn.Wait(run_time=1.0), mn.AnimationGroup(mn.FadeIn(rings_left), mn.FadeIn(rings_right), mn.FadeIn(legend_truth)),
                                                    lag_ratio=1.0),
                                  run_time=4.0),
                lag_ratio=1.0),
        chapter(seconds[1],
                mn.AnimationGroup(mn.FadeOut(title), mn.FadeOut(rings_left), mn.FadeIn(box), mn.FadeIn(captions[1]), run_time=0.6),
                mn.AnimationGroup(mn.FadeIn(box_flying, run_time=0.2),
                                  mn.AnimationGroup(mn.Transform(box_flying, frame_zoom), mn.Transform(path_pixel, path_z), run_time=1.8),
                                  mn.FadeOut(box_flying, run_time=0.3), lag_ratio=1.0),
                mn.AnimationGroup(mn.FadeIn(bars_zoom), mn.FadeIn(ticks), mn.FadeIn(legend_ticks), mn.FadeIn(number), lag_ratio=0.2),
                lag_ratio=1.0),
        chapter(seconds[2],
                mn.AnimationGroup(*replace(captions[1], titles[2]), run_time=1.0),
                mn.AnimationGroup(Blend(screen, filtered_rgb), *replace(name_screen, name_filtered), mn.Create(kernels), mn.FadeIn(kernel_names),
                                  mn.Transform(path_pixel, path_f), Count(number, d_raw, d_filtered, d_label), run_time=4.0),
                lag_ratio=1.0),
        chapter(seconds[3],
                mn.AnimationGroup(*replace(titles[2], titles[3]), mn.FadeIn(captions[3]), run_time=1.0),
                mn.LaggedStart(*[mn.FadeIn(dot, scale=1.6) for dot in dots_max], lag_ratio=0.05, run_time=2.5),
                lag_ratio=1.0),
        chapter(seconds[4],
                mn.AnimationGroup(*replace(titles[3], titles[4]), mn.FadeOut(captions[3]), run_time=1.0),
                mn.AnimationGroup(mn.LaggedStart(*[mn.Transform(dot, home) for dot, home in zip(dots_max, dots_home)], lag_ratio=0.03, run_time=2.0),
                                  mn.AnimationGroup(Blend(screen, activity_rgb), *replace(name_filtered, name_activity), run_time=2.5), lag_ratio=0.4),
                lag_ratio=1.0),
        chapter(seconds[5],
                mn.AnimationGroup(*replace(titles[4], titles[5]), mn.FadeIn(caption_peaks), *[mn.FadeOut(m) for m in (*zoom_parts, number)], run_time=1.0),
                mn.AnimationGroup(mn.Create(circles), mn.FadeIn(rings_again), mn.FadeIn(number_found), lag_ratio=0.3, run_time=2.0),
                lag_ratio=1.0),
        chapter(seconds[6],
                mn.AnimationGroup(*replace(titles[5], titles[6]), mn.FadeOut(caption_peaks), mn.FadeOut(number_found), mn.FadeOut(rings_again), mn.FadeOut(rings_right),
                                  mn.FadeOut(legend_truth), mn.FadeOut(name_mean), mn.FadeOut(mean_edge), run_time=1.0),
                mn.AnimationGroup(*[mn.Transform(circles[order[k]], colored[k]) for k in range(len(SOURCE))],
                                  *[mn.Transform(flight, tile) for flight, tile in zip(tile_flights, tiles)], mn.Transform(mean_image, b_target), run_time=1.8),
                mn.AnimationGroup(mn.FadeIn(z_screen), mn.FadeIn(z_edge), mn.FadeIn(b_edge), *[mn.FadeIn(e) for e in tile_edges], mn.FadeIn(signs), mn.FadeIn(names),
                                  *[mn.Create(pth) for pth in x_paths[0]], mn.FadeIn(bars_x), mn.FadeIn(round_names[0]), run_time=1.2),
                mn.AnimationGroup(Play(z_screen, z_frames, rate_func=mn.linear), run_time=2.4),
                lag_ratio=1.0),
        chapter(seconds[7],
                mn.AnimationGroup(*replace(titles[6], titles[7]), mn.FadeIn(name_r), *[mn.FadeIn(r) for r in r_labels], run_time=1.0),
                *[morph(i, x_paths, tile_flights, mean_image, r_labels, r_sets, round_names, every, order, shape, det.peaks, b_rgb) for i in (1, 2)],
                lag_ratio=1.0),
        chapter(seconds[8],
                mn.AnimationGroup(*replace(titles[7], titles[8]), run_time=1.0),
                morph(3, x_paths, tile_flights, mean_image, r_labels, r_sets, round_names, every, order, shape, det.peaks, b_rgb),
                lag_ratio=1.0),
        chapter(seconds[9],
                mn.AnimationGroup(mn.FadeOut(titles[8]), *[mn.FadeOut(m) for m in equation], run_time=1.0),
                mn.AnimationGroup(mn.FadeIn(header), mn.FadeIn(example_name), run_time=0.8),
                mn.LaggedStart(*[mn.FadeIn(row) for row in table_rows], lag_ratio=0.4, run_time=2.4),
                mn.LaggedStart(*[mn.AnimationGroup(mn.FadeIn(row[0]), mn.Create(row[1])) for row in example_rows], lag_ratio=0.5, run_time=2.4),
                mn.AnimationGroup(mn.FadeIn(example_ticks), mn.FadeIn(legend_example), mn.FadeIn(bar_example), run_time=0.6),
                lag_ratio=1.0),
    ]
    return timeline([], chapters)


def morph(i: int, x_paths: list, tile_flights: list, mean_image: mn.ImageMobject, r_labels: list, r_sets: list, round_names: list,
          fits: list, order: np.ndarray, shape: tuple, peaks: np.ndarray, b_rgb: list) -> mn.AnimationGroup:
    """Everything the fit draws moving from step i - 1 to step i: traces, footprints, background, r and the step's name."""
    return mn.AnimationGroup(*[mn.Transform(x_paths[0][k], x_paths[i][k]) for k in range(len(SOURCE))],
                             *[Blend(tile_flights[k], footprint_rgb(fits[i], order[k], SOURCE[k], shape, peaks[order[k]])) for k in range(len(SOURCE))],
                             Blend(mean_image, b_rgb[i]),
                             *[Count(r_labels[k], r_sets[i - 1][k], r_sets[i][k], partial(r_small, c=SOURCE[k]), 2) for k in range(len(SOURCE))],
                             *replace(round_names[i - 1], round_names[i]), run_time=2.5)


def footprint_rgb(fit, j: int, c: tuple, shape: tuple, peak: np.ndarray) -> np.ndarray:
    """Source j's footprint tinted c and scaled to its peak, the CROP x CROP pixels around its peak."""
    half = CROP // 2
    padded = np.pad(scatter(fit.H[:, j], fit.pixels, shape), half)
    r, c0 = peak
    return tinted_rgb(padded[r : r + CROP, c0 : c0 + CROP], 0, float(fit.H[:, j].max()), c)



def scatter(values: np.ndarray, pixels: np.ndarray, shape: tuple) -> np.ndarray:
    """(P,) values on the fitted pixels into an image of shape, 0 elsewhere."""
    out = np.zeros(shape)
    out[pixels[:, 0], pixels[:, 1]] = values
    return out


if __name__ == "__main__":
    kit.run("silo", CHAPTERS, UNTITLED, build, __doc__)
