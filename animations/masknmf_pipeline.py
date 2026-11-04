# /// script
# requires-python = ">=3.11"
# dependencies = ["manim>=0.19", "numpy", "matplotlib", "pillow<13"]  # pillow 13 breaks manim 0.22 ImageMobject (Image.fromarray mode=)
# ///
"""
masknmf's pipeline as manim scenes, drawn from one masknmf run on a simulated movie:
movie -> pixel trace -> registration -> block compression -> compressed movie -> seeds -> demixing.

Text only, no LaTeX. Render every scene, or some, into --out:

    uv run -p 3.12 animations/masknmf_pipeline.py --out <dir>                       # 480p15, every scene
    uv run -p 3.12 animations/masknmf_pipeline.py Registration --out <dir> -q high_quality

animations/data/masknmf_pipeline/run.npz cannot be regenerated: its exporter needed imgui_masknmf, which is gone.
"""

import argparse
import functools
from pathlib import Path

import matplotlib
import manim as mn
import numpy as np

DIR_HERE = Path(__file__).resolve().parent
FILEPATH_RUN = DIR_HERE / "data" / "masknmf_pipeline" / "run.npz"
COLOR_KEPT, COLOR_REJECTED, COLOR_PICK = mn.GREEN_C, mn.RED_C, mn.YELLOW
COLOR_ROW, COLOR_COL = mn.BLUE_C, mn.ORANGE
mn.Text.set_default(font="Noto Sans")


@functools.cache
def load_run() -> dict:
    """The arrays of the stored run."""
    return dict(np.load(FILEPATH_RUN))


def to_rgba(image: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "gray") -> np.ndarray:
    """
    uint8 RGBA of an image through a matplotlib colormap.

    Parameters
    ----------
    image : np.ndarray
        Shape (height, width).
    vmin, vmax : float, optional
        Clipping range. ``None`` takes the image's 1st and 99.5th percentiles.
    cmap : str
        A matplotlib colormap name.

    Returns
    -------
    np.ndarray
        Shape (height, width, 4), uint8.
    """
    vmin = np.percentile(image, 1) if vmin is None else vmin
    vmax = np.percentile(image, 99.5) if vmax is None else vmax
    scaled = np.clip((image - vmin) / max(vmax - vmin, 1e-12), 0, 1)
    return (matplotlib.colormaps[cmap](scaled) * 255).astype(np.uint8)


def image_mobject(image: np.ndarray, height: float, **kwargs_to_rgba) -> mn.ImageMobject:
    """An image drawn with square, unsmoothed pixels, ``height`` scene units tall."""
    rgba = image if image.ndim == 3 else to_rgba(image, **kwargs_to_rgba)
    mobject = mn.ImageMobject(rgba)
    mobject.set_resampling_algorithm(mn.RESAMPLING_ALGORITHMS["nearest"])
    return mobject.scale_to_fit_height(height)


def matrix_mobject(matrix: np.ndarray, width: float, height: float, **kwargs_to_rgba) -> mn.ImageMobject:
    """A matrix as a heatmap stretched to (width, height) scene units."""
    mobject = image_mobject(matrix, height, **kwargs_to_rgba)
    return mobject.stretch_to_fit_width(width)


def pixel_point(mobject: mn.Mobject, row: float, col: float, shape: tuple) -> np.ndarray:
    """Scene point at the center of pixel (row, col) of an image mobject showing an array of this (height, width)."""
    size_pixel = mobject.height / shape[0]
    return mobject.get_corner(mn.UL) + (col + 0.5) * size_pixel * mn.RIGHT + (row + 0.5) * size_pixel * mn.DOWN


def block_rectangle(mobject: mn.Mobject, block: np.ndarray, shape: tuple, **kwargs_style) -> mn.Rectangle:
    """The outline of a block (row, col, height, width, ...) on an image mobject."""
    row, col, height, width = (int(v) for v in block[:4])
    size_pixel = mobject.height / shape[0]
    rectangle = mn.Rectangle(width=width * size_pixel, height=height * size_pixel, **kwargs_style)
    corner = mobject.get_corner(mn.UL) + col * size_pixel * mn.RIGHT + row * size_pixel * mn.DOWN
    return rectangle.move_to(corner, aligned_edge=mn.UL)


def axes_points(axes: mn.Axes, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(n, 3) scene points of the coordinates (x, y) on linear axes."""
    origin = axes.c2p(0, 0)
    return origin + np.outer(x, axes.c2p(1, 0) - origin) + np.outer(y, axes.c2p(0, 1) - origin)


def plain_axes(x_max: float, y_min: float, y_max: float, width: float, height: float) -> mn.Axes:
    """Axes without ticks or tips, x from 0."""
    return mn.Axes(
        x_range=[0, x_max, x_max], y_range=[y_min, y_max, y_max - y_min], x_length=width, y_length=height,
        tips=False, axis_config={"include_ticks": False, "stroke_width": 2, "color": mn.GREY_B},
    )


def path_mobject(points: np.ndarray, color, stroke_width: float = 2.0) -> mn.VMobject:
    """A polyline through (n, 3) points."""
    return mn.VMobject(stroke_color=color, stroke_width=stroke_width).set_points_as_corners(points)


class PipelineScene(mn.Scene):
    """A scene with one caption at a time at the bottom of the frame."""

    caption: mn.Text = None

    def show_caption(self, text: str) -> list:
        """Animations that replace the caption with text."""
        new = mn.Text(text, font_size=26, color=mn.GREY_A).to_edge(mn.DOWN, buff=0.35)
        animations = [mn.FadeIn(new)] if self.caption is None else [mn.FadeOut(self.caption), mn.FadeIn(new)]
        self.caption = new
        return animations

    def hide_caption(self) -> list:
        animations = [] if self.caption is None else [mn.FadeOut(self.caption)]
        self.caption = None
        return animations


def label_under(mobject: mn.Mobject, text: str, font_size: float = 22) -> mn.Text:
    return mn.Text(text, font_size=font_size, color=mn.GREY_B).next_to(mobject, mn.DOWN, buff=0.15)


def label_over(mobject: mn.Mobject, text: str, font_size: float = 22) -> mn.Text:
    return mn.Text(text, font_size=font_size, color=mn.GREY_B).next_to(mobject, mn.UP, buff=0.15)


def update_movie_frame(mobject: mn.ImageMobject, frames: np.ndarray, tracker: mn.ValueTracker, vmin: float,
                       vmax: float, cmap: str = "gray") -> None:
    """Show the frame the tracker is on."""
    mobject.pixel_array = to_rgba(frames[int(tracker.get_value())], vmin=vmin, vmax=vmax, cmap=cmap)


def update_trace_path(mobject: mn.VMobject, axes: mn.Axes, x: np.ndarray, y: np.ndarray,
                      tracker: mn.ValueTracker) -> None:
    """Draw the trace up to the tracker's frame."""
    n_points = max(2, int(tracker.get_value()) + 1)
    mobject.set_points_as_corners(axes_points(axes, x[:n_points], y[:n_points]))


def update_cursor(mobject: mn.Mobject, axes: mn.Axes, x: np.ndarray, y_min: float, y_max: float,
                  tracker: mn.ValueTracker) -> None:
    """Move a vertical line to the tracker's frame."""
    t = x[int(tracker.get_value())]
    mobject.put_start_and_end_on(*axes_points(axes, np.array([t, t]), np.array([y_min, y_max])))


def update_shift_arrow(mobject: mn.Arrow, image: mn.ImageMobject, center: tuple, shifts: np.ndarray, gain: float,
                       shape: tuple, tracker: mn.ValueTracker) -> None:
    """Point from the reference pixel along the frame's correcting shift, magnified by gain."""
    shift_row, shift_col = shifts[int(tracker.get_value())] * gain
    start = pixel_point(image, center[0], center[1], shape)
    end = pixel_point(image, center[0] + shift_row, center[1] + shift_col, shape)
    if np.linalg.norm(end - start) < 1e-3:
        end = start + 1e-3 * mn.RIGHT
    mobject.put_start_and_end_on(start, end)


def update_time_label(mobject: mn.Text, fs: float, tracker: mn.ValueTracker, anchor: mn.Mobject) -> None:
    mobject.become(mn.Text(f"t = {tracker.get_value() / fs:4.1f} s", font_size=22, color=mn.GREY_B)
                   .next_to(anchor, mn.UP, buff=0.15))


def zscore_rows(matrix: np.ndarray) -> np.ndarray:
    """Each row minus its mean, over its SD."""
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    return centered / np.maximum(centered.std(axis=1, keepdims=True), 1e-12)


def signal_colors(n_signals: int) -> np.ndarray:
    """(n_signals, 3) distinct RGB colors in [0, 1]."""
    return matplotlib.colormaps["tab20"](np.arange(n_signals) % 20)[:, :3]


def footprint_composite(footprints: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """
    uint8 RGBA of every footprint at once: each pixel takes the color of its strongest signal, opaque where that
    signal is near its peak.

    Parameters
    ----------
    footprints : np.ndarray
        Shape (height, width, signals).
    colors : np.ndarray
        Shape (signals, 3).
    """
    normalized = footprints / np.maximum(footprints.max(axis=(0, 1), keepdims=True), 1e-12)  ## (height, width, signals)
    strongest = normalized.argmax(axis=2)
    alpha = np.clip(normalized.max(axis=2) * 1.4, 0, 1)
    rgba = np.dstack([colors[strongest], alpha])
    return (rgba * 255).astype(np.uint8)


class MovieToTrace(PipelineScene):
    """A movie is a stack of frames; one pixel over time is a trace."""

    def construct(self):
        run = load_run()
        frames, fs = run["raw"], float(run["fs"])
        n_frames, height, width = frames.shape
        vmin, vmax = np.percentile(frames, [1, 99.7])
        cell = int(np.argmax(run["true_traces"].std(axis=1)))
        row, col = np.unravel_index(np.argmax(run["true_footprints"][cell]), (height, width))

        title = mn.Text("a movie is a stack of images", font_size=34).to_edge(mn.UP)
        front = image_mobject(frames[0], 3.8, vmin=vmin, vmax=vmax).move_to(3.4 * mn.LEFT + 0.3 * mn.DOWN)
        stack_back = mn.Group(*[
            image_mobject(frames[30 * depth], 3.8, vmin=vmin, vmax=vmax).move_to(front).shift(0.22 * depth * mn.UR)
            .set_opacity(0.8 - 0.15 * depth) for depth in range(4, 0, -1)
        ])
        self.play(mn.Write(title), mn.LaggedStart(*[mn.FadeIn(m) for m in stack_back], lag_ratio=0.25), run_time=1.5)
        self.play(mn.FadeIn(front, shift=0.2 * mn.DL))
        self.play(mn.FadeOut(stack_back))

        tracker = mn.ValueTracker(0)
        front.add_updater(functools.partial(update_movie_frame, frames=frames, tracker=tracker, vmin=vmin, vmax=vmax))
        self.add(front)
        time_label = mn.Text("t = 0.0 s", font_size=22, color=mn.GREY_B).next_to(front, mn.UP, buff=0.15)
        time_label.add_updater(functools.partial(update_time_label, fs=fs, tracker=tracker, anchor=front))
        self.add(time_label)

        x = np.arange(n_frames) / fs
        y = frames[:, row, col]
        y_min, y_max = np.percentile(y, [0.5, 99.5])
        axes = plain_axes(x_max=x[-1], y_min=y_min, y_max=y_max, width=6.2, height=3.4).move_to(3.4 * mn.RIGHT + 0.3 * mn.DOWN)
        axes_title = label_over(axes, f"pixel ({row}, {col}), raw counts")
        axes_x = label_under(axes, f"time, 0 to {x[-1]:.0f} s")
        size_pixel = front.height / height
        marker = mn.Square(side_length=size_pixel * 1.6, color=COLOR_PICK, stroke_width=3)
        marker.move_to(pixel_point(front, row, col, (height, width)))
        link = mn.DashedLine(marker.get_right(), axes.get_left(), color=COLOR_PICK, stroke_width=1.5)
        self.play(mn.Create(marker), mn.Create(axes), mn.FadeIn(axes_title), mn.FadeIn(axes_x))
        self.play(mn.Create(link))

        trace = path_mobject(axes_points(axes, x[:2], y[:2]), COLOR_PICK)
        trace.add_updater(functools.partial(update_trace_path, axes=axes, x=x, y=y, tracker=tracker))
        self.add(trace)
        self.play(tracker.animate.set_value(n_frames - 1), run_time=9, rate_func=mn.linear)
        for mobject in (front, trace, time_label):
            mobject.clear_updaters()

        truth = run["true_traces"][cell, :n_frames]
        slope, intercept = np.polyfit(truth, y, 1)
        truth_path = path_mobject(axes_points(axes, x, slope * truth + intercept), mn.TEAL_B, stroke_width=3)
        truth_label = mn.Text("true activity", font_size=22, color=mn.TEAL_B).next_to(axes, mn.UR, buff=0.15).shift(1.4 * mn.LEFT)
        self.play(mn.Create(truth_path), mn.FadeIn(truth_label), trace.animate.set_stroke(opacity=0.5), run_time=2)
        self.play(*self.show_caption("motion, background and shot noise ride on top: the pipeline separates them"))
        self.wait(2)


class Registration(PipelineScene):
    """Every frame is shifted onto a template, so a pixel stays on one place in the tissue."""

    def construct(self):
        run = load_run()
        raw, registered, template, shifts = run["raw"], run["registered"], run["template"], run["shifts"]
        fs = float(run["fs"])
        n_frames, height, width = raw.shape
        shape = (height, width)
        vmin, vmax = np.percentile(raw, [1, 99.7])
        center = np.unravel_index(np.argmax(template), shape)

        title = mn.Text("register: line every frame up with a template", font_size=32).to_edge(mn.UP)
        panel_raw = image_mobject(raw[0], 2.9, vmin=vmin, vmax=vmax).move_to(4.4 * mn.LEFT + 0.9 * mn.UP)
        panel_template = image_mobject(template, 2.9, vmin=vmin, vmax=vmax).move_to(0.9 * mn.UP)
        panel_registered = image_mobject(registered[0], 2.9, vmin=vmin, vmax=vmax).move_to(4.4 * mn.RIGHT + 0.9 * mn.UP)
        labels = mn.VGroup(
            label_under(panel_raw, "raw"), label_under(panel_template, "template: median, refined 3 times"),
            label_under(panel_registered, "registered"),
        )
        radius_cell = 4 * panel_raw.height / height
        rings = mn.VGroup(*[
            mn.Circle(radius=radius_cell, color=COLOR_PICK, stroke_width=2).move_to(pixel_point(panel, *center, shape))
            for panel in (panel_raw, panel_template, panel_registered)
        ])
        self.play(mn.Write(title), mn.FadeIn(panel_raw), mn.FadeIn(labels[0]))
        self.play(mn.FadeIn(panel_template), mn.FadeIn(labels[1]))
        self.play(mn.Create(rings))

        tracker = mn.ValueTracker(0)
        panel_raw.add_updater(functools.partial(update_movie_frame, frames=raw, tracker=tracker, vmin=vmin, vmax=vmax))
        panel_registered.add_updater(functools.partial(update_movie_frame, frames=registered, tracker=tracker,
                                                       vmin=vmin, vmax=vmax))
        gain = 3.0
        arrow = mn.Arrow(mn.ORIGIN, mn.RIGHT, buff=0, color=mn.RED_B, stroke_width=4, max_tip_length_to_length_ratio=0.35)
        arrow.add_updater(functools.partial(update_shift_arrow, image=panel_raw, center=center, shifts=shifts,
                                            gain=gain, shape=shape, tracker=tracker))
        arrow_label = mn.Text(f"shift applied, x{gain:.0f}", font_size=18, color=mn.RED_B).next_to(panel_raw, mn.UP, buff=0.1)

        x = np.arange(n_frames) / fs
        limit = float(np.abs(shifts[:n_frames]).max()) * 1.15
        axes = plain_axes(x_max=x[-1], y_min=-limit, y_max=limit, width=11.5, height=1.7).move_to(2.3 * mn.DOWN)
        paths = mn.VGroup(*[
            path_mobject(axes_points(axes, x, shifts[:n_frames, axis]), color)
            for axis, color in ((0, COLOR_ROW), (1, COLOR_COL))
        ])
        legend = mn.VGroup(
            mn.Text("row shift", font_size=18, color=COLOR_ROW), mn.Text("column shift", font_size=18, color=COLOR_COL),
            mn.Text(f"+/-{limit:.1f} px", font_size=18, color=mn.GREY_B),
        ).arrange(mn.RIGHT, buff=0.5).next_to(axes, mn.UP, buff=0.1)
        cursor = mn.Line(mn.ORIGIN, mn.UP, color=mn.WHITE, stroke_width=2)
        cursor.add_updater(functools.partial(update_cursor, axes=axes, x=x, y_min=-limit, y_max=limit, tracker=tracker))
        self.play(mn.FadeIn(panel_registered), mn.FadeIn(labels[2]), mn.Create(axes), mn.FadeIn(legend),
                  mn.Create(paths), mn.FadeIn(arrow_label))
        self.add(arrow, cursor)
        self.play(tracker.animate.set_value(n_frames - 1), run_time=10, rate_func=mn.linear)
        for mobject in (panel_raw, panel_registered, arrow, cursor):
            mobject.clear_updaters()

        mean_raw = image_mobject(raw.mean(axis=0), 3.4, vmin=vmin, vmax=vmax).move_to(2.6 * mn.LEFT + 0.6 * mn.UP)
        mean_registered = image_mobject(registered.mean(axis=0), 3.4, vmin=vmin, vmax=vmax).move_to(2.6 * mn.RIGHT + 0.6 * mn.UP)
        corr_before, corr_after = float(run["corr_before"].mean()), float(run["corr_after"].mean())
        means = mn.Group(
            mean_raw, mean_registered, label_under(mean_raw, "mean of the raw frames"),
            label_under(mean_registered, "mean of the registered frames"),
        )
        self.play(mn.FadeOut(mn.Group(panel_raw, panel_template, panel_registered, labels, rings, arrow, arrow_label,
                                      axes, paths, legend, cursor)))
        self.play(mn.FadeIn(means))
        self.play(*self.show_caption(f"correlation with the template, frame by frame: {corr_before:.2f} -> {corr_after:.2f}"))
        self.wait(2)


class Compression(PipelineScene):
    """Each block of the movie becomes a few spatial components times their traces: Y ~ U V."""

    def construct(self):
        run = load_run()
        registered, blocks, pick = run["registered"], run["blocks"], int(run["pick"])
        n_frames, height, width = registered.shape
        shape = (height, width)
        row0, col0, block_height, block_width = (int(v) for v in blocks[pick][:4])
        kept = run["pick_kept"].astype(bool)
        n_kept = int(kept.sum())

        title = mn.Text("compress: a few smooth components per block", font_size=32).to_edge(mn.UP)
        frame = image_mobject(registered.mean(axis=0), 4.6).move_to(3.6 * mn.LEFT + 0.2 * mn.DOWN)
        outlines = mn.VGroup(*[block_rectangle(frame, b, shape, stroke_width=1.5, stroke_opacity=0.45, color=mn.BLUE_B)
                               for b in blocks])
        highlight = block_rectangle(frame, blocks[pick], shape, stroke_width=4, color=COLOR_PICK)
        self.play(mn.Write(title), mn.FadeIn(frame))
        self.play(mn.LaggedStart(*[mn.Create(o) for o in outlines], lag_ratio=0.04), run_time=2)
        self.play(*self.show_caption(f"{len(blocks)} blocks of {block_height} x {block_width} px, overlapping by half"))
        self.play(mn.Create(highlight))

        block_movie = registered[:, row0:row0 + block_height, col0:col0 + block_width]
        y_matrix = zscore_rows(block_movie.reshape(n_frames, -1).T)  ## (pixels, frames)
        crop = image_mobject(block_movie.mean(axis=0), highlight.height).move_to(highlight)
        y_mob = matrix_mobject(y_matrix, width=3.4, height=3.2, vmin=-3, vmax=3, cmap="RdBu_r").move_to(2.6 * mn.RIGHT + 0.1 * mn.UP)
        y_label = label_under(y_mob, f"Y: {y_matrix.shape[0]} pixels x {n_frames} frames, each pixel z-scored")
        self.add(crop)
        self.play(mn.FadeTransform(crop, y_mob), mn.FadeIn(y_label), run_time=1.5)
        self.wait(0.5)

        spatial_kept = run["pick_spatial"][:, :, kept].reshape(-1, n_kept)  ## (pixels, kept)
        temporal_kept = run["pick_temporal"][kept][:, :n_frames]  ## (kept, frames)
        limit_u = float(np.abs(spatial_kept).max())
        u_mob = matrix_mobject(spatial_kept, width=0.22 * n_kept, height=3.2, vmin=-limit_u, vmax=limit_u, cmap="RdBu_r")
        v_mob = matrix_mobject(zscore_rows(temporal_kept), width=3.4, height=0.22 * n_kept, vmin=-3, vmax=3, cmap="RdBu_r")
        approx, times = mn.Text("~", font_size=48), mn.Text("x", font_size=36)
        equation = mn.Group(y_mob.copy(), approx, u_mob, times, v_mob).arrange(mn.RIGHT, buff=0.3).move_to(0.2 * mn.UP)
        equation_labels = mn.VGroup(
            label_under(equation[0], "Y"), label_under(u_mob, f"U: {n_kept} images"),
            label_under(v_mob, f"V: {n_kept} traces"),
        )
        self.play(mn.FadeOut(mn.Group(frame, outlines, highlight, y_label)), *self.hide_caption())
        self.play(y_mob.animate.move_to(equation[0]), mn.FadeIn(approx), mn.FadeIn(times),
                  mn.FadeIn(u_mob, shift=0.3 * mn.LEFT), mn.FadeIn(v_mob, shift=0.3 * mn.LEFT),
                  mn.FadeIn(equation_labels), *self.show_caption("U V keeps the block's signal in a few components"),
                  run_time=1.5)
        self.wait(1.5)
        self.play(mn.FadeOut(mn.Group(y_mob, approx, times, u_mob, v_mob, equation_labels)), *self.hide_caption())
        self.candidates(run, kept)

    def candidates(self, run: dict, kept: np.ndarray) -> None:
        """The block's candidate components, kept or not, and the roughness test that decides."""
        spatial, temporal = run["pick_spatial"], run["pick_temporal"]
        n_show = min(spatial.shape[2], 10)
        tiles = mn.Group()
        for index in range(n_show):
            limit = float(np.abs(spatial[:, :, index]).max())
            tile = image_mobject(spatial[:, :, index], 1.0, vmin=-limit, vmax=limit, cmap="RdBu_r")
            sparkline_y = zscore_rows(temporal[index][None, :450])[0]
            sparkline = path_mobject(
                np.column_stack([np.linspace(-0.5, 0.5, 450), 0.12 * np.clip(sparkline_y, -3, 3), np.zeros(450)]),
                mn.GREY_A, stroke_width=1,
            ).next_to(tile, mn.DOWN, buff=0.12)
            color = COLOR_KEPT if kept[index] else COLOR_REJECTED
            border = mn.SurroundingRectangle(mn.Group(tile, sparkline), color=color, buff=0.06, stroke_width=3)
            tiles.add(mn.Group(tile, sparkline, border))
        tiles.arrange(mn.RIGHT, buff=0.18).move_to(1.7 * mn.UP).scale_to_fit_width(min(tiles.width, 13))
        self.play(mn.LaggedStart(*[mn.FadeIn(t[:2]) for t in tiles], lag_ratio=0.1), run_time=1.5)
        self.play(*self.show_caption("candidates from a randomized SVD of the block, largest first"), run_time=0.6)
        self.wait(1)

        spatial_roughness, temporal_roughness = run["pick_spatial_roughness"], run["pick_temporal_roughness"]
        noise_spatial, noise_temporal = run["noise_spatial"], run["noise_temporal"]
        all_x = np.concatenate([spatial_roughness, noise_spatial])
        all_y = np.concatenate([temporal_roughness, noise_temporal])
        x_max, y_max = float(np.percentile(all_x, 99.5)) * 1.1, float(np.percentile(all_y, 99.5)) * 1.1
        axes = plain_axes(x_max=x_max, y_min=0, y_max=y_max, width=5.5, height=2.3).move_to(1.6 * mn.DOWN + 2.2 * mn.LEFT)
        axis_labels = mn.VGroup(
            mn.Text("spatial roughness", font_size=18, color=mn.GREY_B).next_to(axes, mn.DOWN, buff=0.1),
            mn.Text("temporal roughness", font_size=18, color=mn.GREY_B).rotate(np.pi / 2).next_to(axes, mn.LEFT, buff=0.1),
        )
        noise_dots = mn.VGroup(*[
            mn.Dot(axes.c2p(min(xv, x_max), min(yv, y_max)), radius=0.025, color=mn.GREY_C)
            for xv, yv in zip(noise_spatial, noise_temporal)
        ])
        threshold_x, threshold_y = float(run["spatial_threshold"]), float(run["temporal_threshold"])
        thresholds = mn.VGroup(
            mn.DashedLine(axes.c2p(threshold_x, 0), axes.c2p(threshold_x, y_max), color=mn.WHITE, stroke_width=1.5),
            mn.DashedLine(axes.c2p(0, threshold_y), axes.c2p(x_max, threshold_y), color=mn.WHITE, stroke_width=1.5),
        )
        candidate_dots = mn.VGroup(*[
            mn.Dot(axes.c2p(min(spatial_roughness[i], x_max), min(temporal_roughness[i], y_max)), radius=0.06,
                   color=COLOR_KEPT if kept[i] else COLOR_REJECTED).set_z_index(1 if kept[i] else 0)
            for i in range(len(kept))
        ])
        legend = mn.VGroup(
            mn.Text("gray: first component of 250 pure-noise blocks", font_size=18, color=mn.GREY_B),
            mn.Text("dashed: their 5th percentile", font_size=18, color=mn.GREY_B),
            mn.Text(f"kept {int(kept.sum())} of {len(kept)}", font_size=22, color=COLOR_KEPT),
        ).arrange(mn.DOWN, aligned_edge=mn.LEFT, buff=0.15).next_to(axes, mn.RIGHT, buff=0.5)
        self.play(*self.hide_caption(), mn.Create(axes), mn.FadeIn(axis_labels), mn.FadeIn(noise_dots, lag_ratio=0.01))
        self.play(mn.Create(thresholds), mn.FadeIn(legend[:2]))
        self.play(mn.LaggedStart(*[mn.GrowFromCenter(d) for d in candidate_dots], lag_ratio=0.08),
                  mn.LaggedStart(*[mn.Create(t[2]) for t in tiles], lag_ratio=0.08), run_time=2)
        self.play(*[t[:2].animate.set_opacity(0.25) for t, k in zip(tiles, kept[:n_show]) if not k], mn.FadeIn(legend[2]))
        self.play(*self.show_caption("the rank runs through the first rough candidate: masknmf keeps that one too"))
        self.wait(2)


class CompressedMovie(PipelineScene):
    """How many components cover each pixel, and the movie rebuilt from U V next to the registered one."""

    def construct(self):
        run = load_run()
        registered, reconstruction = run["registered"], run["reconstruction"]
        u_matrix, v_matrix, rank_image = run["u"], run["v"], run["rank_image"]
        n_frames, height, width = registered.shape
        n_pixels, rank = u_matrix.shape
        n_frames_full = v_matrix.shape[1]
        nonzero_u = int(np.count_nonzero(u_matrix))
        fraction_stored = (nonzero_u + rank * n_frames_full) / (n_pixels * n_frames_full)

        title = mn.Text("the compressed movie", font_size=32).to_edge(mn.UP)
        rank_mob = image_mobject(rank_image, 3.8, vmin=0, vmax=float(rank_image.max()), cmap="viridis")
        rank_mob.move_to(3.6 * mn.LEFT + 0.3 * mn.UP)
        rank_label = label_under(rank_mob, f"components covering each pixel (max {int(rank_image.max())})")
        facts = mn.VGroup(*[
            mn.Text(line, font_size=24) for line in (
                f"{n_pixels} pixels x {n_frames_full} frames",
                f"rank {rank}, summed over the blocks",
                f"U: {n_pixels} x {rank}, {100 * nonzero_u / u_matrix.size:.0f}% nonzero",
                f"V: {rank} x {n_frames_full}",
                f"stored: {100 * fraction_stored:.1f}% of the movie",
            )
        ]).arrange(mn.DOWN, aligned_edge=mn.LEFT, buff=0.25).move_to(2.6 * mn.RIGHT + 0.3 * mn.UP)
        self.play(mn.Write(title), mn.FadeIn(rank_mob), mn.FadeIn(rank_label))
        self.play(mn.LaggedStart(*[mn.FadeIn(line, shift=0.1 * mn.RIGHT) for line in facts], lag_ratio=0.3), run_time=2.5)
        self.wait(1.5)
        self.play(mn.FadeOut(mn.Group(rank_mob, rank_label, facts)))

        residual = registered - reconstruction
        vmin, vmax = np.percentile(registered, [1, 99.7])
        limit = float(np.percentile(np.abs(residual), 99.5))
        tracker = mn.ValueTracker(0)
        panels = mn.Group()
        for frames, kwargs in (
            (registered, dict(vmin=vmin, vmax=vmax)),
            (reconstruction, dict(vmin=vmin, vmax=vmax)),
            (residual, dict(vmin=-limit, vmax=limit, cmap="RdBu_r")),
        ):
            panel = image_mobject(frames[0], 3.5, **kwargs)
            panel.add_updater(functools.partial(update_movie_frame, frames=frames, tracker=tracker, **kwargs))
            panels.add(panel)
        panels.arrange(mn.RIGHT, buff=0.5).move_to(0.2 * mn.UP)
        panel_labels = mn.VGroup(
            label_under(panels[0], "registered"), label_under(panels[1], "U V"),
            label_under(panels[2], f"residual, +/-{limit:.0f}"),
        )
        self.play(mn.FadeIn(panels), mn.FadeIn(panel_labels))
        self.play(tracker.animate.set_value(n_frames - 1), run_time=8, rate_func=mn.linear)
        self.play(*self.show_caption("what's left is noise: the cells and the background are in U V"))
        self.wait(1.5)


class SeedsAndDemix(PipelineScene):
    """Seeds where neighboring pixels move together, grown by NMF into footprints and traces."""

    def construct(self):
        run = load_run()
        corr_image, peaks, all_peaks, pure = run["corr_image"], run["peaks"], run["all_peaks"], run["pure"]
        height, width = corr_image.shape
        shape = (height, width)
        fs = float(run["fs"])

        title = mn.Text("find seeds, then demix", font_size=32).to_edge(mn.UP)
        corr_mob = image_mobject(corr_image, 4.6, vmin=0, vmax=1, cmap="magma").move_to(3.6 * mn.LEFT + 0.1 * mn.DOWN)
        corr_label = label_under(corr_mob, "correlation image")
        self.play(mn.Write(title), mn.FadeIn(corr_mob), mn.FadeIn(corr_label))
        self.play(*self.show_caption("does a pixel's activity move with its neighbors'?"))

        dots_all = mn.VGroup(*[mn.Dot(pixel_point(corr_mob, r, c, shape), radius=0.03, color=mn.GREY_B) for r, c in all_peaks])
        dots_peak = mn.VGroup(*[mn.Dot(pixel_point(corr_mob, r, c, shape), radius=0.06, color=mn.WHITE) for r, c in peaks])
        seed_points = np.argwhere(pure > 0)
        colors = signal_colors(max(len(seed_points), run["footprints"].shape[2]))
        rings = mn.VGroup(*[
            mn.Circle(radius=0.16, stroke_width=3, color=mn.WHITE).move_to(pixel_point(corr_mob, r, c, shape))
            for r, c in seed_points
        ])
        self.play(mn.FadeIn(dots_all, lag_ratio=0.02), *self.show_caption(f"{len(all_peaks)} local maxima"))
        self.play(mn.FadeIn(dots_peak), mn.FadeOut(dots_all),
                  *self.show_caption(f"{len(peaks)} above the correlation threshold"))
        self.play(mn.Create(rings), *self.show_caption(f"{len(seed_points)} seeds survive successive projection"))
        self.wait(1)

        snapshots = [run[f"iter_fp_{i}"] for i in run["iter_index"]]
        composites = [footprint_composite(fp, colors).astype(np.float32) for fp in snapshots]
        demix_mob = image_mobject(composites[0].astype(np.uint8), 4.6).move_to(3.6 * mn.RIGHT + 0.1 * mn.DOWN)
        demix_frame = mn.SurroundingRectangle(demix_mob, buff=0, color=mn.GREY_D, stroke_width=1)
        tracker = mn.ValueTracker(0)
        demix_mob.add_updater(functools.partial(update_composite, composites=composites, tracker=tracker))
        iteration_label = mn.Text("HALS iteration 0", font_size=22, color=mn.GREY_B).next_to(demix_mob, mn.DOWN, buff=0.15)
        iteration_label.add_updater(functools.partial(update_iteration_label, iterations=run["iter_index"],
                                                      tracker=tracker, anchor=demix_mob))
        self.play(mn.FadeIn(demix_mob), mn.Create(demix_frame), mn.FadeIn(iteration_label),
                  *self.show_caption("NMF grows each seed into a footprint and a trace"))
        self.play(tracker.animate.set_value(len(composites) - 1), run_time=5, rate_func=mn.smooth)
        demix_mob.clear_updaters()
        iteration_label.clear_updaters()
        self.wait(0.5)

        footprints, traces, true_traces = run["footprints"], run["traces"], run["true_traces"]
        pairs = {int(found): int(true) for true, found in run["match_pairs"]}
        mean_mob = image_mobject(run["mean_image"], 4.6).move_to(4.1 * mn.LEFT + 0.1 * mn.DOWN)
        overlay = image_mobject(footprint_composite(footprints, colors), 4.6).move_to(mean_mob).set_opacity(0.75).set_z_index(1)
        self.play(mn.FadeOut(mn.Group(corr_mob, corr_label, dots_peak, rings, demix_frame, iteration_label)),
                  mn.FadeIn(mean_mob), mn.ReplacementTransform(demix_mob, overlay))

        n_signals = traces.shape[0]
        n_points = 600
        index = np.linspace(0, traces.shape[1] - 1, n_points).astype(int)
        x = index / fs
        axes = plain_axes(x_max=x[-1], y_min=0, y_max=n_signals, width=6.4, height=5.3).move_to(3.2 * mn.RIGHT + 0.45 * mn.DOWN)
        trace_paths = mn.VGroup()
        for signal in range(n_signals):
            baseline = n_signals - signal - 0.5
            if signal in pairs:
                truth = zscore_rows(true_traces[pairs[signal]][None, index])[0]
                trace_paths.add(path_mobject(axes_points(axes, x, baseline + 0.12 * truth), mn.GREY_C, stroke_width=3))
            found = zscore_rows(traces[signal][None, index])[0]
            trace_paths.add(path_mobject(axes_points(axes, x, baseline + 0.12 * found),
                                         mn.ManimColor.from_rgb(colors[signal]), stroke_width=1.5))
        r_values = run["match_temporal_r"]
        legend = mn.Text(f"{len(pairs)} of {true_traces.shape[0]} true cells found, median trace r = "
                         f"{np.median(r_values):.2f}; gray: truth", font_size=20, color=mn.GREY_B)
        legend.next_to(axes, mn.UP, buff=0.1)
        self.play(mn.LaggedStart(*[mn.Create(p) for p in trace_paths], lag_ratio=0.05), mn.FadeIn(legend), run_time=3)
        self.play(*self.show_caption(f"footprints A ({height * width} pixels x {n_signals}) and traces C "
                                     f"({n_signals} x {traces.shape[1]} frames)"))
        self.wait(2.5)


def update_composite(mobject: mn.ImageMobject, composites: list, tracker: mn.ValueTracker) -> None:
    """Crossfade between the footprint snapshots on either side of the tracker."""
    position = tracker.get_value()
    lower = min(int(position), len(composites) - 1)
    upper = min(lower + 1, len(composites) - 1)
    weight = position - lower
    mobject.pixel_array = ((1 - weight) * composites[lower] + weight * composites[upper]).astype(np.uint8)


def update_iteration_label(mobject: mn.Text, iterations: np.ndarray, tracker: mn.ValueTracker, anchor: mn.Mobject) -> None:
    iteration = int(iterations[min(round(tracker.get_value()), len(iterations) - 1)])
    mobject.become(mn.Text(f"HALS iteration {iteration}", font_size=22, color=mn.GREY_B).next_to(anchor, mn.DOWN, buff=0.15))


SCENES = (MovieToTrace, Registration, Compression, CompressedMovie, SeedsAndDemix)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="*", default=[s.__name__ for s in SCENES])
    parser.add_argument("-q", "--quality", default="low_quality",
                        choices=["low_quality", "medium_quality", "high_quality", "production_quality"])
    parser.add_argument("--out", type=Path, required=True, help="where each scene's mp4 goes")
    parser.add_argument("--media-dir", default=str(Path.home() / ".cache" / "imgui_manim"), help="manim's partial movie files")
    args = parser.parse_args()
    scenes_by_name = {s.__name__: s for s in SCENES}
    for name in args.scenes:
        with mn.tempconfig({"quality": args.quality, "media_dir": args.media_dir, "video_dir": str(args.out),
                            "partial_movie_dir": str(Path(args.media_dir) / "partial_movie_files" / name),
                            "disable_caching": True, "verbosity": "WARNING", "progress_bar": "none"}):
            scenes_by_name[name]().render()
        print(args.out / f"{name}.mp4")


if __name__ == "__main__":
    main()
