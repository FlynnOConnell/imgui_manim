# /// script
# requires-python = ">=3.11"
# dependencies = ["imgui_ophys @ git+https://github.com/FlynnOConnell/imgui_ophys", "manim>=0.22", "numpy", "pillow<13"]
# ///
"""
StripRegistration's template as one animation, everything on screen from the first frame: a thin strip drifts over the tissue, each aligned frame is added to the
template where it landed, and a padding pixel fills toward blue with the frames that cover it until, at 100, it joins
the template. Built from the
strip_registration model; drawn with kit.py.

    uv run -p 3.12 animations/strip_registration.py --out <dir> [--stills-only] [--gif-width 720]
"""

import manim as mn
import numpy as np

import kit
from kit import ACCENT, DIM, KEY, Count, Play, at, caption, chapter, color, gray_rgb, image, label, timeline

M = kit.lesson_model("strip_registration")

CHAPTERS = (("", 22.5),)
UNTITLED = {0: "after-2000-frames"}
SEED = 1  # covers most of the padding by the last frame
N_SHOWN = 2000  # frames played
EVERY = 5  # frames per drawn frame
LO, HI = -0.1, 1.1  # template gray scale
CANVAS = (0.13, 0.13, 0.15)
HATCH = (0.24, 0.24, 0.27)
SCALE = 11.0 / M.CANVAS_SHAPE[1]  # scene units per pixel
Y_TEMPLATE = 0.2
PENDING = tuple(0.85 * np.array(ACCENT))  # a padding pixel at 99 frames

STRIPES = np.add(*np.mgrid[0 : M.CANVAS_SHAPE[0], 0 : M.CANVAS_SHAPE[1]]) % 4 == 0  ## (height, width) of the canvas


class Track(mn.Animation):
    """A mobject moving to points[0] .. points[-1] in turn over the animation."""

    def __init__(self, mobject: mn.Mobject, points: np.ndarray, **kwargs) -> None:
        super().__init__(mobject, **kwargs)
        self.points = points

    def interpolate_mobject(self, alpha: float) -> None:
        self.mobject.move_to(self.points[min(int(self.rate_func(alpha) * len(self.points)), len(self.points) - 1)])


def pending_rgb(counts: np.ndarray, stripes: np.ndarray) -> np.ndarray:
    """(..., 3) uint8 for pixels not yet in the template: hatched at 0 frames, PENDING as counts nears THRESHOLD."""
    base = np.where(stripes[..., None], np.array(HATCH), np.array(CANVAS))
    fraction = np.minimum(counts / M.THRESHOLD, 1.0)[..., None]
    return (255 * ((1 - fraction) * base + fraction * np.array(PENDING))).astype(np.uint8)


def template_rgb(template: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """The template in gray; NaN pixels show how many frames have covered them."""
    rgb = gray_rgb(np.nan_to_num(template, nan=LO), LO, HI)
    nan = np.isnan(template)
    rgb[nan] = pending_rgb(counts[nan], STRIPES[nan])
    return rgb


def strip_center(template_img: mn.ImageMobject, shift: np.ndarray) -> np.ndarray:
    """Scene point at the center of the strip at (dy, dx) on the template image."""
    return template_img.get_center() + at(shift[1] * SCALE, -shift[0] * SCALE)


def frame_label(v: float) -> mn.Text:
    return label(f"frame {int(v)}", DIM, 20)


def build(seconds: list) -> kit.Timeline:
    recording = M.simulate(seed=SEED)
    counts, templates = M.grow_template(recording=recording, threshold=M.THRESHOLD)
    shown = np.arange(EVERY - 1, N_SHOWN, EVERY)

    template_img = image(template_rgb(recording.template_initial, np.zeros(M.CANVAS_SHAPE)), M.CANVAS_SHAPE[0] * SCALE, 0.0, Y_TEMPLATE)
    template_frame = kit.frame_of(template_img)
    strip = mn.Rectangle(width=M.FOV_SHAPE[1] * SCALE, height=M.FOV_SHAPE[0] * SCALE, color=color(KEY), stroke_width=2.5)
    strip.move_to(strip_center(template_img, recording.shifts[0]))
    template_name = label("template", DIM, 18).next_to(template_frame, mn.DOWN, buff=0.12).align_to(template_frame, mn.LEFT)
    ramp = np.linspace(0, M.THRESHOLD - 1, 24)[None, :].repeat(8, axis=0)
    swatch = image(pending_rgb(ramp, STRIPES[:8, :24]), 0.22, 0, 0)
    strip_swatch = mn.Rectangle(width=0.4, height=0.2, color=color(KEY), stroke_width=2.0)
    legend = mn.Group(strip_swatch, label("Scanned frame", DIM, 16), label("NaN", DIM, 16), swatch, label(f"{M.THRESHOLD - 1} frames", DIM, 16)).arrange(mn.RIGHT, buff=0.12)
    legend[2:].shift(at(0.4, 0))
    legend.next_to(template_frame, mn.DOWN, buff=0.14).align_to(template_frame, mn.RIGHT)
    counter = frame_label(1).next_to(template_frame, mn.UP, buff=0.12).align_to(template_frame, mn.RIGHT)
    note = caption(f"a canvas pixel joins the template once {M.THRESHOLD} frames have covered it")

    template_frames = [template_rgb(templates[t], counts[t]) for t in shown]
    centers = np.array([strip_center(template_img, recording.shifts[t]) for t in shown])
    run_movie = 18.0

    chapters = [
        chapter(seconds[0],
                Play(template_img, template_frames, run_time=run_movie, rate_func=mn.linear),
                Track(strip, centers, run_time=run_movie, rate_func=mn.linear),
                Count(counter, 1, N_SHOWN, frame_label, decimals=0, run_time=run_movie, rate_func=mn.linear)),
    ]
    padding = np.isnan(recording.template_initial)
    print(f"seed {SEED}: shifts y {recording.shifts[:, 0].min()} to {recording.shifts[:, 0].max()} px, x {recording.shifts[:, 1].min()} to {recording.shifts[:, 1].max()} px; "
          f"padding joined at frames 500 / 1000 / 2000: {[int(np.isfinite(templates[t][padding]).sum()) for t in (499, 999, N_SHOWN - 1)]} of {int(padding.sum())} px")
    return timeline([template_img, template_frame, strip, template_name, legend, counter, note], chapters)


if __name__ == "__main__":
    kit.run("strip_registration", CHAPTERS, UNTITLED, build, __doc__)
