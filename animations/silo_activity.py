# /// script
# requires-python = ">=3.11"
# dependencies = ["imgui_ophys @ git+https://github.com/FlynnOConnell/imgui_ophys", "manim>=0.22", "numpy", "pillow<13"]
# ///
"""
SILo's activity image as one animation, everything on screen from the first frame: the mean image of a simulated
dendrite strip above, and below it the activity image building up frame by frame. Every maximum of the matched-filtered
movie in x, y and t flashes yellow (by its square) and adds its square to its pixel; synapses' events pile up while the
bright shaft adds nothing. The sources found at the end are circled. Built from imgui_ophys's SILo model (a port of
GIAnT-MATLAB checked against it); drawn with kit.py.

    uv run -p 3.12 animations/silo_activity.py --out <dir> [--stills-only] [--gif-width 720]
"""

from pathlib import Path

import manim as mn
import numpy as np

import kit
from kit import DIM, KEY, Count, Play, at, chapter, color, gray_rgb, image, label, timeline

M = kit.lesson_model("silo")

CHAPTERS = (("", 28.5),)  # 80% of it is the 21.5 s movie and the rings
UNTITLED = {0: "sources-found"}
CANVAS = (0.13, 0.13, 0.15)
FLASH_FULL = 4.0  # filtered z at which a maximum's flash is fully yellow; the median maximum (0.45) barely shows
FLASH_FADE = 0.6  # a flash keeps this share per drawn frame, so a one-frame maximum stays visible ~0.25 s
WIDTH = 8.4  # scene units per strip
Y_MEAN, Y_ACTIVITY = 2.2, -1.2


def activity_rgb(activity: np.ndarray, flash: np.ndarray, hi: float) -> np.ndarray:
    """The activity image in gray from 0 to hi, each pixel blended toward KEY by flash (0 to 1), NaN flat."""
    rgb = gray_rgb(np.nan_to_num(activity), 0.0, hi).astype(float)
    rgb += flash[..., None] * (255 * np.array(KEY) - rgb)
    rgb[np.isnan(activity)] = 255 * np.array(CANVAS)
    return rgb.astype(np.uint8)


def time_label(v: float) -> mn.Text:
    return label(f"{v:.1f} s", DIM, 20)


def ring(img: mn.ImageMobject, pixel: np.ndarray, rows: int) -> mn.Circle:
    """A circle around one pixel of img."""
    scale = img.height / rows
    center = img.get_corner(mn.UL) + at((pixel[1] + 0.5) * scale, -(pixel[0] + 0.5) * scale)
    return mn.Circle(radius=1.6 * scale, color=color(KEY), stroke_width=2.5).move_to(center)


def build(seconds: list) -> kit.Timeline:
    strip = M.simulate()
    detection = M.detect(strip)
    rows, cols = strip.params.rows, strip.params.cols
    height = WIDTH * rows / cols
    filtered = np.nan_to_num(detection.filtered)
    squares = np.where(detection.maxima, filtered**2, 0.0)  ## (frames, rows, cols)
    tau_frames = M.TAU_S * strip.fs_ds
    n_searched = filtered.shape[0] - 1 - int(np.ceil(1.5 * tau_frames))
    sums = np.cumsum(squares, axis=0) / (M.ACTIVITY_REG + n_searched)
    sums[:, ~strip.scanned] = np.nan
    hi = float(sums[-1][tuple(detection.peaks.T)].min())  ## the weakest source's pixel is full white
    flashes = np.clip(squares / FLASH_FULL**2, 0.0, 1.0)
    for t in range(1, len(flashes)):
        flashes[t] = np.maximum(flashes[t], FLASH_FADE * flashes[t - 1])

    mean = np.where(strip.scanned, detection.averaged.mean(axis=0), np.nan)
    mean_rgb = gray_rgb(np.nan_to_num(mean), 0.0, float(np.nanmax(mean)))
    mean_rgb[~strip.scanned] = (255 * np.array(CANVAS)).astype(np.uint8)
    mean_img = image(mean_rgb, height, 0.0, Y_MEAN)
    activity_img = image(activity_rgb(np.where(strip.scanned, 0.0, np.nan), np.zeros((rows, cols)), hi), height, 0.0, Y_ACTIVITY)
    frames_of = [kit.frame_of(m) for m in (mean_img, activity_img)]
    mean_name = label("mean image (simulated)", DIM, 18).next_to(frames_of[0], mn.DOWN, buff=0.12).align_to(frames_of[0], mn.LEFT)
    activity_name = label("activity image", DIM, 18).next_to(frames_of[1], mn.DOWN, buff=0.12).align_to(frames_of[1], mn.LEFT)
    flash_swatch = mn.Square(side_length=0.2, fill_color=color(KEY), fill_opacity=1.0, stroke_width=0)
    ring_swatch = mn.Circle(radius=0.11, color=color(KEY), stroke_width=2.0)
    legend = mn.Group(flash_swatch, label("maximum in x, y, t: adds its square", DIM, 16), ring_swatch, label("source", DIM, 16)).arrange(mn.RIGHT, buff=0.12)
    legend[2:].shift(at(0.4, 0))
    legend.next_to(frames_of[1], mn.DOWN, buff=0.14).align_to(frames_of[1], mn.RIGHT)
    counter = time_label(0).next_to(frames_of[1], mn.UP, buff=0.12).align_to(frames_of[1], mn.RIGHT)
    rings = mn.VGroup(*[ring(activity_img, peak, rows) for peak in detection.peaks])

    duration = filtered.shape[0] / strip.fs_ds
    run_movie = filtered.shape[0] / kit.FPS  ## one detection frame per drawn frame, so no flash is skipped
    activity_frames = [activity_rgb(s, f, hi) for s, f in zip(sums, flashes)]

    chapters = [
        chapter(seconds[0],
                mn.AnimationGroup(Play(activity_img, activity_frames, run_time=run_movie, rate_func=mn.linear),
                                  Count(counter, 0, duration, time_label, decimals=1, run_time=run_movie, rate_func=mn.linear)),
                mn.Create(rings, run_time=1.0), lag_ratio=1.0),
    ]
    found = M.match(detection.peaks, strip.centers)
    print(f"{filtered.shape[0]} detection frames ({duration:.1f} s at {strip.fs_ds:.0f} Hz) over {run_movie:.1f} s; {int(detection.maxima.sum())} maxima; "
          f"sources {detection.peaks.tolist()} against synapses {strip.centers.round(1).tolist()}, matched {found.tolist()}")
    return timeline([mean_img, activity_img, *frames_of, mean_name, activity_name, legend, counter], chapters)


if __name__ == "__main__":
    kit.run("silo_activity", CHAPTERS, UNTITLED, build, __doc__)
