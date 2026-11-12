# /// script
# requires-python = ">=3.11"
# dependencies = ["manim>=0.22", "numpy", "pillow<13"]
# ///
"""
The SNR of one simulated transient as a three-chapter animation: the trace, the number, then noise added while the
number counts down. The smallest complete animation on kit.py; copy both files and edit this one.

    uv run -p 3.12 example.py --out out/example [--stills-only]

Text only, no LaTeX. Needs ffmpeg and uv on PATH.
"""

import manim as mn
import numpy as np

import kit
from kit import DIM, KEY, TEXT, TRACE, Count, at, chapter, label, path, plain_axes, put, scale_bars, timeline

TITLE = "SNR: the peak of a transient over the SD of the noise"
CHAPTERS = (("", 4.0), ("", 4.0), ("Add noise", 6.0))
UNTITLED = {0: "the-trace", 1: "the-number"}  # still file names of the chapters without a title

FS = 30.0  # Hz
SECONDS = 10.0
T_EVENT = 3.0  # s
Y_MIN, Y_MAX = -1.0, 2.0  # dF/F, fixed over both noise levels so the noise moves the trace, not the axes


def simulate(sd_noise: float, seed: int = 0) -> tuple:
    """Frame times (s) and the dF/F of one transient of peak 1 decaying over 1 s, in Gaussian noise."""
    t = np.arange(int(FS * SECONDS)) / FS
    transient = np.where(t >= T_EVENT, np.exp(-(t - T_EVENT)), 0.0)
    return t, transient + np.random.default_rng(seed).normal(0.0, sd_noise, t.size)


def snr(t: np.ndarray, dff: np.ndarray) -> float:
    """The peak after the event over the SD of the frames before it."""
    return float(dff[t >= T_EVENT].max() / dff[t < T_EVENT].std())


def number_label(value: float) -> mn.Text:
    return put(label(f"SNR = {value:.1f}", KEY, 44), 0.0, -2.9)


def build(seconds: list) -> kit.Timeline:
    t, dff_quiet = simulate(sd_noise=0.05)
    _, dff_noisy = simulate(sd_noise=0.30)
    snr_quiet, snr_noisy = snr(t, dff_quiet), snr(t, dff_noisy)
    print(f"SNR {snr_quiet:.1f} at noise SD 0.05, {snr_noisy:.1f} at 0.30")

    axes = plain_axes(0.0, SECONDS, Y_MIN, Y_MAX, width=11.0, height=4.0, visible=False).move_to(at(0.0, 0.3))
    trace = path(axes, t, dff_quiet, TRACE, 2.0, Y_MIN, Y_MAX)
    trace_noisy = path(axes, t, dff_noisy, TRACE, 2.0, Y_MIN, Y_MAX)
    bars = scale_bars(axes, SECONDS, Y_MIN, Y_MAX, seconds=2.0, y_drop=0.2)
    provenance = label("One transient (simulated)", DIM, 20).next_to(axes, mn.UP, buff=0.1, aligned_edge=mn.LEFT)
    title = put(label(TITLE, TEXT, 30), 0.0, 3.6)
    titles = kit.titles_for(CHAPTERS)
    number = number_label(snr_quiet)

    chapters = [
        chapter(seconds[0], mn.FadeIn(title), mn.Create(trace), mn.FadeIn(bars), mn.FadeIn(provenance)),
        chapter(seconds[1], mn.FadeOut(title), mn.FadeIn(number)),
        ## NOTE: the title leads alone for 1 s, then the trace and the number change in one group, so the number
        ## counts while the picture morphs. trace_noisy is only a Transform target and gets no animation of its own.
        chapter(
            seconds[2],
            mn.AnimationGroup(mn.FadeIn(titles[2]), run_time=1.0),
            mn.AnimationGroup(mn.Transform(trace, trace_noisy), Count(number, snr_quiet, snr_noisy, number_label), run_time=4.0),
            lag_ratio=1.0,
        ),
    ]
    return timeline([], chapters)


if __name__ == "__main__":
    kit.run("example", CHAPTERS, UNTITLED, build, __doc__)
