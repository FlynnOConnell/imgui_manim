"""
StripRegistration's template growth on a simulated dendrite. A thin strip drifts over the tissue; the template sits on
a canvas padded by MAX_SHIFT pixels of NaN. Every aligned frame adds to the visit count and running sum of each canvas
pixel it covers, and a padding pixel joins the template as sum / count once it has THRESHOLD visits. Frames land at
their simulated true shifts: the correlation step is left out, so only the template update is modeled.
"""

from dataclasses import dataclass

import numpy as np

FOV_SHAPE = (10, 64)
MAX_SHIFT = 10
CANVAS_SHAPE = (FOV_SHAPE[0] + 2 * MAX_SHIFT, FOV_SHAPE[1] + 2 * MAX_SHIFT)
NOISE_SD = 1.6  # single-frame correlation with the tissue near 0.2, as in kg236's glutamate frames
N_FRAMES = 3000
DRIFT_SD = (4.0, 2.5)  # px, (y, x) spread of the slow drift
DRIFT_FRAMES = 400  # frames, time constant of the drift
JITTER_SD = 0.5  # px, frame-to-frame jitter on top of the drift
N_INITIAL = 1000  # frames behind the initial template
THRESHOLD = 100


@dataclass
class Recording:
    tissue: np.ndarray  ## (height, width) of the canvas
    shifts: np.ndarray  ## (n_frames, 2) integer (dy, dx) of each frame from the canvas center
    frames: np.ndarray  ## (n_frames, *FOV_SHAPE)
    template_initial: np.ndarray  ## canvas, NaN outside the field of view at shift 0


def distance_to_polyline(points: np.ndarray, yy: np.ndarray, xx: np.ndarray) -> np.ndarray:
    """
    Distance from every (yy, xx) to the nearest segment of a polyline.

    Parameters
    ----------
    points : np.ndarray
        Polyline vertices, (n_points, 2) as (y, x).
    yy, xx : np.ndarray
        Pixel coordinates, same shape.

    Returns
    -------
    np.ndarray
        Distances, shape of yy.
    """
    a, b = points[:-1], points[1:]  ## (n_segments, 2)
    p = np.stack([yy, xx], axis=-1)[..., None, :]  ## (..., 1, 2)
    ab = b - a
    t = np.clip(((p - a) * ab).sum(-1) / (ab * ab).sum(-1), 0, 1)  ## (..., n_segments)
    nearest = a + t[..., None] * ab
    return np.sqrt(((p - nearest) ** 2).sum(-1)).min(-1)


def make_tissue(seed: int = 0) -> np.ndarray:
    """
    A dendrite along the canvas with two branches and spines, background 0.08, shaft peak near 1.

    Parameters
    ----------
    seed : int
        Seeds the spine positions.

    Returns
    -------
    np.ndarray
        Canvas image, shape CANVAS_SHAPE.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : CANVAS_SHAPE[0], 0 : CANVAS_SHAPE[1]].astype(float)
    shaft = np.array([(12, -2), (16, 20), (13, 44), (17, 64), (14, 86)], float)
    branches = (np.array([(15, 26), (6, 40), (-2, 50)], float), np.array([(16, 58), (24, 70), (32, 76)], float))
    tissue = 0.08 + np.exp(-0.5 * (distance_to_polyline(shaft, yy, xx) / 1.2) ** 2)
    for branch in branches:
        tissue += 0.8 * np.exp(-0.5 * (distance_to_polyline(branch, yy, xx) / 1.0) ** 2)
    for _ in range(20):
        k = rng.integers(len(shaft) - 1)
        y, x = shaft[k] + rng.uniform(0, 1) * (shaft[k + 1] - shaft[k]) + np.array([rng.choice([-2.8, 2.8]), 0.0])
        tissue += 0.7 * np.exp(-0.5 * ((yy - y) ** 2 + (xx - x) ** 2) / 0.8**2)
    return np.minimum(tissue, 1.1)


def simulate_shifts(seed: int = 0) -> np.ndarray:
    """
    Slow drift (an Ornstein-Uhlenbeck walk, DRIFT_SD spread, DRIFT_FRAMES time constant) plus jitter, rounded to
    whole pixels and clipped to MAX_SHIFT.

    Returns
    -------
    np.ndarray
        (N_FRAMES, 2) integer (dy, dx).
    """
    rng = np.random.default_rng(seed)
    decay = 1 - 1 / DRIFT_FRAMES
    kicks = rng.standard_normal((N_FRAMES, 2)) * np.array(DRIFT_SD) * np.sqrt(1 - decay**2)
    drift = np.zeros((N_FRAMES, 2))
    for t in range(1, N_FRAMES):
        drift[t] = decay * drift[t - 1] + kicks[t]
    shifts = drift + JITTER_SD * rng.standard_normal((N_FRAMES, 2))
    return np.clip(np.round(shifts), -MAX_SHIFT, MAX_SHIFT).astype(int)


def simulate(seed: int = 0) -> Recording:
    """
    A recording of a strip drifting over the tissue, starting where the initial template was taken.

    Parameters
    ----------
    seed : int
        Seeds the tissue, motion and noise.

    Returns
    -------
    Recording
    """
    rng = np.random.default_rng(seed)
    tissue = make_tissue(seed=seed)
    shifts = simulate_shifts(seed=seed)
    frames = np.stack([tissue[window(shift=s)] for s in shifts]) + NOISE_SD * rng.standard_normal((N_FRAMES, *FOV_SHAPE))
    template_initial = np.full(CANVAS_SHAPE, np.nan)
    template_initial[window(shift=(0, 0))] = tissue[window(shift=(0, 0))] + NOISE_SD / np.sqrt(N_INITIAL) * rng.standard_normal(FOV_SHAPE)
    return Recording(tissue=tissue, shifts=shifts, frames=frames, template_initial=template_initial)


def window(shift) -> tuple:
    """The canvas slices a frame at (dy, dx) covers."""
    y0, x0 = MAX_SHIFT + int(shift[0]), MAX_SHIFT + int(shift[1])
    return slice(y0, y0 + FOV_SHAPE[0]), slice(x0, x0 + FOV_SHAPE[1])


def grow_template(recording: Recording, threshold: int) -> tuple:
    """
    Visit counts and template after every frame.

    Padding pixels show sum / count from `threshold` visits on and NaN before; pixels of the initial template average
    their N_INITIAL frames with every visit since.

    Parameters
    ----------
    recording : Recording
    threshold : int
        Visits a padding pixel needs before it joins the template.

    Returns
    -------
    counts : np.ndarray
        (n_frames, *CANVAS_SHAPE) int, visits up to and including each frame.
    templates : np.ndarray
        (n_frames, *CANVAS_SHAPE), NaN where the template has no value yet.
    """
    visited = np.zeros((len(recording.frames), *CANVAS_SHAPE), bool)
    placed = np.zeros((len(recording.frames), *CANVAS_SHAPE))
    for t, (frame, shift) in enumerate(zip(recording.frames, recording.shifts)):
        visited[(t, *window(shift=shift))] = True
        placed[(t, *window(shift=shift))] = frame
    counts = np.cumsum(visited, axis=0)
    sums = np.cumsum(placed, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        templates = np.where(counts >= threshold, sums / counts, np.nan)
    initial = ~np.isnan(recording.template_initial)
    templates[:, initial] = (N_INITIAL * recording.template_initial[initial] + sums[:, initial]) / (N_INITIAL + counts[:, initial])
    return counts, templates


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson r over the pixels where both are finite."""
    finite = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[finite], b[finite])[0, 1])
