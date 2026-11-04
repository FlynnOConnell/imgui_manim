"""
masknmf's compression on a simulated two-photon movie, where the cells' true traces are known, or on a recording. The
movie, the rank test and the choice of a block are numpy here; the run itself, masknmf's compression with and without its
temporal denoiser, is animations/compression_run.py, which saves what the animation draws to run.npz beside this file.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SHAPE = (128, 128)
N_FRAMES = 30000  # 1000 s, one session; a much shorter movie leaves the denoiser undertrained
FS = 30.0  # Hz
BLOCK = 20  # px, masknmf's CompressConfig default
MAX_COMPONENTS = 20  # candidates per block, masknmf's default
N_CELLS = 40
CELL_SD = (2.4, 3.2)  # px, spread of a cell's footprint
SPIKE_RATE = 0.25  # Hz
TAU_RISE, TAU_DECAY = 0.05, 0.4  # s, GCaMP6f-like
BACKGROUND = 40.0  # photons / pixel / frame
CELL_BRIGHTNESS = (12.0, 90.0)  # photons / pixel / frame at a footprint's peak, one spike's worth, log-spaced over cells
READ_NOISE = 3.0  # photons
FILEPATH_RUN = Path(__file__).with_name("run.npz")
N_ROWS = 4  # candidates the animation draws
MAX_CELL_OFFSET = 4.5  # px, from the block's center to a followed cell's
CELL_SPACING = 10.0  # px, between the centers of two cells measured for the closing table


@dataclass
class Movie:
    frames: np.ndarray  ## (n_frames, height, width) float32
    footprints: np.ndarray  ## (n_cells, height, width), peak 1
    traces: np.ndarray  ## (n_cells, n_frames) in photons at the footprint's peak
    spikes: np.ndarray  ## (n_cells, n_frames) spike counts


def calcium_kernel(fs: float) -> np.ndarray:
    """A double exponential with peak 1, sampled at fs."""
    t = np.arange(int(6 * TAU_DECAY * fs)) / fs
    kernel = np.exp(-t / TAU_DECAY) - np.exp(-t / TAU_RISE)
    return kernel / kernel.max()


def place_cells(rng: np.random.Generator, n_cells: int, min_distance: float = 11.0, margin: float = 6.0) -> np.ndarray:
    """(n_cells, 2) (y, x) centers at least min_distance apart."""
    centers = []
    while len(centers) < n_cells:
        point = rng.uniform(margin, np.array(SHAPE) - margin)
        if all(np.linalg.norm(point - c) >= min_distance for c in centers):
            centers.append(point)
    return np.array(centers)


def simulate(seed: int = 0) -> Movie:
    """
    A movie of N_CELLS cells over a flat background with shot and read noise.

    Parameters
    ----------
    seed : int
        Seeds the cells, the spikes and the noise.

    Returns
    -------
    Movie
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0 : SHAPE[0], 0 : SHAPE[1]]
    centers = place_cells(rng=rng, n_cells=N_CELLS)
    sds = rng.uniform(*CELL_SD, N_CELLS)
    distance2 = (yy[None] - centers[:, 0, None, None]) ** 2 + (xx[None] - centers[:, 1, None, None]) ** 2
    footprints = np.exp(-distance2 / (2 * sds[:, None, None] ** 2))
    brightness = rng.permutation(np.geomspace(*CELL_BRIGHTNESS, N_CELLS))

    spikes = rng.poisson(SPIKE_RATE / FS, (N_CELLS, N_FRAMES)).astype(float)
    kernel = calcium_kernel(fs=FS)
    traces = np.stack([np.convolve(s, kernel)[:N_FRAMES] for s in spikes]) * brightness[:, None]

    baseline = BACKGROUND * (1.0 + 0.5 * footprints.sum(axis=0))
    frames = np.empty((N_FRAMES, *SHAPE), np.float32)
    for start in range(0, N_FRAMES, 1000):
        chunk = slice(start, start + 1000)
        expected = baseline[None] + np.einsum("ct,chw->thw", traces[:, chunk], footprints)
        frames[chunk] = rng.poisson(expected) + rng.normal(0.0, READ_NOISE, expected.shape)
    return Movie(frames, footprints, traces, spikes)


def temporal_roughness(traces: np.ndarray) -> np.ndarray:
    """mean |x[t-1] - 2 x[t] + x[t+1]| / mean |x| per row, masknmf's temporal_roughness_statistic. White noise scores sqrt(6)."""
    second_difference = traces[:, :-2] + traces[:, 2:] - 2 * traces[:, 1:-1]
    return np.abs(second_difference).mean(axis=1) / np.abs(traces).mean(axis=1)


def keep_mask(passed: np.ndarray, max_consecutive_failures: int = 1) -> np.ndarray:
    """Components kept: everything up to and including the first run of max_consecutive_failures, masknmf's filter_by_failures."""
    run = np.convolve((~passed).astype(float), np.ones(max_consecutive_failures))[: passed.size]
    over = (run >= max_consecutive_failures).astype(float)
    return np.cumsum(np.cumsum(over)) <= 1


def block_starts(size: int, block: int = BLOCK) -> list:
    """Block origins along one axis: a stride of half a block, plus one block flush with the far edge."""
    starts = list(range(0, size - block + 1, block - int(np.ceil(block / 2))))
    if starts[-1] != size - block:
        starts.append(size - block)
    return starts


def row_major_order(groups_y: list, groups_x: list) -> list:
    """
    Which visit holds each block, in row-major order: masknmf visits blocks tile by tile, and the two orders differ as soon
    as a movie needs more than one tile.

    Parameters
    ----------
    groups_y, groups_x : list
        The block starts of each tile along each axis, as lists of lists.

    Returns
    -------
    list
        Indices into the visits, sorted by (y, x).
    """
    visited = [(y, x) for group_y in groups_y for group_x in groups_x for y in group_y for x in group_x]
    return sorted(range(len(visited)), key=visited.__getitem__)


def load_run(filepath: Path = FILEPATH_RUN) -> dict:
    """run.npz's arrays by name, with the numbers of the run parsed under "numbers"."""
    with np.load(filepath) as saved:
        run = {name: saved[name] for name in saved.files}
    run["numbers"] = json.loads(str(run["numbers"]))
    return run


def oriented(spatial: np.ndarray, temporal: np.ndarray) -> tuple:
    """
    Components with the sign that makes each image's largest entry positive; SVD's sign is arbitrary, and u v is unchanged.

    Parameters
    ----------
    spatial : np.ndarray
        Images, (height, width, n_components).
    temporal : np.ndarray
        Traces, (n_components, n_frames).

    Returns
    -------
    tuple
        The images and traces, flipped per component, and the signs (n_components,).
    """
    flat = spatial.reshape(-1, spatial.shape[2])
    signs = np.sign(flat[np.argmax(np.abs(flat), axis=0), np.arange(flat.shape[1])])
    return spatial * signs, temporal * signs[:, None], signs


def image_moments(images: np.ndarray) -> tuple:
    """Center, (n_images, 2) in px, and spread, (n_images,) in px, of the squared positive part of (height, width, n_images) images."""
    yy, xx = np.mgrid[0 : images.shape[0], 0 : images.shape[1]]
    mass = np.clip(images, 0, None) ** 2
    mass = mass / mass.sum(axis=(0, 1))
    y, x = (mass * yy[..., None]).sum(axis=(0, 1)), (mass * xx[..., None]).sum(axis=(0, 1))
    return np.stack([y, x], axis=1), np.sqrt((mass * ((yy[..., None] - y) ** 2 + (xx[..., None] - x) ** 2)).sum(axis=(0, 1)))


def skewness(traces: np.ndarray) -> np.ndarray:
    """Skewness of each row. Transients lean positive; noise sits near 0."""
    centered = traces - traces.mean(axis=1, keepdims=True)
    return (centered**3).mean(axis=1) / (centered**2).mean(axis=1) ** 1.5


def cell_scores(spatial: np.ndarray, temporal: np.ndarray) -> np.ndarray:
    """
    How much each of a block's candidates looks like one cell: the skewness of its trace over the spread of its image.

    Parameters
    ----------
    spatial : np.ndarray
        Images, (height, width, n_components).
    temporal : np.ndarray
        Traces, (n_components, n_frames).

    Returns
    -------
    np.ndarray
        One score per candidate, (n_components,). 0 for an image centered more than MAX_CELL_OFFSET from the block's center.
    """
    images, traces, _ = oriented(spatial=spatial, temporal=temporal)
    centers, spreads = image_moments(images=images)
    offsets = np.linalg.norm(centers - (np.array(images.shape[:2]) - 1) / 2, axis=1)
    return np.where(offsets <= MAX_CELL_OFFSET, skewness(traces=traces) / spreads, 0.0)


def nice_blocks(scores: np.ndarray, kept_1: np.ndarray, kept_2: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """
    Blocks that can carry the animation, best first: a kept candidate looks like a cell, the second pass keeps more than
    the first, and every candidate kept in either pass is among the N_ROWS drawn.

    Parameters
    ----------
    scores : np.ndarray
        cell_scores of each block's first N_ROWS candidates, (n_blocks, N_ROWS).
    kept_1, kept_2 : np.ndarray
        Candidates kept by each pass, (n_blocks, n_candidates).
    allowed : np.ndarray
        Blocks to choose from, (n_blocks,).

    Returns
    -------
    np.ndarray
        Block indices.
    """
    rank_1, rank_2 = kept_1.sum(axis=1), kept_2.sum(axis=1)
    best = (scores * kept_1[:, : scores.shape[1]]).max(axis=1)
    fits = allowed & (rank_2 > rank_1) & (rank_2 <= scores.shape[1]) & (best > 0)
    return np.flatnonzero(fits)[np.argsort(-best[fits], kind="stable")]


def busiest_window(trace: np.ndarray, n_frames: int) -> int:
    """Start of the n_frames of trace with the most signal more than 3 robust SDs above its median."""
    median = np.median(trace)
    excess = np.clip(trace - median - 3 * 1.4826 * np.median(np.abs(trace - median)), 0, None)
    return int(np.argmax(np.convolve(excess, np.ones(n_frames), mode="valid")))


def noise_sd(traces: np.ndarray) -> np.ndarray:
    """Robust SD of the frame-to-frame noise of each row, from the median absolute first difference."""
    return 1.4826 * np.median(np.abs(np.diff(traces, axis=1)), axis=1) / np.sqrt(2)


def gain_on_raw(traces: np.ndarray, raw: np.ndarray, n_smooth: int = 5) -> np.ndarray:
    """
    Amplitude of each row relative to raw when no true trace exists: the slope of the row on raw, both smoothed over
    n_smooth frames so that raw's noise does not flatten the slope. Within 0.02 of the true gain on the simulated movie.
    """
    kernel = np.ones(n_smooth) / n_smooth
    x = np.stack([np.convolve(row, kernel, mode="valid") for row in raw])
    y = np.stack([np.convolve(row, kernel, mode="valid") for row in traces])
    x, y = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)
    return (x * y).sum(axis=1) / (x**2).sum(axis=1)


def rest_text(kept: np.ndarray, n_rows: int = N_ROWS) -> str:
    """What became of the candidates under the n_rows drawn."""
    n_kept = int(kept[n_rows:].sum())
    return f"Candidates {n_rows + 1} to {kept.size}: {'dropped' if n_kept == 0 else f'{n_kept} kept'}"


def to_dff(trace: np.ndarray, f0: float) -> np.ndarray:
    return (trace - f0) / f0


def spread(run: dict, name: str, key: str) -> tuple:
    """(lowest, highest) of key over the seeds of a training budget, "shipped" or "long"."""
    values = [row[key] for row in run["numbers"]["spread"][name]]
    return min(values), max(values)


def table_rows(run: dict) -> list:
    """
    The closing table, one dict per condition: the followed cell's trace, then (lowest, highest) over training seeds of
    r on that cell, median r over cells and median gain over cells. Unseeded conditions give lowest = highest.
    """
    n = run["numbers"]
    cell = n["cell"]
    rows = []
    for name, key, gain in (("raw", "trace_raw", None), ("plain", "trace_plain", n["gain"].get("plain"))):
        r = n["r"][name]
        gain_median = 1.0 if gain is None else float(np.median(gain))
        rows.append({"name": name, "trace": run[key], "r": (r[cell], r[cell]), "r_median": (float(np.median(r)),) * 2, "gain_median": (gain_median, gain_median)})
    for name, key in (("shipped", "trace_denoised"), ("long", "trace_long")):
        rows.append({"name": name, "trace": run[key], "r": spread(run, name, "r"), "r_median": spread(run, name, "r_median"), "gain_median": spread(run, name, "gain_median")})
    short = n["short"]
    rows.append({"name": "short", "trace": run["trace_short"][0], "r": (min(row["r"] for row in short), max(row["r"] for row in short)),
                 "r_median": (min(row["r_median"] for row in short), max(row["r_median"] for row in short)),
                 "gain_median": (min(row["gain_median"] for row in short), max(row["gain_median"] for row in short))})
    return rows


def table_rows_recording(run: dict) -> list:
    """
    The closing table of a recording, one dict per condition: the followed cell's trace, then (lowest, highest) over
    training seeds of noise relative to raw on that cell, its median over cells and median gain over cells.
    """
    n = run["numbers"]
    cell = n["cell"]
    rows = [{"name": "raw", "trace": run["trace_raw"], "noise": (1.0, 1.0), "noise_median": (1.0, 1.0), "gain_median": (1.0, 1.0)}]
    for name, key, stage in (("plain", "trace_plain", "plain"), ("shipped", "trace_denoised", "denoised")):
        noise, gain = n["noise"][stage], n["gain"][stage]
        rows.append({"name": name, "trace": run[key], "noise": (noise[cell], noise[cell]), "noise_median": (float(np.median(noise)),) * 2, "gain_median": (float(np.median(gain)),) * 2})
    short = n["short"]
    rows.append({"name": "short", "trace": run["trace_short"][0], **{key: (min(row[key] for row in short), max(row[key] for row in short)) for key in ("noise", "noise_median", "gain_median")}})
    return rows
