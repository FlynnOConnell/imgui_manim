"""
masknmf's compression run, on the simulated movie or a recording, saved for animations/compression.py: unmodified
CompressStrategy and CompressDenoiseStrategy, with module functions wrapped only to record what they compute. Needs masknmf,
torch and mbo_utilities, so it runs in its own environment and the animation reads the npz:

    python animations/compression_run.py [--out <npz>]

A recording is read through mbo_utilities and registered first. The compression is cached, so choosing another block takes
seconds:

    ... animations/compression_run.py --input <movie> --block auto|<y>,<x> --out <npz> [--cache <dir>]
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

import masknmf
import mbo_utilities
from masknmf.compression import compression_strategies, decomposition, denoising

matplotlib.use("Agg")

DIR_LESSON = Path(__file__).resolve().parents[1] / "src" / "imgui_manim" / "lessons" / "compression"
spec = importlib.util.spec_from_file_location("compression_model", DIR_LESSON / "model.py")
M = importlib.util.module_from_spec(spec)
sys.modules["compression_model"] = M
spec.loader.exec_module(M)

## NOTE: a 512 x 512 px, 30000 frame movie ran out of a 32 GB card with masknmf's default budget and with 12 GB: the budget leaves out what is already allocated when a tile loads
KWARGS_STRATEGY = {"block_sizes": (M.BLOCK, M.BLOCK), "device": "cuda", "device_memory_gb": 5.0}
SEED = 0
SEEDS_SHORT = (0, 1, 2)
MOVIE_START, MOVIE_FRAMES, MOVIE_STEP = 300, 200, 2  # the frames the panels play; a recording starts just before a transient
MOVIE_LEAD = 70  # frames played before the transient
TRACE_FRAMES = 1800  # 60 s of the followed cell: the first of a simulated movie, the busiest of a recording
ZOOM_FRAMES = 360  # the 12 s the blind spot sweeps over
PANEL_SIDE = 128  # px, of the window the panels show of a recording
N_TABLE_CELLS = 40  # cells of a recording measured for the closing table
N_SHEET_BLOCKS = 12
NOISE_QUANTILE = 0.3  # CompressDenoiseStrategy's noise_variance_quantile
PADDING = 100  # CompressionTemporalDenoiser's
KWARGS_TRAIN_SHIPPED = {"learning_rate": 1e-4, "max_epochs": 10, "batch_size": 128}  # CompressDenoiseStrategy's: it fixes the learning rate and batch size, and num_epochs defaults to 10
KWARGS_TRAIN_LONG = {"learning_rate": 1e-3, "max_epochs": 30, "batch_size": 128}  # not masknmf's
SEEDS_SPREAD = (0, 1, 2, 3, 4)
SHORT_FRAMES = 3000  # the short recording: 100 s of an 80 x 80 px crop
SHORT_SHAPE = (80, 80)
MAX_SHIFTS = (15, 15)  # px, the masknmf pipeline's default in mbo_utilities
READ_BATCH = 300  # frames
FILENAME_DENOISER = "denoiser.pt"
FILENAME_RUN = "run.json"
FOOTPRINT_FLOOR = 0.2  # of the image's peak


class Recorder:
    """Wraps masknmf's module functions for one compression_routine call and keeps what they saw, block by block."""

    def __init__(self) -> None:
        self.originals = {}
        self.noise_temporal, self.noise_spatial = [], []
        self.thresholds = None
        self.blocks = []  # per block: spatial stats, temporal stats, kept mask
        self.inside_threshold = False
        self.plan = None

    def __enter__(self):
        for name in ("threshold_heuristic", "evaluate_fitness", "filter_by_failures", "temporal_roughness_statistic", "spatial_roughness_statistic", "plan_tiles"):
            self.originals[name] = getattr(decomposition, name)
            setattr(decomposition, name, getattr(self, name))
        return self

    def __exit__(self, *exception) -> None:
        for name, original in self.originals.items():
            setattr(decomposition, name, original)

    def plan_tiles(self, *args, **kwargs):
        self.plan = self.originals["plan_tiles"](*args, **kwargs)
        return self.plan

    def blocks_row_major(self, shape: tuple) -> list:
        """The blocks of a (height, width) field of view in row-major order, which is what block_origins lists."""
        groups_y, groups_x = [decomposition._tile_groups(M.block_starts(size), n) for size, n in zip(shape, self.plan.data_batching)]
        return [self.blocks[k] for k in M.row_major_order(groups_y=groups_y, groups_x=groups_x)]

    def temporal_roughness_statistic(self, traces):
        stats = self.originals["temporal_roughness_statistic"](traces)
        if self.inside_threshold:
            self.noise_temporal.append(stats.cpu().numpy())
        return stats

    def spatial_roughness_statistic(self, comps):
        stats = self.originals["spatial_roughness_statistic"](comps)
        if self.inside_threshold:
            self.noise_spatial.append(stats.cpu().numpy())
        return stats

    def threshold_heuristic(self, *args, **kwargs):
        self.inside_threshold = True
        self.thresholds = self.originals["threshold_heuristic"](*args, **kwargs)
        self.inside_threshold = False
        return self.thresholds

    def evaluate_fitness(self, spatial, temporal, spatial_threshold, temporal_threshold):
        self.blocks.append({"spatial": spatial.cpu().numpy(), "temporal": temporal.cpu().numpy(),
                            "spatial_stats": self.originals["spatial_roughness_statistic"](spatial).cpu().numpy(),
                            "temporal_stats": self.originals["temporal_roughness_statistic"](temporal).cpu().numpy()})
        return self.originals["evaluate_fitness"](spatial, temporal, spatial_threshold, temporal_threshold)

    def filter_by_failures(self, decisions, max_consecutive_failures):
        kept = self.originals["filter_by_failures"](decisions, max_consecutive_failures)
        self.blocks[-1]["passed"], self.blocks[-1]["kept"] = decisions.cpu().numpy(), kept.cpu().numpy()
        return kept


class RecorderTwoPass:
    """Gives each compression_routine call of a strategy its own Recorder, and keeps its result, the trained model and its budget."""

    def __init__(self) -> None:
        self.passes = []
        self.results = []
        self.training = {}

    def __enter__(self):
        self.routine = decomposition.compression_routine
        self.train = denoising.train_total_variance_denoiser
        compression_strategies.compression_routine = self.compression_routine
        masknmf.compression.compression_routine = self.compression_routine
        denoising.train_total_variance_denoiser = self.train_total_variance_denoiser
        return self

    def __exit__(self, *exception) -> None:
        compression_strategies.compression_routine = self.routine
        masknmf.compression.compression_routine = self.routine
        denoising.train_total_variance_denoiser = self.train

    def compression_routine(self, *args, **kwargs):
        with Recorder() as recorder:
            result = self.routine(*args, **kwargs)
        self.passes.append(recorder)
        self.results.append(result)
        return result

    def train_total_variance_denoiser(self, *args, **kwargs):
        model, dataset = self.train(*args, **kwargs)
        batches = int(np.ceil(len(dataset) / kwargs["batch_size"]))
        self.training = {"model": model, "windows": len(dataset), "batch_size": kwargs["batch_size"], "epochs": kwargs["max_epochs"],
                         "learning_rate": kwargs["learning_rate"], "steps": batches * kwargs["max_epochs"]}
        return model, dataset


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def read_recording(filepath: Path, uint16: bool, max_frames: int | None) -> tuple:
    """
    A single-plane, single-channel recording through mbo_utilities.

    Parameters
    ----------
    filepath : Path
        Any file mbo_utilities.imread opens.
    uint16 : bool
        Read int16 pixels as the uint16 they hold; a Bruker export wraps pixels above 32767 to negative values.
    max_frames : int, optional
        Reads only the first max_frames.

    Returns
    -------
    tuple
        Frames, float32 (n_frames, height, width), and the frame rate in Hz.
    """
    array = mbo_utilities.imread(filepath)
    height, width = array.shape[-2:]
    if int(np.prod(array.shape[1:-2])) != 1:
        raise ValueError(f"{filepath} has shape {array.shape}; expected one plane and one channel")
    n_frames = array.shape[0] if max_frames is None else min(max_frames, array.shape[0])
    frames = np.empty((n_frames, height, width), np.float32)
    for start in tqdm(range(0, n_frames, READ_BATCH), desc="reading"):
        stop = min(start + READ_BATCH, n_frames)
        chunk = np.asarray(array[start:stop]).reshape(stop - start, height, width)
        frames[start:stop] = chunk.view(np.uint16) if uint16 else chunk
    return frames, float(array.fs)


def registered(frames: np.ndarray) -> tuple:
    """Frames after masknmf's rigid registration, the pipeline's default, and the shifts, (n_frames, 2) in px."""
    corrector = masknmf.RigidMotionCorrector(max_shifts=MAX_SHIFTS, batch_size=READ_BATCH, device=KWARGS_STRATEGY["device"])
    corrector.compute_template(frames)
    moco = corrector.motion_correct(frames, target_movie=frames)
    out = np.empty_like(frames)
    for start in tqdm(range(0, frames.shape[0], READ_BATCH), desc="registering"):
        stop = min(start + READ_BATCH, frames.shape[0])
        out[start:stop] = moco[start:stop].cpu().numpy()
    return out, moco.shifts.cpu().numpy()


def padded(record: np.ndarray, axis: int, n: int) -> np.ndarray:
    """record with zeros after its candidates along axis, up to n."""
    widths = [(0, 0)] * record.ndim
    widths[axis] = (0, n - record.shape[axis])
    return np.pad(record, widths)


def save_pass(recorder: Recorder, shape: tuple, dir_cache: Path, name: str) -> None:
    """
    One compression_routine call's per-block records, in row-major order. Pass 2 cuts each block's candidates at 99% of the
    denoised traces' variance, so blocks are padded to the longest and n_candidates says how many are real. spatial and
    temporal get their own files, so that one block reads from a memory map.
    """
    blocks = recorder.blocks_row_major(shape=shape)
    n_candidates = np.array([block["temporal"].shape[0] for block in blocks])
    n_most = int(n_candidates.max())
    np.save(dir_cache / f"{name}_spatial.npy", np.stack([padded(block["spatial"], 2, n_most) for block in blocks]))
    np.save(dir_cache / f"{name}_temporal.npy", np.stack([padded(block["temporal"], 0, n_most) for block in blocks]))
    per_block = {key: np.stack([padded(block[key], 0, n_most) for block in blocks]) for key in ("spatial_stats", "temporal_stats", "passed", "kept")}
    np.savez(dir_cache / f"{name}.npz", thresholds=np.array([float(x) for x in recorder.thresholds]), noise_temporal=np.concatenate(recorder.noise_temporal),
             noise_spatial=np.concatenate(recorder.noise_spatial), n_candidates=n_candidates, **per_block)


def load_pass(dir_cache: Path, name: str) -> dict:
    """What save_pass wrote: spatial (n_blocks, height, width, n_candidates) and temporal (n_blocks, n_candidates, n_frames) memory-mapped."""
    with np.load(dir_cache / f"{name}.npz") as saved:
        record = {key: saved[key] for key in saved.files}
    for key in ("spatial", "temporal"):
        record[key] = np.load(dir_cache / f"{name}_{key}.npy", mmap_mode="r")
    return record


def compress_into(dir_cache: Path, frames: np.ndarray, details: dict) -> None:
    """
    CompressDenoiseStrategy on frames, recorded into dir_cache; run.json goes last, so its presence marks a finished cache.

    Parameters
    ----------
    dir_cache : Path
        Where the frames, both passes, both compressed movies and the trained denoiser go.
    frames : np.ndarray
        The movie, (n_frames, height, width).
    details : dict
        Saved in run.json with the training budget.
    """
    dir_cache.mkdir(parents=True, exist_ok=True)
    np.save(dir_cache / "frames.npy", frames)
    seed_everything(SEED)
    with RecorderTwoPass() as recorded:
        compression_strategies.CompressDenoiseStrategy(**KWARGS_STRATEGY).compress(frames)
    n_blocks = len(M.block_starts(frames.shape[1])) * len(M.block_starts(frames.shape[2]))
    for name, recorder, array in zip(("pass_1", "pass_2"), recorded.passes, recorded.results):
        if len(recorder.blocks) != n_blocks:
            raise RuntimeError(f"expected {n_blocks} blocks, saw {len(recorder.blocks)}")
        save_pass(recorder=recorder, shape=frames.shape[1:], dir_cache=dir_cache, name=name)
        array.export(dir_cache / f"{name}.hdf5")
    torch.save(recorded.training["model"].state_dict(), dir_cache / FILENAME_DENOISER)
    training = {k: v for k, v in recorded.training.items() if k != "model"}
    (dir_cache / FILENAME_RUN).write_text(json.dumps({**details, "training": training}, indent=1))


def frames_at(array, start: int, stop: int) -> np.ndarray:
    """Frames of a numpy array or a CompressionArray on any device."""
    frames = array[start:stop]
    return frames.cpu().numpy() if torch.is_tensor(frames) else np.asarray(frames)


def cell_traces(array, weights: np.ndarray, batch: int = 3000) -> np.ndarray:
    """(n_cells, n_frames): a movie's footprint-weighted average per cell. weights is (n_cells, height * width)."""
    n_frames = array.shape[0]
    return np.concatenate([weights @ frames_at(array, s, min(s + batch, n_frames)).reshape(-1, weights.shape[1]).T for s in range(0, n_frames, batch)], axis=1)


def score(traces: np.ndarray, truth: np.ndarray, raw: np.ndarray) -> tuple:
    """Per cell: r with the true trace, and gain, the slope on the true trace relative to the raw movie's slope."""
    r = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(traces, truth)])
    gain = np.array([np.polyfit(b, a, 1)[0] / np.polyfit(b, c, 1)[0] for a, b, c in zip(traces, truth, raw)])
    return r, gain


def weights_of(footprints: np.ndarray) -> np.ndarray:
    return (footprints / footprints.sum(axis=(1, 2), keepdims=True)).reshape(footprints.shape[0], -1).astype(np.float32)


def stored_share(array) -> float:
    """Values U and V hold over the values of the movie."""
    n_frames, height, width = array.shape
    return float((array.spatial_compressed._nnz() + array.temporal_compressed.numel()) / (n_frames * height * width))


def block_origins(shape: tuple) -> list:
    """(y, x) of every block of a field of view, in masknmf's row-major order."""
    return [(y, x) for y in M.block_starts(shape[0]) for x in M.block_starts(shape[1])]


def parse_block(text: str, origins: list) -> tuple:
    """The (y, x) of "y,x", which must be a block's origin."""
    origin = tuple(int(v) for v in text.split(","))
    if origin not in origins:
        raise ValueError(f"no block starts at {origin}; rows and columns start at {sorted({y for y, _ in origins})}")
    return origin


def retrain(frames: np.ndarray, plain, truth: np.ndarray, raw_traces: np.ndarray, weights: np.ndarray, cell: int, kwargs_train: dict, seed: int) -> dict:
    """
    Pass 2 again on a fixed pass 1, with the denoiser trained on kwargs_train from seed.

    Parameters
    ----------
    frames : np.ndarray
        The movie.
    plain : CompressionArray
        Pass 1.
    truth, raw_traces : np.ndarray
        Per cell, (n_cells, n_frames): the true traces and the raw movie's.
    weights : np.ndarray
        Footprint weights, (n_cells, height * width).
    cell : int
        The followed cell.
    kwargs_train : dict
        learning_rate, max_epochs and batch_size.
    seed : int
        Seeds torch and numpy.

    Returns
    -------
    dict
        The followed cell's trace, its r and gain, the median r and gain over cells, rank and gradient steps.
    """
    seed_everything(seed)
    model, dataset = denoising.train_total_variance_denoiser(plain.temporal_compressed.cpu(), **kwargs_train)
    array = masknmf.compression.compression_routine(frames, temporal_denoiser=denoising.CompressionTemporalDenoiser(model, NOISE_QUANTILE), **KWARGS_STRATEGY)
    traces = cell_traces(array, weights)
    r, gain = score(traces, truth, raw_traces)
    return {"seed": seed, "trace": traces[cell], "r": float(r[cell]), "gain": float(gain[cell]), "r_median": float(np.median(r)), "gain_median": float(np.median(gain)),
            "rank": int(array.temporal_compressed.shape[0]), "steps": int(np.ceil(len(dataset) / kwargs_train["batch_size"])) * kwargs_train["max_epochs"]}


def pick_block(footprint: np.ndarray, kept_1: np.ndarray, kept_2: np.ndarray, origins: list) -> tuple:
    """The (y, x) of the block holding the most of footprint among those pass 2 keeps more components in than pass 1."""
    rows = []
    for k, (y, x) in enumerate(origins):
        share = float(footprint[y : y + M.BLOCK, x : x + M.BLOCK].sum() / footprint.sum())
        rows.append((share, int(kept_1[k].sum()), int(kept_2[k].sum()), y, x))
    for share, rank_1, rank_2, y, x in sorted(rows, reverse=True)[:12]:
        print(f"block ({y}, {x}): {100 * share:.0f}% of the cell, rank {rank_1} then {rank_2}")
    gained = [row for row in rows if row[2] > row[1]]
    if not gained:
        raise RuntimeError("no block gained a component")
    return max(gained)[3:]


def block_scores(record: dict) -> np.ndarray:
    """(n_blocks, N_ROWS): model.cell_scores of the candidates the animation draws, in every block of a pass."""
    return np.stack([M.cell_scores(spatial=np.asarray(spatial[:, :, : M.N_ROWS]), temporal=np.asarray(temporal[: M.N_ROWS]))
                     for spatial, temporal in zip(tqdm(record["spatial"], desc="scoring blocks"), record["temporal"])])


def footprint_of(record: dict, index: int, k: int, origin: tuple, shape: tuple) -> np.ndarray:
    """Candidate k of a block as a footprint in the field of view: its image above FOOTPRINT_FLOOR of the peak."""
    images, _, _ = M.oriented(spatial=np.asarray(record["spatial"][index]), temporal=np.asarray(record["temporal"][index]))
    image = images[:, :, k]
    footprint = np.zeros(shape, np.float32)
    footprint[origin[0] : origin[0] + M.BLOCK, origin[1] : origin[1] + M.BLOCK] = np.where(image > FOOTPRINT_FLOOR * image.max(), image, 0)
    return footprint


def distinct_cells(scores: np.ndarray, record: dict, origins: list, shape: tuple, first: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """
    The best candidates of a recording as cells, CELL_SPACING apart.

    Parameters
    ----------
    scores : np.ndarray
        block_scores, (n_blocks, N_ROWS).
    record : dict
        Pass 1.
    origins : list
        Block origins.
    shape : tuple
        The field of view.
    first : np.ndarray
        The followed cell's footprint, (height, width).
    allowed : np.ndarray
        Blocks to choose from, (n_blocks,).

    Returns
    -------
    np.ndarray
        Footprints, (n_cells, height, width), the followed cell first.
    """
    footprints = [first]
    centers = [M.image_moments(images=first[..., None])[0][0]]
    best = np.where(allowed, (scores * record["kept"][:, : M.N_ROWS]).max(axis=1), 0.0)
    for index in np.argsort(-best):
        if len(footprints) == N_TABLE_CELLS or best[index] == 0:
            break
        footprint = footprint_of(record=record, index=int(index), k=int(np.argmax(scores[index] * record["kept"][index, : M.N_ROWS])), origin=origins[index], shape=shape)
        center = M.image_moments(images=footprint[..., None])[0][0]
        if min(np.linalg.norm(center - c) for c in centers) >= M.CELL_SPACING:
            footprints.append(footprint)
            centers.append(center)
    return np.stack(footprints)


def save_block_sheet(filepath: Path, order: np.ndarray, chosen: int, scores: np.ndarray, record_1: dict, record_2: dict, origins: list) -> None:
    """The best blocks with their first candidates and the best one's trace, each named by the value --block takes."""
    shown = [chosen, *[int(i) for i in order if i != chosen][: N_SHEET_BLOCKS - 1]]
    figure, axes = plt.subplots(len(shown), M.N_ROWS + 1, figsize=(12, 1.5 * len(shown)), width_ratios=[1] * M.N_ROWS + [5], squeeze=False)
    for row, index in zip(axes, shown):
        images, traces, _ = M.oriented(spatial=np.asarray(record_1["spatial"][index]), temporal=np.asarray(record_1["temporal"][index]))
        for k in range(M.N_ROWS):
            peak = np.abs(images[:, :, k]).max()
            row[k].imshow(images[:, :, k], cmap="RdBu_r", vmin=-peak, vmax=peak)
            row[k].axis("off")
        k_cell = int(np.argmax(scores[index]))
        row[M.N_ROWS].plot(traces[k_cell], lw=0.3, color="k")
        row[M.N_ROWS].axis("off")
        y, x = origins[index]
        row[M.N_ROWS].set_title(f"--block {y},{x}: rank {int(record_1['kept'][index].sum())} then {int(record_2['kept'][index].sum())}, score {scores[index].max():.2f}, candidate {k_cell + 1}"
                                f"{' (chosen)' if index == chosen else ''}", fontsize=9, loc="left")
    figure.tight_layout()
    figure.savefig(filepath, dpi=100)
    plt.close(figure)


def check_block_place(block: dict, frames: np.ndarray, origin: tuple, n_frames: int = 3000) -> None:
    """Candidate 1's trace is the movie projected on its image, which holds only where the block really sits."""
    y, x = origin
    movie = np.asarray(frames[:n_frames, y : y + M.BLOCK, x : x + M.BLOCK], dtype=np.float64).reshape(-1, M.BLOCK**2)
    movie = (movie - movie.mean(axis=0)) / movie.std(axis=0)
    r = abs(np.corrcoef(movie @ block["spatial"][:, :, 0].ravel(), block["temporal"][0, :n_frames])[0, 1])
    if not r >= 0.9:
        raise RuntimeError(f"the block recorded for {origin} does not match the movie there: |r| = {r:.3f}")


def check_model(block: dict) -> None:
    """The model's numpy versions give masknmf's numbers on a recorded block."""
    if not np.allclose(M.temporal_roughness(traces=block["temporal"]), block["temporal_stats"], rtol=1e-3):
        raise RuntimeError("model.temporal_roughness differs from masknmf's")
    if not np.array_equal(M.keep_mask(passed=block["passed"]), block["kept"]):
        raise RuntimeError("model.keep_mask differs from filter_by_failures")


def blend_parts(model: torch.nn.Module, traces: torch.Tensor) -> tuple:
    """
    One denoiser call taken apart, in the traces' own units, checked against what CompressionTemporalDenoiser returns.

    Parameters
    ----------
    model : torch.nn.Module
        The trained network.
    traces : torch.Tensor
        Shape (n_traces, n_frames).

    Returns
    -------
    tuple
        Traces, network prediction, predicted total variance, weight on the prediction and blended trace, each
        (n_traces, n_frames).
    """
    padded_traces = denoising.blind_spot_safe_pad(traces, PADDING, radius=model.receptive_radius)
    blended, prediction, total_variance = denoising.denoise_batched(model, padded_traces, noise_variance_quantile=NOISE_QUANTILE, input_size=3000, input_batch_size=20)
    means = padded_traces.mean(dim=1, keepdim=True)
    norms = torch.linalg.norm(padded_traces - means, dim=1, keepdim=True)
    noise_variance = torch.quantile(total_variance, NOISE_QUANTILE, dim=1, keepdim=True)
    weight = noise_variance / total_variance  ## total_variance is already clamped at noise_variance
    if not torch.allclose(weight * (prediction * norms + means) + (1 - weight) * padded_traces, blended, rtol=1e-3, atol=1e-3 * float(norms.max()) / padded_traces.shape[1] ** 0.5):
        raise RuntimeError("the blend does not reproduce the denoiser's output")
    inner = slice(PADDING, -PADDING)
    return tuple(a[:, inner].cpu().numpy() for a in (padded_traces, prediction * norms + means, total_variance * norms**2, weight, blended))


def quantized(frames: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """uint8 frames between lo and hi."""
    return (255 * np.clip((frames - lo) / (hi - lo), 0, 1)).astype(np.uint8)


def block_at(record: dict, index: int) -> dict:
    """One block of a pass, without its padding."""
    n = int(record["n_candidates"][index])
    return {"spatial": np.asarray(record["spatial"][index, :, :, :n]), "temporal": np.asarray(record["temporal"][index, :n]),
            **{key: record[key][index, :n] for key in ("spatial_stats", "temporal_stats", "passed", "kept")}}


def read_cache(dir_cache: Path) -> dict:
    """What compress_into saved, the compressed movies on the GPU."""
    details = json.loads((dir_cache / FILENAME_RUN).read_text())
    model = denoising.TotalVarianceTemporalDenoiser(learning_rate=details["training"]["learning_rate"], max_epochs=details["training"]["epochs"])
    model.load_state_dict(torch.load(dir_cache / FILENAME_DENOISER, map_location=KWARGS_STRATEGY["device"]))
    arrays = [masknmf.CompressionArray.from_hdf5(dir_cache / f"{name}.hdf5", device=KWARGS_STRATEGY["device"]) for name in ("pass_1", "pass_2")]
    return {"details": details, "frames": np.load(dir_cache / "frames.npy", mmap_mode="r"), "pass_1": load_pass(dir_cache=dir_cache, name="pass_1"),
            "pass_2": load_pass(dir_cache=dir_cache, name="pass_2"), "plain": arrays[0], "denoised": arrays[1], "model": model.to(KWARGS_STRATEGY["device"]).eval()}


def short_runs(frames: np.ndarray, footprints: np.ndarray, cell: int, center: np.ndarray, start: int) -> dict:
    """
    The same movie, cut down: same code and defaults, fewer and shorter traces to train on.

    Parameters
    ----------
    frames : np.ndarray
        The movie.
    footprints : np.ndarray
        Cell footprints, (n_cells, height, width).
    cell : int
        The followed cell.
    center : np.ndarray
        The crop's center, (y, x).
    start : int
        First frame of the crop.

    Returns
    -------
    dict
        The crop's origin, the cells inside it, the followed cell's position among them and one row per seed with the
        inside cells' traces, the rank and the gradient steps.
    """
    shape = frames.shape[1:]
    crop_y, crop_x = [int(np.clip(c - s / 2, 0, n - s)) for c, s, n in zip(center, SHORT_SHAPE, shape)]
    window_y, window_x = slice(crop_y, crop_y + SHORT_SHAPE[0]), slice(crop_x, crop_x + SHORT_SHAPE[1])
    short = np.ascontiguousarray(frames[start : start + SHORT_FRAMES, window_y, window_x])
    cells = np.flatnonzero(footprints[:, window_y, window_x].sum(axis=(1, 2)) > 0.95 * footprints.sum(axis=(1, 2)))
    weights = weights_of(footprints[cells, window_y, window_x])
    rows = []
    for seed in SEEDS_SHORT:
        seed_everything(seed)
        with RecorderTwoPass() as recorded:
            compression_strategies.CompressDenoiseStrategy(**KWARGS_STRATEGY).compress(short)
        rows.append({"seed": seed, "traces": cell_traces(recorded.results[1], weights), "rank": int(recorded.results[1].temporal_compressed.shape[0]), "steps": recorded.training["steps"]})
    return {"crop": [crop_y, crop_x], "cells": cells, "cell": int(np.flatnonzero(cells == cell)[0]), "rows": rows}


def followed_in_simulation(movie: M.Movie, cache: dict, block: str, cell: int | None) -> dict:
    """The cell plain compression loses most, or the one asked for, and its block."""
    origins = block_origins(shape=M.SHAPE)
    weights = weights_of(movie.footprints)
    raw_traces, plain_traces = cell_traces(movie.frames, weights), cell_traces(cache["plain"], weights)
    r_plain, _ = score(plain_traces, movie.traces, raw_traces)
    cell = int(np.argmin(r_plain)) if cell is None else cell
    origin = pick_block(footprint=movie.footprints[cell], kept_1=cache["pass_1"]["kept"], kept_2=cache["pass_2"]["kept"], origins=origins) if block == "auto" else parse_block(text=block, origins=origins)
    return {"cell": cell, "origin": origin, "k_cell": 0, "footprints": movie.footprints, "center": np.array(np.unravel_index(np.argmax(movie.footprints[cell]), M.SHAPE))}


def followed_in_recording(cache: dict, block: str, dir_sheet: Path, stem: str) -> dict:
    """The block asked for or the nicest one, its most cell-like candidate as the followed cell, and the cells for the table."""
    frames, record_1, record_2 = cache["frames"], cache["pass_1"], cache["pass_2"]
    shape = frames.shape[1:]
    origins = block_origins(shape=shape)
    scores = block_scores(record=record_1)
    margin = int(np.ceil(cache["details"]["max_shift"]))
    allowed = np.array([margin <= y <= shape[0] - M.BLOCK - margin and margin <= x <= shape[1] - M.BLOCK - margin for y, x in origins])
    order = M.nice_blocks(scores=scores, kept_1=record_1["kept"], kept_2=record_2["kept"], allowed=allowed)
    print(f"{order.size} of {len(origins)} blocks can carry the animation")
    if block == "auto":
        if order.size == 0:
            raise RuntimeError("no block fits; choose one with --block y,x")
        index = int(order[0])
    else:
        index = origins.index(parse_block(text=block, origins=origins))
    k_cell = int(np.argmax(scores[index] * record_1["kept"][index, : M.N_ROWS]))
    for i in [index, *[i for i in order[:N_SHEET_BLOCKS] if i != index]]:
        print(f"block {origins[i]}{' (chosen)' if i == index else ''}: rank {int(record_1['kept'][i].sum())} then {int(record_2['kept'][i].sum())}, score {scores[i].max():.3f}")
    save_block_sheet(filepath=dir_sheet / f"{stem}_blocks.png", order=order, chosen=index, scores=scores, record_1=record_1, record_2=record_2, origins=origins)
    first = footprint_of(record=record_1, index=index, k=k_cell, origin=origins[index], shape=shape)
    footprints = distinct_cells(scores=scores, record=record_1, origins=origins, shape=shape, first=first, allowed=allowed)
    return {"cell": 0, "origin": origins[index], "k_cell": k_cell, "footprints": footprints, "center": np.round(M.image_moments(images=first[..., None])[0][0]).astype(int)}


def export(cache: dict, followed: dict, movie: M.Movie | None, name: str, filepath_out: Path) -> None:
    """
    Writes what the animation draws.

    Parameters
    ----------
    cache : dict
        read_cache's.
    followed : dict
        followed_in_simulation's or followed_in_recording's.
    movie : M.Movie, optional
        The simulated movie; None for a recording, which has no true traces.
    name : str
        The movie's name on screen.
    filepath_out : Path
        The npz.
    """
    frames, plain, denoised, record_1, record_2, model = (cache[key] for key in ("frames", "plain", "denoised", "pass_1", "pass_2", "model"))
    n_frames, shape = frames.shape[0], frames.shape[1:]
    fs = cache["details"]["fs"]
    cell, footprints, center, k_cell = followed["cell"], followed["footprints"], followed["center"], followed["k_cell"]
    block_y, block_x = followed["origin"]
    origins = block_origins(shape=shape)
    index = origins.index((block_y, block_x))
    block_1, block_2 = block_at(record=record_1, index=index), block_at(record=record_2, index=index)
    check_model(block_1)
    check_block_place(block=block_1, frames=frames, origin=(block_y, block_x))

    weights = weights_of(footprints)
    raw_traces, plain_traces, denoised_traces = cell_traces(frames, weights), cell_traces(plain, weights), cell_traces(denoised, weights)

    x, prediction, total_variance, weight, blended = blend_parts(model=model, traces=torch.as_tensor(block_1["temporal"], device=KWARGS_STRATEGY["device"]))

    if movie is None:
        trace_start = M.busiest_window(trace=denoised_traces[cell], n_frames=TRACE_FRAMES)
        _, oriented_temporal, _ = M.oriented(spatial=block_1["spatial"], temporal=block_1["temporal"])
        component = oriented_temporal[k_cell, trace_start : trace_start + TRACE_FRAMES]
        zoom_start = M.busiest_window(trace=component, n_frames=ZOOM_FRAMES)
        peak = trace_start + int(np.argmax(component))
        movie_start = int(np.clip(peak - MOVIE_LEAD * MOVIE_STEP, 0, n_frames - MOVIE_FRAMES * MOVIE_STEP))
        short_start = int(np.clip(trace_start, 0, n_frames - SHORT_FRAMES))
    else:
        trace_start, zoom_start, movie_start, short_start = 0, 60, MOVIE_START, 0
    n_trace = slice(trace_start, trace_start + TRACE_FRAMES)

    spread = {}
    if movie is not None:
        spread = {name_budget: [retrain(frames=frames, plain=plain, truth=movie.traces, raw_traces=raw_traces, weights=weights, cell=cell, kwargs_train=kwargs_train, seed=seed) for seed in SEEDS_SPREAD]
                  for name_budget, kwargs_train in (("shipped", KWARGS_TRAIN_SHIPPED), ("long", KWARGS_TRAIN_LONG))}
        for name_budget, rows in spread.items():
            for row in rows:
                print(name_budget, {k: v for k, v in row.items() if k != "trace"})

    short = short_runs(frames=frames, footprints=footprints, cell=cell, center=center, start=short_start)
    cells_short, cell_short = short["cells"], short["cell"]
    short_rows = []
    for row in short["rows"]:
        trace = row["traces"][cell_short]
        summary = {"seed": row["seed"], "trace": trace, "n_cells": int(cells_short.size), "rank": row["rank"], "steps": row["steps"]}
        if movie is None:
            raw_short = raw_traces[cells_short, short_start : short_start + SHORT_FRAMES]
            noise = M.noise_sd(traces=row["traces"]) / M.noise_sd(traces=raw_short)
            summary |= {"noise": float(noise[cell_short]), "noise_median": float(np.median(noise)), "gain_median": float(np.median(M.gain_on_raw(traces=row["traces"], raw=raw_short)))}
        else:
            r, gain = score(row["traces"], movie.traces[cells_short, :SHORT_FRAMES], raw_traces[cells_short, :SHORT_FRAMES])
            summary |= {"r": float(r[cell_short]), "gain": float(gain[cell_short]), "r_median": float(np.median(r)), "gain_median": float(np.median(gain))}
        short_rows.append(summary)
        print({k: v for k, v in summary.items() if k != "trace"})

    frames_shown = np.arange(movie_start, movie_start + MOVIE_FRAMES * MOVIE_STEP, MOVIE_STEP)
    raw_shown = np.stack([np.asarray(frames[int(t)]) for t in frames_shown])
    plain_frames = np.stack([frames_at(plain, int(t), int(t) + 1)[0] for t in frames_shown])
    denoised_frames = np.stack([frames_at(denoised, int(t), int(t) + 1)[0] for t in frames_shown])
    block_slices = (slice(block_y, block_y + M.BLOCK), slice(block_x, block_x + M.BLOCK))
    mean_block, noise_block = [np.asarray(a.cpu())[block_slices] for a in (plain.mean_image, plain.noise_variance_image)]
    block_movie = (np.asarray(frames[n_trace, block_slices[0], block_slices[1]]) - mean_block) / noise_block
    if movie is None:
        movie_lo, movie_hi = float(np.percentile(plain_frames, 0.1)), float(np.percentile(plain_frames, 99.95))
    else:
        movie_lo, movie_hi = float(np.percentile(raw_shown, 0.1)), float(np.percentile(raw_shown, 99.95))
    panel_side = min(PANEL_SIDE, *shape) if movie is None else max(shape)
    panel_y, panel_x = [int(np.clip(c - panel_side / 2, 0, n - panel_side)) for c, n in zip(center, shape)] if movie is None else (0, 0)
    temporal_shown = block_1["temporal"][: M.N_ROWS, n_trace]
    training = cache["details"]["training"]
    numbers = {
        "movie_name": name, "simulated": movie is not None, "shape": list(shape), "fs": fs, "n_frames": n_frames, "cell": cell, "k_cell": k_cell, "block": [block_y, block_x], "center": [int(c) for c in center],
        "crop": short["crop"], "n_blocks": len(origins), "panel": [panel_y, panel_x, panel_side], "trace_start": trace_start, "zoom_frames": [zoom_start, zoom_start + ZOOM_FRAMES],
        "row_range": float(1.05 * np.abs(temporal_shown).max()), "movie_range": [movie_lo, movie_hi], "rank_1": int(plain.temporal_compressed.shape[0]), "rank_2": int(denoised.temporal_compressed.shape[0]),
        "stored_1": stored_share(plain), "stored_2": stored_share(denoised),
        "thresholds_1": record_1["thresholds"].tolist(), "thresholds_2": record_2["thresholds"].tolist(),
        "training": training, "receptive_radius": int(model.receptive_radius), "short": [{k: v for k, v in row.items() if k != "trace"} for row in short_rows],
    }
    arrays = {}
    if movie is None:
        numbers["noise"] = {key: (M.noise_sd(traces=traces) / M.noise_sd(traces=raw_traces)).tolist() for key, traces in (("plain", plain_traces), ("denoised", denoised_traces))}
        numbers["gain"] = {key: M.gain_on_raw(traces=traces, raw=raw_traces).tolist() for key, traces in (("plain", plain_traces), ("denoised", denoised_traces))}
    else:
        r_raw, _ = score(raw_traces, movie.traces, raw_traces)
        r_plain, gain_plain = score(plain_traces, movie.traces, raw_traces)
        r_denoised, gain_denoised = score(denoised_traces, movie.traces, raw_traces)
        numbers |= {"r": {"raw": r_raw.tolist(), "plain": r_plain.tolist(), "denoised": r_denoised.tolist()}, "gain": {"plain": gain_plain.tolist(), "denoised": gain_denoised.tolist()},
                    "spread": {key: [{k: v for k, v in row.items() if k != "trace"} for row in rows] for key, rows in spread.items()},
                    "kwargs_train": {"shipped": KWARGS_TRAIN_SHIPPED, "long": KWARGS_TRAIN_LONG}}
        arrays = {"spikes": movie.spikes[cell, n_trace], "truth": (movie.traces[cell, n_trace] * float(weights[cell] @ footprints[cell].ravel())).astype(np.float32),
                  "trace_long": spread["long"][0]["trace"][n_trace]}
    print(json.dumps({k: v for k, v in numbers.items() if k not in ("r", "gain", "noise")}, indent=1))

    short_slice = slice(trace_start - short_start, trace_start - short_start + TRACE_FRAMES)
    np.savez_compressed(
        filepath_out, numbers=json.dumps(numbers), frames_shown=frames_shown, mean_image=np.mean(frames[:: max(1, n_frames // 3000)], axis=0),
        raw_frames=quantized(raw_shown, movie_lo, movie_hi), plain_frames=quantized(plain_frames, movie_lo, movie_hi), denoised_frames=quantized(denoised_frames, movie_lo, movie_hi),
        block_y=quantized(block_movie.reshape(TRACE_FRAMES, -1).T, -3.0, 3.0), footprint=footprints[cell].astype(np.float32),
        trace_raw=raw_traces[cell, n_trace], trace_plain=plain_traces[cell, n_trace], trace_denoised=denoised_traces[cell, n_trace],
        trace_short=np.stack([row["trace"][short_slice] for row in short_rows]),
        ranks_1=record_1["kept"].sum(axis=1), ranks_2=record_2["kept"].sum(axis=1),
        noise_temporal_1=record_1["noise_temporal"], noise_spatial_1=record_1["noise_spatial"], noise_temporal_2=record_2["noise_temporal"], noise_spatial_2=record_2["noise_spatial"],
        block_spatial_1=block_1["spatial"], block_temporal_1=block_1["temporal"][:, n_trace].astype(np.float32),
        block_spatial_2=block_2["spatial"], block_temporal_2=block_2["temporal"][:, n_trace].astype(np.float32),
        temporal_stats_1=block_1["temporal_stats"], spatial_stats_1=block_1["spatial_stats"], passed_1=block_1["passed"], kept_1=block_1["kept"],
        temporal_stats_2=block_2["temporal_stats"], spatial_stats_2=block_2["spatial_stats"], passed_2=block_2["passed"], kept_2=block_2["kept"],
        blend_input=x[:, n_trace].astype(np.float32), blended=blended[:, n_trace].astype(np.float32), prediction=prediction[:, n_trace].astype(np.float32),
        total_variance=total_variance[:, n_trace].astype(np.float32), weight=weight[:, n_trace].astype(np.float32), **arrays,
    )
    print(f"{filepath_out}: {filepath_out.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out", type=Path, default=None, help="the npz; the default is the lesson's run.npz for the simulated movie, run.npz in the cache for a recording")
    parser.add_argument("--cell", type=int, default=None, help="the simulated movie's followed cell; the default picks the one plain compression loses most")
    parser.add_argument("--input", type=Path, default=None, help="a recording mbo_utilities can read; the default simulates a movie")
    parser.add_argument("--block", default="auto", help="y,x of the block to follow, or auto; a recording also gets <out>_blocks.png, the best blocks to choose from")
    parser.add_argument("--name", default=None, help="the movie's name on screen; the default is the file's")
    parser.add_argument("--uint16", action="store_true", help="read int16 pixels as uint16: a Bruker export wraps pixels above 32767 to negative values")
    parser.add_argument("--max-frames", type=int, default=None, help="use only the first frames of the recording")
    parser.add_argument("--cache", type=Path, default=None, help="holds the compression; the default sits beside the recording")
    parser.add_argument("--compress-only", action="store_true", help="fill the cache and stop")
    args = parser.parse_args()
    print(masknmf.__file__, mbo_utilities.__file__)

    if args.input is None:
        movie = M.simulate(seed=SEED)
        dir_cache = args.cache if args.cache is not None else Path.home() / ".cache" / "imgui_manim" / "compression_simulated"
        filepath_out = args.out if args.out is not None else M.FILEPATH_RUN
        if not (dir_cache / FILENAME_RUN).exists():
            compress_into(dir_cache=dir_cache, frames=movie.frames, details={"input": None, "fs": M.FS, "max_shift": 0.0})
    else:
        movie = None
        dir_cache = args.cache if args.cache is not None else args.input.with_name(f"{args.input.stem}_compression")
        filepath_out = args.out if args.out is not None else dir_cache / "run.npz"
        if not (dir_cache / FILENAME_RUN).exists():
            frames, fs = read_recording(filepath=args.input, uint16=args.uint16, max_frames=args.max_frames)
            print(f"{args.input}: {frames.shape}, {fs:.2f} Hz, mean {frames.mean():.0f}, max {frames.max():.0f}")
            frames, shifts = registered(frames=frames)
            print(f"shifts: y {shifts[:, 0].min():.1f} to {shifts[:, 0].max():.1f} px, x {shifts[:, 1].min():.1f} to {shifts[:, 1].max():.1f} px")
            compress_into(dir_cache=dir_cache, frames=frames, details={"input": str(args.input), "fs": fs, "uint16": args.uint16, "max_frames": args.max_frames, "max_shift": float(np.abs(shifts).max())})
    if args.compress_only:
        return

    cache = read_cache(dir_cache=dir_cache)
    if movie is None:
        followed = followed_in_recording(cache=cache, block=args.block, dir_sheet=filepath_out.parent, stem=filepath_out.stem)
    else:
        followed = followed_in_simulation(movie=movie, cache=cache, block=args.block, cell=args.cell)
    name = "Simulated two-photon movie" if movie is not None else args.name if args.name is not None else args.input.stem
    export(cache=cache, followed=followed, movie=movie, name=name, filepath_out=filepath_out)


if __name__ == "__main__":
    main()
