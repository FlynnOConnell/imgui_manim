"""
A movie split the way masknmf's compression rebuilds it (CompressionArray.getitem_tensor with rescale and the trend):

    registered = background + signal + residual
    background = mean image + spatial trend @ temporal trend        (the detrender's slow drift)
    signal     = noise SD * (U @ V)                                 (the block-wise low-rank fluctuations)
    residual   = whatever is left: masknmf's CompressionResidualArray

decompose loads a movie from disk; split_movie registers and compresses it with masknmf.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import tifffile
import torch

import masknmf
import masknmf.compression.compression_strategies
import masknmf.compression.preprocessing
import masknmf.motion_correction.moco_preprocessing

DETRENDER_SECONDS = {"window": 40.0, "knots": 25.0, "sigma": 0.3}  ## masknmf's two-photon pipeline


@dataclass
class Decomposition:
    """One movie and its three pieces, all (frames, height, width) float32 in raw units."""

    name: str
    fs: float
    recorded: np.ndarray  # as loaded (flipped for a negative indicator)
    registered: np.ndarray
    shifts: np.ndarray  # (frames, 2), the (row, col) shift registration applied
    mean_image: np.ndarray  # (height, width), the trend's mean over time folded in
    trend_spatial: np.ndarray  # (height, width, trend rank); rank 0 without the detrender
    trend_temporal: np.ndarray  # (trend rank, frames), zero mean over time
    noise_sd: np.ndarray  # (height, width), what each pixel is divided by before the SVD
    background: np.ndarray
    signal: np.ndarray
    residual: np.ndarray
    rank: int
    nonzero_u: int
    range_movie: tuple  # (low, high) to show the registered movie and the background with
    limit_signed: float  # +/- this shows the signal, the residual and the fluctuations

    @property
    def n_frames(self) -> int:
        return self.registered.shape[0]

    @property
    def shape(self) -> tuple:
        return self.registered.shape[1:]

    @property
    def stored_fraction(self) -> float:
        """The values U (its nonzeros) and V hold, as a fraction of the movie's values."""
        pixels = self.shape[0] * self.shape[1]
        return (self.nonzero_u + self.rank * self.n_frames) / (pixels * self.n_frames)

    @property
    def fluctuations(self) -> np.ndarray:
        """The registered movie minus its background: what the SVD sees, before dividing by the noise SD."""
        return self.registered - self.background


def variance_shares(d: Decomposition) -> dict:
    """Each piece's variance over time, summed over pixels, as a share of the registered movie's."""
    total = float(d.registered.var(axis=0).sum())
    return {name: float(getattr(d, name).var(axis=0).sum()) / total for name in ("background", "signal", "residual")}


def lag1_correlation(frames: np.ndarray) -> np.ndarray:
    """(height, width) correlation of each pixel's trace with itself one frame later."""
    centered = frames - frames.mean(axis=0)
    numerator = (centered[1:] * centered[:-1]).sum(axis=0)
    return numerator / np.maximum((centered**2).sum(axis=0), 1e-12)


def busiest_pixel(d: Decomposition) -> tuple:
    """(row, col) where the signal varies most: a good first pixel to follow."""
    return tuple(int(v) for v in np.unravel_index(np.argmax(d.signal.std(axis=0)), d.shape))


def pixel_snr(d: Decomposition) -> np.ndarray:
    """(height, width) the signal's SD over time divided by the noise SD."""
    return d.signal.std(axis=0) / np.maximum(d.noise_sd, 1e-12)


def top_pixels(d: Decomposition, n: int, first: Optional[tuple] = None, min_distance: int = 6) -> list:
    """
    n (row, col) pixels with the highest signal SNR, at least min_distance px apart (Chebyshev), best first.

    Parameters
    ----------
    first : tuple, optional
        A pixel to lead the list, whatever its SNR; the others keep their distance from it.
    """
    snr = pixel_snr(d)
    chosen = [] if first is None else [tuple(int(v) for v in first)]
    for flat in np.argsort(snr, axis=None)[::-1]:
        if len(chosen) >= n:
            break
        candidate = tuple(int(v) for v in np.unravel_index(flat, snr.shape))
        if all(max(abs(candidate[0] - row), abs(candidate[1] - col)) >= min_distance for row, col in chosen):
            chosen.append(candidate)
    return chosen[:n]


def split_pmd(pmd, registered: np.ndarray, name: str, fs: float, recorded: np.ndarray, shifts: np.ndarray) -> Decomposition:
    """A Decomposition from a masknmf CompressionArray of the registered movie, as masknmf rebuilds it."""
    n_frames, height, width = registered.shape
    noise_sd = pmd.noise_variance_image.float()
    u, v = pmd.spatial_compressed.float(), pmd.temporal_compressed.float()
    signal = torch.sparse.mm(u, v) * noise_sd.flatten()[:, None]  ## (pixels, frames)
    mean_image = pmd.mean_image.float()
    if pmd.spatial_trend_basis is not None:
        trend_spatial, trend_temporal = pmd.spatial_trend_basis.float(), pmd.temporal_trend_basis.float()
    else:
        trend_spatial, trend_temporal = torch.zeros((height * width, 0)), torch.zeros((0, n_frames))
    background = mean_image.flatten()[:, None] + trend_spatial @ trend_temporal  ## (pixels, frames)
    sample = slice(None, None, max(1, n_frames // 300))
    signal = np.ascontiguousarray(signal.T.reshape(n_frames, height, width).cpu().numpy())
    background = np.ascontiguousarray(background.T.reshape(n_frames, height, width).cpu().numpy())
    return Decomposition(
        name=name, fs=fs, recorded=recorded, registered=registered, shifts=shifts,
        mean_image=mean_image.reshape(height, width).cpu().numpy(),
        trend_spatial=trend_spatial.reshape(height, width, -1).cpu().numpy(),
        trend_temporal=trend_temporal.cpu().numpy(),
        noise_sd=noise_sd.reshape(height, width).cpu().numpy(),
        background=background, signal=signal, residual=registered - background - signal,
        rank=int(u.shape[1]), nonzero_u=int(u._nnz()),
        range_movie=tuple(float(v) for v in np.percentile(registered[sample], [0.5, 99.8])),
        limit_signed=float(np.percentile(np.abs(signal[sample]), 99.7)),
    )


def load_movie(filepath_movie: str, negative: bool = False) -> np.ndarray:
    """
    (frames, height, width) float32 from a .tif or .npy.

    Parameters
    ----------
    negative : bool
        The indicator dims with activity (ASAP, JEDI): flips the movie so events go up.
    """
    path = Path(filepath_movie)
    frames = np.load(path) if path.suffix == ".npy" else tifffile.imread(path)
    frames = np.asarray(frames, dtype=np.float32)
    return frames.max() - frames if negative else frames


def compress_movie(frames: np.ndarray, fs: float, block_sizes: tuple = (20, 20), max_shifts: tuple = (10, 10),
                   device: str = "auto", log: Optional[Callable[[str], None]] = None) -> tuple:
    """
    Register a movie with masknmf and compress it.

    Rigid registration, then PMD in blocks with a maximin spline detrender (masknmf's two-photon defaults) and
    masknmf's shift mask as the pixel weighting, so the band registration moved in from outside the field is 0 in
    the signal.

    Parameters
    ----------
    frames : np.ndarray
        (frames, height, width), as recorded.
    fs : float
        Frame rate in Hz.
    log : callable, optional
        Called with each step's name before it runs.

    Returns
    -------
    tuple
        (pmd, recorded, registered, shifts): masknmf's CompressionArray and the float32 movies and (frames, 2) shifts.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    recorded = np.ascontiguousarray(frames, dtype=np.float32)
    n_frames, height, width = recorded.shape
    if log is not None:
        log("register")
    corrector = masknmf.RigidMotionCorrector(max_shifts=max_shifts, device=device)
    corrector.compute_template(recorded)
    registration = corrector.motion_correct(recorded)
    registered = np.ascontiguousarray(torch.as_tensor(registration[:]).cpu().numpy(), dtype=np.float32)
    shifts = registration.shifts.cpu().numpy().astype(np.float32)
    shift_mask = masknmf.motion_correction.moco_preprocessing.construct_moco_template(shifts, (height, width))
    if log is not None:
        log("compress")
    detrender = masknmf.compression.preprocessing.MaximinSplineDetrend(
        num_frames=n_frames, num_knots=max(4, int(n_frames / fs / DETRENDER_SECONDS["knots"])),
        window=int(DETRENDER_SECONDS["window"] * fs), sigma=max(2.0, DETRENDER_SECONDS["sigma"] * fs), device=device,
    )
    strategy = masknmf.compression.compression_strategies.CompressStrategy(
        block_sizes=block_sizes, max_components=20, pixel_weighting=shift_mask.astype(np.float32),
        detrender=detrender, device=device,
    )
    return strategy.compress(registered), recorded, registered, shifts


def split_movie(frames: np.ndarray, fs: float, name: str, log: Optional[Callable[[str], None]] = None) -> Decomposition:
    """compress_movie, then split_pmd; log also hears "split" before the split."""
    pmd, recorded, registered, shifts = compress_movie(frames, fs, log=log)
    if log is not None:
        log("split")
    return split_pmd(pmd, registered, name=name, fs=fs, recorded=recorded, shifts=shifts)


def decompose(filepath_movie: str, fs: float, negative: bool = False,
              log: Optional[Callable[[str], None]] = None) -> Decomposition:
    """load_movie, then split_movie; log hears each step's name before it runs."""
    if log is not None:
        log("movie")
    return split_movie(load_movie(filepath_movie, negative), fs, Path(filepath_movie).stem, log=log)
