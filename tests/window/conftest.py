import numpy as np
import pytest
import tifffile

from imgui_manim.window import style
from imgui_manim.window.decomposition import busiest_pixel, compress_movie, split_pmd

FS = 30.0


def simulate_movie(n_frames: int = 1800, size: int = 64, n_cells: int = 60, seed: int = 0) -> np.ndarray:
    """(frames, size, size) float32 photon counts: Gaussian cells with calcium transients on a flat field."""
    rng = np.random.default_rng(seed)
    rows, cols = np.indices((size, size))
    centers = rng.uniform(8, size - 8, (n_cells, 2))
    distances = (rows - centers[:, 0, None, None]) ** 2 + (cols - centers[:, 1, None, None]) ** 2
    footprints = np.exp(-distances / (2 * 3.0**2))  ## (cells, size, size)
    spikes = rng.random((n_cells, n_frames)) < 0.5 / FS
    kernel = np.exp(-np.arange(int(2 * FS)) / (0.5 * FS))
    traces = np.stack([np.convolve(s, kernel)[:n_frames] for s in spikes])  ## (cells, frames)
    return rng.poisson(100.0 + 60.0 * np.einsum("cf,cij->fij", traces, footprints)).astype(np.float32)


@pytest.fixture(scope="session")
def run():
    """A simulated movie, registered and compressed: (pmd, recorded, registered, shifts)."""
    return compress_movie(simulate_movie(), FS)


@pytest.fixture(scope="session")
def d(run):
    pmd, recorded, registered, shifts = run
    return split_pmd(pmd, registered, name="simulated", fs=FS, recorded=recorded, shifts=shifts)


@pytest.fixture(scope="session")
def manim_ready():
    style.setup_manim()


@pytest.fixture(scope="session")
def pixel(d):
    return busiest_pixel(d)


@pytest.fixture(scope="session")
def filepath_movie(tmp_path_factory):
    """A short simulated movie as a tif, for the window to open."""
    filepath = tmp_path_factory.mktemp("movie") / "simulated.tif"
    tifffile.imwrite(filepath, simulate_movie(n_frames=600))
    return str(filepath)
