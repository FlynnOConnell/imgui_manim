# /// script
# requires-python = ">=3.11"
# dependencies = ["cloud-volume", "numba", "numpy", "pillow", "scipy"]
# ///
"""
Writes the synapse_to_soma lesson's neuron folder: the panels of the MICrONS cell imgui_ophys's Subcellular page draws,
the electron micrograph of Spine 1's synapse with its bouton and spine outlined, and the path distance of every pixel of
cell.png from the soma and from Spine 1, which is what lets a voltage run along the real dendrites.

    uv run -p 3.12 scripts/synapse_to_soma/export_neuron.py

The volumes are public and need no account. The mesh is ~/data/microns_pyramidal (fetch_microns_cell.py in the topic
below); the renderer and the cell's coordinates are that topic's mesh_render.py and neuron_em.py.
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.csgraph
from cloudvolume import CloudVolume
from PIL import Image, ImageFilter

DIR_TOPIC_CODE = Path.home() / "agent_assets/imgui_ophys/20261029_subcellular-lesson/code"
sys.path.insert(0, str(DIR_TOPIC_CODE))
import mesh_render as R  # noqa: E402
import neuron_em as E  # noqa: E402

DIR_PANELS = Path.home() / "repos/imgui_ophys/src/imgui_ophys/lessons/subcellular/neuron"
DIR_LESSON = Path(__file__).resolve().parents[2] / "src/imgui_manim/lessons/synapse_to_soma/neuron"
PANELS = ("cell.png", "branch.png", "spine.png", "soma.png", "neuron.json")
URL_EM = "precomputed://https://bossdb-open-data.s3.amazonaws.com/iarpa_microns/minnie/minnie65/em"
URL_SEG = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"
CELL, BOUTON = 864691135572530981, 864691135502216470  # the bouton caps Spine 1's head in the segmentation
HALF_NM = np.array([1500.0, 1500.0, 800.0])  # the block around Spine 1
SYNAPSE_SLICE = 16  # the block's plane holding head, neck and shaft together
SYNAPSE_CROP = (90, 310, 20, 240)  # x0, x1, y0, y1 in the block, 8 nm voxels
SYNAPSE_ZOOM = 3
SYNAPSE_LEVELS = (70.0, 150.0)
PATH_STEP_UM = 0.1
UNREACHED = 65535


def cutout(volume: CloudVolume) -> np.ndarray:
    """The block around Spine 1 at the volume's finest resolution, (x, y, z)."""
    resolution = np.array(volume.meta.resolution(0), float)
    center = E.SPINE_1.astype(float) * 1000.0
    start, stop = np.floor((center - HALF_NM) / resolution).astype(int), np.ceil((center + HALF_NM) / resolution).astype(int)
    return np.asarray(volume[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]])[..., 0]


def fetch_block(filepath_block: Path) -> None:
    """Downloads the micrograph and the segmentation around Spine 1, about 4 MB."""
    em = cutout(CloudVolume(URL_EM, use_https=True, mip=0, progress=False, fill_missing=True))
    seg = cutout(CloudVolume(URL_SEG, use_https=True, mip=0, progress=False, fill_missing=True))
    filepath_block.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(filepath_block, em=em, seg=seg)


def synapse_panels(filepath_block: Path) -> tuple:
    """The micrograph of Spine 1's synapse and the outlines of its bouton and of the spine, enlarged SYNAPSE_ZOOM times."""
    block = np.load(filepath_block)
    x0, x1, y0, y1 = SYNAPSE_CROP
    gray = block["em"][x0:x1, y0:y1, SYNAPSE_SLICE].T.astype(np.float32)
    size = ((x1 - x0) * SYNAPSE_ZOOM, (y1 - y0) * SYNAPSE_ZOOM)
    levels = np.clip((gray - SYNAPSE_LEVELS[0]) / (SYNAPSE_LEVELS[1] - SYNAPSE_LEVELS[0]), 0, 1)
    micrograph = Image.fromarray((255 * levels).astype(np.uint8)).resize(size, Image.BICUBIC).convert("RGB")
    outlines = {}
    for name, segment in (("bouton", BOUTON), ("spine", CELL)):
        mask = Image.fromarray(255 * (block["seg"][x0:x1, y0:y1, SYNAPSE_SLICE].T == segment).astype(np.uint8))
        inside = np.asarray(mask.resize(size, Image.BICUBIC).filter(ImageFilter.GaussianBlur(3))) > 127
        outlines[name] = inside ^ scipy.ndimage.binary_erosion(inside, iterations=3)
    return micrograph, outlines


def voxel_graph(vertices: np.ndarray) -> tuple:
    """The occupied 1 um voxels as a graph, since the mesh is chunks that share no vertices, and each vertex's voxel."""
    index = np.floor(vertices / E.VOXEL_UM).astype(np.int64)
    index -= index.min(axis=0)
    dims = index.max(axis=0) + 3
    keys = (index[:, 0] + 1) * dims[1] * dims[2] + (index[:, 1] + 1) * dims[2] + index[:, 2] + 1
    voxels, voxel_of_vertex = np.unique(keys, return_inverse=True)
    rows, cols, weights = [], [], []
    for a, b, c in [(a, b, c) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if (a, b, c) > (0, 0, 0)]:
        neighbors = voxels + a * dims[1] * dims[2] + b * dims[2] + c
        at = np.searchsorted(voxels, neighbors) % len(voxels)
        hit = voxels[at] == neighbors
        rows.append(np.where(hit)[0])
        cols.append(at[hit])
        weights.append(np.full(hit.sum(), E.VOXEL_UM * np.sqrt(a * a + b * b + c * c)))
    graph = scipy.sparse.coo_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(len(voxels), len(voxels))).tocsr()
    return graph, voxel_of_vertex


def path_from(graph: scipy.sparse.csr_matrix, voxel_of_vertex: np.ndarray, vertices: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Path distance in um of every vertex from the vertex nearest point; infinite on pieces no path reaches."""
    source = int(voxel_of_vertex[np.argmin(np.linalg.norm(vertices - point, axis=1))])
    return scipy.sparse.csgraph.dijkstra(graph, directed=False, indices=source)[voxel_of_vertex].astype(np.float32)


def path_pixels(vertices: np.ndarray, normals: np.ndarray, faces: np.ndarray, um: np.ndarray) -> np.ndarray:
    """Path distance per pixel of cell.png in PATH_STEP_UM steps, uint16; pixels off the cell are UNREACHED."""
    size_cell, um_cell = (900, 1400), 0.50  # as neuron_em.py draws cell.png
    center_cell = E.SOMA + np.array([0.0, -135.0, 0.0], np.float32)
    reached = np.where(np.isfinite(um), um, UNREACHED * PATH_STEP_UM)
    depth, _, value = R.render(vertices, normals, faces, center_cell, E.VIEW, um_cell, size_cell, scalars=reached)
    steps = np.where(np.isfinite(depth), np.minimum(np.round(value / PATH_STEP_UM), UNREACHED), UNREACHED)
    return steps.reshape(size_cell[1], 2, size_cell[0], 2).min(axis=(1, 3)).astype(np.uint16)  ## the render's 2x supersampling: any drawn sample counts


def main(dir_mesh: Path, dir_out: Path) -> None:
    dir_out.mkdir(parents=True, exist_ok=True)
    for name in PANELS:
        shutil.copy(DIR_PANELS / name, dir_out / name)

    filepath_block = dir_mesh / "synapse/block.npz"
    if not filepath_block.exists():
        fetch_block(filepath_block)
    micrograph, outlines = synapse_panels(filepath_block)
    micrograph.save(dir_out / "synapse.png", optimize=True)

    data = np.load(dir_mesh / "cell_lod1.npz")
    vertices = (data["vertices"] / 1000.0).astype(np.float32)
    faces = data["faces"]
    normals = R.vertex_normals(vertices, faces)
    inflated = vertices + E.INFLATE_UM * normals
    graph, voxel_of_vertex = voxel_graph(vertices)
    soma, spine = (path_pixels(inflated, normals, faces, path_from(graph, voxel_of_vertex, vertices, point)) for point in (E.SOMA, E.SPINE_1))
    np.savez_compressed(dir_out / "path.npz", soma=soma, spine=spine, step_um=PATH_STEP_UM, unreached=UNREACHED,
                        outline_bouton=np.packbits(outlines["bouton"]), outline_spine=np.packbits(outlines["spine"]), outline_shape=np.array(outlines["spine"].shape))
    print(f"wrote {dir_out}: path from soma covers {100 * np.mean(soma != UNREACHED):.1f}% of the panel")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=Path.home() / "data/microns_pyramidal")
    parser.add_argument("--out", type=Path, default=DIR_LESSON)
    args = parser.parse_args()
    main(dir_mesh=args.mesh, dir_out=args.out)
