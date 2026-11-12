# /// script
# requires-python = ">=3.11"
# dependencies = ["numba", "numpy", "scipy", "pillow"]
# ///
"""Renders triangle meshes on the CPU: z-buffer, interpolated normals, screen-space ambient occlusion."""

import numba
import numpy as np
import scipy.ndimage
from PIL import Image

BACKGROUND = np.array([0.086, 0.090, 0.102], np.float32)
KEY_LIGHT = np.array([-0.50, 0.60, 0.62], np.float32) / np.linalg.norm([-0.50, 0.60, 0.62])
FILL_LIGHT = np.array([0.70, -0.25, 0.65], np.float32) / np.linalg.norm([0.70, -0.25, 0.65])


@numba.njit
def vertex_normals(vertices, faces):
    """Area-weighted unit normals. Shape (n_vertices, 3)."""
    normals = np.zeros(vertices.shape, np.float32)
    for f in range(faces.shape[0]):
        a, b, c = faces[f, 0], faces[f, 1], faces[f, 2]
        e1 = vertices[b] - vertices[a]
        e2 = vertices[c] - vertices[a]
        n = np.cross(e1, e2)
        normals[a] += n
        normals[b] += n
        normals[c] += n
    for v in range(normals.shape[0]):
        length = np.sqrt((normals[v] ** 2).sum())
        if length > 0:
            normals[v] /= length
    return normals


@numba.njit
def rasterize(points, normals, scalars, faces, height, width):
    """Z-buffers triangles in pixel coordinates; larger z is nearer. Returns depth and a per-vertex scalar (h, w), normals (h, w, 3)."""
    depth = np.full((height, width), -np.inf, np.float32)
    surface = np.zeros((height, width, 3), np.float32)
    value = np.zeros((height, width), np.float32)
    for f in range(faces.shape[0]):
        i0, i1, i2 = faces[f, 0], faces[f, 1], faces[f, 2]
        x0, y0, z0 = points[i0, 0], points[i0, 1], points[i0, 2]
        x1, y1, z1 = points[i1, 0], points[i1, 1], points[i1, 2]
        x2, y2, z2 = points[i2, 0], points[i2, 1], points[i2, 2]
        left, right = int(max(np.floor(min(x0, x1, x2)), 0)), int(min(np.ceil(max(x0, x1, x2)), width - 1))
        top, bottom = int(max(np.floor(min(y0, y1, y2)), 0)), int(min(np.ceil(max(y0, y1, y2)), height - 1))
        if left > right or top > bottom:
            continue
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if area == 0:
            continue
        for y in range(top, bottom + 1):
            for x in range(left, right + 1):
                w1 = ((x - x0) * (y2 - y0) - (x2 - x0) * (y - y0)) / area
                w2 = ((x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)) / area
                w0 = 1.0 - w1 - w2
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                z = w0 * z0 + w1 * z1 + w2 * z2
                if z > depth[y, x]:
                    depth[y, x] = z
                    surface[y, x] = w0 * normals[i0] + w1 * normals[i1] + w2 * normals[i2]
                    value[y, x] = w0 * scalars[i0] + w1 * scalars[i1] + w2 * scalars[i2]
    return depth, surface, value


def ambient_occlusion(depth: np.ndarray, drawn: np.ndarray, sigmas_px: tuple, scale_px: float) -> np.ndarray:
    """Darkens pixels that sit below their surroundings: 1 where open, toward 0 in a crevice."""
    filled = np.where(drawn, depth, 0.0).astype(np.float32)
    occlusion = np.zeros(depth.shape, np.float32)
    for sigma in sigmas_px:
        weight = scipy.ndimage.gaussian_filter(drawn.astype(np.float32), sigma)
        around = scipy.ndimage.gaussian_filter(filled, sigma) / np.maximum(weight, 1e-3)
        occlusion += np.clip((around - depth) / (scale_px * sigma), 0.0, 1.0) * np.minimum(weight * 2.0, 1.0)
    return 1.0 - np.clip(occlusion / len(sigmas_px), 0.0, 1.0)


def shade(depth: np.ndarray, surface: np.ndarray, color: np.ndarray, sigmas_px: tuple, ao_scale: float, fog: float, ao_strength: float = 0.75) -> np.ndarray:
    """Lit float rgb from a depth and normal buffer. `color` is (3,) or per pixel (h, w, 3)."""
    drawn = np.isfinite(depth)
    length = np.linalg.norm(surface, axis=-1, keepdims=True)
    n = surface / np.maximum(length, 1e-6)
    n = n * np.where(n[..., 2:3] < 0, -1.0, 1.0)
    key = np.clip(n @ KEY_LIGHT, 0.0, 1.0)
    fill = np.clip(n @ FILL_LIGHT, 0.0, 1.0)
    rim = (1.0 - np.clip(n[..., 2], 0.0, 1.0)) ** 2.5
    ao = ambient_occlusion(np.where(drawn, depth, 0.0), drawn, sigmas_px, ao_scale)
    z = depth[drawn]
    lo, hi = np.percentile(z, 1), np.percentile(z, 99)
    near = np.zeros(depth.shape, np.float32)
    near[drawn] = np.clip((z - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    light = (0.16 + 0.72 * key + 0.20 * fill + 0.22 * rim) * (1.0 - ao_strength * (1.0 - ao)) * (1.0 - fog + fog * near)
    image = np.where(drawn[..., None], color * light[..., None], BACKGROUND)
    return np.clip(image, 0.0, 1.0)


def view(vertices: np.ndarray, center_um: np.ndarray, rotation: np.ndarray, um_per_px: float, size: tuple) -> np.ndarray:
    """Vertices (um) as pixel coordinates, x right, y down, z nearer as it grows."""
    relative = (vertices - center_um) @ rotation.T
    return np.c_[size[0] / 2 + relative[:, 0] / um_per_px, size[1] / 2 + relative[:, 1] / um_per_px, relative[:, 2] / um_per_px].astype(np.float32)


def render(vertices: np.ndarray, normals: np.ndarray, faces: np.ndarray, center_um: np.ndarray, rotation: np.ndarray, um_per_px: float, size: tuple, scalars: np.ndarray = None, supersample: int = 2) -> tuple:
    """Depth, normal and scalar buffers at supersample x the image size."""
    points = view(vertices, center_um, rotation, um_per_px / supersample, (size[0] * supersample, size[1] * supersample))
    values = np.zeros(len(vertices), np.float32) if scalars is None else scalars.astype(np.float32)
    return rasterize(points, (normals @ rotation.T).astype(np.float32), values, faces, size[1] * supersample, size[0] * supersample)


def reduce(rgb: np.ndarray, supersample: int = 2) -> Image.Image:
    image = Image.fromarray((rgb * 255).astype(np.uint8))
    return image.resize((image.width // supersample, image.height // supersample), Image.LANCZOS)


def reduce_rgba(rgb: np.ndarray, drawn: np.ndarray, supersample: int = 2) -> Image.Image:
    """The drawn pixels on a transparent background, each output pixel's alpha the share of its samples that are drawn."""
    height, width = drawn.shape[0] // supersample, drawn.shape[1] // supersample
    coverage = drawn.reshape(height, supersample, width, supersample).sum(axis=(1, 3))
    color = (rgb * drawn[..., None]).reshape(height, supersample, width, supersample, 3).sum(axis=(1, 3)) / np.maximum(coverage, 1)[..., None]
    alpha = coverage / supersample**2
    return Image.fromarray((np.dstack([color, alpha]) * 255).astype(np.uint8), "RGBA")
