# /// script
# requires-python = ">=3.11"
# dependencies = ["manim>=0.22", "numpy", "pillow<13"]
# ///
"""
Subthreshold voltage in a dendrite as one animation: a real cell from soma to spine, a branch's voltage-indicator trace,
one EPSP zoomed, the glutamate pulse at the spine that caused it, five inputs summing into a dendritic spike, the same
cell at a spine, the branch and the soma, two ratios (EPSP over noise, bAP at the branch over the somatic spike), the
picture morphing as noise, a second input and distance change, and a table of the four shapes. Built from imgui_ophys's
subcellular model with the input and spike times known; drawn with kit.py (manim through a Cairo camera, no Scene). The
cell is from the MICrONS dataset under CC BY 4.0, cited on screen wherever it shows: see its README in imgui_ophys,
src/imgui_ophys/lessons/subcellular/neuron/.

    uv run -p 3.12 animations/subcellular.py --out <dir> [--stills-only] [--gif-width 720]

Text only, no LaTeX. Needs ffmpeg and uv on PATH.
"""

import functools
from dataclasses import dataclass
from typing import Optional

import manim as mn
import numpy as np
from PIL import Image

import kit
from kit import (ACCENT, BAR, DIM, HIGHLIGHT, KEY, TEXT, TRACE, TRUTH, Count, Play, at, axes_points, chapter, color, hbar, label, path,
                 plain_axes, put, replace, timeline, vbar)

M = kit.lesson_model("subcellular")

GLU = TRUTH  # glutamate at a spine, and the input ticks: the ground truth
SPIKE = HIGHLIGHT  # somatic spikes and the bAP
DSPIKE = KEY  # the dendritic spike
W_TRACE, H_TRACE, X_TRACE, Y_TRACE = 10.0, 1.5, 1.7, 2.1  # the branch's trace across the top, right of its close-up, scene units
W_WIDE = 13.0  # the same trace over the closing table, with the cell gone
W_ZOOM, H_ZOOM, Y_ZOOM = 4.4, 2.3, -1.7  # the zooms, lower left and lower right
H_STRIP, Y_STRIP = 0.7, 0.0  # the glutamate strip above each zoom
X_ZOOM_A, X_ZOOM_B = -1.1, 4.3
W_SLOT, H_SLOT, X_SLOT = 5.4, 1.3, -0.6  # the three sites' traces, one slot each, right of their close-ups
Y_SLOTS = {"spine": 1.9, "branch": 0.2, "soma": -1.5}
X_FIG, W_FIG, H_FIG = 4.9, 3.2, 1.1  # the two figures the ratios come from, right
Y_FIG_EPSP, Y_FIG_BAP = 1.5, -1.0
X_NUMBER = 3.0
TIME_BAR_TOP = 0.2  # s
ZOOM_HALF = 0.05  # s each side of the followed input
H_CELL, X_CELL, Y_CELL = 7.0, -4.9, -0.3  # the cell at the start, scene units
H_BRANCH, X_BRANCH, Y_BRANCH = 3.3, 2.9, 1.1  # the branch and spine panels, right
H_SPINE, X_SPINE, Y_SPINE = 3.0, 1.1, -2.2
X_NOTES = 2.85  # the spine's name and the cell's credit, right of the spine panel
H_SMALL, X_SMALL, Y_SMALL = 4.5, -5.85, 0.25  # the cell from then on, left of every trace
X_LENS, R_LENS = -3.95, 0.5  # the round close-ups of each site, left of their traces
R_SPINE_MARK = 0.12  # the spine's circle inside the branch's close-up
SOFT_EDGE_PX = 40  # a crop's sides fade out, so branches do not stop at a hard line

TITLE = "Subthreshold voltage: a dendrite's potential below the level that fires a spike"
CHAPTERS = (
    ("", 4.5),
    ("", 4.0),
    ("", 4.0),
    ("", 7.0),
    ("", 4.5),
    ("", 5.0),
    ("", 7.5),
    ("", 6.0),
    ("", 5.0),
    ("Add noise", 5.5),
    ("Add a second input", 5.5),
    ("Record farther from the soma", 5.5),
    ("", 10.5),
)
UNTITLED = {0: "the-cell", 1: "a-branch", 2: "a-spine", 3: "the-branch", 4: "one-epsp", 5: "glutamate-at-the-spine", 6: "inputs-sum", 7: "spine-branch-soma",
            8: "two-ratios", 12: "shapes"}
# the cells the animation morphs between, in order, each change kept by the next
CELLS = (
    ("base", dict()),
    ("noisy", dict(photons=10000.0)),
    ("pairs", dict(photons=10000.0, inputs_per_event=2)),
    ("far", dict(photons=10000.0, inputs_per_event=2, distance=350.0)),
)
ROW_COLORS = {"EPSP": ACCENT, "dendritic spike": DSPIKE, "bAP": SPIKE, "noise": DIM}
FS = M.Params().fs


@dataclass
class Run:
    """One simulated cell with its parameters and the two ratios measured on it."""

    name: str
    p: M.Params
    cell: M.Cell
    numbers: M.Numbers


def simulate_all() -> dict:
    out = {}
    for name, kw in CELLS:
        p = M.Params(**kw)
        cell = M.simulate(p)
        out[name] = Run(name, p, cell, M.numbers(cell, p.fs))
    return out


def cap(text: str) -> mn.Text:
    """A chapter's one-line caption, small and grey at the bottom."""
    return put(label(text, DIM, 18), 0.0, -3.6)


def ms(seconds: float) -> str:
    return f"{1000 * seconds:g} ms"


def pct(dff: float) -> str:
    return f"{100 * dff:g}%"


def bars_of(axes: mn.Axes, x_max: float, y_lo: float, y_hi: float, seconds: float, drop: float, sign: str = "-", with_dff: bool = True) -> mn.VGroup:
    """A time bar under the right end in ms and, with_dff, a dF/F bar at the top right, both labelled."""
    bars = hbar(axes, x_max - seconds, x_max, y_lo, ms(seconds), drop)
    if with_dff:
        dff = kit.nice(0.3 * (y_hi - y_lo))
        bars.add(*vbar(axes, x_max, y_hi - dff, y_hi, f"{pct(dff)} {sign}dF/F"))
    return bars


def padded(values: list, pad: float = 0.08) -> tuple[float, float]:
    lo, hi = float(min(v.min() for v in values)), float(max(v.max() for v in values))
    return lo - pad * (hi - lo), hi + pad * (hi - lo)


def tick_marks(axes: mn.Axes, times: np.ndarray, y_lo: float, y_hi: float, c: tuple) -> mn.VGroup:
    bottom, top = y_lo - 0.09 * (y_hi - y_lo), y_lo - 0.02 * (y_hi - y_lo)
    return mn.VGroup(*[mn.Line(*axes_points(axes, np.array([t, t]), np.array([bottom, top])), color=color(c), stroke_width=2.0) for t in times])


def window(t: np.ndarray, start: float, end: float) -> slice:
    return slice(int(round(start * FS)), int(round(end * FS)))


def box_on(axes: mn.Axes, start: float, end: float, y_lo: float, y_hi: float, height: float, width_min: float = 0.12) -> mn.Rectangle:
    pts = axes_points(axes, np.array([start, end]), np.array([y_lo, y_hi]))
    box = mn.Rectangle(width=max(pts[1, 0] - pts[0, 0], width_min), height=height, color=color(DIM), stroke_width=1.5)
    return box.move_to(0.5 * (pts[0] + pts[1]))


def panel(name: str, height: float, x: float, y: float, crop: Optional[np.ndarray] = None) -> mn.ImageMobject:
    """A panel of the reference cell, transparent around the cell, smoothed, height scene units tall, centered at (x, y)."""
    rgba = np.asarray(Image.open(M.DIR_NEURON / f"{name}.png"))
    if crop is not None:
        rgba = rgba[crop[1]:crop[3], crop[0]:crop[2]]
        columns = np.arange(rgba.shape[1])
        soft = np.clip(np.minimum(columns, columns[::-1]) / SOFT_EDGE_PX, 0.0, 1.0)
        rgba = np.dstack([rgba[..., :3], (rgba[..., 3] * soft).astype(np.uint8)])
    mobject = mn.ImageMobject(rgba)
    mobject.set_resampling_algorithm(mn.RESAMPLING_ALGORITHMS["bicubic"])
    mobject.height = height
    return put(mobject, x, y)


def on_panel(image: mn.ImageMobject, width_px: float, xy: np.ndarray) -> np.ndarray:
    """Pixels (..., 2) measured from an image's top left, width_px across, as scene points (..., 3)."""
    left, top = image.get_corner(mn.UL)[:2]
    scale = image.width / width_px
    return np.stack([left + scale * xy[..., 0], top - scale * xy[..., 1], np.zeros(xy.shape[:-1])], axis=-1)


def on_cell(image: mn.ImageMobject, n: M.Neuron, xy: np.ndarray) -> np.ndarray:
    """Pixels of cell.png as scene points on an image of its tall crop."""
    return on_panel(image, n.tall_crop[2] - n.tall_crop[0], xy - n.tall_crop[:2])


def route_on(image: mn.ImageMobject, n: M.Neuron, xy: np.ndarray) -> mn.VMobject:
    """A blue line through pixels of cell.png on an image of its tall crop."""
    return mn.VMobject(stroke_color=color(ACCENT), stroke_width=2.5).set_points_as_corners(on_cell(image, n, xy))


def ring_at(point: np.ndarray, radius: float, c: tuple) -> mn.Circle:
    """An open circle around a scene point."""
    return mn.Circle(radius=radius, color=color(c), stroke_width=2.5).move_to(point)


def um_bar(image: mn.ImageMobject, width_px: float, um_per_px: float, um: float) -> mn.VGroup:
    """A scale bar at an image's lower right with its length under it."""
    length = um / um_per_px * image.width / width_px
    right = image.get_corner(mn.DR) + at(-0.15, 0.42)
    line = mn.Line(right - at(length, 0.0), right, color=color(BAR), stroke_width=3.0)
    return mn.VGroup(line, label(f"{um:g} um", BAR, 15).next_to(line, mn.DOWN, buff=0.06).align_to(line, mn.RIGHT))


def zoom_lines(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, line_color: tuple) -> mn.VGroup:
    """Thin lines from a to b and c to d, a circle's edges to its panel's corners."""
    return mn.VGroup(*[mn.Line(start, end, color=color(line_color), stroke_width=1.5) for start, end in ((a, b), (c, d))]).set_stroke(opacity=0.7)


def cone(a: np.ndarray, radius_a: float, b: np.ndarray, radius_b: float, c: tuple) -> mn.VGroup:
    """Two thin lines opening from a circle at a to a larger one at b: b is a close-up of a."""
    along = (b - a) / np.linalg.norm(b - a)
    side = np.array([-along[1], along[0], 0.0])
    return zoom_lines(a - radius_a * side, b - radius_b * side, a + radius_a * side, b + radius_b * side, c)


def cutout(name: str, box: np.ndarray) -> np.ndarray:
    """The box (x0, y0, x1, y1) of a panel as rgba."""
    x0, y0, x1, y1 = np.round(box).astype(int)
    return np.asarray(Image.open(M.DIR_NEURON / f"{name}.png"))[y0:y1, x0:x1]


def rounded(rgba: np.ndarray) -> np.ndarray:
    """A square rgba with everything outside its inscribed circle transparent."""
    size = rgba.shape[0]
    rows, columns = np.ogrid[:size, :size]
    inside = (rows - 0.5 * (size - 1)) ** 2 + (columns - 0.5 * (size - 1)) ** 2 <= (0.5 * size) ** 2
    return np.dstack([rgba[..., :3], rgba[..., 3] * inside]).astype(np.uint8)


def lens_image(rgba: np.ndarray, y: float) -> mn.ImageMobject:
    """A round close-up in the lens column, centered at height y."""
    mobject = mn.ImageMobject(rounded(rgba))
    mobject.set_resampling_algorithm(mn.RESAMPLING_ALGORITHMS["bicubic"])
    mobject.height = 2 * R_LENS
    return put(mobject, X_LENS, y)


def lens_ring(y: float, c: tuple) -> mn.Circle:
    """The rim of a close-up, filled so nothing shows through it."""
    return mn.Circle(radius=R_LENS, color=color(c), stroke_width=2.5, fill_color=color(kit.BG), fill_opacity=1.0).move_to(at(X_LENS, y))


def branch_name(um: float) -> mn.Text:
    return label(f"Branch, {um:.0f} um from the soma", ACCENT, 18)


def number_row(name: str, value_text: str, x: float, y: float) -> tuple[mn.Text, mn.Text]:
    """The ratio's name in grey and its value large in the key color on one line, left-aligned at x."""
    name_label = label(name, DIM, 20).move_to(at(x, y), aligned_edge=mn.LEFT)
    value = label(value_text, KEY, 40).next_to(name_label, mn.RIGHT, buff=0.3)
    value.align_to(name_label, mn.DOWN).shift(at(0, -0.04))
    return name_label, value


def value_1(v: float) -> mn.Text:
    return label(f"{v:.1f}", KEY, 40)


def value_2(v: float) -> mn.Text:
    return label(f"{v:.2f}", KEY, 40)


def swap(titles: list, captions: list, k: int) -> list:
    """Chapter k's title and caption replace chapter k - 1's."""
    return [*replace(titles[k - 1], titles[k]), *replace(captions[k - 1], captions[k])]


def build_step(builds: list, glu_builds: list, counts: list, k: int) -> mn.AnimationGroup:
    """The burst zoom with one more input: the response, the glutamate and the count change together."""
    return mn.AnimationGroup(mn.Transform(builds[0], builds[k]), mn.Transform(glu_builds[0], glu_builds[k]), *replace(counts[k - 1], counts[k]), run_time=0.6)


@dataclass
class Pieces:
    """Everything a cell change morphs: the three traces, the two figures and the two numbers."""

    traces: dict  # cell name -> {site: path at its slot}
    epsp: dict  # cell name -> the noiseless EPSP in the figure
    band: dict  # cell name -> the +/- 1 SD noise band
    bap: dict  # cell name -> {site: path}


def morph(duration: float, lead: list, p: Pieces, src: str, dst: str, cells: dict, value_epsp: mn.Text, value_bap: mn.Text, extra: tuple = ()) -> mn.AnimationGroup:
    """The chapter's title first, then everything on screen, with extra, turns from cell src into cell dst at once, numbers counting."""
    change = mn.AnimationGroup(
        *[mn.Transform(p.traces["base"][site], p.traces[dst][site]) for site in M.SITES],
        mn.Transform(p.epsp["base"], p.epsp[dst]), mn.Transform(p.band["base"], p.band[dst]),
        *[mn.Transform(p.bap["base"][site], p.bap[dst][site]) for site in ("soma", "branch")],
        Count(value_epsp, cells[src].numbers.epsp_over_noise, cells[dst].numbers.epsp_over_noise, value_1),
        Count(value_bap, cells[src].numbers.bap_ratio, cells[dst].numbers.bap_ratio, value_2, decimals=2),
        *extra,
        run_time=4.0,
    )
    return chapter(duration, mn.AnimationGroup(*lead, run_time=1.0), change, lag_ratio=1.0)


def table_rows(cell: M.Cell, rows: list, y_top: float, pitch: float) -> list:
    """
    One row per shape: a 60 ms piece of the branch's recording with the noiseless response over it, its own dF/F bar,
    then the name, the size at the branch in mV, the duration and the sites it clears the noise at.
    """
    branch = cell.sites["branch"]
    span = 0.06
    out = []
    for k, row in enumerate(rows):
        y = y_top - k * pitch
        starts = {"EPSP": M.followed_input(cell), "bAP": M.followed_ap(cell), "dendritic spike": row.example, "noise": M.quiet_start(cell, FS, span) + 0.015}
        start = starts[row.name] - 0.015
        sl = window(cell.t, start, start + span)
        rec, clean = -branch.dff[sl], -branch.clean[sl]
        lo, hi = padded([rec, clean], 0.1)
        axes = put(plain_axes(0, span, lo, hi, 2.6, 0.72 * pitch, visible=False), -5.2, y)
        t = cell.t[sl] - cell.t[sl][0]
        snippet = mn.VGroup(path(axes, t, rec, TRACE, 1.2, lo, hi))
        if row.name != "noise":
            snippet.add(path(axes, t, clean, ROW_COLORS[row.name], 2.0, lo, hi))
        dff = kit.nice(0.4 * (hi - lo))
        bar = vbar(axes, 0.0, hi - dff, hi, pct(dff))
        texts = [
            label(row.name, ROW_COLORS[row.name], 20).move_to(at(-3.3, y), aligned_edge=mn.LEFT),
            label(f"{row.mv['branch']:.0f} mV" if row.mv["branch"] >= 10 else f"{row.mv['branch']:.1f} mV", TEXT, 20).move_to(at(-0.6, y), aligned_edge=mn.LEFT),
            label(ms(round(row.duration, 3)), TEXT, 20).move_to(at(1.9, y), aligned_edge=mn.LEFT),
            label(", ".join(row.seen_at), TEXT, 20).move_to(at(3.9, y), aligned_edge=mn.LEFT),
        ]
        out.append((snippet, bar, texts, axes, lo, hi))
    return out


def build(cells: dict, seconds: list) -> kit.Timeline:
    """The animation, each chapter seconds[k] long."""
    base = cells["base"].cell
    t = base.t
    T = float(t[-1])
    names = [name for name, _ in CELLS]
    p_base = cells["base"].p

    title = put(label(TITLE, TEXT, 27), 0.0, 3.6)
    chapter_titles = kit.titles_for(CHAPTERS)
    captions = [None] * len(CHAPTERS)
    captions[3] = cap("The indicator dims when the membrane depolarizes, so the trace is flipped: -dF/F puts depolarization up.")
    captions[4] = cap("One input depolarizes the branch by a few mV for about 15 ms: an EPSP (excitatory postsynaptic potential).")
    captions[5] = cap("The input is a pulse of glutamate landing on a spine; the EPSP follows under it within a millisecond.")
    captions[6] = cap("EPSPs within a few milliseconds of each other add up; past a threshold the branch fires its own spike, a dendritic spike.")
    captions[7] = cap("A somatic spike travels back into the branch, smaller and later with distance: a bAP (backpropagating action potential).")
    captions[8] = cap("The ratios are one EPSP over the noise SD at the branch, and the bAP at the branch over the spike at the soma.")

    # chapters 1 to 3: the real cell, a circle on its side branch grown into a panel, a circle on one spine grown into another
    n = M.load_neuron()
    details = n.panels
    cell_image = panel("cell", H_CELL, X_CELL, Y_CELL, n.tall_crop)
    branch_image = panel("branch", H_BRANCH, X_BRANCH, Y_BRANCH)
    spine_image = panel("spine", H_SPINE, X_SPINE, Y_SPINE)
    ring_soma = ring_at(on_cell(cell_image, n, n.soma), 0.2, SPIKE)
    name_soma = label("Soma", SPIKE, 20).next_to(ring_soma, mn.DR, buff=0.05)
    patch_soma = mn.BackgroundRectangle(name_soma, color=color(kit.BG), fill_opacity=1.0, buff=0.06)
    credit = mn.VGroup(label(M.NEURON_NAME, TEXT, 17), label(M.NEURON_METHOD, DIM, 15), label(M.NEURON_CITATION, DIM, 15), label(M.NEURON_SITE, DIM, 15),
                       label(f"Whole cell thickened {n.inflate_um:g} um for display", DIM, 15))
    credit.arrange(mn.DOWN, aligned_edge=mn.LEFT, buff=0.1).move_to(at(X_NOTES, -3.75), aligned_edge=mn.DL)
    bar_cell = um_bar(cell_image, n.tall_crop[2] - n.tall_crop[0], details["cell"]["um_per_px"], 100.0)

    px_cell = H_CELL / (n.tall_crop[3] - n.tall_crop[1])
    ring_branch = ring_at(on_cell(cell_image, n, n.branch_box.reshape(2, 2).mean(axis=0)), 0.5 * px_cell * (n.branch_box[2] - n.branch_box[0]), ACCENT)
    flying_branch = ring_branch.copy()
    frame_branch = kit.frame_of(branch_image, ACCENT)
    border_branch = kit.frame_of(branch_image, ACCENT).set_fill(color(kit.BG), opacity=0.0)  # drawn after the image, which would cover the flown frame's stroke
    lines_branch = zoom_lines(ring_branch.get_top(), branch_image.get_corner(mn.UL), ring_branch.get_bottom(), branch_image.get_corner(mn.DL), ACCENT)
    name_branch = branch_name(round(n.branch_um, -1)).next_to(branch_image, mn.UP, buff=0.08).align_to(branch_image, mn.LEFT)
    bar_branch = um_bar(branch_image, details["branch"]["width"], details["branch"]["um_per_px"], 5.0)

    spine_at = on_panel(branch_image, details["branch"]["width"], n.spine_box.reshape(2, 2).mean(axis=0))
    ring_spine_panel = ring_at(spine_at, 0.5 * H_BRANCH / details["branch"]["height"] * (n.spine_box[2] - n.spine_box[0]), GLU)
    flying_spine = ring_spine_panel.copy()
    frame_spine = kit.frame_of(spine_image, GLU)
    border_spine = kit.frame_of(spine_image, GLU)
    lines_spine = zoom_lines(ring_spine_panel.get_left(), spine_image.get_corner(mn.UL), ring_spine_panel.get_right(), spine_image.get_corner(mn.UR), GLU)
    name_spine = label("Spine 1", GLU, 20).move_to(at(X_NOTES, Y_SPINE + 0.5 * H_SPINE - 0.15), aligned_edge=mn.LEFT)
    bar_spine = um_bar(spine_image, details["spine"]["width"], details["spine"]["um_per_px"], 1.0)
    opening_only = [patch_soma, name_soma, credit, bar_cell, name_branch, bar_branch, ring_spine_panel, lines_spine, spine_image, border_spine, name_spine, bar_spine]

    # chapter 4: the cell shrinks to the left and stays there; the branch panel shrinks into the close-up its trace starts from
    small = panel("cell", H_SMALL, X_SMALL, Y_SMALL, n.tall_crop)
    credit_small = mn.VGroup(*[label(line, DIM, 14) for line in (*M.NEURON_CITATION.split(", "), M.NEURON_SITE)]).arrange(mn.DOWN, aligned_edge=mn.LEFT, buff=0.06)
    credit_small.next_to(small, mn.DOWN, buff=0.08).align_to(small, mn.LEFT)
    soma_at, site_at = on_cell(small, n, n.soma), on_cell(small, n, n.side_xy[-1])
    r_soma, r_site = 0.14, 0.09
    route_root = route_on(small, n, n.trunk_between(0.0, n.side_um[0]))
    route_side = route_on(small, n, n.side_xy)
    lens_top = at(X_LENS, Y_TRACE)
    lens_branch = lens_image(cutout("branch", n.lens_branch), Y_TRACE)
    panel_scale = (n.lens_branch[2] - n.lens_branch[0]) / details["branch"]["height"] * H_BRANCH / (2 * R_LENS)  # how much larger the close-up's dendrite is in the panel
    panel_to_lens = lens_top + (branch_image.get_center() - spine_at) / panel_scale

    # chapter 4: the branch's trace across the top, raw first (dipping), then flipped
    branch = base.sites["branch"]
    y_lo, y_hi = padded([-branch.dff], 0.06)
    axes_top = put(plain_axes(0, T, y_lo, y_hi, W_TRACE, H_TRACE, visible=False), X_TRACE, Y_TRACE)
    axes_raw = put(plain_axes(0, T, -y_hi, -y_lo, W_TRACE, H_TRACE, visible=False), X_TRACE, Y_TRACE)
    axes_wide = put(plain_axes(0, T, y_lo, y_hi, W_WIDE, H_TRACE, visible=False), 0.0, Y_TRACE)
    trace_top = path(axes_raw, t, branch.dff, TRACE, 1.3, -y_hi, -y_lo)
    trace_flipped = path(axes_top, t, -branch.dff, TRACE, 1.3, y_lo, y_hi)
    bars_top = bars_of(axes_top, T, y_lo, y_hi, TIME_BAR_TOP, 0.26, with_dff=False)
    dff_top = kit.nice(0.3 * (y_hi - y_lo))
    bar_dff = vbar(axes_top, T, y_hi - dff_top, y_hi, f"{pct(dff_top)} dF/F")
    bar_dff_flipped = vbar(axes_top, T, y_hi - dff_top, y_hi, f"{pct(dff_top)} -dF/F")
    site_top = label(f"Dendritic branch, {p_base.distance:g} um from the soma (simulated)", ACCENT, 18).next_to(axes_top, mn.UP, buff=0.06).align_to(axes_top, mn.LEFT)
    ticks_in = tick_marks(axes_top, np.unique(base.input_times), y_lo, y_hi, GLU)
    ticks_ap = tick_marks(axes_top, base.ap_times, y_lo, y_hi, SPIKE)
    legend = mn.VGroup(label("Synaptic inputs (simulated)", GLU, 18), label("Somatic spikes (simulated)", SPIKE, 18)).arrange(mn.RIGHT, buff=0.5)
    legend.next_to(ticks_in, mn.DOWN, buff=0.06).align_to(axes_top, mn.LEFT)

    # chapter 5: one input, boxed, flown into a zoom with the noiseless membrane over it
    t_in = M.followed_input(base)
    sl_a = window(t, t_in - ZOOM_HALF, t_in + ZOOM_HALF)
    rec_a, clean_a = -branch.dff[sl_a], -branch.clean[sl_a]
    a_lo, a_hi = padded([rec_a, clean_a], 0.1)
    span_a = 2 * ZOOM_HALF
    axes_a = put(plain_axes(0, span_a, a_lo, a_hi, W_ZOOM, H_ZOOM, visible=False), X_ZOOM_A, Y_ZOOM)
    t_a = t[sl_a] - t[sl_a][0]
    zoom_a = path(axes_a, t_a, rec_a, TRACE, 1.6, a_lo, a_hi)
    segment_a = path(axes_top, t[sl_a], rec_a, TRACE, 1.3, y_lo, y_hi)
    segment_a.set_stroke(opacity=0.0)  # a Transform source is drawn from t = 0; the flight reveals it
    frame_a = mn.Rectangle(width=W_ZOOM, height=H_ZOOM, color=color(DIM), stroke_width=1.5).move_to(axes_a)
    box_a = box_on(axes_top, t[sl_a][0], t[sl_a][-1], y_lo, y_hi, H_TRACE)
    flying_a = box_a.copy()
    bars_a = bars_of(axes_a, span_a, a_lo, a_hi, 0.02, 0.0)
    name_a = label(f"input +/- {ms(ZOOM_HALF)}", DIM, 18).next_to(axes_a, mn.DOWN, buff=0.1).align_to(axes_a, mn.LEFT)
    membrane_a = path(axes_a, t_a, clean_a, ACCENT, 2.2, a_lo, a_hi)
    membrane_label = label("Voltage (noiseless)", ACCENT, 18)
    membrane_label.move_to(axes_a.get_corner(mn.UL) + at(0.1, -0.1), aligned_edge=mn.UL)

    # chapter 6: the glutamate pulse at the spine above the zoom
    glu_a = base.glutamate[0][sl_a]
    g_lo, g_hi = -0.05 * M.GLU_DFF, 1.1 * M.GLU_DFF
    strip_a = put(plain_axes(0, span_a, g_lo, g_hi, W_ZOOM, H_STRIP, visible=False), X_ZOOM_A, Y_STRIP)
    glu_path_a = path(strip_a, t_a, glu_a, GLU, 2.0, g_lo, g_hi)
    glu_label_a = label("Glutamate at spine 1 (simulated)", GLU, 18).next_to(strip_a, mn.UP, buff=0.04).align_to(strip_a, mn.LEFT)
    glu_bar_a = vbar(strip_a, span_a, g_lo + 0.3 * M.GLU_DFF, g_lo + 0.8 * M.GLU_DFF, f"{pct(0.5 * M.GLU_DFF)} dF/F")
    mark_spine = ring_at(lens_top, R_SPINE_MARK, GLU)  # the branch's close-up is centered on Spine 1
    rim_spine = lens_ring(Y_STRIP, GLU)
    spine_rgba = np.asarray(Image.open(M.DIR_NEURON / "spine.png"))
    lens_spine = lens_image(spine_rgba, Y_STRIP)
    cone_spine = cone(lens_top, R_SPINE_MARK, at(X_LENS, Y_STRIP), R_LENS, GLU)

    # chapter 7: the burst, boxed, flown into a second zoom; the noiseless response built one input at a time
    steps = M.burst_steps(p_base)
    b_start, b_end = M.burst_window(p_base)
    sl_b = window(t, b_start, b_end)
    rec_b = -branch.dff[sl_b]
    span_b = b_end - b_start
    b_lo, b_hi = padded([rec_b] + [-clean for _, clean, _ in steps], 0.1)
    b_hi += 0.18 * (b_hi - b_lo)  # headroom: the count label sits above the dendritic spike
    axes_b = put(plain_axes(0, span_b, b_lo, b_hi, W_ZOOM, H_ZOOM, visible=False), X_ZOOM_B, Y_ZOOM)
    t_b = t[sl_b] - t[sl_b][0]
    zoom_b = path(axes_b, t_b, rec_b, TRACE, 1.6, b_lo, b_hi)
    segment_b = path(axes_top, t[sl_b], rec_b, TRACE, 1.3, y_lo, y_hi)
    segment_b.set_stroke(opacity=0.0)
    frame_b = mn.Rectangle(width=W_ZOOM, height=H_ZOOM, color=color(DIM), stroke_width=1.5).move_to(axes_b)
    box_b = box_on(axes_top, t[sl_b][0], t[sl_b][-1], y_lo, y_hi, H_TRACE)
    flying_b = box_b.copy()
    bars_b = bars_of(axes_b, span_b, b_lo, b_hi, 0.02, 0.0)
    name_b = label(f"burst, {ms(span_b)}", DIM, 18).next_to(axes_b, mn.DOWN, buff=0.1).align_to(axes_b, mn.LEFT)
    gb_hi = 1.1 * float(max(glu.max() for _, _, glu in steps))
    strip_b = put(plain_axes(0, span_b, g_lo, gb_hi, W_ZOOM, H_STRIP, visible=False), X_ZOOM_B, Y_STRIP)
    glu_label_b = label("Glutamate at spines 1 to 4 (simulated)", GLU, 18).next_to(strip_b, mn.UP, buff=0.04).align_to(strip_b, mn.LEFT)
    builds = [path(axes_b, t_b, -clean, ACCENT, 2.2, b_lo, b_hi) for _, clean, _ in steps]
    glu_builds = [path(strip_b, t_b, glu, GLU, 2.0, g_lo, gb_hi) for _, _, glu in steps]
    count_texts = [f"{k} input" if k == 1 else f"{k} inputs within {ms(round((k - 1) * M.BURST_STEP, 3))}" for k in range(1, p_base.burst_size)]
    count_texts.append(f"{p_base.burst_size} inputs: a dendritic spike")
    counts = [label(text, DSPIKE if k == len(count_texts) - 1 else TEXT, 18).move_to(axes_b.get_corner(mn.UL) + at(0.1, -0.1), aligned_edge=mn.UL)
              for k, text in enumerate(count_texts)]

    # chapter 8: the three sites, one slot each, on one dF/F scale, beside the small cell
    all_dff = [-c.cell.sites[site].dff for c in cells.values() for site in M.SITES]
    s_lo, s_hi = padded(all_dff, 0.04)
    axes_slot = {site: put(plain_axes(0, T, s_lo, s_hi, W_SLOT, H_SLOT, visible=False), X_SLOT, y) for site, y in Y_SLOTS.items()}
    traces = {name: {site: path(axes_slot[site], t, -c.cell.sites[site].dff, TRACE, 1.2, s_lo, s_hi) for site in M.SITES} for name, c in cells.items()}
    branch_mid = traces["base"]["branch"]
    traces["base"]["branch"] = trace_top  # the trace that is already on screen moves into its slot
    t_ap = M.followed_ap(base)
    box_ap = box_on(axes_slot["spine"], t_ap - 0.01, t_ap + 0.02, s_lo, s_hi, Y_SLOTS["spine"] - Y_SLOTS["soma"] + H_SLOT)
    box_ap.move_to(at(box_ap.get_center()[0], 0.5 * (Y_SLOTS["spine"] + Y_SLOTS["soma"])))
    site_labels = {"spine": label("Spine 1", GLU, 18), "branch": branch_name(p_base.distance), "soma": label("Soma", SPIKE, 18)}
    for site, text in site_labels.items():
        text.next_to(axes_slot[site], mn.UP, buff=0.02).next_to(box_ap, mn.RIGHT, buff=0.1, coor_mask=np.array([1, 0, 0]))
    bars_slots = bars_of(axes_slot["soma"], T, s_lo, s_hi, TIME_BAR_TOP, 0.26, with_dff=False)
    dff_slots = kit.nice(0.3 * (s_hi - s_lo))
    bar_dff_slots = vbar(axes_slot["spine"], T, s_hi - dff_slots, s_hi, f"{pct(dff_slots)} -dF/F")
    ticks_in_slot = tick_marks(axes_slot["branch"], np.unique(base.input_times), s_lo, s_hi, GLU)
    ticks_ap_slot = tick_marks(axes_slot["soma"], base.ap_times, s_lo, s_hi, SPIKE)
    y_branch = Y_SLOTS["branch"]
    lens_mid = at(X_LENS, y_branch)
    rim_slot = lens_ring(y_branch, ACCENT)
    lens_branch_slot = lens_image(cutout("branch", n.lens_branch), y_branch)
    mark_slot = ring_at(lens_mid, R_SPINE_MARK, GLU)
    rim_spine_slot = lens_ring(Y_SLOTS["spine"], GLU)
    lens_spine_slot = lens_image(spine_rgba, Y_SLOTS["spine"])
    cone_spine_slot = cone(lens_mid, R_SPINE_MARK, at(X_LENS, Y_SLOTS["spine"]), R_LENS, GLU)
    rim_soma = lens_ring(Y_SLOTS["soma"], SPIKE)
    lens_soma = lens_image(np.asarray(Image.open(M.DIR_NEURON / "soma.png")), Y_SLOTS["soma"])
    cone_soma = cone(soma_at, r_soma, at(X_LENS, Y_SLOTS["soma"]), R_LENS, SPIKE)

    # chapter 12: the circle steps to the trunk under the title, then climbs it; its close-up shows the real dendrite under it, and Spine 1 is no longer there
    distance_far = cells["far"].p.distance
    trunk_at, far_at = on_cell(small, n, n.on_trunk(p_base.distance)), on_cell(small, n, n.on_trunk(distance_far))
    route_trunk = route_on(small, n, n.trunk_between(n.side_um[0], p_base.distance))
    route_far = route_on(small, n, n.trunk_between(p_base.distance, distance_far))
    ring_trunk = ring_at(trunk_at, r_site, ACCENT)
    tiles = [rounded(cutout("trunk", n.tile_at(um))) for um in n.tile_um[(n.tile_um >= p_base.distance) & (n.tile_um <= distance_far)]]
    lens_trunk = lens_image(cutout("trunk", n.tile_at(p_base.distance)), y_branch)
    beside_traces = [cell_image, credit_small, ring_soma, route_root, route_trunk, route_far, ring_trunk, border_branch, lens_trunk, lines_branch,
                     rim_spine_slot, lens_spine_slot, rim_soma, lens_soma, cone_soma]

    # chapter 9: the two figures and their ratios
    epsp_shapes, bap_shapes = {}, {}
    for name, c in cells.items():
        inputs, aps = M.single_inputs(c.cell), M.single_aps(c.cell)
        lag_e, mean_e = M.average(c.cell.sites["branch"].clean, FS, inputs, 0.02, 0.05)
        epsp_shapes[name] = (lag_e, -mean_e)
        bap_shapes[name] = {site: M.average(c.cell.sites[site].clean, FS, aps, 0.005, 0.012) for site in ("soma", "branch")}
    e_lo, e_hi = padded([np.array([-c.numbers.noise, c.numbers.noise]) for c in cells.values()] + [shape for _, shape in epsp_shapes.values()], 0.12)
    axes_e = put(plain_axes(-0.02, 0.05, e_lo, e_hi, W_FIG, H_FIG, visible=False), X_FIG, Y_FIG_EPSP)
    epsp_paths = {name: path(axes_e, lag, shape, ACCENT, 2.2, e_lo, e_hi) for name, (lag, shape) in epsp_shapes.items()}
    bands = {}
    for name, c in cells.items():
        pts = axes_points(axes_e, np.array([-0.02, 0.05]), np.array([-c.numbers.noise, c.numbers.noise]))
        bands[name] = mn.Rectangle(width=pts[1, 0] - pts[0, 0], height=pts[1, 1] - pts[0, 1], stroke_width=0, fill_color=color(DIM), fill_opacity=0.3).move_to(0.5 * (pts[0] + pts[1]))
    bars_e = bars_of(axes_e, 0.05, e_lo, e_hi, 0.02, 0.0)
    label_e = label("EPSP (simulated, noiseless)", ACCENT, 17).next_to(axes_e, mn.UP, buff=0.04).align_to(axes_e, mn.LEFT)
    label_band = label("noise, +/- 1 SD", BAR, 17).next_to(axes_e, mn.DOWN, buff=0.08).align_to(axes_e, mn.LEFT)
    bap_lo, bap_hi = padded([-shape for shapes in bap_shapes.values() for _, shape in shapes.values()], 0.12)
    axes_bap = put(plain_axes(-0.005, 0.012, bap_lo, bap_hi, W_FIG, H_FIG, visible=False), X_FIG, Y_FIG_BAP)
    bap_paths = {name: {"soma": path(axes_bap, shapes["soma"][0], -shapes["soma"][1], SPIKE, 2.2, bap_lo, bap_hi),
                        "branch": path(axes_bap, shapes["branch"][0], -shapes["branch"][1], TRACE, 2.2, bap_lo, bap_hi)}
                 for name, shapes in bap_shapes.items()}
    bars_bap = bars_of(axes_bap, 0.012, bap_lo, bap_hi, 0.005, 0.0)
    label_bap = mn.VGroup(label("bAP at the soma", SPIKE, 17), label("at the branch", TRACE, 17)).arrange(mn.RIGHT, buff=0.25)
    label_bap.next_to(axes_bap, mn.UP, buff=0.04).align_to(axes_bap, mn.LEFT)
    n0 = cells["base"].numbers
    name_epsp, value_epsp = number_row("EPSP / noise SD", f"{n0.epsp_over_noise:.1f}", X_NUMBER, Y_FIG_EPSP - 0.5 * H_FIG - 0.75)
    name_bap, value_bap = number_row("bAP, branch / soma", f"{n0.bap_ratio:.2f}", X_NUMBER, Y_FIG_BAP - 0.5 * H_FIG - 0.75)
    for name, c in cells.items():
        n = c.numbers
        print(f"{name:6s}: EPSP {100 * n.epsp:.2f}% over noise {100 * n.noise:.2f}% = {n.epsp_over_noise:.2f}; "
              f"bAP branch {100 * n.bap_branch:.1f}% / soma {100 * n.bap_soma:.1f}% = {n.bap_ratio:.2f}")
    pieces = Pieces(traces, epsp_paths, bands, bap_paths)

    # chapter 13: the shapes
    rows = M.shapes(base, FS)
    for row in rows:
        print(f"{row.name:16s}: " + ", ".join(f"{s} {row.mv[s]:.1f} mV / {100 * row.dff[s]:.2f}%" for s in M.SITES) + f"; {1000 * row.duration:.1f} ms; seen at {', '.join(row.seen_at)}")
    trace_again = path(axes_wide, t, -branch.dff, TRACE, 1.3, y_lo, y_hi)
    site_wide = label(f"Dendritic branch, {p_base.distance:g} um from the soma (simulated)", DIM, 18).next_to(axes_wide, mn.UP, buff=0.06).align_to(axes_wide, mn.LEFT)
    chrome_again = mn.VGroup(bars_of(axes_wide, T, y_lo, y_hi, TIME_BAR_TOP, 0.26, with_dff=False),
                             vbar(axes_wide, T, y_hi - dff_top, y_hi, f"{pct(dff_top)} -dF/F"), site_wide)
    y_header, pitch = 0.55, 0.95
    header = mn.VGroup(*[label(text, DIM, 18).move_to(at(x, y_header), aligned_edge=mn.LEFT)
                         for text, x in (("branch, -dF/F", -6.5), ("shape", -3.3), ("size at the branch", -0.6), ("duration", 1.9), ("seen at", 3.9))])
    table = table_rows(base, rows, y_header - 0.8, pitch)
    time_bar_table = hbar(table[-1][3], 0.04, 0.06, table[-1][4], "20 ms", 0.06)

    lower = [box_a, box_b, bars_a, bars_b, name_a, name_b, membrane_a, membrane_label, glu_path_a, glu_label_a, glu_bar_a,
             glu_label_b, builds[0], glu_builds[0], counts[-1], segment_a, segment_b, rim_spine, lens_spine, cone_spine]
    after_pool = [*beside_traces, traces["base"]["spine"], traces["base"]["soma"], *site_labels.values(), bars_slots, bar_dff_slots, box_ap,
                  epsp_paths["base"], bands["base"], bars_e, label_e, label_band, bap_paths["base"]["soma"], bap_paths["base"]["branch"], bars_bap, label_bap,
                  name_epsp, value_epsp, name_bap, value_bap]

    chapters = [
        chapter(seconds[0],
                mn.AnimationGroup(mn.FadeIn(title), mn.FadeIn(cell_image), mn.FadeIn(credit), mn.FadeIn(bar_cell), lag_ratio=0.1),
                mn.AnimationGroup(mn.FadeIn(ring_soma), mn.FadeIn(patch_soma), mn.FadeIn(name_soma), lag_ratio=0.2),
                lag_ratio=1.0),
        chapter(seconds[1],
                mn.AnimationGroup(mn.FadeOut(title), mn.FadeIn(ring_branch), lag_ratio=0.2),
                mn.AnimationGroup(mn.FadeIn(flying_branch, run_time=0.2), mn.Transform(flying_branch, frame_branch, run_time=1.4), lag_ratio=1.0),
                mn.AnimationGroup(mn.FadeIn(branch_image), mn.FadeIn(border_branch), mn.FadeOut(flying_branch), mn.FadeIn(lines_branch), mn.FadeIn(name_branch), mn.FadeIn(bar_branch), lag_ratio=0.15),
                lag_ratio=1.0),
        chapter(seconds[2],
                mn.FadeIn(ring_spine_panel),
                mn.AnimationGroup(mn.FadeIn(flying_spine, run_time=0.2), mn.Transform(flying_spine, frame_spine, run_time=1.4), lag_ratio=1.0),
                mn.AnimationGroup(mn.FadeIn(spine_image), mn.FadeIn(border_spine), mn.FadeOut(flying_spine), mn.FadeIn(lines_spine), mn.FadeIn(name_spine), mn.FadeIn(bar_spine), lag_ratio=0.15),
                lag_ratio=1.0),
        chapter(seconds[3],
                mn.AnimationGroup(mn.Transform(cell_image, small), mn.Transform(ring_soma, ring_at(soma_at, r_soma, SPIKE)), mn.Transform(ring_branch, ring_at(site_at, r_site, ACCENT)),
                                  mn.Transform(border_branch, lens_ring(Y_TRACE, ACCENT).rotate(np.arctan2(branch_image.height, branch_image.width))), mn.Transform(lines_branch, cone(site_at, r_site, lens_top, R_LENS, ACCENT)),
                                  mn.FadeOut(branch_image, target_position=panel_to_lens, scale=1.0 / panel_scale),
                                  mn.FadeIn(lens_branch, target_position=spine_at, scale=panel_scale),
                                  *[mn.FadeOut(m) for m in opening_only], mn.FadeIn(credit_small), run_time=1.8),
                mn.AnimationGroup(mn.Create(route_root), mn.Create(route_side), lag_ratio=1.0, run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(site_top), mn.Create(trace_top), mn.FadeIn(bars_top), mn.FadeIn(bar_dff), lag_ratio=0.15),
                mn.Wait(run_time=0.6),
                mn.AnimationGroup(mn.FadeIn(captions[3]), mn.Transform(trace_top, trace_flipped), *replace(bar_dff, bar_dff_flipped), run_time=1.2),
                mn.AnimationGroup(mn.FadeIn(ticks_in), mn.FadeIn(ticks_ap), mn.FadeIn(legend), lag_ratio=0.2),
                lag_ratio=1.0),
        chapter(seconds[4],
                mn.AnimationGroup(*swap(chapter_titles, captions, 4), mn.FadeIn(box_a), lag_ratio=0.1),
                mn.AnimationGroup(mn.FadeIn(flying_a, run_time=0.2),
                                  mn.AnimationGroup(mn.Transform(flying_a, frame_a), mn.Transform(segment_a, zoom_a), run_time=1.4),
                                  mn.FadeOut(flying_a, run_time=0.3), lag_ratio=1.0),
                mn.AnimationGroup(mn.FadeIn(bars_a), mn.FadeIn(name_a), mn.Create(membrane_a), mn.FadeIn(membrane_label), lag_ratio=0.2),
                lag_ratio=1.0),
        chapter(seconds[5], *swap(chapter_titles, captions, 5), mn.FadeIn(mark_spine), mn.FadeIn(cone_spine),
                mn.AnimationGroup(mn.FadeIn(rim_spine, target_position=lens_top, scale=R_SPINE_MARK / R_LENS),
                                  mn.FadeIn(lens_spine, target_position=lens_top, scale=R_SPINE_MARK / R_LENS)),
                mn.FadeIn(glu_label_a), mn.Create(glu_path_a), mn.FadeIn(glu_bar_a), lag_ratio=0.2),
        chapter(seconds[6],
                mn.AnimationGroup(*swap(chapter_titles, captions, 6), mn.FadeIn(box_b), lag_ratio=0.1),
                mn.AnimationGroup(mn.FadeIn(flying_b, run_time=0.2),
                                  mn.AnimationGroup(mn.Transform(flying_b, frame_b), mn.Transform(segment_b, zoom_b), run_time=1.2),
                                  mn.FadeOut(flying_b, run_time=0.3), lag_ratio=1.0),
                mn.AnimationGroup(mn.FadeIn(bars_b), mn.FadeIn(name_b), mn.FadeIn(glu_label_b), lag_ratio=0.2),
                mn.AnimationGroup(mn.Create(builds[0]), mn.Create(glu_builds[0]), mn.FadeIn(counts[0]), run_time=0.6),
                mn.Wait(run_time=0.4),
                build_step(builds, glu_builds, counts, 1), mn.Wait(run_time=0.4), build_step(builds, glu_builds, counts, 2), mn.Wait(run_time=0.4),
                build_step(builds, glu_builds, counts, 3), mn.Wait(run_time=0.4), build_step(builds, glu_builds, counts, 4),
                lag_ratio=1.0),
        chapter(seconds[7],
                mn.AnimationGroup(*swap(chapter_titles, captions, 7), *[mn.FadeOut(m) for m in lower], mn.FadeOut(legend), mn.FadeOut(site_top), mn.FadeOut(bars_top), mn.FadeOut(bar_dff_flipped), run_time=1.0),
                mn.AnimationGroup(mn.Transform(trace_top, branch_mid), mn.Transform(ticks_in, ticks_in_slot), mn.Transform(ticks_ap, ticks_ap_slot),
                                  mn.Transform(border_branch, rim_slot), mn.Transform(lens_branch, lens_branch_slot), mn.Transform(mark_spine, mark_slot),
                                  mn.Transform(lines_branch, cone(site_at, r_site, lens_mid, R_LENS, ACCENT)), run_time=1.2),
                mn.AnimationGroup(mn.FadeIn(site_labels["branch"]),
                                  mn.AnimationGroup(mn.FadeIn(cone_spine_slot), mn.FadeIn(rim_spine_slot, target_position=lens_mid, scale=R_SPINE_MARK / R_LENS),
                                                    mn.FadeIn(lens_spine_slot, target_position=lens_mid, scale=R_SPINE_MARK / R_LENS)),
                                  mn.Create(traces["base"]["spine"]), mn.FadeIn(site_labels["spine"]),
                                  mn.AnimationGroup(mn.FadeIn(cone_soma), mn.FadeIn(rim_soma, target_position=soma_at, scale=r_soma / R_LENS),
                                                    mn.FadeIn(lens_soma, target_position=soma_at, scale=r_soma / R_LENS)),
                                  mn.Create(traces["base"]["soma"]), mn.FadeIn(site_labels["soma"]), mn.FadeIn(bars_slots), mn.FadeIn(bar_dff_slots), lag_ratio=0.15),
                mn.FadeIn(box_ap),
                lag_ratio=1.0),
        chapter(seconds[8],
                mn.AnimationGroup(*swap(chapter_titles, captions, 8), run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(bands["base"]), mn.Create(epsp_paths["base"]), mn.FadeIn(bars_e), mn.FadeIn(label_e), mn.FadeIn(label_band),
                                  mn.FadeIn(name_epsp), mn.FadeIn(value_epsp), lag_ratio=0.15),
                mn.AnimationGroup(mn.Create(bap_paths["base"]["soma"]), mn.Create(bap_paths["base"]["branch"]), mn.FadeIn(bars_bap), mn.FadeIn(label_bap),
                                  mn.FadeIn(name_bap), mn.FadeIn(value_bap), lag_ratio=0.15),
                lag_ratio=1.0),
        morph(seconds[9], swap(chapter_titles, captions, 9), pieces, "base", "noisy", cells, value_epsp, value_bap),
        morph(seconds[10], swap(chapter_titles, captions, 10), pieces, "noisy", "pairs", cells, value_epsp, value_bap),
        morph(seconds[11], [*swap(chapter_titles, captions, 11), *replace(ring_branch, ring_trunk), *replace(route_side, route_trunk), mn.FadeOut(lens_branch), mn.FadeIn(lens_trunk),
                            mn.FadeOut(mark_spine), mn.FadeOut(cone_spine_slot), mn.Transform(lines_branch, cone(trunk_at, r_site, lens_mid, R_LENS, ACCENT))],
              pieces, "pairs", "far", cells, value_epsp, value_bap,
              (mn.MoveAlongPath(ring_trunk, route_far.copy(), rate_func=mn.smooth), mn.Create(route_far, rate_func=mn.smooth),
               mn.Transform(lines_branch, cone(far_at, r_site, lens_mid, R_LENS, ACCENT)), Play(lens_trunk, tiles),
               Count(site_labels["branch"], p_base.distance, distance_far, branch_name, decimals=0))),
        chapter(seconds[12],
                mn.AnimationGroup(*swap(chapter_titles, captions, 12), *[mn.FadeOut(m) for m in after_pool], mn.FadeOut(ticks_in), mn.FadeOut(ticks_ap),
                                  mn.Transform(trace_top, trace_again), run_time=1.5),
                mn.AnimationGroup(mn.FadeIn(chrome_again), mn.FadeIn(header),
                                  mn.LaggedStart(*[mn.AnimationGroup(mn.Create(snippet), mn.FadeIn(bar), *[mn.FadeIn(text) for text in texts], lag_ratio=0.25)
                                                   for snippet, bar, texts, *_ in table], lag_ratio=0.5),
                                  mn.FadeIn(time_bar_table), lag_ratio=0.3),
                lag_ratio=1.0),
    ]
    return timeline([], chapters)


if __name__ == "__main__":
    cells_all = simulate_all()
    kit.run("subcellular", CHAPTERS, UNTITLED, functools.partial(build, cells_all), __doc__)
