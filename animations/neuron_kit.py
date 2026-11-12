"""
The kit the synapse_to_soma, epsp and bap animations share, on top of kit.py: the MICrONS cell as a small panel that a
simulated voltage can run along, its round close-ups, the synapse's micrograph, and traces stacked on one time axis
with a cursor. All three use the same cell, colors and layout, so one can cut into another.
"""

import functools
from dataclasses import dataclass

import manim as mn
import numpy as np
from PIL import Image

import kit
from kit import ACCENT, BAR, DIM, HIGHLIGHT, KEY, TEXT, TRACE, TRUTH, Count, Play, at, axes_points, color, hbar, label, path, plain_axes, put, vbar

M = kit.lesson_model("synapse_to_soma")

GLU, SPIKE = TRUTH, HIGHLIGHT  # glutamate at the synapse; somatic spikes and bAPs
SITE_COLORS = {"spine": TRUTH, "branch": ACCENT, "soma": HIGHLIGHT}
SITE_NAMES = {"spine": "Spine 1", "branch": f"Branch, {M.DISTANCE_UM:g} um from the soma", "soma": "Soma"}
H_CELL, X_CELL, Y_CELL = 5.2, -5.5, 0.55  # the cell, on the left from the first frame
W_ROW, H_ROW, X_ROW = 8.0, 0.95, 1.5  # one trace, scene units; scale bars sit in the column right of it
Y_ROWS = {"top": 2.55, "spine": 1.25, "branch": -0.05, "soma": -1.35}
X_LENS, R_LENS = -3.1, 0.42
R_SOMA, R_SITE = 0.13, 0.09  # the rings on the cell
SOFT_EDGE_PX = 20  # the cell's crop fades out at its sides, so branches do not stop at a hard line
HEAT_FLOOR_MV, HEAT_TOP_MV = 0.3, 100.0  # the voltage color scale is logarithmic: a 3 mV EPSP and a 100 mV spike on one scale
HEAT_STOPS = np.array([0.0, 0.35, 0.7, 1.0])
HEAT_COLORS = np.array([(0.45, 0.16, 0.10), HIGHLIGHT, KEY, (1.0, 1.0, 0.92)])
SYNAPSE_PX = 660  # synapse.png's side
SYNAPSE_UM = 660 * 0.008 / 3
CLEFT_PX = np.array([335.0, 190.0])  # the cleft above the dark postsynaptic density, in synapse.png
LENS_CLEFT_PX = 300  # the side of the round close-up of the cleft


def ms(seconds: float) -> str:
    return f"{1000 * seconds:g} ms"


def mv_text(mv: float) -> str:
    return f"{mv:.1f}" if mv < 20 else f"{mv:.0f}"


def pct(dff: float) -> str:
    return f"{100 * dff:g}%"


def cap(text: str) -> mn.Text:
    """A chapter's one-line caption, small and grey at the bottom."""
    return put(label(text, DIM, 18), 0.0, -3.62)


def piecewise(alpha: float, knots_alpha: tuple, knots_value: tuple) -> float:
    """A rate function: linear between knots, for a sweep that slows down where the voltage moves fastest."""
    return float(np.interp(alpha, knots_alpha, knots_value))


def warp(knots_alpha: tuple, knots_value: tuple) -> functools.partial:
    return functools.partial(piecewise, knots_alpha=knots_alpha, knots_value=knots_value)


# ----------------------------------------------------------------------
# the cell and the voltage on it
# ----------------------------------------------------------------------


def min_pool(x: np.ndarray) -> np.ndarray:
    """Each 2 x 2 block's smallest value: a thin branch keeps its path distance at half resolution."""
    height, width = x.shape[0] // 2 * 2, x.shape[1] // 2 * 2
    return x[:height, :width].reshape(height // 2, 2, width // 2, 2).min(axis=(1, 3))


@dataclass
class Painter:
    """The cell's crop at half resolution, and each drawn pixel's path distance."""

    cell: M.Cell
    rgba: np.ndarray  # (height, width, 4) uint8
    drawn: np.ndarray  # (height, width) bool
    um_spine: np.ndarray  # (n,)
    um_soma: np.ndarray  # (n,)


def new_painter() -> Painter:
    cell = M.load_cell()
    x0, y0, x1, y1 = cell.tall_crop
    full = np.array(Image.open(M.DIR_NEURON / "cell.png"))[y0:y1, x0:x1]
    columns = np.arange(full.shape[1])
    full[..., 3] = (full[..., 3] * np.clip(np.minimum(columns, columns[::-1]) / SOFT_EDGE_PX, 0.0, 1.0)).astype(np.uint8)
    rgba = np.asarray(Image.fromarray(full).resize((full.shape[1] // 2, full.shape[0] // 2), Image.LANCZOS)).copy()
    paths = {name: min_pool(um[y0:y1, x0:x1]) for name, um in M.load_paths().items()}
    drawn = np.isfinite(paths["soma"]) & np.isfinite(paths["spine"])
    return Painter(cell, rgba, drawn, paths["spine"][drawn], paths["soma"][drawn])


def heat_level(v_mv: np.ndarray) -> np.ndarray:
    """Voltage above rest as 0 to 1 on the logarithmic color scale."""
    return np.clip(np.log10(np.maximum(v_mv, 1e-6) / HEAT_FLOOR_MV) / np.log10(HEAT_TOP_MV / HEAT_FLOOR_MV), 0.0, 1.0)


def heat_rgb(level: np.ndarray) -> np.ndarray:
    """(n,) levels as (n, 3) colors in 0 to 1."""
    return np.stack([np.interp(level, HEAT_STOPS, HEAT_COLORS[:, k]) for k in range(3)], axis=-1)


def painted(p: Painter, t: float, events: M.Events) -> np.ndarray:
    """The cell at time t s as rgba, each pixel colored by its voltage."""
    level = heat_level(M.voltage_mv(p.um_spine, p.um_soma, t, events))
    mix = np.clip(level / 0.12, 0.0, 1.0)[:, None]
    out = p.rgba.copy()
    pixels = out[p.drawn].astype(np.float32)
    pixels[:, :3] = (1 - mix) * pixels[:, :3] + mix * 255.0 * heat_rgb(level)
    pixels[:, 3] = np.maximum(pixels[:, 3], 255.0 * mix[:, 0])
    out[p.drawn] = pixels.astype(np.uint8)
    return out


def heat_frames(p: Painter, events: M.Events, times: np.ndarray) -> list:
    """The cell at each time, then at rest, so a voltage that has not quite decayed does not stay on screen."""
    return [*[painted(p, float(t), events) for t in times], p.rgba]


def cell_panel(p: Painter) -> mn.ImageMobject:
    mobject = mn.ImageMobject(p.rgba)
    mobject.set_resampling_algorithm(mn.RESAMPLING_ALGORITHMS["bicubic"])
    mobject.height = H_CELL
    return put(mobject, X_CELL, Y_CELL)


def on_cell(image: mn.ImageMobject, p: Painter, xy: np.ndarray) -> np.ndarray:
    """A pixel of cell.png as a scene point on the cell panel."""
    left, top = image.get_corner(mn.UL)[:2]
    x0, y0, x1, _ = p.cell.tall_crop
    scale = image.width / (x1 - x0)
    return np.array([left + scale * (xy[0] - x0), top - scale * (xy[1] - y0), 0.0])


def credit() -> mn.VGroup:
    """The citation MICrONS asks for, under the cell."""
    lines = mn.VGroup(*[label(line, DIM, 14) for line in (M.NEURON_CITATION, M.NEURON_SITE)]).arrange(mn.DOWN, aligned_edge=mn.LEFT, buff=0.05)
    return lines.move_to(at(-6.95, -2.82), aligned_edge=mn.UL)


def heat_legend() -> mn.Group:
    """The color scale with its ticks, under the cell."""
    width, x, y = 2.6, X_CELL, -2.42
    level = np.linspace(0.0, 1.0, 256)
    bar = mn.ImageMobject((255 * heat_rgb(level))[None].repeat(4, axis=0).astype(np.uint8))
    bar.set_resampling_algorithm(mn.RESAMPLING_ALGORITHMS["bilinear"])
    bar.stretch_to_fit_width(width).stretch_to_fit_height(0.09)
    put(bar, x, y)
    ticks = [label(f"{mv:g}", BAR, 14).move_to(at(x - 0.5 * width + width * float(heat_level(np.array(mv))), y - 0.17)) for mv in (1, 10, 100)]
    name = label("mV above rest (simulated)", BAR, 14).move_to(at(x, y + 0.17))
    return mn.Group(bar, name, *ticks)


@dataclass
class CellView:
    """The cell on the left with its rings, the names beside them, the citation and the color scale."""

    image: mn.ImageMobject
    soma_at: np.ndarray
    site_at: np.ndarray
    ring_soma: mn.Circle
    ring_site: mn.Circle
    marks: list  # the rings and their names, in draw order
    credit: mn.VGroup
    legend: mn.Group


def new_cell_view(p: Painter) -> CellView:
    image = cell_panel(p)
    soma_at, site_at = on_cell(image, p, p.cell.soma), on_cell(image, p, p.cell.site)
    ring_soma, ring_site = ring_at(soma_at, R_SOMA, SPIKE), ring_at(site_at, R_SITE, GLU)
    name_soma = label("Soma", SPIKE, 17).next_to(ring_soma, mn.DR, buff=0.03)
    name_site = label("Spine 1", GLU, 17).next_to(ring_site, mn.UR, buff=0.03)
    patch_soma, patch_site = [mn.BackgroundRectangle(name, color=color(kit.BG), fill_opacity=0.85, buff=0.03) for name in (name_soma, name_site)]
    return CellView(image, soma_at, site_at, ring_soma, ring_site, [ring_soma, patch_soma, name_soma, ring_site, patch_site, name_site], credit(), heat_legend())


def opening(seconds: float, view: CellView, title: mn.Text) -> mn.AnimationGroup:
    """The first chapter of all three animations: the headline title, the cell and its marks."""
    return kit.chapter(seconds,
                       mn.AnimationGroup(mn.FadeIn(title), mn.FadeIn(view.image), mn.FadeIn(view.credit), lag_ratio=0.1),
                       mn.AnimationGroup(*[mn.FadeIn(m) for m in view.marks], lag_ratio=0.15),
                       lag_ratio=1.0)


# ----------------------------------------------------------------------
# rings, close-ups and the synapse
# ----------------------------------------------------------------------


def ring_at(point: np.ndarray, radius: float, c: tuple) -> mn.Circle:
    return mn.Circle(radius=radius, color=color(c), stroke_width=2.5).move_to(point)


def zoom_lines(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, line_color: tuple) -> mn.VGroup:
    """Thin lines from a to b and c to d: a ring's edges to its panel's corners."""
    return mn.VGroup(*[mn.Line(start, end, color=color(line_color), stroke_width=1.5) for start, end in ((a, b), (c, d))]).set_stroke(opacity=0.7)


def cone(a: np.ndarray, radius_a: float, b: np.ndarray, radius_b: float, c: tuple) -> mn.VGroup:
    """Two thin lines opening from a circle at a to a larger one at b: b is a close-up of a."""
    along = (b - a) / np.linalg.norm(b - a)
    side = np.array([-along[1], along[0], 0.0])
    lines = [mn.Line(a + s * radius_a * side, b + s * radius_b * side, color=color(c), stroke_width=1.5) for s in (-1, 1)]
    return mn.VGroup(*lines).set_stroke(opacity=0.7)


def rounded(rgba: np.ndarray) -> np.ndarray:
    """A square rgba with everything outside its inscribed circle transparent."""
    size = rgba.shape[0]
    rows, columns = np.ogrid[:size, :size]
    inside = (rows - 0.5 * (size - 1)) ** 2 + (columns - 0.5 * (size - 1)) ** 2 <= (0.5 * size) ** 2
    return np.dstack([rgba[..., :3], rgba[..., 3] * inside]).astype(np.uint8)


def lens_image(rgba: np.ndarray, y: float, x: float = X_LENS, radius: float = R_LENS) -> mn.ImageMobject:
    mobject = mn.ImageMobject(rounded(rgba))
    mobject.set_resampling_algorithm(mn.RESAMPLING_ALGORITHMS["bicubic"])
    mobject.height = 2 * radius
    return put(mobject, x, y)


def lens_ring(y: float, c: tuple, x: float = X_LENS, radius: float = R_LENS) -> mn.Circle:
    """The rim of a close-up, filled so nothing shows through it."""
    return mn.Circle(radius=radius, color=color(c), stroke_width=2.5, fill_color=color(kit.BG), fill_opacity=1.0).move_to(at(x, y))


def site_rgba(site: str, p: Painter) -> np.ndarray:
    """The square picture a site's close-up shows."""
    if site == "branch":
        x0, y0, x1, y1 = np.round(p.cell.lens_branch).astype(int)
        return np.asarray(Image.open(M.DIR_NEURON / "branch.png"))[y0:y1, x0:x1]
    return np.asarray(Image.open(M.DIR_NEURON / f"{site}.png"))


def cleft_rgba() -> np.ndarray:
    """The round close-up's square of the micrograph, centered on the cleft."""
    x, y = np.round(CLEFT_PX).astype(int) + np.array([0, 40])
    half = LENS_CLEFT_PX // 2
    gray = np.asarray(Image.open(M.DIR_NEURON / "synapse.png").convert("RGBA"))
    return gray[y - half:y + half, x - half:x + half]


def synapse_panel(height: float, x: float, y: float) -> tuple[mn.ImageMobject, mn.ImageMobject]:
    """The micrograph and, over it, the segmentation's outlines of the bouton in green and the spine in white."""
    micrograph = mn.ImageMobject(np.asarray(Image.open(M.DIR_NEURON / "synapse.png").convert("RGBA")))
    micrograph.set_resampling_algorithm(mn.RESAMPLING_ALGORITHMS["bicubic"])
    micrograph.height = height
    outlines = M.load_outlines()
    rgba = np.zeros((*outlines["spine"].shape, 4), np.uint8)
    rgba[outlines["bouton"]] = (*[int(255 * v) for v in GLU], 255)
    rgba[outlines["spine"]] = (*[int(255 * v) for v in TEXT], 255)
    drawn = mn.ImageMobject(rgba)
    drawn.set_resampling_algorithm(mn.RESAMPLING_ALGORITHMS["bicubic"])
    drawn.height = height
    return put(micrograph, x, y), put(drawn, x, y)


def on_synapse(image: mn.ImageMobject, xy: np.ndarray) -> np.ndarray:
    """A pixel of synapse.png as a scene point."""
    left, top = image.get_corner(mn.UL)[:2]
    scale = image.width / SYNAPSE_PX
    return np.array([left + scale * xy[0], top - scale * xy[1], 0.0])


def glutamate_puff(image: mn.ImageMobject, seed: int) -> mn.VGroup:
    """Dots of glutamate in the cleft, drawn as a schematic over the micrograph."""
    rng = np.random.default_rng(seed)
    points = CLEFT_PX + rng.normal(size=(14, 2)) * np.array([38.0, 9.0])
    return mn.VGroup(*[mn.Dot(on_synapse(image, xy), radius=0.032, color=color(GLU)) for xy in points])


def puffs_during(image: mn.ImageMobject, times: np.ndarray, start: float, stop: float, seconds: float) -> mn.AnimationGroup:
    """Glutamate in the cleft at each time, on the schedule of a sweep from start to stop over seconds."""
    sequence, clock, drop = [], 0.0, at(0.0, -0.1)
    for k, time in enumerate(times):
        at_release = seconds * (time - start) / (stop - start)
        puff = glutamate_puff(image, k)
        sequence += [mn.Wait(run_time=at_release - clock), mn.FadeIn(puff, shift=drop, run_time=0.2), mn.FadeOut(puff, shift=0.5 * drop, run_time=0.45)]
        clock = at_release + 0.65
    sequence.append(mn.Wait(run_time=max(seconds - clock, 0.01)))
    return mn.AnimationGroup(*sequence, lag_ratio=1.0)


def um_bar(left: np.ndarray, length: float, text: str) -> mn.VGroup:
    """A scale bar starting at a scene point, with its length under it."""
    line = mn.Line(left, left + at(length, 0.0), color=color(BAR), stroke_width=3.0)
    return mn.VGroup(line, label(text, BAR, 15).next_to(line, mn.DOWN, buff=0.06))


@dataclass
class SynapseView:
    """The micrograph of Spine 1's synapse, the ring that flies out to it and everything drawn on it."""

    micrograph: mn.ImageMobject
    outlines: mn.ImageMobject
    flying: mn.Circle
    frame: mn.Rectangle
    border: mn.Rectangle
    lines: mn.VGroup
    marks: list  # names, their patches and the scale bar, in draw order


def new_synapse_view(view: CellView, height: float, x: float, y: float) -> SynapseView:
    micrograph, outlines = synapse_panel(height, x, y)
    lines = zoom_lines(view.ring_site.get_top(), micrograph.get_corner(mn.UL), view.ring_site.get_bottom(), micrograph.get_corner(mn.DL), GLU)
    name_bouton = label("Bouton", GLU, 19).move_to(on_synapse(micrograph, np.array([170.0, 150.0])))
    name_spine = label("Spine 1", TEXT, 19).move_to(on_synapse(micrograph, np.array([325.0, 330.0])))
    name_psd = label("Postsynaptic density", TEXT, 16).move_to(on_synapse(micrograph, np.array([450.0, 70.0])))
    line_psd = mn.Line(name_psd.get_bottom(), on_synapse(micrograph, np.array([345.0, 205.0])), color=color(TEXT), stroke_width=1.5)
    patch_bouton, patch_spine, patch_psd = [mn.BackgroundRectangle(name, color=color(kit.BG), fill_opacity=0.7, buff=0.05) for name in (name_bouton, name_spine, name_psd)]
    bar = um_bar(micrograph.get_corner(mn.DR) + at(-1.0, 0.42), height * 0.25 / SYNAPSE_UM, "250 nm")
    patch_bar = mn.BackgroundRectangle(bar, color=color(kit.BG), fill_opacity=0.7, buff=0.06)
    marks = [patch_bouton, name_bouton, patch_spine, name_spine, patch_psd, name_psd, line_psd, patch_bar, bar]
    return SynapseView(micrograph, outlines, view.ring_site.copy(), kit.frame_of(micrograph, GLU), kit.frame_of(micrograph, GLU).set_fill(color(kit.BG), opacity=0.0), lines, marks)


def synapse_arrives(s: SynapseView, title: mn.Text, caption: mn.Text) -> list:
    """The headline title leaves, the ring flies out into the micrograph and its labels come in. Play with lag_ratio 1."""
    return [mn.AnimationGroup(mn.FadeOut(title), mn.FadeIn(s.flying, run_time=0.2), lag_ratio=0.2),
            mn.Transform(s.flying, s.frame, run_time=1.4),
            mn.AnimationGroup(mn.FadeIn(s.micrograph), mn.FadeIn(s.border), mn.FadeOut(s.flying), mn.FadeIn(s.lines), lag_ratio=0.1),
            mn.AnimationGroup(mn.FadeIn(s.outlines), *[mn.FadeIn(m) for m in s.marks], mn.FadeIn(caption), lag_ratio=0.1)]


def synapse_to_lens(s: SynapseView, lens: mn.ImageMobject, rim: mn.Circle) -> list:
    """The micrograph shrinks into its round close-up, and everything drawn on it leaves."""
    scale = lens.height / s.micrograph.height
    return [mn.FadeOut(s.outlines), mn.FadeOut(s.border), mn.FadeOut(s.lines), *[mn.FadeOut(m) for m in s.marks],
            mn.FadeOut(s.micrograph, target_position=lens.get_center(), scale=scale),
            mn.FadeIn(rim, target_position=s.micrograph.get_center(), scale=1 / scale),
            mn.FadeIn(lens, target_position=s.micrograph.get_center(), scale=1 / scale)]


def site_column(p: Painter) -> tuple:
    """Each site's close-up with its rim, in the column left of the traces: (rims, close-ups)."""
    return ({site: lens_ring(Y_ROWS[site], SITE_COLORS[site]) for site in M.SITES},
            {site: lens_image(site_rgba(site, p), Y_ROWS[site]) for site in M.SITES})


def site_cones(view: CellView) -> list:
    """Lines from the rings on the cell to the branch and soma close-ups, and from the branch's close-up to the spine's."""
    lens_at = {site: at(X_LENS, Y_ROWS[site]) for site in M.SITES}
    return [cone(view.site_at, R_SITE, lens_at["branch"], R_LENS, SITE_COLORS["branch"]),
            cone(view.soma_at, R_SOMA, lens_at["soma"], R_LENS, SPIKE),
            cone(lens_at["branch"], 0.1, lens_at["spine"], R_LENS, GLU)]


# ----------------------------------------------------------------------
# traces on one time axis
# ----------------------------------------------------------------------


@dataclass
class Row:
    """One trace's axes and limits."""

    axes: mn.Axes
    lo: float
    hi: float
    start: float
    stop: float


def new_row(start: float, stop: float, values: np.ndarray, y: float, x: float = X_ROW, width: float = W_ROW, height: float = H_ROW, top: float = None) -> Row:
    """Axes for values from start to stop s. top fixes the upper limit, for rows sharing one scale."""
    hi = 1.08 * (float(values.max()) if top is None else top)
    lo = min(float(values.min()) * 1.08, -0.06 * hi)
    return Row(put(plain_axes(start, stop, lo, hi, width, height, visible=False), x, y), lo, hi, start, stop)


def row_path(row: Row, t: np.ndarray, values: np.ndarray, c: tuple = TRACE, stroke_width: float = 1.7) -> mn.VMobject:
    return path(row.axes, t, values, c, stroke_width, row.lo, row.hi)


def right_bar(row: Row, size: float, text: str) -> mn.VGroup:
    """A vertical bar at the row's top right, its size written to its right, clear of the trace and the cursor."""
    bar = vbar(row.axes, row.stop, row.hi - size, row.hi, text)
    bar[1].next_to(bar[0], mn.RIGHT, buff=0.08)
    return bar


def dff_bar(row: Row, dff: float) -> mn.VGroup:
    return right_bar(row, dff, f"{pct(dff)} -dF/F")


def mv_bar(row: Row, mv: float) -> mn.VGroup:
    return right_bar(row, mv, f"{mv:g} mV")


def row_name(row: Row, text: str, c: tuple) -> mn.Text:
    return label(text, c, 17).next_to(row.axes, mn.UP, buff=0.0).align_to(row.axes, mn.LEFT)


def time_bar(row: Row, seconds: float) -> mn.VGroup:
    return hbar(row.axes, row.stop - seconds, row.stop, row.lo, ms(seconds), 0.2)


def cursor_over(rows: list, start: float) -> mn.VGroup:
    """A thin line through each row at time start, broken between rows so it never crosses a row's name."""
    x = axes_points(rows[0].axes, np.array([start]), np.array([0.0]))[0, 0]
    return mn.VGroup(*[mn.Line(np.array([x, row.axes.get_top()[1], 0.0]), np.array([x, row.axes.get_bottom()[1], 0.0]), color=color(DIM), stroke_width=1.5) for row in rows])


def cursor_path(cursor: mn.VGroup, row: Row) -> mn.Line:
    """The line the cursor's center runs along, from the row's start to its stop."""
    y = cursor.get_center()[1]
    x0, x1 = axes_points(row.axes, np.array([row.start, row.stop]), np.array([0.0, 0.0]))[:, 0]
    return mn.Line(np.array([x0, y, 0.0]), np.array([x1, y, 0.0]))


def clock_text(milliseconds: float) -> mn.Text:
    return label(f"{milliseconds:.1f} ms", TEXT, 20)


def new_clock(start: float) -> mn.Text:
    """The time under the cursor, above the rows at the right."""
    return clock_text(1000 * start).move_to(at(X_ROW + 0.5 * W_ROW - 1.1, 3.28), aligned_edge=mn.LEFT)


def sweep(paths: list, cursor: mn.VGroup, row: Row, clock: mn.Text, image: mn.ImageMobject, frames: list, seconds: float, rate=mn.linear) -> mn.AnimationGroup:
    """Traces drawing themselves under a moving cursor, the clock counting and the cell's voltage following, together."""
    return mn.AnimationGroup(*[mn.Create(p, rate_func=rate) for p in paths],
                             mn.MoveAlongPath(cursor, cursor_path(cursor, row), rate_func=rate),
                             Count(clock, 1000 * row.start, 1000 * row.stop, clock_text, decimals=1, rate_func=rate),
                             Play(image, frames, rate_func=mn.linear), run_time=seconds)


def sweep_times(row: Row, seconds: float, rate=mn.linear) -> np.ndarray:
    """The simulated time of each animation frame of a sweep."""
    alphas = np.linspace(0.0, 1.0, int(seconds * kit.FPS))
    return row.start + (row.stop - row.start) * np.array([rate(a) for a in alphas])


def normalized_pair(axes: mn.Axes, t: np.ndarray, first: np.ndarray, second: np.ndarray, colors: tuple) -> tuple:
    """Two responses scaled to their own peaks, each with a line down from its peak, and the times of those peaks."""
    paths, marks, peaks = [], [], []
    for v, c in zip((first, second), colors):
        paths.append(path(axes, t, v / v.max(), c, 2.4, -0.1, 1.15))
        t_peak = float(t[int(np.argmax(v))])
        peaks.append(t_peak)
        marks.append(mn.DashedLine(*axes_points(axes, np.array([t_peak, t_peak]), np.array([1.0, -0.1])), color=color(c), stroke_width=1.5, dash_length=0.06))
    return paths, marks, peaks


def delay_span(axes: mn.Axes, peaks: list) -> mn.Line:
    """The bar under two peaks. Peaks in one frame still get a sliver, so the bar can morph to nothing."""
    a, b = axes_points(axes, np.array(peaks), np.array([-0.1, -0.1]))
    return mn.Line(a, b + at(0.02, 0.0), color=color(TEXT), stroke_width=4.0)


def delay_number(milliseconds: float) -> mn.Text:
    return label(f"{milliseconds:.1f} ms", KEY, 34)


def delay_figure(x: float, y: float, width: float, height: float, t: np.ndarray, first: np.ndarray, second: np.ndarray, names: tuple, colors: tuple, heading: str, delay: float, bar_seconds: float) -> dict:
    """One event at two sites, each scaled to its peak, with the delay between the peaks under them and a time bar."""
    axes = put(plain_axes(float(t[0]), float(t[-1]), -0.1, 1.15, width, height, visible=False), x, y)
    paths, marks, peaks = normalized_pair(axes, t, first, second, colors)
    head = label(heading, TEXT, 20).next_to(axes, mn.UP, buff=0.3).align_to(axes, mn.LEFT)
    legend = mn.VGroup(*[label(name, c, 17) for name, c in zip(names, colors)]).arrange(mn.RIGHT, buff=0.35).next_to(axes, mn.UP, buff=0.0).align_to(axes, mn.LEFT)
    span = delay_span(axes, peaks)
    number = delay_number(1000 * delay).next_to(span, mn.DOWN, buff=0.12)
    bar = hbar(axes, float(t[-1]) - bar_seconds, float(t[-1]), -0.1, ms(bar_seconds), 0.06)
    return dict(axes=axes, paths=paths, marks=marks, head=head, legend=legend, span=span, number=number, bar=bar)


def morph(duration: float, lead: list, change: list) -> mn.AnimationGroup:
    """A chapter whose title comes first and alone, then everything on screen changes at once."""
    return kit.chapter(duration, mn.AnimationGroup(*lead, run_time=1.0), mn.AnimationGroup(*change, run_time=4.0), lag_ratio=1.0)


def table_rows(rows: list, y_top: float, pitch: float, xs: tuple, top: float = None) -> list:
    """(snippet, texts) per row of (name, color, t, trace, cells): a piece of trace, the row's name, then each cell at its x."""
    out = []
    for k, (name, c, t, trace, cells) in enumerate(rows):
        y = y_top - k * pitch
        row = new_row(float(t[0]), float(t[-1]), trace, y, x=-2.05, width=1.9, height=0.62 * pitch, top=top)
        texts = [label(name, c, 19).move_to(at(-0.85, y), aligned_edge=mn.LEFT)]
        texts += [label(text, TEXT, 19).move_to(at(x, y), aligned_edge=mn.LEFT) for text, x in zip(cells, xs)]
        out.append((row_path(row, t, trace, c, 1.8), texts))
    return out


def table_in(table: list) -> mn.LaggedStart:
    return mn.LaggedStart(*[mn.AnimationGroup(mn.Create(snippet), *[mn.FadeIn(text) for text in texts], lag_ratio=0.2) for snippet, texts in table], lag_ratio=0.5)
