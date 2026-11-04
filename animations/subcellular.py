# /// script
# requires-python = ">=3.11"
# dependencies = ["manim>=0.22", "numpy", "pillow<13"]
# ///
"""
Subthreshold voltage in a dendrite as one animation: a branch's voltage-indicator trace, one EPSP zoomed, the glutamate
pulse at the spine that caused it, five inputs summing into a dendritic spike, the same cell at a spine, the branch and
the soma, two ratios (EPSP over noise, bAP at the branch over the somatic spike), the picture morphing as noise, a
second input and distance change, and a table of the four shapes. Built from imgui_ophys's subcellular model with
the input and spike times known; drawn with kit.py (manim through a Cairo camera, no Scene).

    uv run -p 3.12 animations/subcellular.py --out <dir> [--stills-only] [--gif-width 720]

Text only, no LaTeX. Needs ffmpeg and uv on PATH.
"""

import functools
from dataclasses import dataclass

import manim as mn
import numpy as np

import kit
from kit import (ACCENT, BAR, DIM, HIGHLIGHT, KEY, TEXT, TRACE, TRUTH, Count, at, axes_points, chapter, color, hbar, label, path,
                 plain_axes, put, replace, timeline, vbar)

M = kit.lesson_model("subcellular")

GLU = TRUTH  # glutamate at a spine, and the input ticks: the ground truth
SPIKE = HIGHLIGHT  # somatic spikes and the bAP
DSPIKE = KEY  # the dendritic spike
W_TRACE, H_TRACE, Y_TRACE = 13.0, 1.5, 2.1  # the branch's trace across the top, scene units
W_ZOOM, H_ZOOM, Y_ZOOM = 4.6, 2.3, -1.7  # the zooms, lower left and lower middle
H_STRIP, Y_STRIP = 0.7, 0.0  # the glutamate strip above each zoom
X_ZOOM_A, X_ZOOM_B = -4.3, 1.4
W_SLOT, H_SLOT, X_SLOT = 8.4, 1.3, -2.4  # the three sites' traces, one slot each
Y_SLOTS = {"spine": 1.9, "branch": 0.2, "soma": -1.5}
X_FIG, W_FIG, H_FIG = 4.9, 3.2, 1.1  # the two figures the ratios come from, right
Y_FIG_EPSP, Y_FIG_BAP = 1.5, -1.0
X_NUMBER = 3.0
TIME_BAR_TOP = 0.2  # s
ZOOM_HALF = 0.05  # s each side of the followed input

TITLE = "Subthreshold voltage: a dendrite's potential below the level that fires a spike"
CHAPTERS = (
    ("", 5.5),
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
UNTITLED = {0: "the-branch", 1: "one-epsp", 2: "glutamate-at-the-spine", 3: "inputs-sum", 4: "spine-branch-soma", 5: "two-ratios", 9: "shapes"}
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


def morph(duration: float, lead: list, p: Pieces, src: str, dst: str, cells: dict, value_epsp: mn.Text, value_bap: mn.Text) -> mn.AnimationGroup:
    """The chapter's title first, then everything on screen turns from cell src into cell dst at once, numbers counting."""
    change = mn.AnimationGroup(
        *[mn.Transform(p.traces["base"][site], p.traces[dst][site]) for site in M.SITES],
        mn.Transform(p.epsp["base"], p.epsp[dst]), mn.Transform(p.band["base"], p.band[dst]),
        *[mn.Transform(p.bap["base"][site], p.bap[dst][site]) for site in ("soma", "branch")],
        Count(value_epsp, cells[src].numbers.epsp_over_noise, cells[dst].numbers.epsp_over_noise, value_1),
        Count(value_bap, cells[src].numbers.bap_ratio, cells[dst].numbers.bap_ratio, value_2, decimals=2),
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
    captions[0] = cap("The indicator dims when the membrane depolarizes, so the trace is flipped: -dF/F puts depolarization up.")
    captions[1] = cap("One input depolarizes the branch by a few mV for about 15 ms: an EPSP (excitatory postsynaptic potential).")
    captions[2] = cap("The input is a pulse of glutamate landing on a spine; the EPSP follows under it within a millisecond.")
    captions[3] = cap("EPSPs within a few milliseconds of each other add up; past a threshold the branch fires its own spike, a dendritic spike.")
    captions[4] = cap("A somatic spike travels back into the branch, smaller and later with distance: a bAP (backpropagating action potential).")
    captions[5] = cap("The ratios are one EPSP over the noise SD at the branch, and the bAP at the branch over the spike at the soma.")

    # chapter 1: the branch's trace across the top, raw first (dipping), then flipped
    branch = base.sites["branch"]
    y_lo, y_hi = padded([-branch.dff], 0.06)
    axes_top = put(plain_axes(0, T, y_lo, y_hi, W_TRACE, H_TRACE, visible=False), 0.0, Y_TRACE)
    axes_raw = put(plain_axes(0, T, -y_hi, -y_lo, W_TRACE, H_TRACE, visible=False), 0.0, Y_TRACE)
    trace_top = path(axes_raw, t, branch.dff, TRACE, 1.3, -y_hi, -y_lo)
    trace_flipped = path(axes_top, t, -branch.dff, TRACE, 1.3, y_lo, y_hi)
    bars_top = bars_of(axes_top, T, y_lo, y_hi, TIME_BAR_TOP, 0.26, with_dff=False)
    dff_top = kit.nice(0.3 * (y_hi - y_lo))
    bar_dff = vbar(axes_top, T, y_hi - dff_top, y_hi, f"{pct(dff_top)} dF/F")
    bar_dff_flipped = vbar(axes_top, T, y_hi - dff_top, y_hi, f"{pct(dff_top)} -dF/F")
    site_top = label(f"Dendritic branch, {p_base.distance:g} um from the soma (simulated)", DIM, 18).next_to(axes_top, mn.UP, buff=0.06).align_to(axes_top, mn.LEFT)
    ticks_in = tick_marks(axes_top, np.unique(base.input_times), y_lo, y_hi, GLU)
    ticks_ap = tick_marks(axes_top, base.ap_times, y_lo, y_hi, SPIKE)
    legend = mn.VGroup(label("Synaptic inputs (simulated)", GLU, 18), label("Somatic spikes (simulated)", SPIKE, 18)).arrange(mn.RIGHT, buff=0.5)
    legend.next_to(ticks_in, mn.DOWN, buff=0.06).align_to(axes_top, mn.LEFT)

    # chapter 2: one input, boxed, flown into a zoom with the noiseless membrane over it
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
    membrane_label = label("Membrane voltage (noiseless)", ACCENT, 18)
    membrane_label.move_to(axes_a.get_corner(mn.UL) + at(0.1, -0.1), aligned_edge=mn.UL)

    # chapter 3: the glutamate pulse at the spine above the zoom
    glu_a = base.glutamate[0][sl_a]
    g_lo, g_hi = -0.05 * M.GLU_DFF, 1.1 * M.GLU_DFF
    strip_a = put(plain_axes(0, span_a, g_lo, g_hi, W_ZOOM, H_STRIP, visible=False), X_ZOOM_A, Y_STRIP)
    glu_path_a = path(strip_a, t_a, glu_a, GLU, 2.0, g_lo, g_hi)
    glu_label_a = label("Glutamate at spine 1 (simulated)", GLU, 18).next_to(strip_a, mn.UP, buff=0.04).align_to(strip_a, mn.LEFT)
    glu_bar_a = vbar(strip_a, span_a, g_lo + 0.05 * M.GLU_DFF, g_lo + 0.55 * M.GLU_DFF, f"{pct(0.5 * M.GLU_DFF)} dF/F")

    # chapter 4: the burst, boxed, flown into a second zoom; the noiseless response built one input at a time
    steps = M.burst_steps(p_base)
    b_start, b_end = M.burst_window(p_base)
    sl_b = window(t, b_start, b_end)
    rec_b = -branch.dff[sl_b]
    span_b = b_end - b_start
    b_lo, b_hi = padded([rec_b] + [-clean for _, clean, _ in steps], 0.1)
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
    counts = [label(text, DSPIKE if k == len(count_texts) - 1 else TEXT, 20).move_to(axes_b.get_corner(mn.UL) + at(0.1, -0.1), aligned_edge=mn.UL)
              for k, text in enumerate(count_texts)]

    # chapter 5: the three sites, one slot each, on one dF/F scale
    all_dff = [-c.cell.sites[site].dff for c in cells.values() for site in M.SITES]
    s_lo, s_hi = padded(all_dff, 0.04)
    axes_slot = {site: put(plain_axes(0, T, s_lo, s_hi, W_SLOT, H_SLOT, visible=False), X_SLOT, y) for site, y in Y_SLOTS.items()}
    traces = {name: {site: path(axes_slot[site], t, -c.cell.sites[site].dff, TRACE, 1.2, s_lo, s_hi) for site in M.SITES} for name, c in cells.items()}
    branch_mid = traces["base"]["branch"]
    traces["base"]["branch"] = trace_top  # the trace that is already on screen moves into its slot
    site_labels = {
        "spine": label("Spine 1", DIM, 18),
        "branch": label(f"Branch, {p_base.distance:g} um from the soma", DIM, 18),
        "soma": label("Soma", DIM, 18),
    }
    for site, text in site_labels.items():
        text.next_to(axes_slot[site], mn.UP, buff=0.02).align_to(axes_slot[site], mn.LEFT)
    bars_slots = bars_of(axes_slot["soma"], T, s_lo, s_hi, TIME_BAR_TOP, 0.26, with_dff=False)
    dff_slots = kit.nice(0.3 * (s_hi - s_lo))
    bar_dff_slots = vbar(axes_slot["spine"], T, s_hi - dff_slots, s_hi, f"{pct(dff_slots)} -dF/F")
    ticks_in_slot = tick_marks(axes_slot["branch"], np.unique(base.input_times), s_lo, s_hi, GLU)
    ticks_ap_slot = tick_marks(axes_slot["soma"], base.ap_times, s_lo, s_hi, SPIKE)
    t_ap = M.followed_ap(base)
    box_ap = box_on(axes_slot["spine"], t_ap - 0.01, t_ap + 0.02, s_lo, s_hi, Y_SLOTS["spine"] - Y_SLOTS["soma"] + H_SLOT)
    box_ap.move_to(at(box_ap.get_center()[0], 0.5 * (Y_SLOTS["spine"] + Y_SLOTS["soma"])))

    # chapter 6: the two figures and their ratios
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

    # chapter 10: the shapes
    rows = M.shapes(base, FS)
    for row in rows:
        print(f"{row.name:16s}: " + ", ".join(f"{s} {row.mv[s]:.1f} mV / {100 * row.dff[s]:.2f}%" for s in M.SITES) + f"; {1000 * row.duration:.1f} ms; seen at {', '.join(row.seen_at)}")
    trace_again = path(axes_top, t, -branch.dff, TRACE, 1.3, y_lo, y_hi)
    chrome_again = mn.VGroup(bars_of(axes_top, T, y_lo, y_hi, TIME_BAR_TOP, 0.26, with_dff=False),
                             vbar(axes_top, T, y_hi - dff_top, y_hi, f"{pct(dff_top)} -dF/F"), site_top.copy())
    y_header, pitch = 0.55, 0.95
    header = mn.VGroup(*[label(text, DIM, 18).move_to(at(x, y_header), aligned_edge=mn.LEFT)
                         for text, x in (("branch, -dF/F", -6.5), ("shape", -3.3), ("size at the branch", -0.6), ("duration", 1.9), ("seen at", 3.9))])
    table = table_rows(base, rows, y_header - 0.8, pitch)
    time_bar_table = hbar(table[-1][3], 0.04, 0.06, table[-1][4], "20 ms", 0.06)

    lower = [box_a, box_b, bars_a, bars_b, name_a, name_b, membrane_a, membrane_label, glu_path_a, glu_label_a, glu_bar_a,
             glu_label_b, builds[0], glu_builds[0], counts[-1], segment_a, segment_b]
    after_pool = [traces["base"]["spine"], traces["base"]["soma"], *site_labels.values(), bars_slots, bar_dff_slots, box_ap,
                  epsp_paths["base"], bands["base"], bars_e, label_e, label_band, bap_paths["base"]["soma"], bap_paths["base"]["branch"], bars_bap, label_bap,
                  name_epsp, value_epsp, name_bap, value_bap]

    chapters = [
        chapter(seconds[0],
                mn.AnimationGroup(mn.FadeIn(title), mn.FadeIn(site_top), mn.Create(trace_top), mn.FadeIn(bars_top), mn.FadeIn(bar_dff), lag_ratio=0.15),
                mn.Wait(run_time=0.6),
                mn.AnimationGroup(mn.FadeIn(captions[0]), mn.Transform(trace_top, trace_flipped), *replace(bar_dff, bar_dff_flipped), run_time=1.2),
                mn.AnimationGroup(mn.FadeIn(ticks_in), mn.FadeIn(ticks_ap), mn.FadeIn(legend), lag_ratio=0.2),
                lag_ratio=1.0),
        chapter(seconds[1],
                mn.AnimationGroup(mn.FadeOut(title), *swap(chapter_titles, captions, 1), mn.FadeIn(box_a), lag_ratio=0.1),
                mn.AnimationGroup(mn.FadeIn(flying_a, run_time=0.2),
                                  mn.AnimationGroup(mn.Transform(flying_a, frame_a), mn.Transform(segment_a, zoom_a), run_time=1.4),
                                  mn.FadeOut(flying_a, run_time=0.3), lag_ratio=1.0),
                mn.AnimationGroup(mn.FadeIn(bars_a), mn.FadeIn(name_a), mn.Create(membrane_a), mn.FadeIn(membrane_label), lag_ratio=0.2),
                lag_ratio=1.0),
        chapter(seconds[2], *swap(chapter_titles, captions, 2), mn.FadeIn(glu_label_a), mn.Create(glu_path_a), mn.FadeIn(glu_bar_a), lag_ratio=0.2),
        chapter(seconds[3],
                mn.AnimationGroup(*swap(chapter_titles, captions, 3), mn.FadeIn(box_b), lag_ratio=0.1),
                mn.AnimationGroup(mn.FadeIn(flying_b, run_time=0.2),
                                  mn.AnimationGroup(mn.Transform(flying_b, frame_b), mn.Transform(segment_b, zoom_b), run_time=1.2),
                                  mn.FadeOut(flying_b, run_time=0.3), lag_ratio=1.0),
                mn.AnimationGroup(mn.FadeIn(bars_b), mn.FadeIn(name_b), mn.FadeIn(glu_label_b), lag_ratio=0.2),
                mn.AnimationGroup(mn.Create(builds[0]), mn.Create(glu_builds[0]), mn.FadeIn(counts[0]), run_time=0.6),
                mn.Wait(run_time=0.4),
                build_step(builds, glu_builds, counts, 1), mn.Wait(run_time=0.4), build_step(builds, glu_builds, counts, 2), mn.Wait(run_time=0.4),
                build_step(builds, glu_builds, counts, 3), mn.Wait(run_time=0.4), build_step(builds, glu_builds, counts, 4),
                lag_ratio=1.0),
        chapter(seconds[4],
                mn.AnimationGroup(*swap(chapter_titles, captions, 4), *[mn.FadeOut(m) for m in lower], mn.FadeOut(legend), mn.FadeOut(site_top), mn.FadeOut(bars_top), mn.FadeOut(bar_dff_flipped), run_time=1.0),
                mn.AnimationGroup(mn.Transform(trace_top, branch_mid), mn.Transform(ticks_in, ticks_in_slot), mn.Transform(ticks_ap, ticks_ap_slot), run_time=1.2),
                mn.AnimationGroup(mn.FadeIn(site_labels["branch"]), mn.Create(traces["base"]["spine"]), mn.FadeIn(site_labels["spine"]),
                                  mn.Create(traces["base"]["soma"]), mn.FadeIn(site_labels["soma"]), mn.FadeIn(bars_slots), mn.FadeIn(bar_dff_slots), lag_ratio=0.15),
                mn.FadeIn(box_ap),
                lag_ratio=1.0),
        chapter(seconds[5],
                mn.AnimationGroup(*swap(chapter_titles, captions, 5), run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(bands["base"]), mn.Create(epsp_paths["base"]), mn.FadeIn(bars_e), mn.FadeIn(label_e), mn.FadeIn(label_band),
                                  mn.FadeIn(name_epsp), mn.FadeIn(value_epsp), lag_ratio=0.15),
                mn.AnimationGroup(mn.Create(bap_paths["base"]["soma"]), mn.Create(bap_paths["base"]["branch"]), mn.FadeIn(bars_bap), mn.FadeIn(label_bap),
                                  mn.FadeIn(name_bap), mn.FadeIn(value_bap), lag_ratio=0.15),
                lag_ratio=1.0),
        morph(seconds[6], swap(chapter_titles, captions, 6), pieces, "base", "noisy", cells, value_epsp, value_bap),
        morph(seconds[7], swap(chapter_titles, captions, 7), pieces, "noisy", "pairs", cells, value_epsp, value_bap),
        morph(seconds[8], swap(chapter_titles, captions, 8), pieces, "pairs", "far", cells, value_epsp, value_bap),
        chapter(seconds[9],
                mn.AnimationGroup(*swap(chapter_titles, captions, 9), *[mn.FadeOut(m) for m in after_pool], mn.FadeOut(ticks_in), mn.FadeOut(ticks_ap),
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
