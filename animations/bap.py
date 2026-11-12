# /// script
# requires-python = ">=3.11"
# dependencies = ["imgui_ophys @ git+https://github.com/FlynnOConnell/imgui_ophys", "manim>=0.22", "numpy", "pillow<13"]
# ///
"""
The bAP half of synapse_to_soma, on the same cell: one somatic spike running back up the dendrites as a wave, its delay
and what a 1 kHz camera makes of it, a burst whose later bAPs fade, and the link to the glutamate synapse: NMDA receptors
conduct only with glutamate bound and with the voltage high enough to expel their magnesium, so the order of glutamate
and bAP decides. The cell is real; every voltage is simulated. Drawn with kit.py and neuron_kit.py.

    uv run -p 3.12 animations/bap.py --out <dir> [--stills-only] [--gif-width 720]

Text only, no LaTeX. Needs ffmpeg and uv on PATH.
"""

import manim as mn
import numpy as np
from PIL import Image

import kit
import neuron_kit as N
from kit import BAR, DIM, KEY, TEXT, Count, Play, at, chapter, color, label, put, replace, timeline
from neuron_kit import GLU, M, SITE_COLORS, SITE_NAMES, SPIKE

TITLE = "bAP: a backpropagating action potential, a spike running back up the dendrites"
CHAPTERS = (
    ("", 4.0),
    ("", 9.0),
    ("", 6.0),
    ("Sample at 1 kHz", 5.0),
    ("", 8.0),
    ("", 6.0),
    ("Add a bAP 10 ms after glutamate", 5.0),
    ("Move the bAP 10 ms before glutamate", 5.0),
    ("", 10.0),
)
UNTITLED = {0: "the-cell", 1: "one-bap", 2: "delay", 4: "burst", 5: "nmda", 8: "timing"}
TUFT_UM, TUFT = 400.0, kit.PARTS[4]
TUFT_LENS_PX = 90  # the side of the tuft's close-up, in cell.png: 45 um
Y_THREE, H_THREE = (1.9, 0.35, -1.2), 1.15  # three rows, top to bottom
ROWS = ("tuft", "branch", "soma")
NAMES = {"tuft": f"Tuft, {TUFT_UM:g} um from the soma", "branch": SITE_NAMES["branch"], "soma": "Soma"}
COLORS = {"tuft": TUFT, "branch": SITE_COLORS["branch"], "soma": SITE_COLORS["soma"]}
SPIKE_AT, ONE_STOP = 0.002, 0.01  # s
WAVE_WARP = N.warp((0.0, 0.1, 0.85, 1.0), (0.0, 0.14, 0.55, 1.0))  # slow while the wave climbs the cell
DELAY_SPIKE, DELAY_STOP = 0.0025, 0.0065  # s; a spike halfway through a frame puts both peaks in one frame
BURST_FIRST, BURST_STOP = 0.008, 0.04  # s
RELEASE, BAP_AFTER, BAP_BEFORE, NMDA_STOP = 0.02, 0.03, 0.01, 0.09  # s


def tuft_pixel(p: N.Painter) -> np.ndarray:
    """The topmost drawn pixel of cell.png at TUFT_UM of path from the soma."""
    rows, columns = np.nonzero(p.drawn)
    near = np.abs(p.um_soma - TUFT_UM) < 2.0
    k = np.flatnonzero(near)[np.argmin(rows[near])]
    return p.cell.tall_crop[:2] + 2.0 * np.array([columns[k], rows[k]])


def tuft_rgba(xy: np.ndarray) -> np.ndarray:
    x, y = np.round(xy).astype(int)
    half = TUFT_LENS_PX // 2
    return np.asarray(Image.open(M.DIR_NEURON / "cell.png"))[y - half:y + half, x - half:x + half]


def site_mv(events: M.Events, start: float, stop: float) -> tuple:
    """(fine times, row name to mV above rest)."""
    near = M.traces(events, start, stop)
    far = M.traces(events, start, stop, distance=TUFT_UM)
    return near.fine_t, {"tuft": far.mv["branch"], "branch": near.mv["branch"], "soma": near.mv["soma"]}


def column(p: N.Painter, view: N.CellView, tuft_at: np.ndarray, tuft_xy: np.ndarray) -> list:
    """The three close-ups beside the rows, with their lines to the cell, in draw order."""
    pictures = {"tuft": tuft_rgba(tuft_xy), "branch": N.site_rgba("branch", p), "soma": N.site_rgba("soma", p)}
    sources = {"tuft": (tuft_at, N.R_SITE), "branch": (view.site_at, N.R_SITE), "soma": (view.soma_at, N.R_SOMA)}
    out = []
    for name, y in zip(ROWS, Y_THREE):
        out += [N.cone(*sources[name], at(N.X_LENS, y), N.R_LENS, COLORS[name]), N.lens_ring(y, COLORS[name]), N.lens_image(pictures[name], y)]
    return out


def percent_text(percent: float) -> mn.Text:
    return label(f"{percent:.1f}%", KEY, 30)


def build(seconds: list) -> kit.Timeline:
    """The animation, each chapter seconds[k] long."""
    p = N.new_painter()
    view = N.new_cell_view(p)
    title = put(label(TITLE, TEXT, 25), 0.0, 3.62)
    titles = kit.titles_for(CHAPTERS)
    tuft_xy = tuft_pixel(p)
    tuft_at = N.on_cell(view.image, p, tuft_xy)
    ring_tuft = N.ring_at(tuft_at, N.R_SITE, TUFT)
    name_tuft = label("Tuft", TUFT, 17).next_to(ring_tuft, mn.UR, buff=0.03)
    patch_tuft = mn.BackgroundRectangle(name_tuft, color=color(kit.BG), fill_opacity=0.85, buff=0.03)
    view.marks += [ring_tuft, patch_tuft, name_tuft]
    captions = [None] * len(CHAPTERS)
    captions[1] = N.cap("A spike at the soma also runs back up the dendrites as a bAP, smaller, broader and later with distance.")
    captions[2] = N.cap("Each trace is scaled to its peak, so the branch's broader bAP shows.")
    captions[3] = N.cap(f"A frame lasts 1 ms, so a {1000 * M.bap_shape().delay:g} ms delay reads as 0 or 1 ms, set by where the spike falls between frames.")
    captions[4] = N.cap("A burst, driven by other inputs: each bAP is smaller than the last, and the loss is steeper far from the soma.")
    captions[5] = N.cap("Magnesium blocks NMDA receptors at rest (Jahr and Stevens 1990), so glutamate alone opens few of them.")
    captions[8] = N.cap("Calcium through NMDA receptors drives spike-timing-dependent plasticity (Markram 1997, Bi and Poo 1998).")

    # chapter 2: one spike, the wave over the cell and the mV at three sites on one scale
    one = M.Events(spikes=np.array([SPIKE_AT]))
    t1, mv1 = site_mv(one, 0.0, ONE_STOP)
    rows_1 = {name: N.new_row(0.0, ONE_STOP, mv1[name], y, height=H_THREE, top=M.AP_MV) for name, y in zip(ROWS, Y_THREE)}
    paths_1 = {name: N.row_path(rows_1[name], t1, mv1[name]) for name in ROWS}
    names_1 = {name: N.row_name(rows_1[name], NAMES[name], COLORS[name]) for name in ROWS}
    peaks_1 = [label(f"{N.mv_text(float(mv1[name].max()))} mV", TEXT, 19).move_to(at(N.X_ROW + 0.5 * N.W_ROW + 0.15, y - 0.25), aligned_edge=mn.LEFT) for name, y in zip(ROWS, Y_THREE)]
    bar_mv_1 = N.mv_bar(rows_1["tuft"], 50.0)
    bar_time_1 = N.time_bar(rows_1["soma"], 0.002)
    cursor_1 = N.cursor_over(list(rows_1.values()), 0.0)
    clock_1 = N.new_clock(0.0)
    seconds_1 = 0.68 * seconds[1]
    frames_1 = N.heat_frames(p, one, N.sweep_times(rows_1["soma"], seconds_1, WAVE_WARP))
    column_1 = column(p, view, tuft_at, tuft_xy)

    # chapters 3 and 4: the delay, then the same two traces as 1 kHz frames
    shape = M.bap_shape()
    delayed = M.traces(M.Events(spikes=np.array([DELAY_SPIKE])), 0.0, DELAY_STOP)
    pair_colors = (COLORS["soma"], COLORS["branch"])
    fig = N.delay_figure(1.5, 0.3, 6.6, 3.0, delayed.fine_t, delayed.mv["soma"], delayed.mv["branch"], ("Soma", "Branch"), pair_colors, "bAP: soma, then branch", shape.delay, 0.001)
    frame_paths, frame_marks, frame_peaks = N.normalized_pair(fig["axes"], delayed.t, delayed.dff["soma"], delayed.dff["branch"], pair_colors)
    frame_span = N.delay_span(fig["axes"], frame_peaks)
    frame_delay = 1000 * (frame_peaks[1] - frame_peaks[0])

    # chapter 5: a burst, on one scale
    burst = M.Events(spikes=M.spike_burst(BURST_FIRST))
    tb, mvb = site_mv(burst, 0.0, BURST_STOP)
    top_b = max(float(v.max()) for v in mvb.values())
    rows_b = {name: N.new_row(0.0, BURST_STOP, mvb[name], y, height=H_THREE, top=top_b) for name, y in zip(ROWS, Y_THREE)}
    paths_b = {name: N.row_path(rows_b[name], tb, mvb[name]) for name in ROWS}
    names_b = [N.row_name(rows_b[name], NAMES[name], COLORS[name]) for name in ROWS]
    bar_mv_b = N.mv_bar(rows_b["tuft"], 50.0)
    bar_time_b = N.time_bar(rows_b["soma"], 0.01)
    cursor_b = N.cursor_over(list(rows_b.values()), 0.0)
    clock_b = N.new_clock(0.0)
    seconds_b = 0.74 * seconds[4]
    frames_b = N.heat_frames(p, burst, N.sweep_times(rows_b["soma"], seconds_b))
    column_b = column(p, view, tuft_at, tuft_xy)

    # chapters 6 to 8: NMDA receptors at Spine 1, with and without a bAP
    release = np.array([RELEASE])
    t_n = np.arange(0.0, NMDA_STOP, 1e-4)
    cases = {"alone": M.Events(releases=release), "after": M.Events(releases=release, spikes=np.array([BAP_AFTER])), "before": M.Events(releases=release, spikes=np.array([BAP_BEFORE]))}
    voltages = {name: M.voltage_mv(*M.site_paths()["spine"], t_n, events) for name, events in cases.items()}
    conducting = {name: M.nmda_open(t_n, events) for name, events in cases.items()}
    percents = {name: 100 * float(v.max()) for name, v in conducting.items()}
    bound = M.nmda_bound(t_n, release)
    row_bound = N.new_row(0.0, NMDA_STOP, bound, Y_THREE[0], height=H_THREE, top=1.0)
    row_v = N.new_row(0.0, NMDA_STOP, voltages["after"], Y_THREE[1], height=H_THREE)
    row_open = N.new_row(0.0, NMDA_STOP, conducting["after"], Y_THREE[2], height=H_THREE, top=1.0)
    path_bound = N.row_path(row_bound, t_n, bound, GLU, 2.0)
    paths_v = {name: N.row_path(row_v, t_n, v) for name, v in voltages.items()}
    paths_open = {name: N.row_path(row_open, t_n, v, KEY, 2.0) for name, v in conducting.items()}
    names_n = [N.row_name(row_bound, "Glutamate bound to NMDA receptors (simulated)", GLU), N.row_name(row_v, "Voltage at Spine 1 (simulated)", SITE_COLORS["spine"]), N.row_name(row_open, "NMDA receptors conducting (simulated)", KEY)]
    bars_n = [N.right_bar(row_bound, 0.5, "50%"), N.mv_bar(row_v, 20.0), N.right_bar(row_open, 0.5, "50%"), N.time_bar(row_open, 0.02)]
    cursor_n = N.cursor_over([row_bound, row_v, row_open], 0.0)
    clock_n = N.new_clock(0.0)
    seconds_n = 0.6 * seconds[5]
    frames_n = N.heat_frames(p, cases["alone"], N.sweep_times(row_open, seconds_n))
    frames_after = N.heat_frames(p, cases["after"], N.sweep_times(row_open, 3.2))
    frames_before = N.heat_frames(p, cases["before"], N.sweep_times(row_open, 3.2))
    number = percent_text(percents["alone"]).move_to(at(N.X_ROW + 0.5 * N.W_ROW + 0.12, Y_THREE[2] - 0.3), aligned_edge=mn.LEFT)
    lens_cleft, rim_cleft = N.lens_image(N.cleft_rgba(), Y_THREE[0]), N.lens_ring(Y_THREE[0], GLU)
    lens_spine, rim_spine = N.lens_image(N.site_rgba("spine", p), Y_THREE[1]), N.lens_ring(Y_THREE[1], GLU)
    cone_spine = N.cone(view.site_at, N.R_SITE, at(N.X_LENS, Y_THREE[1]), N.R_LENS, GLU)
    cone_cleft = N.cone(at(N.X_LENS, Y_THREE[1]), 0.1, at(N.X_LENS, Y_THREE[0]), N.R_LENS, GLU)
    pictures_n = [cone_spine, rim_spine, lens_spine, cone_cleft, rim_cleft, lens_cleft]

    # chapter 9: the table
    y_header, pitch = 2.5, 1.05
    header = mn.VGroup(*[label(text, DIM, 17).move_to(at(x, y_header), aligned_edge=mn.LEFT) for text, x in (("NMDA conducting", -3.0), ("Peak", 3.3))])
    units = label("share of receptors (simulated)", BAR, 15).move_to(at(3.3, y_header + 0.3), aligned_edge=mn.LEFT)
    note = label("one scale", BAR, 15).move_to(at(-3.0, y_header + 0.3), aligned_edge=mn.LEFT)
    none = M.nmda_open(t_n, M.Events(spikes=np.array([BAP_AFTER])))
    table = N.table_rows([
        ("Glutamate alone", GLU, t_n, conducting["alone"], [f"{percents['alone']:.1f}%"]),
        ("bAP 10 ms after glutamate", KEY, t_n, conducting["after"], [f"{percents['after']:.1f}%"]),
        ("bAP 10 ms before glutamate", GLU, t_n, conducting["before"], [f"{percents['before']:.1f}%"]),
        ("bAP alone", SPIKE, t_n, none, [f"{100 * float(none.max()):.1f}%"]),
    ], y_header - 0.7, pitch, (3.3,), top=1.0)

    print(f"bAP peak mV {shape.peak_mv}, delay {1000 * shape.delay:.2f} ms, width {1000 * shape.width:.2f} ms; at {TUFT_UM:g} um {float(mv1['tuft'].max()):.1f} mV")
    print(f"frames: peaks {frame_peaks}, delay {frame_delay:.1f} ms; burst peak mV { {name: round(float(v.max()), 1) for name, v in mvb.items()} }")
    print(f"NMDA conducting, peak %: {percents}")

    chapters = [
        N.opening(seconds[0], view, title),
        chapter(seconds[1],
                mn.AnimationGroup(mn.FadeOut(title), mn.FadeIn(view.legend), *[mn.FadeIn(m) for m in column_1], *[mn.FadeIn(m) for m in names_1.values()],
                                  mn.FadeIn(bar_mv_1), mn.FadeIn(bar_time_1), mn.FadeIn(cursor_1), mn.FadeIn(clock_1), mn.FadeIn(captions[1]), lag_ratio=0.03, run_time=1.2),
                N.sweep(list(paths_1.values()), cursor_1, rows_1["soma"], clock_1, view.image, frames_1, seconds_1, WAVE_WARP),
                mn.AnimationGroup(mn.FadeOut(cursor_1), *[mn.FadeIn(m) for m in peaks_1], run_time=0.5),
                lag_ratio=1.0),
        chapter(seconds[2],
                mn.AnimationGroup(*replace(captions[1], captions[2]), mn.FadeOut(clock_1), mn.FadeOut(bar_mv_1), mn.FadeOut(bar_time_1),
                                  *[mn.FadeOut(m) for m in (*paths_1.values(), *names_1.values(), *peaks_1, *column_1)], run_time=1.0),
                mn.AnimationGroup(mn.FadeIn(fig["head"]), mn.FadeIn(fig["legend"]), *[mn.Create(m) for m in fig["paths"]], lag_ratio=0.08, run_time=1.6),
                mn.AnimationGroup(*[mn.Create(m) for m in fig["marks"]], mn.FadeIn(fig["span"]), mn.FadeIn(fig["number"]), mn.FadeIn(fig["bar"]), lag_ratio=0.1, run_time=1.4),
                lag_ratio=1.0),
        chapter(seconds[3],
                mn.AnimationGroup(*replace(captions[2], None), mn.FadeIn(titles[3]), run_time=1.0),
                mn.AnimationGroup(*[mn.Transform(a, b) for a, b in zip(fig["paths"], frame_paths)], *[mn.Transform(a, b) for a, b in zip(fig["marks"], frame_marks)],
                                  mn.Transform(fig["span"], frame_span), Count(fig["number"], 1000 * shape.delay, frame_delay, N.delay_number), run_time=2.6),
                mn.FadeIn(captions[3], run_time=0.4),
                lag_ratio=1.0),
        chapter(seconds[4],
                mn.AnimationGroup(*replace(captions[3], captions[4]), mn.FadeOut(titles[3]), *[mn.FadeOut(fig[key]) for key in ("head", "legend", "span", "number", "bar")], *[mn.FadeOut(m) for m in (*fig["paths"], *fig["marks"])], run_time=0.7),
                mn.AnimationGroup(*[mn.FadeIn(m) for m in (*column_b, *names_b)], mn.FadeIn(bar_mv_b), mn.FadeIn(bar_time_b), mn.FadeIn(cursor_b), mn.FadeIn(clock_b), lag_ratio=0.03, run_time=0.8),
                N.sweep(list(paths_b.values()), cursor_b, rows_b["soma"], clock_b, view.image, frames_b, seconds_b),
                mn.FadeOut(cursor_b, run_time=0.3),
                lag_ratio=1.0),
        chapter(seconds[5],
                mn.AnimationGroup(*replace(captions[4], captions[5]), mn.FadeOut(clock_b), mn.FadeOut(bar_mv_b), mn.FadeOut(bar_time_b), *[mn.FadeOut(m) for m in (*paths_b.values(), *names_b, *column_b)], run_time=0.8),
                mn.AnimationGroup(*[mn.FadeIn(m) for m in (*pictures_n, *names_n, *bars_n)], mn.FadeIn(cursor_n), mn.FadeIn(clock_n), lag_ratio=0.03, run_time=0.8),
                N.sweep([path_bound, paths_v["alone"], paths_open["alone"]], cursor_n, row_open, clock_n, view.image, frames_n, seconds_n),
                mn.AnimationGroup(mn.FadeOut(cursor_n), mn.FadeOut(clock_n), mn.FadeIn(number), run_time=0.4),
                lag_ratio=1.0),
        N.morph(seconds[6], [mn.FadeOut(captions[5]), mn.FadeIn(titles[6])],
                [mn.Transform(paths_v["alone"], paths_v["after"]), mn.Transform(paths_open["alone"], paths_open["after"]), Count(number, percents["alone"], percents["after"], percent_text), Play(view.image, frames_after, rate_func=mn.linear)]),
        N.morph(seconds[7], replace(titles[6], titles[7]),
                [mn.Transform(paths_v["alone"], paths_v["before"]), mn.Transform(paths_open["alone"], paths_open["before"]), Count(number, percents["after"], percents["before"], percent_text), Play(view.image, frames_before, rate_func=mn.linear)]),
        chapter(seconds[8],
                mn.AnimationGroup(mn.FadeOut(titles[7]), mn.FadeOut(view.legend), mn.FadeOut(number), mn.FadeOut(path_bound), mn.FadeOut(paths_v["alone"]), mn.FadeOut(paths_open["alone"]), *[mn.FadeOut(m) for m in (*pictures_n, *names_n, *bars_n)], run_time=1.0),
                mn.AnimationGroup(mn.FadeIn(header), mn.FadeIn(units), mn.FadeIn(note), N.table_in(table), mn.FadeIn(captions[8]), lag_ratio=0.2),
                lag_ratio=1.0),
    ]
    return timeline([], chapters)


if __name__ == "__main__":
    kit.run("bap", CHAPTERS, UNTITLED, build, __doc__)
