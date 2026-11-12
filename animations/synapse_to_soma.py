# /// script
# requires-python = ">=3.11"
# dependencies = ["imgui_ophys @ git+https://github.com/FlynnOConnell/imgui_ophys", "manim>=0.22", "numpy", "pillow<13"]
# ///
"""
A synapse and a soma burst on one real cell: the MICrONS micrograph of a synapse, four presynaptic spikes and their
glutamate, the EPSPs at the spine, the branch and the soma with the voltage spreading over the cell, a four-spike burst
at the soma with its bAPs running back up the dendrites, the delay each event takes between sites, both events in one
recording, and a table of sizes and delays. The cell and the synapse are real; every voltage is simulated. The epsp and
bap animations each take one of the two events further. Drawn with kit.py and neuron_kit.py.

    uv run -p 3.12 animations/synapse_to_soma.py --out <dir> [--stills-only] [--gif-width 720]

Text only, no LaTeX. Needs ffmpeg and uv on PATH.
"""

import manim as mn
import numpy as np

import kit
import neuron_kit as N
from kit import BAR, DIM, TEXT, at, axes_points, chapter, color, label, put, replace, timeline
from neuron_kit import GLU, M, SITE_COLORS, SITE_NAMES, SPIKE, Y_ROWS

TITLE = "A synapse and a soma burst: the traces each leaves at spine, branch and soma"
CHAPTERS = (
    ("", 4.0),
    ("", 5.0),
    ("", 6.0),
    ("", 9.0),
    ("", 12.0),
    ("", 6.0),
    ("", 8.0),
    ("", 10.0),
)
UNTITLED = {0: "the-cell", 1: "the-synapse", 2: "four-releases", 3: "epsps", 4: "soma-burst", 5: "delays", 6: "both", 7: "sizes-and-delays"}
H_EM, X_EM, Y_EM = 3.7, -1.2, 0.35  # the micrograph
X_SMALL, W_SMALL = 3.7, 5.4  # the two rows beside it; they carry no bars
TRAIN_START, TRAIN_STOP = 0.0, 0.11  # s
BURST_START, BURST_STOP, BURST_FIRST = 0.0, 0.05, 0.008  # s
BOTH_STOP, BOTH_BURST = 0.11, 0.043  # s
BURST_WARP = N.warp((0.0, 0.08, 0.52, 0.88, 1.0), (0.0, 0.15, 0.2, 0.62, 1.0))  # slow through the first bAP, so the wave can be followed
W_FIG, H_FIG, X_FIG_A, X_FIG_B, Y_FIG = 4.3, 2.5, -0.55, 4.25, 0.45  # the two delay figures
SPIKES_NAME = "Somatic spikes, driven by other inputs (simulated)"


def build(seconds: list) -> kit.Timeline:
    """The animation, each chapter seconds[k] long."""
    p = N.new_painter()
    view = N.new_cell_view(p)
    cell_image = view.image
    title = put(label(TITLE, TEXT, 25), 0.0, 3.62)
    captions = [None] * len(CHAPTERS)
    captions[1] = N.cap("Electron micrograph of the synapse on Spine 1. A bouton is the end of a presynaptic axon.")
    captions[2] = N.cap(f"4 presynaptic spikes, {N.ms(M.TRAIN_INTERVAL)} apart: each one releases glutamate into the cleft (dots are schematic).")
    captions[3] = N.cap("Each release gives an EPSP (excitatory postsynaptic potential), 20 times smaller at the soma. Scales differ by row.")
    captions[4] = N.cap(f"A burst: 4 somatic spikes, {N.ms(M.SPIKE_ISI)} apart. Each runs back up the dendrites as a bAP (backpropagating action potential).")
    captions[5] = N.cap("Each trace is scaled to its peak. At 1 kHz a frame lasts 1 ms, so the bAP's delay fits inside one frame.")
    captions[6] = N.cap("In one recording, EPSPs are slow bumps, largest at the spine; bAPs are fast spikes, largest at the soma.")
    synapse = N.new_synapse_view(view, H_EM, X_EM, Y_EM)
    micrograph = synapse.micrograph

    # chapter 3: four presynaptic spikes and their glutamate, beside the micrograph
    spikes_pre = M.presynaptic_train(0.01)
    releases = M.releases_after(spikes_pre)
    train = M.Events(releases=releases)
    tr = M.traces(train, TRAIN_START, TRAIN_STOP)
    pre_mv = M.presynaptic_mv(tr.fine_t, spikes_pre)
    row_pre = N.new_row(TRAIN_START, TRAIN_STOP, pre_mv, 1.25, x=X_SMALL, width=W_SMALL)
    row_glu_small = N.new_row(TRAIN_START, TRAIN_STOP, tr.glutamate, -0.25, x=X_SMALL, width=W_SMALL)
    pre_path = N.row_path(row_pre, tr.fine_t, pre_mv, GLU)
    glu_small = N.row_path(row_glu_small, tr.t, tr.glutamate, GLU, 2.0)
    name_pre = N.row_name(row_pre, "Presynaptic spikes (simulated)", GLU)
    name_glu_small = N.row_name(row_glu_small, "Glutamate at Spine 1 (simulated)", GLU)
    bar_time_small = N.time_bar(row_glu_small, M.TRAIN_INTERVAL)
    cursor_small = N.cursor_over([row_pre, row_glu_small], TRAIN_START)
    seconds_small = 0.72 * seconds[2]

    # chapter 4: the glutamate to the top row, three sites under it, the voltage over the cell
    row_glu = N.new_row(TRAIN_START, TRAIN_STOP, tr.glutamate, Y_ROWS["top"])
    glu_top = N.row_path(row_glu, tr.t, tr.glutamate, GLU, 2.0)
    name_glu = N.row_name(row_glu, "Glutamate at Spine 1 (simulated)", GLU)
    rows_e = {site: N.new_row(TRAIN_START, TRAIN_STOP, tr.dff[site], Y_ROWS[site]) for site in M.SITES}
    paths_e = {site: N.row_path(rows_e[site], tr.t, tr.dff[site]) for site in M.SITES}
    names = {site: N.row_name(rows_e[site], SITE_NAMES[site], SITE_COLORS[site]) for site in M.SITES}
    bars_e = [N.dff_bar(rows_e[site], kit.nice(0.5 * float(tr.dff[site].max()))) for site in M.SITES]
    bar_time_e = N.time_bar(rows_e["soma"], M.TRAIN_INTERVAL)
    cursor_e = N.cursor_over([row_glu, *rows_e.values()], TRAIN_START)
    clock_e = N.new_clock(TRAIN_START)
    seconds_e = 0.62 * seconds[3]
    frames_e = N.heat_frames(p, train, N.sweep_times(rows_e["soma"], seconds_e))

    lens_cleft, rim_cleft = N.lens_image(N.cleft_rgba(), Y_ROWS["top"]), N.lens_ring(Y_ROWS["top"], GLU)
    rims, lens = N.site_column(p)
    cone_branch, cone_soma, cone_spine = N.site_cones(view)
    cone_cleft = N.cone(at(N.X_LENS, Y_ROWS["spine"]), 0.1, at(N.X_LENS, Y_ROWS["top"]), N.R_LENS, GLU)

    # chapter 5: a burst at the soma, all three sites on one scale
    burst_spikes = M.spike_burst(BURST_FIRST)
    burst = M.Events(spikes=burst_spikes)
    tb = M.traces(burst, BURST_START, BURST_STOP)
    top_b = max(float(tb.dff[site].max()) for site in M.SITES)
    soma_mv = M.presynaptic_mv(tb.fine_t, burst_spikes)
    row_s = N.new_row(BURST_START, BURST_STOP, soma_mv, Y_ROWS["top"])
    spikes_path = N.row_path(row_s, tb.fine_t, soma_mv, SPIKE)
    name_spikes = N.row_name(row_s, SPIKES_NAME, SPIKE)
    rows_b = {site: N.new_row(BURST_START, BURST_STOP, tb.dff[site], Y_ROWS[site], top=top_b) for site in M.SITES}
    paths_b = {site: N.row_path(rows_b[site], tb.t, tb.dff[site]) for site in M.SITES}
    bar_dff_b = N.dff_bar(rows_b["spine"], kit.nice(0.5 * top_b))
    bar_time_b = N.time_bar(rows_b["soma"], 0.01)
    cursor_b = N.cursor_over([row_s, *rows_b.values()], BURST_START)
    clock_b = N.new_clock(BURST_START)
    seconds_b = 0.72 * seconds[4]
    frames_b = N.heat_frames(p, burst, N.sweep_times(rows_b["soma"], seconds_b, BURST_WARP))

    # chapter 6: the delay of each event between two sites
    e_shape, b_shape = M.epsp_shape(), M.bap_shape()
    one_e = M.traces(M.Events(releases=np.array([0.002])), 0.0, 0.03)
    one_b = M.traces(M.Events(spikes=np.array([0.002])), 0.0, 0.006)
    fig_e = N.delay_figure(X_FIG_A, Y_FIG, W_FIG, H_FIG, one_e.fine_t, one_e.mv["spine"], one_e.mv["soma"], ("Spine 1", "Soma"), (SITE_COLORS["spine"], SITE_COLORS["soma"]), "EPSP: spine, then soma", e_shape.delay, 0.005)
    fig_b = N.delay_figure(X_FIG_B, Y_FIG, W_FIG, H_FIG, one_b.fine_t, one_b.mv["soma"], one_b.mv["branch"], ("Soma", "Branch"), (SITE_COLORS["soma"], SITE_COLORS["branch"]), "bAP: soma, then branch", b_shape.delay, 0.001)

    # chapter 7: both in one recording, on one scale
    both_spikes = M.spike_burst(BOTH_BURST)
    both = M.Events(releases=releases, spikes=both_spikes)
    to = M.traces(both, 0.0, BOTH_STOP)
    top_o = max(float(to.dff[site].max()) for site in M.SITES)
    rows_o = {site: N.new_row(0.0, BOTH_STOP, to.dff[site], Y_ROWS[site], top=top_o) for site in M.SITES}
    names_o = [N.row_name(rows_o[site], SITE_NAMES[site], SITE_COLORS[site]) for site in M.SITES]
    rims_o, lens_o = N.site_column(p)
    cones_o = N.site_cones(view)
    paths_o = {site: N.row_path(rows_o[site], to.t, to.dff[site]) for site in M.SITES}
    row_g = N.new_row(0.0, BOTH_STOP, to.glutamate, Y_ROWS["top"])
    glu_o = N.row_path(row_g, to.t, to.glutamate, GLU, 2.0)
    ticks = mn.VGroup(*[mn.Line(*axes_points(row_g.axes, np.array([s, s]), np.array([0.55 * row_g.hi, row_g.hi])), color=color(SPIKE), stroke_width=2.5) for s in both_spikes])
    name_both = mn.VGroup(label("Glutamate at Spine 1", GLU, 17), label("Somatic spikes (simulated)", SPIKE, 17)).arrange(mn.RIGHT, buff=0.4).next_to(row_g.axes, mn.UP, buff=0.0).align_to(row_g.axes, mn.LEFT)
    bar_dff_o = N.dff_bar(rows_o["spine"], kit.nice(0.5 * top_o))
    bar_time_o = N.time_bar(rows_o["soma"], M.TRAIN_INTERVAL)
    cursor_o = N.cursor_over([row_g, *rows_o.values()], 0.0)
    clock_o = N.new_clock(0.0)
    seconds_o = 0.68 * seconds[6]
    frames_o = N.heat_frames(p, both, N.sweep_times(rows_o["soma"], seconds_o))

    # chapter 8: sizes and delays
    train_peaks = {site: float(v.max()) for site, v in M.traces(train, 0.0, TRAIN_STOP).mv.items()}
    burst_peaks = {site: float(v.max()) for site, v in M.traces(burst, 0.0, BURST_STOP).mv.items()}
    y_header, pitch = 2.5, 1.05
    header = mn.VGroup(*[label(text, DIM, 17).move_to(at(x, y_header), aligned_edge=mn.LEFT)
                         for text, x in (("branch, -dF/F", -3.0), ("Spine 1", 1.5), ("Branch", 2.6), ("Soma", 3.75), ("delay, 150 um", 5.0))])
    units = label("peak above rest, mV (simulated)", BAR, 15).move_to(at(1.5, y_header + 0.3), aligned_edge=mn.LEFT)
    note = label("each scaled to its peak", BAR, 15).move_to(at(-3.0, y_header + 0.3), aligned_edge=mn.LEFT)
    table_sites = [
        ("1 EPSP", kit.ACCENT, M.Events(releases=np.array([0.004])), 0.05, e_shape.peak_mv, e_shape.delay),
        ("4 EPSPs", kit.ACCENT, M.Events(releases=M.releases_after(M.presynaptic_train(0.004))), 0.1, train_peaks, e_shape.delay),
        ("1 bAP", SPIKE, M.Events(spikes=np.array([0.004])), 0.02, b_shape.peak_mv, b_shape.delay),
        ("Burst of 4", SPIKE, M.Events(spikes=M.spike_burst(0.004)), 0.05, burst_peaks, b_shape.delay),
    ]
    table = []
    for name, c, events, stop, peaks, delay in table_sites:
        trace = M.traces(events, 0.0, stop)
        table.append((name, c, trace.t, trace.dff["branch"], [*[N.mv_text(peaks[site]) for site in M.SITES], f"{1000 * delay:.1f} ms"]))
    table = N.table_rows(table, y_header - 0.7, pitch, (1.65, 2.75, 3.85, 5.0))

    print(f"EPSP peak mV {e_shape.peak_mv}, delay {1000 * e_shape.delay:.2f} ms; train {train_peaks}")
    print(f"bAP peak mV {b_shape.peak_mv}, delay {1000 * b_shape.delay:.2f} ms; burst {burst_peaks}")

    site_rows = [rims["spine"], lens["spine"], rims["branch"], lens["branch"], rims["soma"], lens["soma"]]
    chrome_e = [*bars_e, bar_time_e]
    chapters = [
        N.opening(seconds[0], view, title),
        chapter(seconds[1], *N.synapse_arrives(synapse, title, captions[1]), lag_ratio=1.0),
        chapter(seconds[2],
                mn.AnimationGroup(*replace(captions[1], captions[2]), mn.FadeIn(name_pre), mn.FadeIn(name_glu_small), mn.FadeIn(bar_time_small), mn.FadeIn(cursor_small), run_time=0.9),
                mn.AnimationGroup(mn.Create(pre_path, rate_func=mn.linear), mn.Create(glu_small, rate_func=mn.linear),
                                  mn.MoveAlongPath(cursor_small, N.cursor_path(cursor_small, row_pre), rate_func=mn.linear),
                                  N.puffs_during(micrograph, releases, TRAIN_START, TRAIN_STOP, seconds_small), run_time=seconds_small),
                mn.FadeOut(cursor_small, run_time=0.3),
                lag_ratio=1.0),
        chapter(seconds[3],
                mn.AnimationGroup(*replace(captions[2], captions[3]), mn.FadeOut(pre_path), mn.FadeOut(name_pre), mn.FadeOut(name_glu_small), mn.FadeOut(bar_time_small),
                                  *N.synapse_to_lens(synapse, lens_cleft, rim_cleft), mn.Transform(glu_small, glu_top), mn.FadeIn(name_glu), run_time=1.3),
                mn.AnimationGroup(mn.FadeIn(view.legend), *[mn.FadeIn(m) for m in (cone_branch, rims["branch"], lens["branch"], cone_soma, rims["soma"], lens["soma"], cone_spine, rims["spine"], lens["spine"], cone_cleft)],
                                  *[mn.FadeIn(m) for m in names.values()], *[mn.FadeIn(m) for m in chrome_e], mn.FadeIn(cursor_e), mn.FadeIn(clock_e), lag_ratio=0.03, run_time=1.2),
                N.sweep(list(paths_e.values()), cursor_e, rows_e["soma"], clock_e, cell_image, frames_e, seconds_e),
                mn.FadeOut(cursor_e, run_time=0.3),
                lag_ratio=1.0),
        chapter(seconds[4],
                mn.AnimationGroup(*replace(captions[3], captions[4]), mn.FadeOut(clock_e), mn.FadeOut(glu_small), mn.FadeOut(name_glu), mn.FadeOut(rim_cleft), mn.FadeOut(lens_cleft), mn.FadeOut(cone_cleft),
                                  *[mn.FadeOut(m) for m in paths_e.values()], *[mn.FadeOut(m) for m in chrome_e], run_time=1.0),
                mn.AnimationGroup(mn.FadeIn(name_spikes), mn.FadeIn(bar_dff_b), mn.FadeIn(bar_time_b), mn.FadeIn(cursor_b), mn.FadeIn(clock_b), run_time=0.8),
                N.sweep([spikes_path, *paths_b.values()], cursor_b, rows_b["soma"], clock_b, cell_image, frames_b, seconds_b, BURST_WARP),
                mn.FadeOut(cursor_b, run_time=0.3),
                lag_ratio=1.0),
        chapter(seconds[5],
                mn.AnimationGroup(*replace(captions[4], captions[5]), mn.FadeOut(clock_b), mn.FadeOut(spikes_path), mn.FadeOut(name_spikes), mn.FadeOut(bar_dff_b), mn.FadeOut(bar_time_b),
                                  *[mn.FadeOut(m) for m in paths_b.values()], *[mn.FadeOut(m) for m in names.values()], *[mn.FadeOut(m) for m in site_rows], mn.FadeOut(cone_branch), mn.FadeOut(cone_soma), mn.FadeOut(cone_spine), run_time=1.0),
                mn.AnimationGroup(*[mn.FadeIn(fig[key]) for fig in (fig_e, fig_b) for key in ("head", "legend")],
                                  *[mn.Create(m) for fig in (fig_e, fig_b) for m in fig["paths"]], lag_ratio=0.08, run_time=1.6),
                mn.AnimationGroup(*[mn.Create(m) for fig in (fig_e, fig_b) for m in fig["marks"]], *[mn.FadeIn(fig["span"]) for fig in (fig_e, fig_b)],
                                  *[mn.FadeIn(fig["number"]) for fig in (fig_e, fig_b)], *[mn.FadeIn(fig["bar"]) for fig in (fig_e, fig_b)], lag_ratio=0.1, run_time=1.4),
                lag_ratio=1.0),
        chapter(seconds[6],
                mn.AnimationGroup(*replace(captions[5], captions[6]), *[mn.FadeOut(m) for fig in (fig_e, fig_b) for key in ("head", "legend", "span", "number", "bar") for m in (fig[key],)],
                                  *[mn.FadeOut(m) for fig in (fig_e, fig_b) for m in (*fig["paths"], *fig["marks"])], run_time=0.9),
                mn.AnimationGroup(*[mn.FadeIn(m) for m in (*cones_o, *names_o, *rims_o.values(), *lens_o.values())],
                                  mn.FadeIn(name_both), mn.FadeIn(glu_o), mn.FadeIn(ticks), mn.FadeIn(bar_dff_o), mn.FadeIn(bar_time_o), mn.FadeIn(cursor_o), mn.FadeIn(clock_o), lag_ratio=0.03, run_time=1.0),
                N.sweep(list(paths_o.values()), cursor_o, rows_o["soma"], clock_o, cell_image, frames_o, seconds_o),
                mn.FadeOut(cursor_o, run_time=0.3),
                lag_ratio=1.0),
        chapter(seconds[7],
                mn.AnimationGroup(mn.FadeOut(captions[6]), mn.FadeOut(clock_o), mn.FadeOut(view.legend), mn.FadeOut(name_both), mn.FadeOut(glu_o), mn.FadeOut(ticks), mn.FadeOut(bar_dff_o), mn.FadeOut(bar_time_o),
                                  *[mn.FadeOut(m) for m in (*paths_o.values(), *cones_o, *names_o, *rims_o.values(), *lens_o.values())], run_time=1.0),
                mn.AnimationGroup(mn.FadeIn(header), mn.FadeIn(units), mn.FadeIn(note),
                                  N.table_in(table), lag_ratio=0.2),
                lag_ratio=1.0),
    ]
    return timeline([], chapters)


if __name__ == "__main__":
    kit.run("synapse_to_soma", CHAPTERS, UNTITLED, build, __doc__)
