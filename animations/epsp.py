# /// script
# requires-python = ">=3.11"
# dependencies = ["imgui_ophys @ git+https://github.com/FlynnOConnell/imgui_ophys", "manim>=0.22", "numpy", "pillow<13"]
# ///
"""
The EPSP half of synapse_to_soma, on the same cell: the synapse on Spine 1, one release of glutamate and the EPSP it
gives, the EPSP shrinking from the spine to the soma, its delay, and inputs on neighbouring spines summing on the branch
until the branch fires a dendritic spike. The cell and the synapse are real; every voltage is simulated. Drawn with
kit.py and neuron_kit.py.

    uv run -p 3.12 animations/epsp.py --out <dir> [--stills-only] [--gif-width 720]

Text only, no LaTeX. Needs ffmpeg and uv on PATH.
"""

import manim as mn
import numpy as np

import kit
import neuron_kit as N
from kit import BAR, DIM, KEY, TEXT, Count, Play, at, axes_points, chapter, color, label, put, replace, timeline
from neuron_kit import GLU, M, SITE_COLORS, SITE_NAMES, Y_ROWS

TITLE = "EPSP: an excitatory postsynaptic potential, the response to one glutamate release"
CHAPTERS = (
    ("", 4.0),
    ("", 5.0),
    ("", 7.0),
    ("", 10.0),
    ("", 6.0),
    ("", 7.0),
    ("Inputs 5 ms apart", 5.0),
    ("Inputs 2 ms apart", 6.0),
    ("", 10.0),
)
UNTITLED = {0: "the-cell", 1: "the-synapse", 2: "one-epsp", 3: "spread", 4: "delay", 5: "summation", 8: "sizes"}
H_EM, X_EM, Y_EM = 3.7, -1.2, 0.35  # the micrograph
X_SMALL, W_SMALL = 3.35, 4.7  # the two rows beside it
RELEASE, ONE_STOP = 0.01, 0.07  # s
SPREAD_WARP = N.warp((0.0, 0.1, 0.6, 1.0), (0.0, 0.14, 0.36, 1.0))  # slow through the EPSP's rise and peak
N_INPUTS, INPUT_START, SUM_STOP = 5, 0.01, 0.12  # s
INTERVALS = (0.02, 0.005, 0.002)  # s between inputs, one per chapter
H_SUM, Y_TICKS, Y_BRANCH, Y_SOMA = 1.35, 2.35, 1.1, -0.75
X_NUMBER, Y_NUMBER = 3.6, 3.28


def input_events(interval: float) -> M.Events:
    """Inputs on neighbouring spines, interval s apart, and the dendritic spike if they reach threshold."""
    inputs = M.Events(neighbours=M.presynaptic_train(INPUT_START, N_INPUTS, interval))
    fired = M.dspike_time(inputs, 0.0, SUM_STOP)
    return inputs if fired is None else M.Events(neighbours=inputs.neighbours, dspikes=np.array([fired]))


def input_ticks(row: N.Row, events: M.Events) -> mn.VGroup:
    """One tick per input."""
    return mn.VGroup(*[mn.Line(*axes_points(row.axes, np.array([s, s]), np.array([row.lo, row.hi])), color=color(GLU), stroke_width=2.5) for s in events.neighbours])


def peak_text(mv: float) -> mn.Text:
    return label(f"{mv:.1f} mV", KEY, 34)


def build(seconds: list) -> kit.Timeline:
    """The animation, each chapter seconds[k] long."""
    p = N.new_painter()
    view = N.new_cell_view(p)
    title = put(label(TITLE, TEXT, 25), 0.0, 3.62)
    titles = kit.titles_for(CHAPTERS)
    captions = [None] * len(CHAPTERS)
    captions[1] = N.cap("Electron micrograph of the synapse on Spine 1. A bouton is the end of a presynaptic axon.")
    captions[2] = N.cap("Glutamate opens AMPA receptors on the spine. Positive ions flow in and depolarize it: an EPSP (dots are schematic).")
    captions[3] = N.cap("The EPSP spreads passively along the dendrite and shrinks on the way. All rows share one scale.")
    captions[4] = N.cap("Each trace is scaled to its peak. The soma's EPSP rises more slowly, so its peak comes later.")
    captions[5] = N.cap(f"{N_INPUTS} inputs on neighbouring spines of the branch, {N.ms(INTERVALS[0])} apart. Each EPSP has mostly decayed before the next.")
    captions[7] = N.cap(f"Past {M.DSPIKE_THRESHOLD:g} mV the branch fires a dendritic spike, which reaches the soma far larger than any EPSP.")
    synapse = N.new_synapse_view(view, H_EM, X_EM, Y_EM)

    # chapter 3: one release and its EPSP, beside the micrograph
    release = np.array([RELEASE])
    one = M.Events(releases=release)
    tr = M.traces(one, 0.0, ONE_STOP)
    shape = M.epsp_shape()
    row_glu_small = N.new_row(0.0, ONE_STOP, tr.glutamate, 1.25, x=X_SMALL, width=W_SMALL)
    row_v_small = N.new_row(0.0, ONE_STOP, tr.dff["spine"], -0.25, x=X_SMALL, width=W_SMALL)
    glu_small = N.row_path(row_glu_small, tr.t, tr.glutamate, GLU, 2.0)
    v_small = N.row_path(row_v_small, tr.t, tr.dff["spine"])
    name_glu_small = N.row_name(row_glu_small, "Glutamate at Spine 1 (simulated)", GLU)
    name_v_small = N.row_name(row_v_small, "Voltage at Spine 1 (simulated)", SITE_COLORS["spine"])
    bar_dff_small = N.dff_bar(row_v_small, kit.nice(0.5 * float(tr.dff["spine"].max())))
    bar_time_small = N.time_bar(row_v_small, 0.01)
    cursor_small = N.cursor_over([row_glu_small, row_v_small], 0.0)
    seconds_small = 0.6 * seconds[2]
    peak_dff = float(tr.dff["spine"].max())
    rise_at, decay_at = axes_points(row_v_small.axes, np.array([RELEASE, RELEASE + 0.018]), np.array([0.5 * peak_dff, 0.62 * peak_dff]))
    name_rise = label(f"rise {N.ms(M.EPSP_RISE)}", BAR, 15).next_to(rise_at, mn.LEFT, buff=0.06)
    name_decay = label(f"decay {N.ms(M.EPSP_DECAY)}", BAR, 15).next_to(decay_at, mn.UR, buff=0.04)

    # chapter 4: the same release at three sites, on one scale, with the voltage over the cell
    row_glu = N.new_row(0.0, ONE_STOP, tr.glutamate, Y_ROWS["top"])
    glu_top = N.row_path(row_glu, tr.t, tr.glutamate, GLU, 2.0)
    name_glu = N.row_name(row_glu, "Glutamate at Spine 1 (simulated)", GLU)
    top_e = float(tr.dff["spine"].max())
    rows_e = {site: N.new_row(0.0, ONE_STOP, tr.dff[site], Y_ROWS[site], top=top_e) for site in M.SITES}
    paths_e = {site: N.row_path(rows_e[site], tr.t, tr.dff[site]) for site in M.SITES}
    names_e = {site: N.row_name(rows_e[site], SITE_NAMES[site], SITE_COLORS[site]) for site in M.SITES}
    peaks_e = [label(f"{N.mv_text(shape.peak_mv[site])} mV", TEXT, 19).move_to(at(N.X_ROW + 0.5 * N.W_ROW + 0.15, Y_ROWS[site] - 0.2), aligned_edge=mn.LEFT) for site in M.SITES]
    bar_dff_e = N.dff_bar(rows_e["spine"], kit.nice(0.5 * top_e))
    bar_time_e = N.time_bar(rows_e["soma"], 0.01)
    cursor_e = N.cursor_over([row_glu, *rows_e.values()], 0.0)
    clock_e = N.new_clock(0.0)
    seconds_e = 0.62 * seconds[3]
    frames_e = N.heat_frames(p, one, N.sweep_times(rows_e["soma"], seconds_e, SPREAD_WARP))
    lens_cleft, rim_cleft = N.lens_image(N.cleft_rgba(), Y_ROWS["top"]), N.lens_ring(Y_ROWS["top"], GLU)
    rims, lens = N.site_column(p)
    cones = N.site_cones(view)
    cone_cleft = N.cone(at(N.X_LENS, Y_ROWS["spine"]), 0.1, at(N.X_LENS, Y_ROWS["top"]), N.R_LENS, GLU)

    # chapter 5: the delay from spine to soma
    one_short = M.traces(M.Events(releases=np.array([0.002])), 0.0, 0.03)
    fig = N.delay_figure(1.5, 0.3, 6.6, 3.0, one_short.fine_t, one_short.mv["spine"], one_short.mv["soma"], ("Spine 1", "Soma"), (SITE_COLORS["spine"], SITE_COLORS["soma"]), "EPSP: spine, then soma", shape.delay, 0.005)

    # chapters 6 to 8: inputs closer and closer in time, on one scale
    events = [input_events(interval) for interval in INTERVALS]
    outs = [M.traces(e, 0.0, SUM_STOP) for e in events]
    peaks = [float(out.mv["branch"].max()) for out in outs]
    top_s = max(peaks)
    row_ticks = N.new_row(0.0, SUM_STOP, np.ones(2), Y_TICKS, height=0.35)
    row_b = N.new_row(0.0, SUM_STOP, outs[-1].mv["branch"], Y_BRANCH, height=H_SUM, top=top_s)
    row_s = N.new_row(0.0, SUM_STOP, outs[-1].mv["soma"], Y_SOMA, height=H_SUM, top=top_s)
    ticks = [input_ticks(row_ticks, e) for e in events]
    paths_b = [N.row_path(row_b, out.fine_t, out.mv["branch"]) for out in outs]
    paths_s = [N.row_path(row_s, out.fine_t, out.mv["soma"]) for out in outs]
    name_ticks = N.row_name(row_ticks, "Inputs on neighbouring spines (simulated)", GLU)
    name_b = N.row_name(row_b, SITE_NAMES["branch"], SITE_COLORS["branch"])
    name_s = N.row_name(row_s, "Soma", SITE_COLORS["soma"])
    threshold_at = axes_points(row_b.axes, np.array([0.0, SUM_STOP]), np.full(2, M.DSPIKE_THRESHOLD))
    threshold = mn.DashedLine(*threshold_at, color=color(BAR), stroke_width=1.5, dash_length=0.08)
    name_threshold = label("Dendritic spike threshold", BAR, 15).next_to(threshold_at[1], mn.UL, buff=0.05)
    name_dspike = label("Dendritic spike", TEXT, 17).next_to(axes_points(row_b.axes, np.array([events[-1].dspikes[0] + 0.024]), np.array([0.8 * top_s]))[0], mn.RIGHT, buff=0.1)
    bar_mv_s = N.mv_bar(row_b, 20.0)
    bar_time_s = N.time_bar(row_s, 0.02)
    cursor_s = N.cursor_over([row_ticks, row_b, row_s], 0.0)
    clock_s = N.new_clock(0.0)
    seconds_s = 0.68 * seconds[5]
    frames_s = [N.heat_frames(p, e, N.sweep_times(row_s, duration)) for e, duration in zip(events, (seconds_s, 3.2, 3.8))]
    number = peak_text(peaks[0]).move_to(at(X_NUMBER, Y_NUMBER), aligned_edge=mn.LEFT)
    name_number = label("Branch peak", DIM, 17).next_to(number, mn.LEFT, buff=0.2)

    # chapter 9: the table
    y_header, pitch = 2.5, 1.0
    header = mn.VGroup(*[label(text, DIM, 17).move_to(at(x, y_header), aligned_edge=mn.LEFT) for text, x in (("branch, mV", -3.0), ("Branch", 2.2), ("Soma", 3.3), ("Dendritic spike", 4.4))])
    units = label("peak above rest, mV (simulated)", BAR, 15).move_to(at(2.2, y_header + 0.3), aligned_edge=mn.LEFT)
    note = label("each scaled to its peak", BAR, 15).move_to(at(-3.0, y_header + 0.3), aligned_edge=mn.LEFT)
    one_table = M.traces(M.Events(releases=np.array([INPUT_START])), 0.0, SUM_STOP)
    table_sources = [("1 input", one_table, 0.0), *[(f"{N_INPUTS} inputs, {N.ms(interval)} apart", out, e.dspikes.size) for interval, out, e in zip(INTERVALS, outs, events)]]
    table = N.table_rows([(name, kit.ACCENT, out.fine_t, out.mv["branch"], [N.mv_text(float(out.mv["branch"].max())), N.mv_text(float(out.mv["soma"].max())), "yes" if fired else "no"]) for name, out, fired in table_sources], y_header - 0.7, pitch, (2.2, 3.3, 4.4))

    print(f"EPSP peak mV {shape.peak_mv}, delay {1000 * shape.delay:.2f} ms, width {1000 * shape.width:.1f} ms")
    print(f"branch peak mV {dict(zip(INTERVALS, peaks))}, dendritic spike at {events[-1].dspikes[0]:.4f} s")

    chapters = [
        N.opening(seconds[0], view, title),
        chapter(seconds[1], *N.synapse_arrives(synapse, title, captions[1]), lag_ratio=1.0),
        chapter(seconds[2],
                mn.AnimationGroup(*replace(captions[1], captions[2]), mn.FadeIn(name_glu_small), mn.FadeIn(name_v_small), mn.FadeIn(bar_dff_small), mn.FadeIn(bar_time_small), mn.FadeIn(cursor_small), run_time=0.9),
                mn.AnimationGroup(mn.Create(glu_small, rate_func=mn.linear), mn.Create(v_small, rate_func=mn.linear),
                                  mn.MoveAlongPath(cursor_small, N.cursor_path(cursor_small, row_v_small), rate_func=mn.linear),
                                  N.puffs_during(synapse.micrograph, release, 0.0, ONE_STOP, seconds_small), run_time=seconds_small),
                mn.AnimationGroup(mn.FadeOut(cursor_small), mn.FadeIn(name_rise), mn.FadeIn(name_decay), run_time=0.6),
                lag_ratio=1.0),
        chapter(seconds[3],
                mn.AnimationGroup(*replace(captions[2], captions[3]), mn.FadeOut(v_small), mn.FadeOut(name_v_small), mn.FadeOut(name_glu_small), mn.FadeOut(bar_dff_small), mn.FadeOut(bar_time_small), mn.FadeOut(name_rise), mn.FadeOut(name_decay),
                                  *N.synapse_to_lens(synapse, lens_cleft, rim_cleft), mn.Transform(glu_small, glu_top), mn.FadeIn(name_glu), run_time=1.3),
                mn.AnimationGroup(mn.FadeIn(view.legend), *[mn.FadeIn(m) for m in (cones[0], rims["branch"], lens["branch"], cones[1], rims["soma"], lens["soma"], cones[2], rims["spine"], lens["spine"], cone_cleft)],
                                  *[mn.FadeIn(m) for m in names_e.values()], mn.FadeIn(bar_dff_e), mn.FadeIn(bar_time_e), mn.FadeIn(cursor_e), mn.FadeIn(clock_e), lag_ratio=0.03, run_time=1.2),
                N.sweep(list(paths_e.values()), cursor_e, rows_e["soma"], clock_e, view.image, frames_e, seconds_e, SPREAD_WARP),
                mn.AnimationGroup(mn.FadeOut(cursor_e), *[mn.FadeIn(m) for m in peaks_e], run_time=0.5),
                lag_ratio=1.0),
        chapter(seconds[4],
                mn.AnimationGroup(*replace(captions[3], captions[4]), mn.FadeOut(clock_e), mn.FadeOut(glu_small), mn.FadeOut(name_glu), mn.FadeOut(bar_dff_e), mn.FadeOut(bar_time_e),
                                  *[mn.FadeOut(m) for m in (*paths_e.values(), *names_e.values(), *peaks_e, *rims.values(), *lens.values(), *cones, cone_cleft, rim_cleft, lens_cleft)], run_time=1.0),
                mn.AnimationGroup(mn.FadeIn(fig["head"]), mn.FadeIn(fig["legend"]), *[mn.Create(m) for m in fig["paths"]], lag_ratio=0.08, run_time=1.6),
                mn.AnimationGroup(*[mn.Create(m) for m in fig["marks"]], mn.FadeIn(fig["span"]), mn.FadeIn(fig["number"]), mn.FadeIn(fig["bar"]), lag_ratio=0.1, run_time=1.4),
                lag_ratio=1.0),
        chapter(seconds[5],
                mn.AnimationGroup(*replace(captions[4], captions[5]), *[mn.FadeOut(fig[key]) for key in ("head", "legend", "span", "number", "bar")], *[mn.FadeOut(m) for m in (*fig["paths"], *fig["marks"])], run_time=0.8),
                mn.AnimationGroup(mn.FadeIn(name_ticks), mn.FadeIn(name_b), mn.FadeIn(name_s), mn.FadeIn(threshold), mn.FadeIn(name_threshold), mn.FadeIn(bar_mv_s), mn.FadeIn(bar_time_s), mn.FadeIn(ticks[0]), mn.FadeIn(cursor_s), mn.FadeIn(clock_s), lag_ratio=0.05, run_time=0.8),
                N.sweep([paths_b[0], paths_s[0]], cursor_s, row_s, clock_s, view.image, frames_s[0], seconds_s),
                mn.AnimationGroup(mn.FadeOut(cursor_s), mn.FadeOut(clock_s), mn.FadeIn(name_number), mn.FadeIn(number), run_time=0.5),
                lag_ratio=1.0),
        N.morph(seconds[6], [mn.FadeOut(captions[5]), mn.FadeIn(titles[6])],
              [mn.Transform(paths_b[0], paths_b[1]), mn.Transform(paths_s[0], paths_s[1]), mn.Transform(ticks[0], ticks[1]), Count(number, peaks[0], peaks[1], peak_text), Play(view.image, frames_s[1], rate_func=mn.linear)]),
        chapter(seconds[7],
                mn.AnimationGroup(*replace(titles[6], titles[7]), run_time=1.0),
                mn.AnimationGroup(mn.Transform(paths_b[0], paths_b[2]), mn.Transform(paths_s[0], paths_s[2]), mn.Transform(ticks[0], ticks[2]), Count(number, peaks[1], peaks[2], peak_text), Play(view.image, frames_s[2], rate_func=mn.linear), run_time=3.8),
                mn.AnimationGroup(mn.FadeIn(name_dspike), mn.FadeIn(captions[7]), run_time=0.5),
                lag_ratio=1.0),
        chapter(seconds[8],
                mn.AnimationGroup(mn.FadeOut(captions[7]), mn.FadeOut(titles[7]), mn.FadeOut(view.legend), *[mn.FadeOut(m) for m in (paths_b[0], paths_s[0], ticks[0], name_ticks, name_b, name_s, threshold, name_threshold, name_dspike, bar_mv_s, bar_time_s, name_number, number)], run_time=1.0),
                mn.AnimationGroup(mn.FadeIn(header), mn.FadeIn(units), mn.FadeIn(note), N.table_in(table), lag_ratio=0.2),
                lag_ratio=1.0),
    ]
    return timeline([], chapters)


if __name__ == "__main__":
    kit.run("epsp", CHAPTERS, UNTITLED, build, __doc__)
