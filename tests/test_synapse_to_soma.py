import numpy as np
import pytest
from PIL import Image

from imgui_manim.lessons.synapse_to_soma import model as M


def test_one_release_reaches_each_site_at_its_size_and_the_soma_last():
    out = M.traces(M.Events(releases=np.array([0.01])), 0.0, 0.08)
    peaks = {site: v.max() for site, v in out.mv.items()}
    assert peaks["spine"] == pytest.approx(M.EPSP_MV["spine"], rel=0.02)
    assert peaks["branch"] == pytest.approx(M.EPSP_MV["branch"], rel=0.05)
    assert peaks["soma"] == pytest.approx(M.EPSP_MV["soma"], rel=0.05)
    lags = {site: M.peak_lag(v, out.fine_t, 0.01) for site, v in out.mv.items()}
    assert lags["spine"] <= lags["branch"] < lags["soma"]
    assert M.peak_lag(out.glutamate, out.t, 0.01) < lags["soma"]


def test_a_spike_reaches_the_soma_first_with_no_glutamate():
    out = M.traces(M.Events(spikes=np.array([0.01])), 0.0, 0.03)
    assert out.mv["soma"].max() == pytest.approx(M.AP_MV, rel=0.02)
    assert out.mv["branch"].max() == pytest.approx(M.AP_MV * np.exp(-150.0 / M.BAP_LAMBDA), rel=0.02)
    assert M.peak_lag(out.mv["soma"], out.fine_t, 0.01) < M.peak_lag(out.mv["branch"], out.fine_t, 0.01)
    assert not out.glutamate.any()


def test_later_spikes_of_a_burst_are_smaller_and_fade_faster_up_the_dendrite():
    near = M.bap_mv(0.0, 0.0, 3) / M.bap_mv(0.0, 0.0, 0)
    far = M.bap_mv(150.0, 0.0005, 3) / M.bap_mv(150.0, 0.0005, 0)
    assert far < near < 1.0


def test_four_releases_stack_at_the_branch_and_stay_far_below_a_spike_at_the_soma():
    releases = M.releases_after(M.presynaptic_train(0.01))
    out = M.traces(M.Events(releases=releases), 0.0, 0.12)
    assert out.mv["branch"].max() > M.EPSP_MV["branch"]
    assert out.mv["soma"].max() < 2.0


def test_four_releases_leave_shallower_troughs_at_the_soma_than_at_the_spine():
    releases = M.releases_after(M.presynaptic_train(0.01))
    out = M.traces(M.Events(releases=releases), 0.0, 0.12)
    troughs = [out.mv[site][np.argmin(np.abs(out.fine_t - releases[1]))] / out.mv[site].max() for site in ("spine", "soma")]
    assert troughs[0] < troughs[1]


def test_inputs_2_ms_apart_fire_a_dendritic_spike_and_20_ms_apart_do_not():
    close = M.Events(releases=M.presynaptic_train(0.01, 5, 0.002))
    apart = M.Events(releases=M.presynaptic_train(0.01, 5, 0.02))
    assert M.dspike_time(close, 0.0, 0.1) is not None
    assert M.dspike_time(apart, 0.0, 0.15) is None


def test_neighbouring_inputs_2_ms_apart_fire_a_dendritic_spike_and_5_ms_apart_do_not():
    close = M.Events(neighbours=M.presynaptic_train(0.01, 5, 0.002))
    apart = M.Events(neighbours=M.presynaptic_train(0.01, 5, 0.005))
    assert M.dspike_time(close, 0.0, 0.1) is not None
    assert M.dspike_time(apart, 0.0, 0.1) is None


def test_a_dendritic_spike_reaches_the_soma_larger_than_an_epsp():
    out = M.traces(M.Events(dspikes=np.array([0.01])), 0.0, 0.06)
    assert out.mv["soma"].max() > 10 * M.EPSP_MV["soma"]


@pytest.mark.parametrize("spike, frames_apart", [(0.002, 1), (0.0025, 0)])
def test_a_bap_delay_of_half_a_frame_reads_as_0_or_1_frame(spike, frames_apart):
    out = M.traces(M.Events(spikes=np.array([spike])), 0.0, 0.008)
    assert int(np.argmax(out.dff["branch"])) - int(np.argmax(out.dff["soma"])) == frames_apart


def test_voltage_broadcasts_over_pixels_and_over_time():
    events = M.Events(releases=np.array([0.0]), spikes=M.spike_burst(0.01))
    pixels = np.array([0.0, 50.0, 150.0])
    assert M.voltage_mv(pixels, pixels[::-1], 0.005, events).shape == (3,)
    assert M.voltage_mv(2.0, 150.0, np.linspace(0, 0.03, 7), events).shape == (7,)


def test_every_drawn_pixel_of_the_cell_has_a_path_from_soma_and_spine():
    paths = M.load_paths()
    drawn = np.asarray(Image.open(M.DIR_NEURON / "cell.png"))[..., 3] > 0
    assert np.isfinite(paths["soma"][drawn]).all() and np.isfinite(paths["spine"][drawn]).all()
    cell = M.load_cell()
    x, y = np.round(cell.soma).astype(int)
    assert paths["spine"][y - 3:y + 3, x - 3:x + 3].min() == pytest.approx(M.DISTANCE_UM, abs=10.0)


def test_magnesium_leaves_only_when_the_membrane_depolarizes():
    assert M.magnesium_unblocked(0.0) < 0.1
    assert M.magnesium_unblocked(60.0) > 0.6


def test_a_bap_after_glutamate_opens_nmda_receptors_and_alone_opens_none():
    t = np.arange(0.0, 0.1, 1e-4)
    release = np.array([0.01])
    alone = M.nmda_open(t, M.Events(releases=release))
    paired = M.nmda_open(t, M.Events(releases=release, spikes=np.array([0.02])))
    no_glutamate = M.nmda_open(t, M.Events(spikes=np.array([0.02])))
    assert paired.max() > 3 * alone.max()
    assert no_glutamate.max() == 0.0


def test_a_bap_before_glutamate_opens_no_more_nmda_receptors_than_glutamate_alone():
    t = np.arange(0.0, 0.09, 1e-4)
    release = np.array([0.02])
    alone = M.nmda_open(t, M.Events(releases=release))
    before = M.nmda_open(t, M.Events(releases=release, spikes=np.array([0.01])))
    assert before.max() == pytest.approx(alone.max(), rel=0.01)


def test_the_cell_is_cited():
    assert "MICrONS" in M.NEURON_CITATION and "CC BY 4.0" in M.NEURON_CITATION
    assert "microns-explorer.org" in M.NEURON_SITE
    assert "CC BY 4.0" in (M.DIR_NEURON / "README.md").read_text()
