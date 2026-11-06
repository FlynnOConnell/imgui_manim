"""
One pyramidal cell's membrane voltage as a function of path distance and time, so the same events can be read off at a
spine, a branch and the soma, or painted over every pixel of a real cell:

- a presynaptic spike releases glutamate at a spine, and the glutamate gives an EPSP: a fast rise and a 15 ms decay, 8 mV
  in the spine head, 3 mV at the branch, under 1 mV at the soma, rising more slowly with path;
- EPSPs close in time add up on the branch; past a threshold the branch fires a dendritic spike;
- a somatic spike runs back up the dendrites as a bAP, smaller, broader and later with path, and a burst's later spikes
  fade faster;
- NMDA receptors at the spine conduct only with glutamate bound and with the voltage high enough to expel their
  magnesium, so a bAP landing after glutamate is what opens them.

Amplitudes and time constants are rough textbook values for shapes and orders of magnitude, with EPSP_MV and the bAP's
constants shared with imgui_ophys's Subcellular page. They are not fits to a dataset, and they are not fitted to the
cell: the cell gives the anatomy and the path distances, from the MICrONS volume (neuron/README.md).

The indicator: dF/F = GAIN * voltage, low-passed with a 1 ms time constant and averaged into 1 kHz frames, noiseless.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

V_REST = -65.0  # mV
GAIN = -0.0015  # dF/F per mV: the indicator loses 1.5% per 10 mV of depolarization
INDICATOR_TAU = 0.001  # s, the indicator's response time
FINE_HZ = 10000.0  # the membrane is simulated at this rate and averaged into frames
FS = 1000.0  # frames / s
DISTANCE_UM = 150.0  # path from the soma to Spine 1 in the model; the cell's own is 157 um by path.npz
SITES = ("spine", "branch", "soma")

RELEASE_DELAY = 0.0005  # s from a presynaptic spike to glutamate release: the synaptic delay
TRAIN_INTERVAL = 0.020  # s between presynaptic spikes: 50 Hz
GLU_RISE, GLU_DECAY, GLU_DFF = 0.0005, 0.010, 0.5  # the glutamate indicator's pulse per release
EPSP_RISE, EPSP_DECAY = 0.001, 0.015  # s
EPSP_MV = {"spine": 8.0, "branch": 3.0, "soma": 0.4}  # one release, seen at each site
EPSP_LAMBDA = 74.0  # um, 3 mV at the branch falls to 0.4 mV at a soma 150 um of path away
EPSP_RISE_SPREAD = 100.0  # um of path per ms added to the rise, so the soma's EPSP peaks later than the spine's
NECK_UM = 0.5  # um, the spine head's extra depolarization falls away across the neck
BRANCH_FROM_SPINE_UM = 2.0  # um of path from the spine head to the shaft under it

DSPIKE_THRESHOLD = 12.0  # mV of summed EPSPs at the branch that fires a dendritic spike
DSPIKE_MV = 35.0
DSPIKE_RISE, DSPIKE_PLATEAU, DSPIKE_FALL = 0.0005, 0.020, 0.004  # s
DSPIKE_LAMBDA = 102.0  # um, the 35 mV plateau at the branch falls to 8 mV at a soma 150 um of path away

AP_MV, AP_SIGMA = 100.0, 0.0004  # the somatic spike: peak above rest, mV, and its Gaussian width, s
AP_AHP_MV, AP_AHP_TAU = 6.0, 0.006  # afterhyperpolarization at the soma
AHP_REACH_UM = 30.0  # um of path the afterhyperpolarization reaches into the dendrite
BAP_LAMBDA = 200.0  # um, the bAP's amplitude falls as exp(-distance / BAP_LAMBDA)
BAP_SPEED = 300.0  # um / ms: 0.3 m/s, Stuart and Sakmann 1994
BAP_BROADEN = 600.0  # um of path per doubling of the bAP's width
SPIKE_ISI = 0.006  # s between the spikes of a burst
BURST_DROP = 0.9  # each spike of a burst reaches this share of the last one's height at the soma
BURST_DROP_UM = 600.0  # um of path per e-fold of extra loss per spike: later bAPs fade faster up the dendrite (Spruston 1995)
BURST_HUMP_MV, BURST_HUMP_RISE, BURST_HUMP_DECAY = 12.0, 0.002, 0.012  # the slow depolarization a burst rides on, s

NMDA_RISE, NMDA_DECAY = 0.003, 0.060  # s, the receptors' glutamate-bound fraction after a release
MG_MM = 1.0  # extracellular magnesium; the block is Jahr and Stevens 1990

DIR_NEURON = Path(__file__).resolve().parent / "neuron"
NEURON_NAME = "Mouse visual cortex pyramidal cell"
NEURON_METHOD = "Electron microscopy, one cell in all panels"
NEURON_CITATION = "MICrONS Consortium 2025, CC BY 4.0"
NEURON_SITE = "www.microns-explorer.org"


@dataclass
class Events:
    """Times in s: glutamate releases at Spine 1 and at its neighbours on the branch, dendritic spikes there, and somatic spikes."""

    releases: np.ndarray = field(default_factory=lambda: np.zeros(0))
    neighbours: np.ndarray = field(default_factory=lambda: np.zeros(0))
    dspikes: np.ndarray = field(default_factory=lambda: np.zeros(0))
    spikes: np.ndarray = field(default_factory=lambda: np.zeros(0))


@dataclass
class Traces:
    """What each site shows between two times, noiseless: the indicator's -dF/F (depolarization up) and the glutamate indicator's dF/F."""

    t: np.ndarray  # frame times, s
    dff: dict  # site -> -dF/F per frame
    glutamate: np.ndarray  # dF/F per frame
    mv: dict  # site -> mV above rest on the fine grid
    fine_t: np.ndarray


@dataclass
class Shape:
    """One event's size at each site and its timing along the 150 um between the spine and the soma."""

    name: str
    peak_mv: dict  # site -> mV above rest
    delay: float  # s between its peak at the near end of the path and at the far end
    width: float  # s, full width at half maximum at the branch


@dataclass
class Cell:
    """The reference cell's panels. Pixels are cell.png's unless a field says otherwise."""

    panels: dict  # name -> width, height (px), um_per_px
    soma: np.ndarray  # (2,)
    site: np.ndarray  # (2,) where the side branch ends, under Spine 1
    tall_crop: np.ndarray  # (4,) x0, y0, x1, y1: soma to tuft
    lens_branch: np.ndarray  # (4,) the round close-up around Spine 1, in branch.png's pixels
    inflate_um: float  # how far cell.png's surface is pushed out so thin branches show


def load_cell() -> Cell:
    """The reference cell as neuron/neuron.json stores it."""
    stored = json.loads((DIR_NEURON / "neuron.json").read_text())
    site = np.array([stored["side"]["x"][-1], stored["side"]["y"][-1]])
    return Cell(stored["panels"], np.array(stored["soma"]), site, np.array(stored["tall_crop"]), np.array(stored["lens_branch"]), stored["inflate_um"])


def load_paths() -> dict:
    """Path distance per pixel of cell.png in um, from "soma" and "spine"; infinite off the cell."""
    stored = np.load(DIR_NEURON / "path.npz")
    return {name: np.where(stored[name] == stored["unreached"], np.inf, stored[name] * float(stored["step_um"])) for name in ("soma", "spine")}


def load_outlines() -> dict:
    """Boolean images the size of synapse.png, "bouton" and "spine": the segmentation's outline of each in the micrograph."""
    stored = np.load(DIR_NEURON / "path.npz")
    shape = tuple(int(v) for v in stored["outline_shape"])
    return {name: np.unpackbits(stored[f"outline_{name}"])[: shape[0] * shape[1]].reshape(shape).astype(bool) for name in ("bouton", "spine")}


def kernel_difference(rise: float, decay: float, dt: float) -> np.ndarray:
    """Difference of exponentials, peak 1, long enough to decay to 1%."""
    t = np.arange(0, decay * 6, dt)
    k = np.exp(-t / decay) - np.exp(-t / rise)
    return k / k.max()


def lowpass(x: np.ndarray, tau: float, dt: float) -> np.ndarray:
    """First-order low-pass with time constant tau, starting from x[0]."""
    a = np.exp(-dt / tau)
    y = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = a * acc + (1 - a) * v
        y[i] = acc
    return y


def bin_frames(x: np.ndarray, per_frame: int) -> np.ndarray:
    n = x.size // per_frame
    return x[: n * per_frame].reshape(n, per_frame).mean(axis=1)


def site_paths(distance: float = DISTANCE_UM) -> dict:
    """Site name to (um of path from Spine 1, um of path from the soma), with Spine 1 distance um from the soma."""
    return {"spine": (0.0, distance), "branch": (BRANCH_FROM_SPINE_UM, distance), "soma": (distance, 0.0)}


def epsp_mv(um_from_spine: np.ndarray, lag: np.ndarray) -> np.ndarray:
    """
    One release's depolarization in mV, um_from_spine of path from its spine, lag s after the glutamate lands. Passive
    spread: it shrinks with path and rises more slowly, so its peak comes later.
    """
    amplitude = EPSP_MV["branch"] * np.exp(-um_from_spine / EPSP_LAMBDA) + (EPSP_MV["spine"] - EPSP_MV["branch"]) * np.exp(-um_from_spine / NECK_UM)
    rise = EPSP_RISE + um_from_spine / EPSP_RISE_SPREAD * 1e-3
    t_peak = rise * EPSP_DECAY / (EPSP_DECAY - rise) * np.log(EPSP_DECAY / rise)
    peak = np.exp(-t_peak / EPSP_DECAY) - np.exp(-t_peak / rise)
    after = np.maximum(lag, 0.0)
    return amplitude * (np.exp(-after / EPSP_DECAY) - np.exp(-after / rise)) / peak


def bap_mv(um_from_soma: np.ndarray, lag: np.ndarray, order: int = 0) -> np.ndarray:
    """
    A somatic spike's voltage in mV, um_from_soma of path along the dendrite, lag s after its peak at the soma: smaller,
    broader and later with path. order is its place in a burst.
    """
    amplitude = AP_MV * np.exp(-um_from_soma / BAP_LAMBDA) * BURST_DROP**order * np.exp(-order * um_from_soma / BURST_DROP_UM)
    sigma = AP_SIGMA * (1 + um_from_soma / BAP_BROADEN)
    delay = um_from_soma / BAP_SPEED * 1e-3
    spike = amplitude * np.exp(-((lag - delay) ** 2) / (2 * sigma**2))
    return spike - AP_AHP_MV * np.exp(-um_from_soma / AHP_REACH_UM) * np.exp(-np.maximum(lag - delay, 0.0) / AP_AHP_TAU) * (lag > delay)


def burst_hump_mv(um_from_soma: np.ndarray, t: np.ndarray, spikes: np.ndarray) -> np.ndarray:
    """The slow depolarization under a burst, rising before its first spike and decaying after its last."""
    rising = 1.0 / (1.0 + np.exp(-(t - spikes[0] + 2 * BURST_HUMP_RISE) / (0.5 * BURST_HUMP_RISE)))
    decaying = np.exp(-np.maximum(t - spikes[-1], 0.0) / BURST_HUMP_DECAY)
    return BURST_HUMP_MV * np.exp(-um_from_soma / BAP_LAMBDA) * rising * decaying


def dspike_mv(um_from_spine: np.ndarray, lag: np.ndarray) -> np.ndarray:
    """A dendritic spike's plateau in mV, um_from_spine of path from the branch site where it fires, lag s after it fires."""
    after = np.maximum(lag, 0.0)
    plateau = (1 - np.exp(-after / DSPIKE_RISE)) * 0.5 * (1 - np.tanh((after - DSPIKE_PLATEAU) / DSPIKE_FALL))
    return DSPIKE_MV * np.exp(-um_from_spine / DSPIKE_LAMBDA) * plateau * (lag > 0)


def voltage_mv(um_from_spine: np.ndarray, um_from_soma: np.ndarray, t: np.ndarray, events: Events) -> np.ndarray:
    """mV above rest at points of the cell at time t s, broadcast: pixels at one moment, or one site over time."""
    v = np.zeros(np.broadcast_shapes(np.shape(um_from_spine), np.shape(um_from_soma), np.shape(t)))
    for release in events.releases:
        v = v + epsp_mv(um_from_spine, t - release)
    for release in events.neighbours:
        v = v + epsp_mv(um_from_spine + BRANCH_FROM_SPINE_UM, t - release)
    for dspike in events.dspikes:
        v = v + dspike_mv(um_from_spine, t - dspike)
    for order, spike in enumerate(events.spikes):
        v = v + bap_mv(um_from_soma, t - spike, order)
    if events.spikes.size > 1:
        v = v + burst_hump_mv(um_from_soma, t, events.spikes)
    return v


def traces(events: Events, start: float, stop: float, distance: float = DISTANCE_UM) -> Traces:
    """Each site's voltage from start to stop s through the voltage indicator, and the glutamate indicator at the synapse."""
    per_frame = int(round(FINE_HZ / FS))
    dt = 1.0 / FINE_HZ
    fine_t = start + dt * np.arange(int(round((stop - start) * FINE_HZ)))
    mv = {site: voltage_mv(*paths, fine_t, events) for site, paths in site_paths(distance).items()}
    dff = {site: -bin_frames(lowpass(GAIN * v, INDICATOR_TAU, dt), per_frame) for site, v in mv.items()}
    impulses = np.zeros(fine_t.size)
    np.add.at(impulses, np.clip(((events.releases - start) * FINE_HZ).astype(int), 0, fine_t.size - 1), 1.0)
    glutamate = bin_frames(GLU_DFF * np.convolve(impulses, kernel_difference(GLU_RISE, GLU_DECAY, dt))[: fine_t.size], per_frame)
    return Traces(start + np.arange(glutamate.size) / FS, dff, glutamate, mv, fine_t)


def presynaptic_train(start: float, count: int = 4, interval: float = TRAIN_INTERVAL) -> np.ndarray:
    """A presynaptic axon's spike times, s."""
    return start + interval * np.arange(count)


def presynaptic_mv(t: np.ndarray, spikes: np.ndarray) -> np.ndarray:
    """The presynaptic axon's voltage in mV above rest at time t: one narrow spike per time in spikes."""
    return AP_MV * np.exp(-((np.asarray(t)[:, None] - spikes[None, :]) ** 2) / (2 * AP_SIGMA**2)).sum(axis=1)


def releases_after(spikes: np.ndarray) -> np.ndarray:
    """Glutamate release times, s, with every presynaptic spike releasing. Real cortical synapses fail on many spikes."""
    return spikes + RELEASE_DELAY


def spike_burst(start: float, count: int = 4) -> np.ndarray:
    """A burst's spike times, s."""
    return start + SPIKE_ISI * np.arange(count)


def peak_lag(x: np.ndarray, t: np.ndarray, start: float) -> float:
    """s from start to x's largest value."""
    return float(t[int(np.argmax(np.abs(x)))] - start)


def fwhm(x: np.ndarray, t: np.ndarray) -> float:
    """Full width at half maximum, s."""
    above = t[x >= 0.5 * x.max()]
    return float(above[-1] - above[0])


def dspike_time(events: Events, start: float, stop: float) -> Optional[float]:
    """The first time the summed EPSPs at the branch reach DSPIKE_THRESHOLD, s, or None."""
    t = start + np.arange(int(round((stop - start) * FINE_HZ))) / FINE_HZ
    crossed = voltage_mv(*site_paths()["branch"], t, Events(events.releases, events.neighbours)) >= DSPIKE_THRESHOLD
    return float(t[np.argmax(crossed)]) if crossed.any() else None


def epsp_shape() -> Shape:
    """One release at Spine 1, delay from the spine's peak to the soma's."""
    out = traces(Events(releases=np.array([0.01])), 0.0, 0.1)
    delay = peak_lag(out.mv["soma"], out.fine_t, 0.01) - peak_lag(out.mv["spine"], out.fine_t, 0.01)
    return Shape("EPSP", {site: float(v.max()) for site, v in out.mv.items()}, delay, fwhm(out.mv["branch"], out.fine_t))


def bap_shape() -> Shape:
    """One somatic spike, delay from the soma's peak to the branch's."""
    out = traces(Events(spikes=np.array([0.01])), 0.0, 0.05)
    delay = peak_lag(out.mv["branch"], out.fine_t, 0.01) - peak_lag(out.mv["soma"], out.fine_t, 0.01)
    return Shape("bAP", {site: float(v.max()) for site, v in out.mv.items()}, delay, fwhm(out.mv["branch"], out.fine_t))


def nmda_bound(t: np.ndarray, releases: np.ndarray) -> np.ndarray:
    """The share of NMDA receptors with glutamate bound at time t, peak 1 per release, capped at 1."""
    bound = np.zeros(np.shape(t))
    t_peak = NMDA_RISE * NMDA_DECAY / (NMDA_DECAY - NMDA_RISE) * np.log(NMDA_DECAY / NMDA_RISE)
    peak = np.exp(-t_peak / NMDA_DECAY) - np.exp(-t_peak / NMDA_RISE)
    for release in releases:
        after = np.maximum(t - release, 0.0)
        bound = bound + (np.exp(-after / NMDA_DECAY) - np.exp(-after / NMDA_RISE)) / peak
    return np.minimum(bound, 1.0)


def magnesium_unblocked(v_above_rest: np.ndarray) -> np.ndarray:
    """The share of NMDA receptors free of magnesium at this voltage (Jahr and Stevens 1990)."""
    return 1.0 / (1.0 + MG_MM / 3.57 * np.exp(-0.062 * (V_REST + v_above_rest)))


def nmda_open(t: np.ndarray, events: Events) -> np.ndarray:
    """The share of Spine 1's NMDA receptors conducting at time t: glutamate bound and magnesium out."""
    return nmda_bound(t, events.releases) * magnesium_unblocked(voltage_mv(*site_paths()["spine"], t, events))
