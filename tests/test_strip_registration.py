import numpy as np
import pytest

from imgui_manim.lessons.strip_registration import model as M


@pytest.fixture(scope="module")
def recording():
    return M.simulate(seed=0)


def test_single_frame_is_mostly_noise(recording):
    r = np.mean([M.correlation(f, recording.tissue[M.window(shift=s)]) for f, s in zip(recording.frames, recording.shifts)])
    assert 0.15 < r < 0.3


def test_strip_drifts_smoothly_within_max_shift(recording):
    steps = np.abs(np.diff(recording.shifts, axis=0)).max(axis=1)
    assert np.abs(recording.shifts).max() <= M.MAX_SHIFT
    assert np.mean(steps <= 2) > 0.99
    assert np.ptp(recording.shifts[:, 0]) > M.FOV_SHAPE[0]


def test_padding_stays_nan_until_the_threshold(recording):
    counts, templates = M.grow_template(recording=recording, threshold=M.THRESHOLD)
    padding = np.isnan(recording.template_initial)
    assert np.array_equal(np.isfinite(templates)[:, padding], (counts >= M.THRESHOLD)[:, padding])
    assert np.isfinite(templates[:, ~padding]).all()


def test_pixel_joins_with_the_mean_of_all_its_visits(recording):
    counts, templates = M.grow_template(recording=recording, threshold=M.THRESHOLD)
    padding = np.isnan(recording.template_initial)
    row, col = np.argwhere(padding & (counts[-1] >= M.THRESHOLD))[0]
    t_join = int(np.argmax(counts[:, row, col] >= M.THRESHOLD))
    visiting = [t for t in range(t_join + 1) if counts[t, row, col] > (counts[t - 1, row, col] if t else 0)]
    values = [recording.frames[t, row - M.MAX_SHIFT - recording.shifts[t, 0], col - M.MAX_SHIFT - recording.shifts[t, 1]] for t in visiting]
    assert len(values) == M.THRESHOLD
    assert np.isnan(templates[t_join - 1, row, col])
    assert templates[t_join, row, col] == pytest.approx(np.mean(values))
