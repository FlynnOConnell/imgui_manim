import numpy as np
import pytest

from imgui_manim.lessons.compression import model as M


@pytest.fixture(scope="module")
def run():
    return M.load_run()


def test_white_noise_scores_root_six():
    noise = np.random.default_rng(0).normal(size=(4, 200000))
    assert np.allclose(M.temporal_roughness(traces=noise), np.sqrt(6), atol=0.02)


def test_smooth_trace_scores_far_under_noise():
    trace = np.sin(np.linspace(0, 20 * np.pi, 3000))[None]
    assert M.temporal_roughness(traces=trace)[0] < 0.01


@pytest.mark.parametrize(
    "passed, kept",
    [
        ([1, 1, 0, 1, 0, 0], [1, 1, 1, 0, 0, 0]),  # the first failure is kept, then the block stops
        ([0, 1, 1], [1, 0, 0]),
        ([1, 1, 1], [1, 1, 1]),
    ],
)
def test_keep_mask_stops_after_the_first_failure(passed, kept):
    assert np.array_equal(M.keep_mask(passed=np.array(passed, bool)), np.array(kept, bool))


def test_keep_mask_two_failures_in_a_row():
    passed = np.array([1, 0, 1, 0, 0, 1, 1], bool)
    assert np.array_equal(M.keep_mask(passed=passed, max_consecutive_failures=2), np.array([1, 1, 1, 1, 1, 0, 0], bool))


@pytest.mark.parametrize("size, last, count", [(128, 108, 12), (80, 60, 7), (100, 80, 9)])
def test_block_starts_end_flush_with_the_edge(size, last, count):
    starts = M.block_starts(size=size)
    assert starts[-1] == last and len(starts) == count
    assert starts[0] == 0


def test_row_major_order_undoes_the_tiles():
    groups_y, groups_x = [[0, 10], [20]], [[0], [10, 20]]
    visited = [(y, x) for gy in groups_y for gx in groups_x for y in gy for x in gx]
    assert visited != sorted(visited)
    assert [visited[k] for k in M.row_major_order(groups_y=groups_y, groups_x=groups_x)] == sorted(visited)


def test_row_major_order_of_one_tile_changes_nothing():
    assert M.row_major_order(groups_y=[[0, 10, 20]], groups_x=[[0, 10]]) == list(range(6))


def test_oriented_keeps_the_product():
    rng = np.random.default_rng(0)
    spatial, temporal = rng.normal(size=(5, 5, 3)), rng.normal(size=(3, 40))
    flipped_spatial, flipped_temporal, _ = M.oriented(spatial=spatial, temporal=temporal)
    assert np.allclose(np.einsum("hwk,kt->hwt", spatial, temporal), np.einsum("hwk,kt->hwt", flipped_spatial, flipped_temporal))
    assert (flipped_spatial.reshape(-1, 3)[np.abs(flipped_spatial.reshape(-1, 3)).argmax(axis=0), np.arange(3)] > 0).all()


def blob_and_noise_block(rng, n_frames=3000):
    """Candidates of a block: a compact image with a transient trace, then broad images with noise."""
    yy, xx = np.mgrid[0 : M.BLOCK, 0 : M.BLOCK]
    spatial = rng.normal(size=(M.BLOCK, M.BLOCK, 3))
    spatial[:, :, 0] = np.exp(-((yy - 9.5) ** 2 + (xx - 9.5) ** 2) / (2 * 3.0**2))
    temporal = rng.normal(size=(3, n_frames))
    temporal[0, 1000:1100] += 12 * np.exp(-np.arange(100) / 30)
    return spatial, temporal


def test_image_moments_find_a_blob():
    yy, xx = np.mgrid[0:20, 0:20]
    blob = np.exp(-((yy - 6) ** 2 + (xx - 13) ** 2) / (2 * 2.0**2))
    centers, spreads = M.image_moments(images=blob[..., None])
    assert np.allclose(centers[0], (6, 13), atol=0.01)
    assert spreads[0] == pytest.approx(2.0, abs=0.05)  # the square halves the variance: sqrt(2 * sd^2 / 2)


def test_skewness_of_noise_and_transients():
    rng = np.random.default_rng(0)
    noise = rng.normal(size=(1, 100000))
    transient = noise.copy()
    transient[0, :500] += 30
    assert abs(M.skewness(traces=noise)[0]) < 0.05
    assert M.skewness(traces=transient)[0] > 5


def test_cell_scores_rank_the_cell_first():
    spatial, temporal = blob_and_noise_block(rng=np.random.default_rng(0))
    assert M.cell_scores(spatial=spatial, temporal=temporal).argmax() == 0


def test_cell_scores_zero_a_cell_at_the_block_edge():
    spatial, temporal = blob_and_noise_block(rng=np.random.default_rng(0))
    shifted = np.roll(spatial[:, :, :1], 9, axis=0)
    shifted[:9] = 0
    assert M.cell_scores(spatial=shifted, temporal=temporal[:1])[0] == 0


def test_nice_blocks_need_a_gain_room_and_a_kept_cell():
    scores = np.array([[0.1, 0, 0, 0], [0.9, 0, 0, 0], [0.5, 0, 0, 0], [0, 0.8, 0, 0], [0.7, 0, 0, 0], [0.6, 0, 0, 0]])
    rank_1, rank_2 = np.array([1, 1, 1, 1, 2, 3]), np.array([2, 1, 2, 2, 5, 4])
    kept_1, kept_2 = (np.arange(20)[None] < rank[:, None] for rank in (rank_1, rank_2))
    allowed = np.array([True, True, True, True, True, False])
    # block 1 gains nothing, 3's cell is dropped, 4 overflows the rows, 5 is not allowed
    assert M.nice_blocks(scores=scores, kept_1=kept_1, kept_2=kept_2, allowed=allowed).tolist() == [2, 0]


def test_busiest_window_lands_on_the_transients():
    trace = np.random.default_rng(0).normal(size=5000)
    trace[3000:3050] += 20
    start = M.busiest_window(trace=trace, n_frames=300)
    assert start <= 3000 and start + 300 >= 3050


def test_noise_sd_reads_white_noise_through_transients():
    rng = np.random.default_rng(0)
    trace = rng.normal(0, 2.0, size=(1, 50000))
    trace[0, 1000:1200] += 50
    assert M.noise_sd(traces=trace)[0] == pytest.approx(2.0, rel=0.03)


def test_gain_on_raw_reads_the_true_gain_of_the_simulated_cell(run):
    truth, raw = run["truth"], run["trace_raw"]
    for key in ("trace_plain", "trace_denoised", "trace_long"):
        true_gain = np.polyfit(truth, run[key], 1)[0] / np.polyfit(truth, raw, 1)[0]
        assert M.gain_on_raw(traces=run[key][None], raw=raw[None])[0] == pytest.approx(true_gain, abs=0.02)


def test_table_rows_recording_start_at_raw():
    trace = np.zeros(5)
    short = [{"noise": 0.2, "noise_median": 0.3, "gain_median": 0.5}, {"noise": 0.4, "noise_median": 0.3, "gain_median": 0.7}]
    numbers = {"cell": 1, "noise": {"plain": [0.5, 0.25, 0.75], "denoised": [0.1, 0.2, 0.3]}, "gain": {"plain": [0.9, 0.8, 0.7], "denoised": [1.0, 0.9, 0.8]}, "short": short}
    rows = M.table_rows_recording({"numbers": numbers, "trace_raw": trace, "trace_plain": trace, "trace_denoised": trace, "trace_short": trace[None]})
    assert [row["name"] for row in rows] == ["raw", "plain", "shipped", "short"]
    assert rows[1]["noise"] == (0.25, 0.25) and rows[1]["noise_median"] == (0.5, 0.5) and rows[2]["gain_median"] == (0.9, 0.9)
    assert rows[3]["noise"] == (0.2, 0.4) and rows[3]["gain_median"] == (0.5, 0.7)


@pytest.mark.parametrize(
    "rank, text",
    [(3, "Candidates 5 to 20: dropped"), (4, "Candidates 5 to 20: dropped"), (7, "Candidates 5 to 20: 3 kept")],
)
def test_rest_text(rank, text):
    assert M.rest_text(kept=np.arange(20) < rank) == text


def test_run_blend_is_the_weighted_sum(run):
    expected = run["weight"] * run["prediction"] + (1 - run["weight"]) * run["blend_input"]
    assert np.allclose(expected, run["blended"], atol=1e-2 * run["blend_input"].std())


def test_run_ranks_add_up(run):
    assert run["ranks_1"].sum() == run["numbers"]["rank_1"]
    assert run["ranks_2"].sum() == run["numbers"]["rank_2"]
    assert run["ranks_1"].size == run["numbers"]["n_blocks"] == len(M.block_starts(M.SHAPE[0])) * len(M.block_starts(M.SHAPE[1]))


def test_run_block_gains_a_component(run):
    assert run["kept_2"].sum() > run["kept_1"].sum()


def test_run_noise_cut_is_the_fifth_percentile(run):
    for p in (1, 2):
        cut_spatial, cut_temporal = run["numbers"][f"thresholds_{p}"]
        assert cut_temporal == pytest.approx(np.percentile(run[f"noise_temporal_{p}"], 5), rel=1e-4)
        assert cut_spatial == pytest.approx(np.percentile(run[f"noise_spatial_{p}"], 5), rel=1e-4)


def test_run_denoiser_lifts_the_dim_cell_and_short_training_breaks_it(run):
    rows = {row["name"]: row for row in M.table_rows(run)}
    assert rows["shipped"]["r"][0] > rows["plain"]["r"][1]
    assert rows["short"]["r"][0] < 0.0
    assert rows["short"]["gain_median"][1] < 0.1
