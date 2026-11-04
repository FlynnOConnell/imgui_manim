import masknmf
import numpy as np
import tifffile

from imgui_manim.window.decomposition import lag1_correlation, load_movie, pixel_snr, top_pixels, variance_shares


def test_background_plus_signal_is_masknmfs_reconstruction(run, d):
    pmd = run[0]
    full = masknmf.CompressionArray.from_flyweight(pmd.shape, pmd.flyweight, rescale=True, include_trend=True)
    reconstruction = np.asarray(full[:])
    scale = float(np.abs(reconstruction).max())
    np.testing.assert_allclose(d.background + d.signal, reconstruction, atol=1e-4 * scale)


def test_residual_is_masknmfs_residual(run, d):
    pmd = run[0]
    residual = np.asarray(masknmf.CompressionResidualArray(d.registered, pmd)[:])
    np.testing.assert_allclose(d.residual, residual, atol=1e-4 * float(np.abs(d.registered).max()))


def test_pieces_add_up_to_the_registered_movie(d):
    np.testing.assert_allclose(d.background + d.signal + d.residual, d.registered, atol=1e-3)


def test_trend_has_zero_mean_over_time(d):
    if d.trend_temporal.shape[0] == 0:
        return
    trend = d.background - d.mean_image
    assert float(np.abs(trend.mean(axis=0)).max()) < 1e-3 * float(np.abs(d.registered).max())


def test_residual_is_closer_to_white_than_the_fluctuations(d):
    assert np.median(lag1_correlation(d.residual)) < 0.1 < np.median(lag1_correlation(d.fluctuations))


def test_variance_shares_are_fractions(d):
    shares = variance_shares(d)
    assert set(shares) == {"background", "signal", "residual"}
    assert all(0 <= v <= 1 for v in shares.values())


def test_stored_fraction_matches_what_u_and_v_hold(run, d):
    pmd = run[0]
    stored = pmd.spatial_compressed._nnz() + pmd.temporal_compressed.numel()
    assert d.stored_fraction == stored / d.registered.size


def test_top_pixels_are_the_best_kept_apart(d):
    snr = pixel_snr(d)
    pixels = top_pixels(d, 5, min_distance=6)
    assert len(pixels) == 5
    assert snr[pixels[0]] == snr.max()
    for i, (row_a, col_a) in enumerate(pixels):
        for row_b, col_b in pixels[i + 1:]:
            assert max(abs(row_a - row_b), abs(col_a - col_b)) >= 6
    assert top_pixels(d, 3, first=(3, 3))[0] == (3, 3)


def test_load_movie_reads_tif_and_npy_alike(tmp_path):
    frames = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    tifffile.imwrite(tmp_path / "movie.tif", frames, photometric="minisblack")
    np.save(tmp_path / "movie.npy", frames)
    np.testing.assert_array_equal(load_movie(str(tmp_path / "movie.tif")), frames)
    np.testing.assert_array_equal(load_movie(str(tmp_path / "movie.npy")), frames)


def test_negative_flips_the_movie_and_stays_positive(tmp_path):
    frames = np.array([[[10.0, 40.0]], [[20.0, 30.0]]], dtype=np.float32)
    np.save(tmp_path / "movie.npy", frames)
    flipped = load_movie(str(tmp_path / "movie.npy"), negative=True)
    np.testing.assert_array_equal(flipped, 40.0 - frames)
    assert flipped.min() == 0
