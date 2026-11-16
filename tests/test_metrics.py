import numpy as np
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment
from skimage.metrics import structural_similarity

from docr import metrics
from docr.geometry import box_to_quad

rng = np.random.default_rng(3)
ALPHABET = "abcdeABCDE01234 .,-"


def random_text(n):
    return "".join(rng.choice(list(ALPHABET), size=n))


def test_levenshtein_matches_rapidfuzz():
    for _ in range(300):
        a, b = random_text(int(rng.integers(0, 30))), random_text(int(rng.integers(0, 30)))
        assert metrics.levenshtein(a, b) == Levenshtein.distance(a, b)


def test_hungarian_matches_scipy():
    for shape in [(5, 5), (7, 4), (3, 9), (40, 40), (1, 6)]:
        cost = rng.uniform(0, 1, shape)
        rows, cols = metrics.hungarian(cost)
        ref_rows, ref_cols = linear_sum_assignment(cost)
        assert np.isclose(cost[rows, cols].sum(), cost[ref_rows, ref_cols].sum())
        assert len(set(rows)) == len(rows) and len(set(cols)) == len(cols) == min(shape)


def test_word_metrics_counts_matches_and_text():
    gt = [box_to_quad((0, 0, 50, 20)), box_to_quad((60, 0, 110, 20)), box_to_quad((0, 40, 50, 60))]
    det = [box_to_quad((1, 1, 51, 21)), box_to_quad((62, 0, 112, 21)), box_to_quad((300, 300, 340, 320))]
    m = metrics.WordMetrics()
    m.update(gt, ["Total", "Rp", "x"], [False, False, True], det, ["Total", "rp", "noise"])
    out = m.compute()
    assert np.isclose(out["det"]["recall"], 1.0)
    assert np.isclose(out["det"]["precision"], 2 / 3)
    assert np.isclose(out["e2e"]["recall"], 0.5)
    assert np.isclose(out["e2e_caseless"]["recall"], 1.0)


def test_dontcare_drops_unmatched_detection():
    m = metrics.WordMetrics()
    m.update([box_to_quad((0, 0, 50, 20))], ["a"], [False], [box_to_quad((0, 0, 50, 20)), box_to_quad((100, 0, 140, 20))], ["a", "b"], dontcare=[box_to_quad((95, 0, 150, 25))])
    assert m.counts["det"] == 1


def test_text_metrics():
    t = metrics.TextMetrics()
    t.update(["hello", "Wor1d"], ["hello", "World"])
    out = t.compute()
    assert out["accuracy"] == 0.5 and np.isclose(out["cer"], 1 / 10)


def test_ssim_matches_skimage_inside_borders():
    x = rng.uniform(0, 255, (80, 90))
    y = np.clip(x + rng.normal(0, 25, x.shape), 0, 255)
    ours = metrics.ssim(x, y)[5:-5, 5:-5].mean()
    theirs = structural_similarity(x, y, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
    assert abs(ours - theirs) < 1e-6


def test_msssim_of_identical_images_is_one():
    image = rng.uniform(0, 255, (300, 220)).round()
    assert abs(metrics.docunet_msssim(image, image) - 1.0001) < 1e-6
