from pathlib import Path

import numpy as np
import pytest

from docr import synth

FONTS = Path("data/fonts")
pytestmark = pytest.mark.skipif(not any(FONTS.glob("*.ttf")), reason="fonts not downloaded")


def test_word_renders_at_reader_height():
    rng = np.random.default_rng(0)
    bank = synth.FontBank(FONTS)
    source = synth.TextSource(0)
    for _ in range(20):
        crop = synth.render_word(source.token(rng), bank, rng)
        assert crop.dtype == np.uint8 and crop.shape[0] == 32 and crop.shape[1] >= 8


def test_tokens_stay_in_charset():
    rng = np.random.default_rng(3)
    source = synth.TextSource(3)
    allowed = set(synth.CHARSET) | {" "}
    for _ in range(5000):
        text = source.token(rng, ["form", "receipt", None][int(rng.integers(3))])
        assert text.strip() and set(text) <= allowed


def test_jitter_stays_invertible_for_flat_boxes():
    for seed in range(300):
        rng = np.random.default_rng(seed)
        quad = np.array([[10, 20], [10 + rng.uniform(0, 3), 20], [10 + rng.uniform(0, 3), 20 + rng.uniform(0, 1)], [10, 20]], float)
        matrix = synth._jitter_matrix(rng, quad, int(rng.integers(11, 56)))
        assert np.isfinite(matrix).all() and abs(np.linalg.det(matrix)) > 1e-3


def test_tiny_and_spaced_words_render():
    bank = synth.FontBank(FONTS)
    for seed in range(40):
        rng = np.random.default_rng(seed)
        crop = synth.render_word([".", "-", "_", "'", "W i d e", "MMMMMMMMMMMM"][seed % 6], bank, rng)
        assert crop.shape[0] == 32 and crop.shape[1] >= 8


@pytest.mark.parametrize("kind", ["form_scan", "receipt_photo"])
def test_pages_keep_boxes_inside_image(kind):
    rng = np.random.default_rng(1)
    out = synth.render_page(synth.FontBank(FONTS), synth.TextSource(1), rng, kind)
    h, w = out["image"].shape[:2]
    assert len(out["quads"]) == len(out["texts"]) == len(out["ignore"]) > 10
    centers = out["quads"].mean(axis=1)
    assert ((centers[:, 0] > -5) & (centers[:, 0] < w + 5) & (centers[:, 1] > -5) & (centers[:, 1] < h + 5)).mean() > 0.95
