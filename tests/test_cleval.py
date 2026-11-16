from pathlib import Path

import numpy as np
import pytest

from docr import datasets
from docr.cleval import CLEval

CLEvalMetric = pytest.importorskip("cleval.torchmetric").CLEvalMetric
rng = np.random.default_rng(7)


def official(gt_quads, gt_texts, gt_dc, det_quads, det_texts, case_sensitive=True):
    metric = CLEvalMetric(case_sensitive=case_sensitive)
    metric(
        np.asarray(det_quads, np.float32).reshape(-1, 8),
        np.asarray(gt_quads, np.float32).reshape(-1, 8),
        list(det_texts),
        list(gt_texts),
        list(gt_dc),
    )
    out = metric.compute()
    return {k: float(out[k]) for k in ("det_r", "det_p", "det_h", "e2e_r", "e2e_p", "e2e_h")}


def ours(gt_quads, gt_texts, gt_dc, det_quads, det_texts, case_sensitive=True):
    metric = CLEval(case_sensitive=case_sensitive)
    metric.update(gt_quads, gt_texts, gt_dc, det_quads, det_texts)
    out = metric.compute()
    return {
        "det_r": out["det"]["recall"], "det_p": out["det"]["precision"], "det_h": out["det"]["hmean"],
        "e2e_r": out["e2e"]["recall"], "e2e_p": out["e2e"]["precision"], "e2e_h": out["e2e"]["hmean"],
    }


def perturb(quads, texts, ignore):
    det_quads, det_texts = [], []
    for q, t, ig in zip(quads, texts, ignore):
        r = rng.random()
        if r < 0.1:
            continue
        box = q + rng.normal(0, 1.5, q.shape)
        if r < 0.2 and len(t) >= 4:
            cut = len(t) // 2
            mid = (box[0] + box[1]) / 2, (box[3] + box[2]) / 2
            det_quads += [np.array([box[0], mid[0], mid[1], box[3]]), np.array([mid[0], box[1], box[2], mid[1]])]
            det_texts += [t[:cut], t[cut:]]
            continue
        text = t
        if rng.random() < 0.2 and t:
            k = int(rng.integers(len(t)))
            text = t[:k] + "x" + t[k + 1:]
        if rng.random() < 0.1:
            text = text.lower()
        det_quads.append(box)
        det_texts.append(text)
    for _ in range(3):
        x, y = rng.uniform(0, 600, 2)
        det_quads.append(np.array([[x, y], [x + 40, y], [x + 40, y + 12], [x, y + 12]]))
        det_texts.append("noise")
    return det_quads, det_texts


@pytest.mark.skipif(not Path("data/raw/funsd/dataset").exists() or not Path("data/raw/cord/test").exists(), reason="datasets not prepared")
@pytest.mark.parametrize("name,split", [("funsd", "test"), ("cord", "test")])
@pytest.mark.parametrize("case_sensitive", [True, False])
def test_matches_official_on_real_pages(name, split, case_sensitive):
    pages = datasets.load_split("data/raw", name, split)[:4]
    for page in pages:
        dc = list(page.ignore) + [True] * len(page.dontcare)
        quads = list(page.quads) + list(page.dontcare)
        texts = list(page.texts) + ["###"] * len(page.dontcare)
        det_quads, det_texts = perturb(page.quads, page.texts, page.ignore)
        a = official(quads, texts, dc, det_quads, det_texts, case_sensitive)
        b = ours(quads, texts, dc, det_quads, det_texts, case_sensitive)
        for key in a:
            assert abs(a[key] - b[key]) < 1e-4, (page.name, key, a[key], b[key])


def test_random_layouts_match_official():
    for _ in range(10):
        gt_quads, gt_texts = [], []
        for row in range(6):
            x = 10.0
            for _ in range(int(rng.integers(2, 6))):
                w = float(rng.uniform(20, 80))
                gt_quads.append(np.array([[x, 20 + 30 * row], [x + w, 20 + 30 * row], [x + w, 38 + 30 * row], [x, 38 + 30 * row]]))
                gt_texts.append("".join(rng.choice(list("abcdefgh0123"), size=int(rng.integers(1, 9)))))
                x += w + float(rng.uniform(5, 15))
        dc = [bool(rng.random() < 0.1) for _ in gt_quads]
        det_quads, det_texts = perturb(np.array(gt_quads), gt_texts, np.array(dc))
        a = official(gt_quads, gt_texts, dc, det_quads, det_texts)
        b = ours(gt_quads, gt_texts, dc, det_quads, det_texts)
        for key in a:
            assert abs(a[key] - b[key]) < 1e-4


def test_merged_detection_matches_official():
    gt = [np.array([[0, 0], [50, 0], [50, 20], [0, 20]]), np.array([[60, 0], [110, 0], [110, 20], [60, 20]])]
    det = [np.array([[-1, -1], [111, -1], [111, 21], [-1, 21]])]
    a = official(gt, ["hello", "world"], [False, False], det, ["helloworld"])
    b = ours(gt, ["hello", "world"], [False, False], det, ["helloworld"])
    for key in a:
        assert abs(a[key] - b[key]) < 1e-6
