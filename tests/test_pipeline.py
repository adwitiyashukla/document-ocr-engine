import numpy as np
import pytest
import torch

from docr import config, dbnet, dewarp, export, readers
from docr.geometry import box_to_quad
from docr.pipeline import Engine, TorchModel, reading_order

CONFIG = config.load("config.toml")


def tiny_engine():
    torch.manual_seed(0)
    detector = dbnet.DBNet((8, 16, 16, 16), (1, 1, 1, 1), 16)
    reader = readers.CRNN(len(readers.Charset()), (8, 8, 16, 16, 16, 16, 16), 16)
    return Engine(TorchModel(detector), TorchModel(reader), CONFIG)


def test_engine_runs_end_to_end():
    image = np.full((200, 300, 3), 230, np.uint8)
    image[80:100, 40:200] = 20
    out = tiny_engine()(image, find_page=True, flatten=True)
    assert len(out["quads"]) == len(out["texts"])
    assert out["page"].shape[2] == 3


def test_db_targets_round_trip_to_boxes():
    quads = [box_to_quad((20, 30, 120, 50)), box_to_quad((150, 30, 260, 52)), box_to_quad((20, 90, 200, 120))]
    gt, mask, thresh, thresh_mask = dbnet.targets((160, 300), quads, [False, False, False])
    assert 0.3 - 1e-6 <= thresh.min() and thresh.max() <= 0.7 + 1e-6
    found, _ = dbnet.boxes(gt, 0.3, 0.5, 1.5)
    assert len(found) == 3


def test_dewarp_straightens_curved_lines():
    rows = []
    for k in range(10):
        y = 40 + 30 * k
        words = []
        for j in range(6):
            x = 30 + 70 * j
            bend = 12 * np.sin(np.pi * x / 460)
            words.append(box_to_quad((x, y + bend, x + 55, y + bend + 14)))
        rows.extend(words)
    out = dewarp.dewarp(np.full((400, 460, 3), 255, np.uint8), np.array(rows))
    assert out is not None and out["lines"] == 10
    centers = np.array(rows).mean(axis=1)
    before = centers[:, 1].reshape(10, 6)
    after = dewarp.to_flat(centers, out["model"])[:, 1].reshape(10, 6)
    spread_before = np.abs(before - before.mean(axis=1, keepdims=True)).max()
    spread_after = np.abs(after - after.mean(axis=1, keepdims=True)).max()
    assert spread_after < 0.2 * spread_before


def test_dewarp_holds_still_outside_the_fitted_text():
    rows = []
    for k in range(8):
        y = 40 + 30 * k
        for j in range(4):
            x = 30 + 70 * j
            bend = 10 * np.sin(np.pi * x / 300)
            rows.append(box_to_quad((x, y + bend, x + 55, y + bend + 14)))
    out = dewarp.dewarp(np.full((300, 900, 3), 255, np.uint8), np.array(rows))
    edge = dewarp.surface(out["model"], np.array([out["model"]["x_range"][1]]), np.array([150.0]))
    far = dewarp.surface(out["model"], np.array([890.0]), np.array([150.0]))
    above = dewarp.surface(out["model"], np.array([200.0]), np.array([-400.0]))
    assert abs(far[0] - edge[0]) < 1e-6
    assert abs(above[0] + 400.0) < 30.0


def test_dewarp_leaves_straight_lines_alone():
    rows = [box_to_quad((30 + 70 * j, 40 + 30 * k, 85 + 70 * j, 54 + 30 * k)) for k in range(8) for j in range(4)]
    assert dewarp.dewarp(np.full((300, 400, 3), 255, np.uint8), np.array(rows)) is None


def test_reading_order_follows_lines():
    quads = [box_to_quad((100, 10, 150, 25)), box_to_quad((10, 10, 90, 25)), box_to_quad((10, 40, 60, 55))]
    assert reading_order(quads, ["world", "hello", "next"]) == "hello world\nnext"


def test_reading_order_joins_columns_on_one_row():
    quads = [box_to_quad((300, 42, 360, 57)), box_to_quad((10, 40, 70, 55)), box_to_quad((10, 10, 90, 25)), box_to_quad((300, 11, 350, 26))]
    assert reading_order(quads, ["3,000", "CHANGE", "TOTAL", "17,000"]) == "TOTAL 17,000\nCHANGE 3,000"


def test_onnx_matches_torch(tmp_path):
    model = dbnet.DBNet((8, 16, 16, 16), (1, 1, 1, 1), 16).eval()
    export.export_detector(model, tmp_path / "d.onnx")
    crnn = readers.CRNN(96, (8, 8, 16, 16, 16, 16, 16), 16).eval()
    export.export_crnn(crnn, tmp_path / "c.onnx")
    import onnxruntime as ort

    x = np.random.default_rng(0).normal(size=(1, 3, 96, 128)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x)).numpy()
    out = ort.InferenceSession(str(tmp_path / "d.onnx")).run(None, {"image": x})[0]
    assert np.abs(ref - out).max() < 1e-4
    y = np.random.default_rng(1).normal(size=(2, 1, 32, 200)).astype(np.float32)
    with torch.no_grad():
        ref = crnn(torch.from_numpy(y)).numpy()
    out = ort.InferenceSession(str(tmp_path / "c.onnx")).run(None, {"image": y})[0]
    assert np.abs(ref - out).max() < 1e-4


def test_config_rejects_bad_values():
    import copy

    bad = copy.deepcopy(CONFIG)
    bad["detect"]["threshold"] = 2.0
    with pytest.raises(config.ConfigError):
        config.validate(bad)
