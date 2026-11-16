import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from docr import dbnet, dewarp, readers, scan
from docr.ctc import beam, greedy
from docr.geometry import apply_homography, crop_quad
from docr.imaging import to_gray


class TorchModel:
    def __init__(self, model):
        self.model = model.eval()

    @torch.no_grad()
    def __call__(self, batch):
        return self.model(torch.from_numpy(batch)).numpy()


class OnnxModel:
    def __init__(self, path, threads=None):
        import onnxruntime as ort

        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def __call__(self, batch):
        return self.session.run(None, {self.input: batch.astype(np.float32)})[0]


def _log_softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


class Engine:
    def __init__(self, detector, reader, config):
        self.detector = detector
        self.reader = reader
        self.config = config
        self.charset = readers.Charset()

    @classmethod
    def from_checkpoints(cls, config, detector_path, reader_path):
        state = torch.load(detector_path, map_location="cpu", weights_only=False)
        d = state["config"]["detector"]
        detector = dbnet.DBNet(tuple(d["channels"]), tuple(d["blocks"]), d["fpn"], d["k"])
        detector.load_state_dict(state["model"])
        state = torch.load(reader_path, map_location="cpu", weights_only=False)
        model = readers.build(state["config"]["crnn"])
        model.load_state_dict(state["model"])
        return cls(TorchModel(detector), TorchModel(model), config)

    @classmethod
    def from_onnx(cls, config, folder, quantized=True, threads=None):
        folder = Path(folder)
        suffix = ".int8.onnx" if quantized else ".onnx"
        return cls(OnnxModel(folder / f"dbnet{suffix}", threads), OnnxModel(folder / f"crnn{suffix}", threads), config)

    def probability(self, image):
        detect = self.config["detect"]
        resized, (sx, sy) = dbnet.resize_for_detection(np.asarray(image, np.float32), detect["max_side"])
        prob = self.detector(dbnet.normalize(resized)[None])[0, 0]
        return prob, (sx, sy)

    def detect(self, image):
        detect = self.config["detect"]
        prob, (sx, sy) = self.probability(image)
        quads, scores = dbnet.boxes(prob, detect["threshold"], detect["box_threshold"], detect["unclip"], detect["min_size"])
        if len(quads):
            quads = quads / np.array([sx, sy], np.float32)
        return quads, scores

    def read_crops(self, crops, batch_size=64):
        if not len(crops):
            return [], []
        texts, scores = [None] * len(crops), [None] * len(crops)
        prepared = [readers.prepare_crop(c, max_width=self.config["crnn"]["max_width"]) for c in crops]
        order = np.argsort([p.shape[1] for p in prepared], kind="stable")
        beam_width = self.config["pipeline"]["beam_width"]
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            x = readers.to_tensor([prepared[i] for i in idx]).numpy()
            logp = _log_softmax(self.reader(x))
            if beam_width > 1:
                decoded = [beam(torch.from_numpy(row), beam_width) for row in logp]
                seqs = [d[0] for d in decoded]
                conf = [float(np.exp(d[1] / max(len(d[0]), 1))) for d in decoded]
            else:
                seqs = greedy(torch.from_numpy(logp))
                conf = np.exp(logp.max(-1).mean(-1)).tolist()
            for j, i in enumerate(idx):
                texts[i] = self.charset.decode(seqs[j])
                scores[i] = float(conf[j])
        return texts, scores

    def read(self, image, quads, batch_size=64):
        gray = to_gray(np.asarray(image)).astype(np.float32)
        crops = [crop_quad(gray, q, 32, max_width=self.config["crnn"]["max_width"], fill=float(np.median(gray))) for q in quads]
        return self.read_crops(crops, batch_size)

    def __call__(self, image, find_page=None, flatten=None):
        options = self.config["pipeline"]
        find_page = options["find_page"] if find_page is None else find_page
        flatten = options["dewarp"] if flatten is None else flatten
        image = np.asarray(image)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        image = image[..., :3]
        timing = {}
        t = time.perf_counter()
        quads, det_scores = self.detect(image)
        timing["detect"] = time.perf_counter() - t
        t = time.perf_counter()
        texts, read_scores = self.read(image, quads)
        timing["read"] = time.perf_counter() - t
        result = {"quads": quads, "texts": texts, "scores": read_scores, "box_scores": det_scores, "timing": timing}
        if find_page or flatten:
            t = time.perf_counter()
            result.update(self.flatten(image, quads, find_page, flatten))
            timing["page"] = time.perf_counter() - t
        return result

    def flatten(self, image, quads, find_page=True, flatten=True):
        h, w = image.shape[:2]
        full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)
        quad = text_aware_page(image, quads) if find_page else full
        page, matrix = scan.rectify(image, quad)
        page_quads = apply_homography(matrix, quads) if len(quads) else quads
        out = {"page_quad": quad, "page": page, "page_matrix": matrix, "page_words": page_quads, "flat": None}
        if flatten and len(page_quads):
            fitted = dewarp.dewarp(page, page_quads, self.config["pipeline"]["min_dewarp_lines"], self.config["pipeline"]["min_dewarp_bend"])
            if fitted is not None:
                out["flat"] = fitted["image"]
                out["flat_words"] = dewarp.to_flat(page_quads, fitted["model"])
                out["lines"] = fitted["lines"]
        return out


def text_aware_page(image, quads):
    quad, _ = scan.find_page_from_text(image, quads)
    return quad


def reading_order(quads, texts):
    if not len(quads):
        return ""
    rows = []
    for chain in dewarp.word_lines(quads):
        y = float(np.mean([np.asarray(quads[k])[:, 1].mean() for k in chain]))
        x = float(np.asarray(quads[chain[0]])[:, 0].min())
        rows.append((y, x, " ".join(texts[k] for k in chain)))
    height = float(np.median([max(np.ptp(np.asarray(q)[:, 1]), 1.0) for q in quads]))
    lines = []
    for row in sorted(rows):
        if lines and abs(row[0] - sum(r[0] for r in lines[-1]) / len(lines[-1])) < 0.5 * height:
            lines[-1].append(row)
        else:
            lines.append([row])
    return "\n".join(" ".join(r[2] for r in sorted(line, key=lambda r: r[1])) for line in lines)


def draw(image, quads, texts=None, color=(220, 30, 30)):
    canvas = Image.fromarray(np.asarray(image, np.uint8)).convert("RGB")
    pen = ImageDraw.Draw(canvas)
    for i, q in enumerate(quads):
        pen.polygon([tuple(map(float, p)) for p in q], outline=color, width=2)
        if texts is not None:
            pen.text((float(q[0][0]), float(q[0][1]) - 11), texts[i], fill=(20, 20, 200))
    return canvas
