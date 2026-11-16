from pathlib import Path

import numpy as np
import torch
from PIL import Image

from docr import dbnet, readers


def _export(module, args, target, names, outputs, axes):
    module.eval()
    torch.onnx.export(module, args, str(target), input_names=names, output_names=outputs, dynamic_axes=axes, opset_version=17, dynamo=False)


def export_detector(model, target):
    _export(model, (torch.zeros(1, 3, 640, 640),), target, ["image"], ["prob"], {"image": {0: "batch", 2: "height", 3: "width"}, "prob": {0: "batch", 2: "height", 3: "width"}})


def export_crnn(model, target):
    _export(model, (torch.zeros(1, 1, 32, 128),), target, ["image"], ["logits"], {"image": {0: "batch", 3: "width"}, "logits": {0: "batch", 1: "steps"}})


class PageCalibration:
    def __init__(self, images, max_side, crop=640):
        self.items = []
        for path in images:
            with Image.open(path) as handle:
                image = np.asarray(handle.convert("RGB"), np.float32)
            resized, _ = dbnet.resize_for_detection(image, max_side)
            page = dbnet.normalize(resized)
            _, h, w = page.shape
            top, left = max(h - crop, 0) // 2, max(w - crop, 0) // 2
            part = page[:, top:top + crop, left:left + crop]
            window = np.zeros((3, crop, crop), np.float32)
            window[:, :part.shape[1], :part.shape[2]] = part
            self.items.append({"image": window[None]})
        self.index = 0

    def get_next(self):
        if self.index >= len(self.items):
            return None
        item = self.items[self.index]
        self.index += 1
        return item

    def rewind(self):
        self.index = 0


def quantize_dynamic(source, target):
    from onnxruntime.quantization import QuantType
    from onnxruntime.quantization import quantize_dynamic as run

    run(str(source), str(target), weight_type=QuantType.QInt8, op_types_to_quantize=["LSTM", "MatMul"])


def quantize_static(source, target, calibration):
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType
    from onnxruntime.quantization import quantize_static as run
    from onnxruntime.quantization.shape_inference import quant_pre_process

    prepared = Path(target).with_suffix(".pre.onnx")
    quant_pre_process(str(source), str(prepared), skip_symbolic_shape=True)
    run(
        str(prepared), str(target), calibration, quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8, per_channel=True, calibrate_method=CalibrationMethod.Percentile,
        extra_options={"CalibPercentile": 99.99},
    )
    prepared.unlink(missing_ok=True)


def export_all(config, detector_path, reader_path, folder, calibration_images, log=print):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    state = torch.load(detector_path, map_location="cpu", weights_only=False)
    d = state["config"]["detector"]
    detector = dbnet.DBNet(tuple(d["channels"]), tuple(d["blocks"]), d["fpn"], d["k"])
    detector.load_state_dict(state["model"])
    export_detector(detector, folder / "dbnet.onnx")
    calibration = PageCalibration(calibration_images, config["detect"]["max_side"])
    quantize_static(folder / "dbnet.onnx", folder / "dbnet.int8.onnx", calibration)
    log("exported dbnet.onnx and dbnet.int8.onnx")
    state = torch.load(reader_path, map_location="cpu", weights_only=False)
    model = readers.build(state["config"]["crnn"])
    model.load_state_dict(state["model"])
    export_crnn(model, folder / "crnn.onnx")
    quantize_dynamic(folder / "crnn.onnx", folder / "crnn.int8.onnx")
    log("exported crnn.onnx and crnn.int8.onnx")
    return folder


def parity(config, detector_path, reader_path, folder, image):
    from docr.pipeline import Engine

    torch_engine = Engine.from_checkpoints(config, detector_path, reader_path)
    onnx_engine = Engine.from_onnx(config, folder, quantized=False)
    prob_t, _ = torch_engine.probability(image)
    prob_o, _ = onnx_engine.probability(image)
    quads, _ = torch_engine.detect(image)
    texts_t, _ = torch_engine.read(image, quads[:64])
    texts_o, _ = onnx_engine.read(image, quads[:64])
    same = float(np.mean([a == b for a, b in zip(texts_t, texts_o)])) if texts_t else 1.0
    return {"prob_max_abs_diff": float(np.abs(prob_t - prob_o).max()), "text_agreement": same}
