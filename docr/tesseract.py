import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from docr.geometry import box_to_quad


def available():
    return shutil.which("tesseract") is not None


def version():
    out = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
    return (out.stdout or out.stderr).splitlines()[0].strip()


def _call(image, args, timeout):
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "page.png"
        Image.fromarray(np.asarray(image, np.uint8)).save(path)
        out = subprocess.run(["tesseract", str(path), "stdout", *args], capture_output=True, text=True, timeout=timeout)
    return out.stdout


def words(image, psm=3, lang="eng", timeout=600):
    output = _call(image, ["--psm", str(psm), "-l", lang, "tsv"], timeout)
    quads, texts, scores = [], [], []
    for line in output.splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 12 or parts[0] != "5":
            continue
        text = parts[11].strip()
        if not text:
            continue
        left, top, width, height = (int(v) for v in parts[6:10])
        quads.append(box_to_quad((left, top, left + width, top + height)))
        texts.append(text)
        scores.append(float(parts[10]))
    return np.array(quads, np.float32).reshape(-1, 4, 2), texts, np.array(scores, np.float32)


def text(image, psm=3, lang="eng", timeout=600):
    return _call(image, ["--psm", str(psm), "-l", lang], timeout)


def read_word(crop, lang="eng", timeout=60):
    return _call(crop, ["--psm", "7", "-l", lang], timeout).strip()
