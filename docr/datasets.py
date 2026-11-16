import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from docr.geometry import box_to_quad

CORD_SPLITS = ("train", "validation", "test")
CORD_FILES = [
    "test-00000-of-00001-9c204eb3f4e11791.parquet",
    "validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
    "train-00000-of-00004-b4aaeceff1d90ecb.parquet",
    "train-00001-of-00004-7dbbe248962764c5.parquet",
    "train-00002-of-00004-688fe1305a55e5cc.parquet",
    "train-00003-of-00004-2d0cd200555ed7fd.parquet",
]
SOURCES = {
    "funsd/dataset.zip": "https://guillaumejaume.github.io/FUNSD/dataset.zip",
    "docunet/crop.zip": "http://vision.cs.stonybrook.edu/~kema/docwarp/crop.zip",
    "docunet/scan.zip": "http://vision.cs.stonybrook.edu/~kema/docwarp/scan.zip",
    **{f"cord/{name}": f"https://huggingface.co/datasets/naver-clova-ix/cord-v2/resolve/main/data/{name}" for name in CORD_FILES},
}
DOCUNET_DEWARPNET = [1, 9, 10, 12, 19, 20, 21, 22, 23, 24, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 44, 45, 46, 47, 49]
DOCUNET_DOCTR = [1, 2, 3, 4, 5, 6, 7, 9, 10, 21, 22, 23, 24, 27, 30, 31, 32, 36, 38, 40, 41, 44, 45, 46, 47, 48, 50, 51, 52, 53]


@dataclass
class Page:
    name: str
    image_path: str
    quads: np.ndarray
    texts: list
    ignore: np.ndarray
    dontcare: np.ndarray = field(default_factory=lambda: np.zeros((0, 4, 2), np.float32))
    roi: np.ndarray = None

    def load(self):
        with Image.open(self.image_path) as image:
            return np.asarray(image.convert("RGB"))


def _quad(entry):
    return np.array([[entry["x1"], entry["y1"]], [entry["x2"], entry["y2"]], [entry["x3"], entry["y3"]], [entry["x4"], entry["y4"]]], np.float32)


def _stack(quads):
    return np.array(quads, np.float32).reshape(-1, 4, 2)


def download(raw, log=print):
    raw = Path(raw)
    for relative, url in SOURCES.items():
        target = raw / relative
        if target.exists() and target.stat().st_size > 0:
            log(f"have {relative}")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, open(partial, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = response.read(1 << 20)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
            log(f"got {relative} ({done / 1e6:.1f} MB{'' if not total else f' of {total / 1e6:.1f} MB'})")
        partial.replace(target)


def prepare(raw):
    raw = Path(raw)
    funsd_zip = raw / "funsd" / "dataset.zip"
    if funsd_zip.exists() and not (raw / "funsd" / "dataset").exists():
        with zipfile.ZipFile(funsd_zip) as archive:
            members = [m for m in archive.namelist() if m.startswith("dataset/") and not m.endswith(".DS_Store")]
            archive.extractall(raw / "funsd", members)
    for part in ("crop", "scan"):
        archive_path = raw / "docunet" / f"{part}.zip"
        if archive_path.exists() and not (raw / "docunet" / part).exists():
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(raw / "docunet")
    for split in CORD_SPLITS:
        out = raw / "cord" / split
        files = sorted((raw / "cord").glob(f"{split}-*.parquet"))
        if not files or out.exists():
            continue
        import pyarrow.parquet as pq

        out.mkdir(parents=True)
        index = 0
        for file in files:
            table = pq.read_table(file)
            for row in table.to_pylist():
                data = row["image"]["bytes"]
                if data[:8] == b"\x89PNG\r\n\x1a\n":
                    (out / f"{index:04d}.png").write_bytes(data)
                else:
                    Image.open(io.BytesIO(data)).convert("RGB").save(out / f"{index:04d}.png")
                (out / f"{index:04d}.json").write_text(row["ground_truth"], encoding="utf-8")
                index += 1


def funsd(raw, split):
    folder = Path(raw) / "funsd" / "dataset" / ("training_data" if split == "train" else "testing_data")
    pages = []
    for path in sorted((folder / "annotations").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        quads, texts = [], []
        for entity in data["form"]:
            for word in entity["words"]:
                quads.append(box_to_quad(word["box"]))
                texts.append(word["text"].strip())
        ignore = np.array([not t for t in texts], bool)
        pages.append(Page(path.stem, str(folder / "images" / f"{path.stem}.png"), _stack(quads), texts, ignore))
    return pages


def cord(raw, split):
    folder = Path(raw) / "cord" / split
    pages = []
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        quads, texts = [], []
        for line in data["valid_line"]:
            for word in line["words"]:
                quads.append(_quad(word["quad"]))
                texts.append(word["text"].strip())
        dontcare = [_quad(q) for group in data.get("dontcare") or [] for q in group]
        dontcare += [_quad(item["quad"]) for group in data.get("repeating_symbol") or [] for item in group]
        roi = _quad(data["roi"]) if data.get("roi") else None
        ignore = np.array([not t for t in texts], bool)
        pages.append(Page(path.stem, str(path.with_suffix(".png")), _stack(quads), texts, ignore, _stack(dontcare), roi))
    return pages


def docunet(raw):
    folder = Path(raw) / "docunet"
    items = []
    for doc in range(1, 66):
        for variant in (1, 2):
            crop = folder / "crop" / f"{doc}_{variant} copy.png"
            items.append({"doc": doc, "variant": variant, "photo": str(crop), "scan": str(folder / "scan" / f"{doc}.png")})
    return items


def load_split(raw, name, split):
    if name == "funsd":
        return funsd(raw, split)
    if name == "cord":
        return cord(raw, split)
    raise ValueError(f"unknown dataset {name}")
