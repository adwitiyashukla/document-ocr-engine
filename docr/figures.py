import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFont

from docr import datasets
from docr.config import path
from docr.metrics import match

PUBLISHED = {"funsd": {"Tesseract (paper)": 69.09, "docTR DBNet + CRNN (paper)": 90.30}, "cord": {"Tesseract (paper)": 55.43, "docTR DBNet + CRNN (paper)": 96.75}}


def _fit(image, height):
    scale = height / image.height
    return image.resize((max(1, int(image.width * scale)), height), Image.LANCZOS)


def _row(images, height=900, gap=16):
    images = [_fit(image, height) for image in images]
    canvas = Image.new("RGB", (sum(image.width for image in images) + gap * (len(images) - 1), height), (255, 255, 255))
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    return canvas


def _side_by_side(left, right, height=900, gap=16):
    return _row([left, right], height, gap)


def _boxes(image, quads, colors, width=3):
    canvas = Image.fromarray(np.asarray(image, np.uint8)).convert("RGB")
    pen = ImageDraw.Draw(canvas)
    for quad, color in zip(quads, colors):
        pen.polygon([tuple(map(float, p)) for p in quad], outline=color, width=width)
    return canvas


def _font(size, family="DejaVu Sans"):
    from matplotlib import font_manager

    return ImageFont.truetype(font_manager.findfont(family), max(6, int(size)))


def _highlight(image, quads, color=(255, 230, 102)):
    base = Image.fromarray(np.asarray(image, np.uint8)).convert("RGB")
    mask = Image.new("L", base.size, 0)
    pen = ImageDraw.Draw(mask)
    for quad in quads:
        pen.polygon([tuple(map(float, p)) for p in quad], fill=255)
    return Image.composite(ImageChops.multiply(base, Image.new("RGB", base.size, color)), base, mask)


def _text_panel(text, height, size=24, pad=28):
    lines = text.splitlines() or [""]
    size = min(size, int((height - 2 * pad) / (1.45 * len(lines))))
    font = _font(size, "DejaVu Sans Mono")
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    width = int(max(probe.textlength(line, font=font) for line in lines)) + 2 * pad
    canvas = Image.new("RGB", (max(width, 240), height), (255, 255, 255))
    pen = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        pen.text((pad, pad + i * 1.45 * size), line, fill=(23, 40, 58), font=font)
    return canvas


def receipt(engine, page):
    from docr.pipeline import reading_order

    image = page.load()
    out = engine(image, find_page=True, flatten=True)
    left = _highlight(image, out["quads"])
    pen = ImageDraw.Draw(left)
    pen.polygon([tuple(map(float, p)) for p in out["page_quad"]], outline=(30, 160, 60), width=6)
    flat = out["flat"] if out.get("flat") is not None else out["page"]
    words = out.get("flat_words") if out.get("flat") is not None else out["page_words"]
    middle = _highlight(np.clip(flat, 0, 255).astype(np.uint8), words)
    return _row([left, middle, _text_panel(reading_order(out["quads"], out["texts"]), 900)])


def form(engine, page):
    image = page.load()
    out = engine(image, find_page=False, flatten=False)
    rows, cols, _ = match(page.quads[~page.ignore], out["quads"])
    gt_texts = [t for t, i in zip(page.texts, page.ignore) if not i]
    colors = [(230, 140, 0)] * len(out["quads"])
    for r, c in zip(rows, cols):
        colors[c] = (40, 170, 60) if out["texts"][c] == gt_texts[r] else (220, 30, 30)
    return _boxes(image, out["quads"], colors, 2)


def docunet(engine, item):
    with Image.open(item["photo"]) as handle:
        photo = np.asarray(handle.convert("RGB"))
    out = engine(photo, find_page=True, flatten=True)
    flat = out["flat"] if out.get("flat") is not None else out["page"]
    return _side_by_side(Image.fromarray(photo), Image.fromarray(np.clip(flat, 0, 255).astype(np.uint8)), 700)


def chart(summary, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, name in zip(axes, ("funsd", "cord")):
        rows = dict(PUBLISHED[name])
        if "tesseract" in summary.get(name, {}):
            version = summary[name]["tesseract"].get("version", "tesseract").split()[-1]
            rows[f"Tesseract {version} (my run)"] = 100 * summary[name]["tesseract"]["cleval"]["e2e"]["hmean"]
        if "crnn" in summary.get(name, {}):
            rows["This repo"] = 100 * summary[name]["crnn"]["cleval"]["e2e"]["hmean"]
        names = list(rows)
        values = [rows[n] for n in names]
        colors = ["#9aa5b1" if "paper" in n else ("#5b8def" if "This repo" in n else "#c9a227") for n in names]
        ax.barh(names, values, color=colors)
        for i, v in enumerate(values):
            ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)
        ax.set_xlim(0, 105)
        ax.set_title(f"{name.upper()} test, end-to-end CLEval F1")
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)


def build(config, engine, folder, results):
    folder.mkdir(parents=True, exist_ok=True)
    raw = path(config, "data") / "raw"
    cord = datasets.cord(raw, "test")
    pick = next((p for p in cord if p.name == "0016"), cord[0])
    receipt(engine, pick).save(folder / "receipt.png")
    form(engine, datasets.funsd(raw, "test")[0]).save(folder / "form.png")
    items = datasets.docunet(raw)
    bent = next((item for item in items if f"{item['doc']}_{item['variant']}" == "46_2"), items[0])
    docunet(engine, bent).save(folder / "dewarp.png")
    summary_path = Path(results) / "summary.json"
    if summary_path.exists():
        chart(json.loads(summary_path.read_text()), folder / "results.png")
