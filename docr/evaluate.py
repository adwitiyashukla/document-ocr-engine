import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from docr import datasets, scan, tesseract
from docr.cleval import CLEval
from docr.config import path
from docr.geometry import crop_quad
from docr.imaging import to_gray, to_uint8
from docr.metrics import TextMetrics, WordMetrics, docunet_msssim, levenshtein
from docr.pipeline import text_aware_page


def ground_truth(page):
    quads = list(page.quads) + list(page.dontcare)
    texts = list(page.texts) + ["###"] * len(page.dontcare)
    dc = list(page.ignore) + [True] * len(page.dontcare)
    return quads, texts, dc


def score(pages, predictions):
    strict, caseless, words = CLEval(), CLEval(case_sensitive=False), WordMetrics()
    for page in pages:
        quads, texts = predictions[page.name]
        gq, gt, dc = ground_truth(page)
        strict.update(gq, gt, dc, list(quads), list(texts))
        caseless.update(gq, gt, dc, list(quads), list(texts))
        words.update(page.quads, page.texts, page.ignore, quads, texts, dontcare=page.dontcare)
    return {"cleval": strict.compute(), "cleval_caseless": caseless.compute(), "word": words.compute()}


def run_engine(engine, pages, log=print):
    predictions, timings = {}, []
    for i, page in enumerate(pages):
        out = engine(page.load(), find_page=False, flatten=False)
        predictions[page.name] = (out["quads"], out["texts"])
        timings.append(out["timing"])
        if (i + 1) % 10 == 0:
            log(f"  {i + 1}/{len(pages)} pages")
    return predictions, timings


def _tesseract_page(page):
    quads, texts, _ = tesseract.words(page.load())
    return page.name, quads, texts


def run_tesseract(pages, workers=2):
    with ProcessPoolExecutor(workers) as pool:
        return {name: (quads, texts) for name, quads, texts in pool.map(_tesseract_page, pages)}


def recognition(engine, pages):
    metric = TextMetrics()
    for page in pages:
        keep = ~page.ignore
        if not keep.any():
            continue
        texts, _ = engine.read(page.load(), page.quads[keep])
        metric.update(texts, [t for t, k in zip(page.texts, keep) if k])
    return metric.compute()


def _tesseract_crops(page):
    gray = to_gray(page.load()).astype(np.float32)
    out = []
    for quad, text, skip in zip(page.quads, page.texts, page.ignore):
        if skip:
            continue
        crop = crop_quad(gray, quad, 48, max_width=2000, fill=float(np.median(gray)))
        padded = np.pad(to_uint8(crop), 12, constant_values=int(np.median(crop)))
        out.append((tesseract.read_word(padded), text))
    return out


def tesseract_recognition(pages, workers=2):
    metric = TextMetrics()
    with ProcessPoolExecutor(workers) as pool:
        for pairs in pool.map(_tesseract_crops, pages):
            metric.update([p for p, _ in pairs], [t for _, t in pairs])
    return metric.compute()


def page_finding(pages, engine=None):
    plain, aware = [], []
    for page in pages:
        if page.roi is None:
            continue
        image = page.load()
        h, w = image.shape[:2]
        quad, _ = scan.find_page(image)
        plain.append(scan.page_iou(quad, page.roi, w, h))
        if engine is not None:
            quads, _ = engine.detect(image)
            aware.append(scan.page_iou(text_aware_page(image, quads), page.roi, w, h))

    def summary(values):
        values = np.array(values)
        return {"mean_iou": float(values.mean()), "median_iou": float(np.median(values)), "iou_90": float((values >= 0.9).mean()), "pages": len(values)} if len(values) else {}

    full = []
    for page in pages:
        if page.roi is not None:
            with Image.open(page.image_path) as handle:
                w, h = handle.size
            full.append(scan.page_iou(np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], float), page.roi, w, h))
    return {"full_frame": summary(full), "classical": summary(plain), "text_aware": summary(aware)}


def _cer(predicted, reference):
    return levenshtein(predicted, reference) / max(len(reference), 1)


def docunet(config, engine, folder, log=print):
    raw = path(config, "data") / "raw"
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    items = datasets.docunet(raw)
    rows = []
    scan_text = {}
    for i, item in enumerate(items):
        with Image.open(item["photo"]) as handle:
            photo = np.asarray(handle.convert("RGB"))
        with Image.open(item["scan"]) as handle:
            reference = np.asarray(handle.convert("RGB"))
        out = engine(photo, find_page=True, flatten=True)
        result = out["flat"] if out.get("flat") is not None else out["page"]
        result = to_uint8(result)
        Image.fromarray(result).save(folder / f"{item['doc']}_{item['variant']}.png")
        if item["doc"] not in scan_text:
            scan_text[item["doc"]] = tesseract.text(reference)
        row = {"doc": item["doc"], "variant": item["variant"], "lines": out.get("lines", 0)}
        row["msssim_photo"] = docunet_msssim(photo, reference)
        row["msssim_ours"] = docunet_msssim(result, reference)
        ref_text = scan_text[item["doc"]]
        photo_text = tesseract.text(photo)
        ours_text = tesseract.text(result)
        row["ed_photo"] = levenshtein(photo_text, ref_text)
        row["ed_ours"] = levenshtein(ours_text, ref_text)
        row["cer_photo"] = _cer(photo_text, ref_text)
        row["cer_ours"] = _cer(ours_text, ref_text)
        rows.append(row)
        log(f"  docunet {i + 1}/{len(items)} doc {item['doc']}_{item['variant']} cer {row['cer_photo']:.3f} -> {row['cer_ours']:.3f}")
    (folder / "rows.json").write_text(json.dumps(rows), encoding="utf-8")
    return summarize_docunet(rows)


def summarize_docunet(rows):
    out = {"msssim_photo": float(np.mean([r["msssim_photo"] for r in rows])), "msssim_ours": float(np.mean([r["msssim_ours"] for r in rows]))}
    for name, docs in (("dewarpnet", datasets.DOCUNET_DEWARPNET), ("doctr", datasets.DOCUNET_DOCTR)):
        subset = [r for r in rows if r["doc"] in docs]
        if not subset:
            continue
        for key in ("photo", "ours"):
            out[f"{name}_cer_{key}"] = float(np.mean([r[f"cer_{key}"] for r in subset]))
            out[f"{name}_ed_{key}"] = float(np.mean([r[f"ed_{key}"] for r in subset]))
    return out


def latency(engine, pages, repeats=1):
    totals = {"detect": [], "read": [], "total": []}
    for page in pages:
        image = page.load()
        for _ in range(repeats):
            t = time.perf_counter()
            out = engine(image, find_page=False, flatten=False)
            totals["total"].append(time.perf_counter() - t)
            totals["detect"].append(out["timing"]["detect"])
            totals["read"].append(out["timing"]["read"])
    return {k: float(np.mean(v)) for k, v in totals.items()}


def evaluate_split(config, engine, name, split, log=print):
    raw = path(config, "data") / "raw"
    pages = datasets.load_split(raw, name, split)
    log(f"{name} {split}: {len(pages)} pages")
    predictions, timings = run_engine(engine, pages, log)
    result = score(pages, predictions)
    result["recognition"] = recognition(engine, pages)
    result["seconds_per_page"] = float(np.mean([t["detect"] + t["read"] for t in timings]))
    return result


def tesseract_split(config, name, split, crops=True):
    raw = path(config, "data") / "raw"
    pages = datasets.load_split(raw, name, split)
    workers = config["eval"]["workers"]
    result = score(pages, run_tesseract(pages, workers))
    if crops:
        result["recognition"] = tesseract_recognition(pages, workers)
    result["version"] = tesseract.version()
    return result


def short(result):
    c, w = result["cleval"], result["word"]
    return {
        "cleval_det_f1": round(100 * c["det"]["hmean"], 2),
        "cleval_e2e_f1": round(100 * c["e2e"]["hmean"], 2),
        "word_det_f1": round(100 * w["det"]["f1"], 2),
        "word_e2e_f1": round(100 * w["e2e"]["f1"], 2),
        "word_accuracy": round(100 * result["recognition"]["accuracy"], 2) if "recognition" in result else None,
        "cer": round(100 * result["recognition"]["cer"], 2) if "recognition" in result else None,
    }
