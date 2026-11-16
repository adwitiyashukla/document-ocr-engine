import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from docr import config as configuration
from docr import datasets, evaluate, export, synth, train
from docr.config import path


def _engine(cfg, args):
    from docr.pipeline import Engine

    runs = path(cfg, "runs")
    if getattr(args, "onnx", False):
        return Engine.from_onnx(cfg, runs / "onnx", quantized=not getattr(args, "fp32", False))
    return Engine.from_checkpoints(cfg, runs / "dbnet" / "best.pt", runs / "crnn" / "best.pt")


def cmd_fonts(cfg, args):
    folder = synth.download_fonts(path(cfg, "fonts"))
    bank = synth.FontBank(folder)
    print(f"{len(bank.entries)} fonts ready in {folder}")


def cmd_download(cfg, args):
    datasets.download(path(cfg, "data") / "raw")


def cmd_prepare(cfg, args):
    datasets.prepare(path(cfg, "data") / "raw")
    print("prepared FUNSD, CORD and DocUNet")


def cmd_cache(cfg, args):
    train.build_caches(cfg)


def cmd_synth(cfg, args):
    if args.workers:
        cfg["synth"]["workers"] = args.workers
    count = train.build_synthetic_pages(cfg, args.pages)
    print(f"{count} synthetic pages")


def cmd_train_reader(cfg, args):
    best = train.train_reader(cfg, steps=args.steps, resume=not args.fresh, hours=args.hours)
    print(f"best checkpoint: {best}" if best else "paused")


def cmd_train_detector(cfg, args):
    best = train.train_detector(cfg, steps=args.steps, resume=not args.fresh, hours=args.hours)
    print(f"best checkpoint: {best}" if best else "paused")


def cmd_export(cfg, args):
    runs = path(cfg, "runs")
    calibration = [r["image"] for r in train.load_records(path(cfg, "data") / "cache" / "pages_val")[::6]]
    folder = export.export_all(cfg, runs / "dbnet" / "best.pt", runs / "crnn" / "best.pt", runs / "onnx", calibration)
    for f in sorted(folder.glob("*.onnx")):
        print(f"{f.name}: {f.stat().st_size / 1e6:.2f} MB")


def cmd_evaluate(cfg, args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) if args.only else {"funsd", "cord", "pages", "docunet", "latency"}
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    started = time.time()

    def save():
        summary_path.write_text(json.dumps(summary, indent=2))

    engine = None if args.tesseract_only else _engine(cfg, args)
    label = args.label or ("crnn" if not args.onnx else "onnx_fp32" if args.fp32 else "onnx_int8")
    for name in ("funsd", "cord"):
        if name not in only:
            continue
        if engine is not None:
            result = evaluate.evaluate_split(cfg, engine, name, "test", print)
            summary.setdefault(name, {})[label] = result
            print(name, label, evaluate.short(result))
            save()
        if args.tesseract:
            result = evaluate.tesseract_split(cfg, name, "test")
            summary.setdefault(name, {})["tesseract"] = result
            print(name, "tesseract", evaluate.short(result))
            save()
    if "pages" in only:
        pages = datasets.cord(path(cfg, "data") / "raw", "test")
        summary["pages"] = evaluate.page_finding(pages, engine)
        print("pages", json.dumps(summary["pages"]))
        save()
    if "docunet" in only and engine is not None:
        summary.setdefault("docunet", {})[label] = evaluate.docunet(cfg, engine, out / "docunet" / label, print)
        print("docunet", json.dumps(summary["docunet"][label]))
        save()
    if "latency" in only and engine is not None:
        pages = datasets.cord(path(cfg, "data") / "raw", "test")[:10] + datasets.funsd(path(cfg, "data") / "raw", "test")[:10]
        summary.setdefault("latency", {})[label] = evaluate.latency(engine, pages)
        print("latency", json.dumps(summary["latency"][label]))
        save()
    print(f"done in {time.time() - started:.0f}s, wrote {summary_path}")


def cmd_read(cfg, args):
    from docr.pipeline import draw, reading_order

    engine = _engine(cfg, args)
    with Image.open(args.image) as handle:
        image = np.asarray(handle.convert("RGB"))
    out = engine(image, find_page=not args.no_page, flatten=not args.no_dewarp)
    text = reading_order(out["quads"], out["texts"])
    print(text)
    timing = ", ".join(f"{k} {v:.2f}s" for k, v in out["timing"].items())
    print(f"\n{len(out['texts'])} words ({timing})", file=sys.stderr)
    if args.out:
        draw(image, out["quads"], out["texts"]).save(args.out)
        if out.get("flat") is not None or out.get("page") is not None:
            flat = out["flat"] if out.get("flat") is not None else out["page"]
            Image.fromarray(np.clip(flat, 0, 255).astype(np.uint8)).save(Path(args.out).with_name(Path(args.out).stem + "_page.png"))


def cmd_figures(cfg, args):
    from docr import figures

    figures.build(cfg, _engine(cfg, args), Path(args.out), Path(args.results))


def main(argv=None):
    parser = argparse.ArgumentParser(prog="docr")
    parser.add_argument("--config", default="config.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("fonts")
    sub.add_parser("download")
    sub.add_parser("prepare")
    sub.add_parser("cache")
    p = sub.add_parser("synth")
    p.add_argument("--pages", type=int)
    p.add_argument("--workers", type=int)
    for name in ("train-reader", "train-detector"):
        p = sub.add_parser(name)
        p.add_argument("--steps", type=int)
        p.add_argument("--fresh", action="store_true")
        p.add_argument("--hours", type=float)
    sub.add_parser("export")
    p = sub.add_parser("evaluate")
    p.add_argument("--only")
    p.add_argument("--onnx", action="store_true")
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--tesseract", action="store_true")
    p.add_argument("--tesseract-only", action="store_true")
    p.add_argument("--label")
    p.add_argument("--out", default="results")
    p = sub.add_parser("read")
    p.add_argument("image")
    p.add_argument("--onnx", action="store_true")
    p.add_argument("--no-page", action="store_true")
    p.add_argument("--no-dewarp", action="store_true")
    p.add_argument("--out")
    p = sub.add_parser("figures")
    p.add_argument("--out", default="figures")
    p.add_argument("--results", default="results")
    args = parser.parse_args(argv)
    cfg = configuration.load(args.config)
    handlers = {
        "fonts": cmd_fonts, "download": cmd_download, "prepare": cmd_prepare, "cache": cmd_cache, "synth": cmd_synth,
        "train-reader": cmd_train_reader, "train-detector": cmd_train_detector, "export": cmd_export,
        "evaluate": cmd_evaluate, "read": cmd_read, "figures": cmd_figures,
    }
    handlers[args.command](cfg, args)
