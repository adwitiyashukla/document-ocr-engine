import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info

from docr import ctc, datasets, dbnet, readers, synth
from docr.config import path
from docr.geometry import apply_homography, crop_quad, quad_size, rotation, warp_perspective
from docr.imaging import gaussian_blur, resize, to_gray, to_uint8
from docr.metrics import TextMetrics, WordMetrics


def splits(config):
    raw = path(config, "data") / "raw"
    funsd_train = datasets.funsd(raw, "train")
    held = config["data"]["funsd_val_pages"]
    cut = len(funsd_train) - held
    return {
        "train": funsd_train[:cut] + datasets.cord(raw, "train"),
        "val": funsd_train[cut:] + datasets.cord(raw, "validation"),
    }


def _expand(quad, mx, my):
    quad = np.asarray(quad, np.float64)
    right = quad[1] - quad[0]
    down = quad[3] - quad[0]
    ux = right / max(np.linalg.norm(right), 1e-6)
    uy = down / max(np.linalg.norm(down), 1e-6)
    return np.array([quad[0] - mx * ux - my * uy, quad[1] + mx * ux - my * uy, quad[2] + mx * ux + my * uy, quad[3] - mx * ux + my * uy])


def _page_crops(job):
    page, height, margin_x, margin_y = job
    gray = to_gray(page.load()).astype(np.float32)
    fill = float(np.median(gray))
    out = []
    for quad, text, skip in zip(page.quads, page.texts, page.ignore):
        if skip:
            continue
        w, h = quad_size(quad)
        if h < 2 or w < 2:
            continue
        big = _expand(quad, margin_x * h, margin_y * h)
        crop = to_uint8(crop_quad(gray, big, height, max_width=1600, fill=fill))
        scale = crop.shape[1] / (w + 2 * margin_x * h)
        inner = (margin_x * h * scale, height * margin_y / (1 + 2 * margin_y), crop.shape[1] - margin_x * h * scale, height * (1 + margin_y) / (1 + 2 * margin_y))
        out.append((crop, inner, text))
    return out


def _map(function, jobs, workers):
    if workers > 1 and len(jobs) > 1:
        with Pool(workers) as pool:
            return pool.map(function, jobs, chunksize=4)
    return [function(job) for job in jobs]


def build_word_cache(pages, target, height=48, margin_x=0.3, margin_y=0.25, workers=1):
    items = [item for part in _map(_page_crops, [(p, height, margin_x, margin_y) for p in pages], workers) for item in part]
    shapes = np.array([crop.shape for crop, _, _ in items], np.int64).reshape(-1, 2)
    offsets = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])])
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.concatenate([crop.ravel() for crop, _, _ in items]) if items else np.zeros(0, np.uint8)
    np.save(target.with_suffix(".pixels.npy"), pixels)
    boxes = np.array([inner for _, inner, _ in items], np.float32).reshape(-1, 4)
    np.savez(target.with_suffix(".npz"), shapes=shapes, offsets=offsets, boxes=boxes, texts=np.array([t for _, _, t in items], dtype=object))
    return len(items)


class WordCache:
    def __init__(self, target):
        target = Path(target)
        meta = np.load(target.with_suffix(".npz"), allow_pickle=True)
        self.shapes = meta["shapes"]
        self.offsets = meta["offsets"]
        self.boxes = meta["boxes"]
        self.texts = list(meta["texts"])
        self.pixels = np.load(target.with_suffix(".pixels.npy"), mmap_mode="r")

    def __len__(self):
        return len(self.texts)

    def crop(self, i):
        h, w = self.shapes[i]
        return np.asarray(self.pixels[self.offsets[i]:self.offsets[i] + h * w]).reshape(h, w)

    def sample(self, i, rng, height=32, jitter=True):
        image = self.crop(i).astype(np.float32)
        x0, y0, x1, y1 = self.boxes[i]
        th = y1 - y0
        if jitter:
            x0 += rng.uniform(-0.25, 0.15) * th
            x1 += rng.uniform(-0.15, 0.25) * th
            y0 += rng.uniform(-0.15, 0.1) * th
            y1 += rng.uniform(-0.1, 0.15) * th
        x0, x1 = max(x0, 0), min(x1, image.shape[1] - 1)
        y0, y1 = max(y0, 0), min(y1, image.shape[0] - 1)
        quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        crop = crop_quad(image, quad, height, max_width=1024, fill=float(np.median(image)))
        if jitter:
            if rng.random() < 0.3:
                crop = gaussian_blur(crop, rng.uniform(0.3, 1.0))
            if rng.random() < 0.3:
                crop = crop + rng.normal(0, rng.uniform(2, 10), crop.shape)
            if rng.random() < 0.4:
                lo, hi = np.percentile(crop, [2, 98])
                span = max(hi - lo, 1.0)
                crop = (crop - lo) / span * rng.uniform(120, 255) + rng.uniform(0, 60)
        return to_uint8(crop)


class WordStream(IterableDataset):
    def __init__(self, config, cache, real_fraction, seed, pool=8):
        self.config = config
        self.cache_path = str(cache) if cache else None
        self.real_fraction = real_fraction
        self.seed = seed
        self.pool = pool

    def __iter__(self):
        info = get_worker_info()
        worker = info.id if info else 0
        rng = np.random.default_rng(self.seed * 1000 + worker + int(time.time()) % 100000)
        bank = synth.FontBank(path(self.config, "fonts"))
        source = synth.TextSource(int(rng.integers(1 << 30)))
        cache = WordCache(self.cache_path) if self.cache_path and self.real_fraction > 0 else None
        charset = readers.Charset()
        batch = self.config["train_reader"]["batch_size"]
        max_width = self.config["crnn"]["max_width"]
        while True:
            items = []
            for _ in range(batch * self.pool):
                if cache is not None and rng.random() < self.real_fraction:
                    i = int(rng.integers(len(cache)))
                    image, text = cache.sample(i, rng), cache.texts[i]
                else:
                    text = source.token(rng, ["form", "receipt", None][int(rng.integers(3))])
                    image = synth.render_word(text, bank, rng, max_width=max_width)
                text = charset.normalize(text)
                if not text:
                    continue
                image = readers.prepare_crop(image, max_width=max_width)
                ids = charset.encode(text)
                if len(ids) + sum(a == b for a, b in zip(ids, ids[1:])) > image.shape[1] // 4:
                    continue
                items.append((image, text))
            items.sort(key=lambda item: item[0].shape[1])
            chunks = [items[i:i + batch] for i in range(0, len(items), batch)]
            for k in rng.permutation(len(chunks)):
                chunk = chunks[k]
                if len(chunk) < batch // 2:
                    continue
                yield readers.to_tensor([c[0] for c in chunk]), [c[1] for c in chunk]


def full_speed(*_):
    import sys

    if sys.platform == "win32":
        import ctypes

        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetProcessInformation.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_ulong]
        kernel.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        process = kernel.GetCurrentProcess()
        throttling = (ctypes.c_ulong * 3)(1, 1, 0)
        kernel.SetProcessInformation(process, 4, ctypes.byref(throttling), ctypes.sizeof(throttling))
        priority = (ctypes.c_ulong * 1)(5)
        kernel.SetProcessInformation(process, 0, ctypes.byref(priority), ctypes.sizeof(priority))
        kernel.SetPriorityClass(process, 0x20)


def keep_awake():
    import sys

    if sys.platform == "win32":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        full_speed()


def schedule(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return base * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))


def _loader(dataset, workers):
    return DataLoader(dataset, batch_size=None, num_workers=workers, persistent_workers=workers > 0, prefetch_factor=2 if workers else None, worker_init_fn=full_speed)


def _log(handle, record):
    handle.write(json.dumps(record) + "\n")
    handle.flush()


def reader_validation(config):
    cache = path(config, "data") / "cache" / "words_val"
    return WordCache(cache) if cache.with_suffix(".npz").exists() else None


@torch.no_grad()
def evaluate_reader(model, cache, max_width=512, limit=None):
    model.eval()
    reader = readers.Reader(model, max_width=max_width)
    rng = np.random.default_rng(0)
    n = len(cache) if limit is None else min(limit, len(cache))
    crops = [cache.sample(i, rng, jitter=False) for i in range(n)]
    texts, _ = reader(crops)
    metric = TextMetrics()
    metric.update(texts, [cache.texts[i] for i in range(n)])
    model.train()
    return metric.compute()


def save(target, model, optimizer, step, best, kind, config):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict() if optimizer else None, "step": step, "best": best,
        "kind": kind, "config": {k: config[k] for k in ("crnn", "detector")},
    }
    torch.save(state, tmp)
    tmp.replace(target)


def train_reader(config, steps=None, resume=True, run=None, log=print, hours=None):
    keep_awake()
    kind = "crnn"
    options = config["train_reader"]
    torch.manual_seed(options["seed"])
    torch.set_num_threads(options["threads"])
    total = steps or options["steps"]
    charset = readers.Charset()
    model = readers.build(config["crnn"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    folder = Path(run) if run else path(config, "runs") / kind
    folder.mkdir(parents=True, exist_ok=True)
    last, best_path = folder / "last.pt", folder / "best.pt"
    step, best = 0, -1.0
    resumed = resume and last.exists()
    if resumed:
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, best = state["step"], state["best"]
        log(f"resumed {kind} from step {step}")
    val = reader_validation(config)
    train_cache = path(config, "data") / "cache" / "words_train"
    has_real = train_cache.with_suffix(".npz").exists()
    model.train()
    handle = open(folder / "log.jsonl", "a" if resumed else "w", encoding="utf-8")
    started, first = time.time(), step
    deadline = started + hours * 3600 if hours else None
    while step < total:
        synthetic_phase = step < options["synthetic_only_steps"] or not has_real
        fraction = 0.0 if synthetic_phase else options["real_fraction"]
        stream = WordStream(config, train_cache if has_real else None, fraction, options["seed"] + step)
        phase_end = options["synthetic_only_steps"] if synthetic_phase and has_real else total
        for x, texts in _loader(stream, options["workers"]):
            lr = schedule(step, total, options["warmup"], options["lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            logits = model(x)
            targets, lengths = readers.ctc_targets(charset, texts)
            input_lengths = torch.full((x.shape[0],), logits.shape[1], dtype=torch.long)
            loss = ctc.ctc_loss(logits.log_softmax(-1).transpose(0, 1), targets, input_lengths, lengths)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            step += 1
            if step % options["log_every"] == 0:
                rate = (step - first) / max(time.time() - started, 1e-6)
                _log(handle, {"step": step, "loss": loss.item(), "lr": lr, "real": fraction, "time": round(time.time() - started, 1)})
                log(f"{kind} step {step}/{total} loss {loss.item():.4f} lr {lr:.2e} real {fraction:.2f} {rate:.2f} it/s")
            if step % options["save_every"] == 0 and step % options["eval_every"] and step < total:
                save(last, model, optimizer, step, best, kind, config)
            if step % options["eval_every"] == 0 or step == total:
                record = {"step": step}
                if val is not None:
                    result = evaluate_reader(model, val, config["crnn"]["max_width"])
                    record.update({f"val_{k}": v for k, v in result.items()})
                    score = result["accuracy"]
                    log(f"{kind} val step {step} accuracy {result['accuracy']:.4f} cer {result['cer']:.4f}")
                else:
                    score = -loss.item()
                _log(handle, record)
                if score > best:
                    best = score
                    save(best_path, model, None, step, best, kind, config)
                save(last, model, optimizer, step, best, kind, config)
            if deadline and time.time() > deadline and step < total:
                save(last, model, optimizer, step, best, kind, config)
                log(f"{kind} stopped at step {step}, run again to resume")
                handle.close()
                return None
            if step >= total or step >= phase_end:
                break
    handle.close()
    if not best_path.exists():
        save(best_path, model, None, step, best, kind, config)
    return best_path


def _page_record(page, image, scale):
    return {
        "image": image,
        "quads": (page.quads * scale).tolist(),
        "ignore": page.ignore.tolist(),
        "dontcare": (page.dontcare * scale).tolist(),
        "texts": page.texts,
    }


def _cache_page(job):
    page, folder, max_side = job
    image = page.load()
    scale = min(1.0, max_side / max(image.shape[:2]))
    if scale < 1.0:
        image = to_uint8(resize(image, (int(round(image.shape[0] * scale)), int(round(image.shape[1] * scale)))))
    name = f"{Path(page.image_path).parent.parent.name}_{Path(page.image_path).parent.name}_{page.name}.png"
    Image.fromarray(image).save(Path(folder) / name)
    return _page_record(page, name, scale)


def build_page_cache(pages, folder, max_side, workers=1):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    records = _map(_cache_page, [(p, str(folder), max_side) for p in pages], workers)
    (folder / "pages.json").write_text(json.dumps(records), encoding="utf-8")
    return len(records)


def _synth_job(args):
    folder, fonts, start, count, seed = args
    bank = synth.FontBank(fonts)
    source = synth.TextSource(seed + start)
    rng = np.random.default_rng(seed + start)
    records = []
    for i in range(start, start + count):
        out = synth.render_page(bank, source, rng)
        name = f"synth_{i:06d}.png"
        Image.fromarray(out["image"]).save(Path(folder) / name)
        records.append({
            "image": name, "quads": out["quads"].tolist(), "ignore": out["ignore"].tolist(), "dontcare": [],
            "texts": out["texts"], "page": np.asarray(out["page"]).tolist(), "kind": out["kind"],
        })
    return records


def build_synthetic_pages(config, count=None, log=print):
    folder = path(config, "data") / "synth"
    folder.mkdir(parents=True, exist_ok=True)
    count = count or config["synth"]["pages"]
    index = folder / "pages.jsonl"
    done = set()
    if index.exists():
        for line in index.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["image"])
    chunk = 25
    jobs = []
    for start in range(0, count, chunk):
        size = min(chunk, count - start)
        if not all(f"synth_{i:06d}.png" in done for i in range(start, start + size)):
            jobs.append((str(folder), str(path(config, "fonts")), start, size, config["synth"]["seed"]))
    started = time.time()
    total = len(done)
    with Pool(config["synth"]["workers"]) as pool, open(index, "a", encoding="utf-8") as handle:
        for part in pool.imap_unordered(_synth_job, jobs):
            for record in part:
                if record["image"] not in done:
                    handle.write(json.dumps(record) + "\n")
                    done.add(record["image"])
            handle.flush()
            total = len(done)
            log(f"synthetic pages {min(total, count)}/{count} ({time.time() - started:.0f}s)")
    return total


def load_records(folder):
    folder = Path(folder)
    if (folder / "pages.jsonl").exists():
        lines = (folder / "pages.jsonl").read_text(encoding="utf-8").splitlines()
        records = sorted((json.loads(line) for line in lines if line.strip()), key=lambda r: r["image"])
    elif (folder / "pages.json").exists():
        records = json.loads((folder / "pages.json").read_text(encoding="utf-8"))
    else:
        return []
    for r in records:
        r["image"] = str(folder / r["image"])
    return records


def _slim(records):
    out = []
    for r in records:
        quads = np.array(r["quads"], np.float32).reshape(-1, 4, 2)
        dontcare = np.array(r.get("dontcare") or [], np.float32).reshape(-1, 4, 2)
        out.append({"image": r["image"], "quads": quads, "ignore": np.array(r["ignore"], bool), "dontcare": dontcare})
    return out


class PageCrops(Dataset):
    def __init__(self, config, records, length):
        self.config = config
        self.records = records
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        options = self.config["train_detector"]
        rng = np.random.default_rng((index * 7919 + int(time.time() * 1000)) % (1 << 32))
        record = self.records[int(rng.integers(len(self.records)))]
        with Image.open(record["image"]) as handle:
            image = np.asarray(handle.convert("RGB"), np.float32)
        quads = np.array(record["quads"], np.float64).reshape(-1, 4, 2)
        ignore = np.array(record["ignore"], bool)
        dontcare = np.array(record.get("dontcare", []), np.float64).reshape(-1, 4, 2)
        if len(dontcare):
            quads = np.concatenate([quads, dontcare])
            ignore = np.concatenate([ignore, np.ones(len(dontcare), bool)])
        crop = options["crop"]
        heights = np.array([quad_size(q)[1] for q in quads[~ignore]]) if (~ignore).any() else np.array([20.0])
        median = float(np.median(heights)) if len(heights) else 20.0
        target = rng.uniform(10, 40) if rng.random() < 0.8 else rng.uniform(6, 70)
        scale = float(np.clip(target / max(median, 1.0), options["min_scale"], options["max_scale"]))
        angle = rng.normal(0, 4) if rng.random() < 0.3 else 0.0
        h, w = image.shape[:2]
        if len(quads) and rng.random() < 0.85:
            center = quads[int(rng.integers(len(quads)))].mean(axis=0)
        else:
            center = np.array([rng.uniform(0, w), rng.uniform(0, h)])
        center = center + rng.normal(0, crop / 4 / scale, 2)
        matrix = np.array([[scale, 0, crop / 2 - scale * center[0]], [0, scale, crop / 2 - scale * center[1]], [0, 0, 1.0]])
        matrix = rotation(angle, (crop / 2, crop / 2)) @ matrix
        fill = float(rng.uniform(0, 255))
        patch = warp_perspective(image, matrix, (crop, crop), fill=fill)
        mapped = apply_homography(matrix, quads) if len(quads) else quads
        if rng.random() < 0.5:
            patch = patch * rng.uniform(0.7, 1.2) + rng.uniform(-30, 30)
        if rng.random() < 0.2:
            patch = patch + rng.normal(0, rng.uniform(2, 10), patch.shape)
        if rng.random() < 0.3:
            gray = patch @ np.array([0.299, 0.587, 0.114], np.float32)
            patch = np.repeat(gray[..., None], 3, axis=2)
        patch = np.clip(patch, 0, 255)
        keep = np.ones(len(mapped), bool)
        if len(mapped):
            inside = (mapped[..., 0] >= 0) & (mapped[..., 0] < crop) & (mapped[..., 1] >= 0) & (mapped[..., 1] < crop)
            lo, hi = mapped.min(axis=1), mapped.max(axis=1)
            keep = (hi[:, 0] >= 0) & (lo[:, 0] < crop) & (hi[:, 1] >= 0) & (lo[:, 1] < crop)
            ignore = ignore | (keep & ~inside.all(axis=1))
        d = self.config["detector"]
        gt, mask, thresh, thresh_mask = dbnet.targets((crop, crop), mapped[keep], ignore[keep], d["shrink"], d["thresh_min"], d["thresh_max"])
        x = torch.from_numpy(dbnet.normalize(patch))
        return x, torch.from_numpy(gt), torch.from_numpy(mask), torch.from_numpy(thresh), torch.from_numpy(thresh_mask)


def detector_from(config, state=None):
    d = config["detector"]
    model = dbnet.DBNet(tuple(d["channels"]), tuple(d["blocks"]), d["fpn"], d["k"])
    if state is not None:
        model.load_state_dict(state)
    return model


def detection_validation(config):
    folder = path(config, "data") / "cache" / "pages_val"
    return load_records(folder)


@torch.no_grad()
def evaluate_detector(model, config, records):
    model.eval()
    detect = config["detect"]
    detector = dbnet.Detector(model, detect["max_side"], detect["threshold"], detect["box_threshold"], detect["unclip"], detect["min_size"])
    metric = WordMetrics()
    for r in records:
        with Image.open(r["image"]) as handle:
            image = np.asarray(handle.convert("RGB"))
        quads, _ = detector(image)
        metric.update(r["quads"], r["texts"], r["ignore"], quads, dontcare=r.get("dontcare") or [])
    model.train()
    return metric.compute()["det"]


def train_detector(config, steps=None, resume=True, run=None, log=print, hours=None):
    keep_awake()
    options = config["train_detector"]
    torch.manual_seed(options["seed"])
    torch.set_num_threads(options["threads"])
    total = steps or options["steps"]
    model = detector_from(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["lr"], weight_decay=options["weight_decay"])
    folder = Path(run) if run else path(config, "runs") / "dbnet"
    folder.mkdir(parents=True, exist_ok=True)
    last, best_path = folder / "last.pt", folder / "best.pt"
    step, best = 0, -1.0
    resumed = resume and last.exists()
    if resumed:
        state = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step, best = state["step"], state["best"]
        log(f"resumed dbnet from step {step}")
    synthetic = _slim(load_records(path(config, "data") / "synth"))
    real = _slim(load_records(path(config, "data") / "cache" / "pages_train"))
    val = detection_validation(config)
    if not synthetic and not real:
        raise FileNotFoundError("no training pages, run: python -m docr synth and python -m docr cache")
    handle = open(folder / "log.jsonl", "a" if resumed else "w", encoding="utf-8")
    model.train()
    started, first = time.time(), step
    deadline = started + hours * 3600 if hours else None
    while step < total:
        synthetic_phase = (step < options["synthetic_only_steps"] and synthetic) or not real
        if synthetic_phase:
            records = synthetic
            phase_end = options["synthetic_only_steps"] if real else total
        else:
            weight = options["real_fraction"]
            copies = max(1, int(round(weight * len(synthetic) / max((1 - weight) * len(real), 1)))) if synthetic and weight < 1 else 1
            records = (synthetic if weight < 1 else []) + real * copies
            phase_end = total
        dataset = PageCrops(config, records, options["batch_size"] * (phase_end - step))
        loader = DataLoader(dataset, batch_size=options["batch_size"], num_workers=options["workers"], persistent_workers=False, prefetch_factor=2 if options["workers"] else None)
        for x, gt, mask, thresh, thresh_mask in loader:
            lr = schedule(step, total, options["warmup"], options["lr"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            outputs = model(x)
            loss, parts = dbnet.loss(outputs, gt, mask, thresh, thresh_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            step += 1
            if step % options["log_every"] == 0:
                rate = (step - first) / max(time.time() - started, 1e-6)
                _log(handle, {"step": step, "loss": loss.item(), **parts, "lr": lr, "time": round(time.time() - started, 1)})
                log(f"dbnet step {step}/{total} loss {loss.item():.4f} prob {parts['prob']:.3f} bin {parts['binary']:.3f} thr {parts['thresh']:.3f} lr {lr:.2e} {rate:.2f} it/s")
            if step % options["save_every"] == 0 and step % options["eval_every"] and step < total:
                save(last, model, optimizer, step, best, "dbnet", config)
            if step % options["eval_every"] == 0 or step == total:
                record = {"step": step}
                if val:
                    result = evaluate_detector(model, config, val)
                    record.update({f"val_{k}": v for k, v in result.items()})
                    score = result["f1"]
                    log(f"dbnet val step {step} precision {result['precision']:.4f} recall {result['recall']:.4f} f1 {result['f1']:.4f}")
                else:
                    score = -loss.item()
                _log(handle, record)
                if score > best:
                    best = score
                    save(best_path, model, None, step, best, "dbnet", config)
                save(last, model, optimizer, step, best, "dbnet", config)
            if deadline and time.time() > deadline and step < total:
                save(last, model, optimizer, step, best, "dbnet", config)
                log(f"dbnet stopped at step {step}, run again to resume")
                handle.close()
                return None
            if step >= total or step >= phase_end:
                break
    handle.close()
    if not best_path.exists():
        save(best_path, model, None, step, best, "dbnet", config)
    return best_path


def build_caches(config, log=print):
    data = path(config, "data")
    parts = splits(config)
    height = config["data"]["word_cache_height"]
    workers = config["data"]["workers"]
    for name, pages in parts.items():
        n = build_word_cache(pages, data / "cache" / f"words_{name}", height, workers=workers)
        log(f"word cache {name}: {n} crops")
        n = build_page_cache(pages, data / "cache" / f"pages_{name}", config["data"]["cache_max_side"], workers)
        log(f"page cache {name}: {n} pages")
