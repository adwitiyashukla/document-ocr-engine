import numpy as np

from docr.geometry import area, clip_convex, convex_hull, intersection_area
from docr.imaging import correlate1d, gaussian_kernel, to_gray


def levenshtein(a, b):
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    target = np.array([ord(c) for c in b])
    offsets = np.arange(len(b) + 1)
    previous = offsets.copy()
    for ch in a:
        step = np.empty_like(previous)
        step[0] = previous[0] + 1
        step[1:] = np.minimum(previous[:-1] + (target != ord(ch)), previous[1:] + 1)
        previous = np.minimum.accumulate(step - offsets) + offsets
    return int(previous[-1])


def hungarian(cost):
    cost = np.asarray(cost, np.float64)
    transposed = cost.shape[0] > cost.shape[1]
    if transposed:
        cost = cost.T
    n, m = cost.shape
    if n == 0:
        return np.zeros(0, int), np.zeros(0, int)
    u = np.zeros(n + 1)
    v = np.zeros(m + 1)
    owner = np.zeros(m + 1, int)
    way = np.zeros(m + 1, int)
    for i in range(1, n + 1):
        owner[0] = i
        j0 = 0
        best = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, bool)
        while True:
            used[j0] = True
            i0 = owner[j0]
            free = ~used[1:]
            reduced = cost[i0 - 1] - u[i0] - v[1:]
            better = free & (reduced < best[1:])
            best[1:][better] = reduced[better]
            way[1:][better] = j0
            candidates = np.where(free, best[1:], np.inf)
            j1 = int(np.argmin(candidates)) + 1
            delta = candidates[j1 - 1]
            u[owner[used]] += delta
            v[used] -= delta
            best[1:][free] -= delta
            j0 = j1
            if owner[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            owner[j0] = owner[j1]
            j0 = j1
    rows = owner[1:] - 1
    cols = np.arange(m)
    keep = rows >= 0
    rows, cols = rows[keep], cols[keep]
    order = np.argsort(rows)
    rows, cols = rows[order], cols[order]
    return (cols, rows) if transposed else (rows, cols)


def _bounds(quads):
    quads = np.asarray(quads, np.float64).reshape(-1, 4, 2)
    return np.concatenate([quads.min(axis=1), quads.max(axis=1)], axis=1)


def iou_matrix(a_quads, b_quads):
    a_quads = np.asarray(a_quads, np.float64).reshape(-1, 4, 2)
    b_quads = np.asarray(b_quads, np.float64).reshape(-1, 4, 2)
    out = np.zeros((len(a_quads), len(b_quads)))
    if not len(a_quads) or not len(b_quads):
        return out
    a_box, b_box = _bounds(a_quads), _bounds(b_quads)
    near = (
        (a_box[:, None, 0] < b_box[None, :, 2]) & (b_box[None, :, 0] < a_box[:, None, 2])
        & (a_box[:, None, 1] < b_box[None, :, 3]) & (b_box[None, :, 1] < a_box[:, None, 3])
    )
    a_hulls = [convex_hull(q) for q in a_quads]
    b_hulls = [convex_hull(q) for q in b_quads]
    a_area = [area(h) if len(h) >= 3 else 0.0 for h in a_hulls]
    b_area = [area(h) if len(h) >= 3 else 0.0 for h in b_hulls]
    for i, j in zip(*np.nonzero(near)):
        if a_area[i] <= 0 or b_area[j] <= 0:
            continue
        inter = intersection_area(a_hulls[i], b_hulls[j])
        union = a_area[i] + b_area[j] - inter
        out[i, j] = inter / union if union > 0 else 0.0
    return out


def coverage(det_quads, region_quads):
    det_quads = np.asarray(det_quads, np.float64).reshape(-1, 4, 2)
    out = np.zeros(len(det_quads))
    if not len(region_quads):
        return out
    for i, q in enumerate(det_quads):
        hull = convex_hull(q)
        total = area(hull) if len(hull) >= 3 else 0.0
        if total <= 0:
            continue
        covered = 0.0
        for r in region_quads:
            clipped = clip_convex(hull, convex_hull(r))
            covered += area(clipped) if len(clipped) >= 3 else 0.0
        out[i] = min(covered / total, 1.0)
    return out


def match(gt_quads, det_quads, threshold=0.5):
    iou = iou_matrix(gt_quads, det_quads)
    if iou.size == 0:
        return np.zeros(0, int), np.zeros(0, int), iou
    rows, cols = hungarian(-np.where(iou >= threshold, iou, 0.0))
    keep = iou[rows, cols] >= threshold
    return rows[keep], cols[keep], iou


class WordMetrics:
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.counts = {"gt": 0, "det": 0, "matched": 0, "exact": 0, "caseless": 0}

    def update(self, gt_quads, gt_texts, gt_ignore, det_quads, det_texts=None, dontcare=()):
        gt_quads = np.asarray(gt_quads, np.float64).reshape(-1, 4, 2)
        det_quads = np.asarray(det_quads, np.float64).reshape(-1, 4, 2)
        gt_ignore = np.asarray(gt_ignore, bool)
        valid = ~gt_ignore
        rows, cols, _ = match(gt_quads[valid], det_quads, self.threshold)
        gt_texts = [t for t, v in zip(gt_texts, valid) if v]
        regions = list(gt_quads[gt_ignore]) + list(np.asarray(dontcare, np.float64).reshape(-1, 4, 2))
        unmatched = np.ones(len(det_quads), bool)
        unmatched[cols] = False
        dropped = unmatched & (coverage(det_quads, regions) > 0.5) if regions else np.zeros(len(det_quads), bool)
        self.counts["gt"] += int(valid.sum())
        self.counts["det"] += int(len(det_quads) - dropped.sum())
        self.counts["matched"] += len(rows)
        if det_texts is not None:
            for r, c in zip(rows, cols):
                self.counts["exact"] += int(gt_texts[r] == det_texts[c])
                self.counts["caseless"] += int(gt_texts[r].lower() == det_texts[c].lower())

    def compute(self):
        def prf(hits):
            p = hits / self.counts["det"] if self.counts["det"] else 0.0
            r = hits / self.counts["gt"] if self.counts["gt"] else 0.0
            return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}

        return {"det": prf(self.counts["matched"]), "e2e": prf(self.counts["exact"]), "e2e_caseless": prf(self.counts["caseless"])}


class TextMetrics:
    def __init__(self):
        self.words = 0
        self.exact = 0
        self.caseless = 0
        self.edits = 0
        self.chars = 0

    def update(self, predictions, targets):
        for p, t in zip(predictions, targets):
            self.words += 1
            self.exact += int(p == t)
            self.caseless += int(p.lower() == t.lower())
            self.edits += levenshtein(p, t)
            self.chars += len(t)

    def compute(self):
        n = max(self.words, 1)
        return {"accuracy": self.exact / n, "accuracy_caseless": self.caseless / n, "cer": self.edits / max(self.chars, 1), "words": self.words}


def _cubic(x):
    x = np.abs(x)
    return np.where(x <= 1, 1.5 * x**3 - 2.5 * x**2 + 1, np.where(x < 2, -0.5 * x**3 + 2.5 * x**2 - 4 * x + 2, 0.0))


def _resize_axis(image, size, axis):
    n = image.shape[axis]
    scale = size / n
    width = 4.0 / scale if scale < 1 else 4.0
    out_pos = np.arange(1, size + 1)
    u = out_pos / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - width / 2)
    taps = int(np.ceil(width)) + 2
    index = left[:, None] + np.arange(taps)[None, :]
    if scale < 1:
        weights = scale * _cubic(scale * (u[:, None] - index))
    else:
        weights = _cubic(u[:, None] - index)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    index = index.astype(int) - 1
    mirror = np.concatenate([np.arange(n), np.arange(n - 1, -1, -1)])
    index = mirror[np.mod(index, 2 * n)]
    moved = np.moveaxis(image, axis, 0)
    out = np.einsum("ij,ij...->i...", weights, moved[index])
    return np.moveaxis(out, 0, axis)


def matlab_resize(image, shape):
    out = _resize_axis(np.asarray(image, np.float64), shape[0], 0)
    out = _resize_axis(out, shape[1], 1)
    return np.clip(np.rint(out), 0, 255)


def ssim(x, y, data_range=255.0, sigma=1.5):
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    kernel = gaussian_kernel(sigma, 2 * int(np.ceil(3 * sigma)) + 1).astype(np.float64)

    def blur(a):
        return correlate1d(correlate1d(a, kernel, 0, "edge"), kernel, 1, "edge").astype(np.float64)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mx, my = blur(x), blur(y)
    sxx = blur(x * x) - mx * mx
    syy = blur(y * y) - my * my
    sxy = blur(x * y) - mx * my
    return ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx * mx + my * my + c1) * (sxx + syy + c2))


def pyramid_reduce(image):
    kernel = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])
    out = correlate1d(correlate1d(image, kernel, 0, "symmetric"), kernel, 1, "symmetric").astype(np.float64)
    return np.clip(np.rint(out), 0, 255)[::2, ::2]


def docunet_msssim(rectified, scan, target_area=598400):
    ref = np.rint(to_gray(np.asarray(scan, np.float64)) if np.asarray(scan).ndim == 3 else np.asarray(scan, np.float64))
    image = np.rint(to_gray(np.asarray(rectified, np.float64)) if np.asarray(rectified).ndim == 3 else np.asarray(rectified, np.float64))
    factor = np.sqrt(target_area / (ref.shape[0] * ref.shape[1]))
    ref = matlab_resize(ref, (int(np.ceil(ref.shape[0] * factor)), int(np.ceil(ref.shape[1] * factor))))
    image = matlab_resize(image, ref.shape)
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
    total = 0.0
    for w in weights:
        total += w * float(ssim(image, ref).mean())
        image = pyramid_reduce(image)
        ref = pyramid_reduce(ref)
    return total
