import numpy as np

from docr.geometry import area, clip_convex, convex_hull


def _side(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def aspect_ratio(quad):
    top = _side(quad[0], quad[1])
    right = _side(quad[1], quad[2])
    bottom = _side(quad[2], quad[3])
    left = _side(quad[3], quad[0])
    return ((top + bottom) / 2 + 1e-5) / ((right + left) / 2 + 1e-5)


def pseudo_length(quad):
    r = aspect_ratio(quad)
    return min(round(0.5 + max(r, 1 / r)), 10)


def char_centers(quad, length, vertical_threshold):
    if length == 0:
        return np.zeros((0, 2))
    q = np.asarray(quad, np.float64)
    if aspect_ratio(q) >= vertical_threshold:
        lt, rt, rb, lb = q[0], q[1], q[2], q[3]
    else:
        lt, rt, rb, lb = q[3], q[0], q[1], q[2]
    p1 = (lt + lb) / 2
    p2 = (rt + rb) / 2
    unit = (p2 - p1) / length
    return p1 + unit / 2 + unit * np.arange(length)[:, None]


def inside(points, poly):
    poly = np.asarray(poly, np.float64)
    points = np.asarray(points, np.float64).reshape(-1, 2)
    x, y = points[:, 0], points[:, 1]
    result = np.zeros(len(points), bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        straddle = ((yi <= y) & (y < yj)) | ((yj <= y) & (y < yi))
        with np.errstate(divide="ignore", invalid="ignore"):
            cross_x = (xj - xi) * (y - yi) / (yj - yi) + xi
        result ^= straddle & (x < cross_x)
        j = i
    return result


def _truncate(quad):
    return np.trunc(np.asarray(quad, np.float64))


def _hull(quad):
    hull = convex_hull(quad)
    return hull if len(hull) >= 3 else None


def _overlap(a, b):
    if a is None or b is None:
        return 0.0
    clipped = clip_convex(a, b)
    return area(clipped) if len(clipped) >= 3 else 0.0


def _lcs(s1, s2):
    n, m = len(s1), len(s2)
    if n == 0 or m == 0:
        return ""
    length = [[0] * m for _ in range(n)]
    move = [[0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            if s1[i] == s2[j]:
                length[i][j] = 1 if i == 0 or j == 0 else length[i - 1][j - 1] + 1
                move[i][j] = 0
            else:
                up = length[i - 1][j] if i > 0 else 0
                left = length[i][j - 1] if j > 0 else 0
                if up > left:
                    length[i][j] = up
                    move[i][j] = 1
                else:
                    length[i][j] = left
                    move[i][j] = 2
    out = []
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        if move[i][j] == 0:
            out.append(s1[i])
            if i == 0 or j == 0:
                break
            i, j = i - 1, j - 1
        elif move[i][j] == 1:
            if i == 0:
                break
            i -= 1
        else:
            if j == 0:
                break
            j -= 1
    return "".join(reversed(out))


def evaluate(gt_quads, gt_texts, gt_dc, det_quads, det_texts=None, case_sensitive=True, ap_constraint=0.3, recall_penalty=1.0, precision_penalty=1.0, vertical_threshold=0.5):
    gt_quads = [np.asarray(q, np.float64).reshape(4, 2) for q in gt_quads]
    det_quads = [np.asarray(q, np.float64).reshape(4, 2) for q in det_quads]
    n_gt, n_det = len(gt_quads), len(det_quads)
    e2e = det_texts is not None
    gt_dc = [bool(d) for d in gt_dc]
    gt_texts = [("#" * pseudo_length(q) if dc else t) for q, t, dc in zip(gt_quads, gt_texts, gt_dc)]
    det_texts = list(det_texts) if e2e else [""] * n_det
    if not case_sensitive:
        gt_texts = [t.upper() for t in gt_texts]
        det_texts = [t.upper() for t in det_texts]
    gt_polys = [_truncate(q) for q in gt_quads]
    det_polys = [_truncate(q) for q in det_quads]
    gt_hulls = [_hull(p) for p in gt_polys]
    det_hulls = [_hull(p) for p in det_polys]
    dc_set = {i for i in range(n_gt) if gt_dc[i]}
    holes = {i: [k for k in range(n_gt) if k not in dc_set and _overlap(gt_hulls[k], gt_hulls[i]) > 0] for i in dc_set}
    centers = [char_centers(gt_quads[i], len(gt_texts[i]), vertical_threshold) for i in range(n_gt)]
    det_area = np.array([area(p) if p is not None else 0.0 for p in det_hulls])
    ap = np.zeros((n_gt, n_det))
    pcc = [np.zeros((n_det, len(centers[i])), bool) for i in range(n_gt)]
    gt_box = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in gt_polys]).reshape(-1, 4)
    det_box = np.array([[p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()] for p in det_polys]).reshape(-1, 4)
    near = (
        (gt_box[:, None, 0] <= det_box[None, :, 2]) & (det_box[None, :, 0] <= gt_box[:, None, 2])
        & (gt_box[:, None, 1] <= det_box[None, :, 3]) & (det_box[None, :, 1] <= gt_box[:, None, 3])
    )
    for g, d in zip(*np.nonzero(near)):
        if det_area[d] <= 0:
            continue
        inter = _overlap(gt_hulls[g], det_hulls[d])
        if g in dc_set and inter > 0:
            for k in holes[g]:
                inner = clip_convex(det_hulls[d], gt_hulls[g]) if det_hulls[d] is not None else []
                if len(inner) >= 3:
                    inter -= _overlap(inner, gt_hulls[k])
        ap[g, d] = max(inter, 0.0) / det_area[d]
        if len(centers[g]):
            pcc[g][d] = inside(centers[g], det_polys[d])
    pcc_sum = np.array([[pcc[g][d].sum() for d in range(n_det)] for g in range(n_gt)], np.int64).reshape(n_gt, n_det)
    ap_bin = ap >= ap_constraint
    det_dc = set()
    if dc_set:
        for d in range(n_det):
            total = 0.0
            for g in sorted(dc_set):
                if ap_bin[g, d]:
                    det_dc.add(d)
                    break
                if pcc_sum[g, d] > 0:
                    total += ap[g, d]
            if total >= ap_constraint:
                det_dc.add(d)
    gt_valid = [g for g in range(n_gt) if g not in dc_set]
    det_valid = [d for d in range(n_det) if d not in det_dc]
    linked = ap_bin & (pcc_sum > 0)
    match = np.zeros((n_gt, n_det), bool)
    row_count = linked.sum(axis=1)
    col_count = linked.sum(axis=0)
    for g in gt_valid:
        for d in det_valid:
            if row_count[g] == 1 and col_count[d] == 1 and linked[g, d]:
                match[g, d] = True
    for g in gt_valid:
        dets = [d for d in det_valid if linked[g, d]]
        if len(dets) >= 2 and sum(pcc_sum[g, d] for d in dets) > 0:
            match[g, dets] = True
    for d in det_valid:
        gts = [g for g in gt_valid if pcc_sum[g, d] > 0]
        if len(gts) >= 2 and sum(ap[g, d] for g in gts) >= ap_constraint:
            match[gts, d] = True
    for g in range(n_gt):
        for d in range(n_det):
            if not match[g, d]:
                pcc[g][d][:] = False
    pcc_sum = np.array([[pcc[g][d].sum() for d in range(n_det)] for g in range(n_gt)], np.int64).reshape(n_gt, n_det)
    gts_per_det = match.sum(axis=0)
    dets_per_gt = match.sum(axis=1)
    checked = [np.zeros(len(c), bool) for c in centers]
    tp = 0
    gran_precision = 0.0
    for d in det_valid:
        if gts_per_det[d] > 0:
            matched = np.nonzero(match[:, d])[0]
            for g in matched:
                for k in np.nonzero(pcc[g][d])[0]:
                    if not checked[g][k]:
                        checked[g][k] = True
                        tp += 1
            gran_precision += max(len(matched) - 1, 0) * precision_penalty
    gran_recall = 0.0
    for g in gt_valid:
        if dets_per_gt[g] > 0:
            gran_recall += max(int(dets_per_gt[g]) - 1, 0) * recall_penalty
    char_gt = sum(len(gt_texts[g]) for g in gt_valid)
    char_det = int(pcc_sum[:, det_valid].sum()) if n_det and n_gt else 0
    fp = 0
    for d in det_valid:
        if gts_per_det[d] == 0:
            fp += min(round(0.5 + 1 / (1e-5 + aspect_ratio(det_quads[d]))), 10)
    char_det += fp
    result = {"det": {"gt": char_gt, "det": char_det, "tp": tp, "gran_recall": gran_recall, "gran_precision": gran_precision}}
    if e2e:
        gt_left = list(gt_texts)
        det_left = list(det_texts)
        tp_e2e = 0
        gran_recall_e2e = 0.0
        for g in gt_valid:
            if dets_per_gt[g] == 0:
                continue
            unordered = list(np.nonzero(match[g])[0])
            ordered = []
            for k in range(len(centers[g])):
                if len(unordered) == 1:
                    break
                for d in unordered:
                    if pcc[g][d][k]:
                        ordered.append(d)
                        unordered.remove(d)
                        break
            ordered.append(unordered[0])
            common = _lcs(gt_texts[g], "".join(det_left[d] for d in ordered))
            for ch in common:
                gt_left[g] = gt_left[g].replace(ch, "", 1)
                for d in ordered:
                    if ch in det_left[d]:
                        det_left[d] = det_left[d].replace(ch, "", 1)
                        break
            gran_recall_e2e += max(len(np.nonzero(match[g])[0]) - 1, 0) * recall_penalty
        gran_precision_e2e = 0.0
        for d in det_valid:
            if gts_per_det[d] > 0:
                gran_precision_e2e += max(int(gts_per_det[d]) - 1, 0) * precision_penalty
            tp_e2e += len(det_texts[d]) - len(det_left[d])
        result["e2e"] = {
            "gt": char_gt,
            "det": sum(len(det_texts[d]) for d in det_valid),
            "tp": tp_e2e,
            "gran_recall": gran_recall_e2e,
            "gran_precision": gran_precision_e2e,
        }
    return result


def scores(counts):
    recall = max(0.0, counts["tp"] - counts["gran_recall"]) / counts["gt"] if counts["gt"] else 0.0
    precision = max(0.0, counts["tp"] - counts["gran_precision"]) / counts["det"] if counts["det"] else 0.0
    hmean = 2 * recall * precision / (recall + precision) if recall + precision > 0 else 0.0
    return {"recall": recall, "precision": precision, "hmean": hmean}


class CLEval:
    def __init__(self, **options):
        self.options = options
        self.totals = {}

    def update(self, gt_quads, gt_texts, gt_dc, det_quads, det_texts=None):
        result = evaluate(gt_quads, gt_texts, gt_dc, det_quads, det_texts, **self.options)
        for task, counts in result.items():
            total = self.totals.setdefault(task, {"gt": 0, "det": 0, "tp": 0, "gran_recall": 0.0, "gran_precision": 0.0})
            for key, value in counts.items():
                total[key] += value
        return result

    def compute(self):
        return {task: scores(counts) for task, counts in self.totals.items()}
