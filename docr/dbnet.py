import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from docr.geometry import area, convex_hull, min_area_rect, offset_convex, order_quad
from docr.imaging import label, resize

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def conv_bn(cin, cout, stride=1, kernel=3):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride, kernel // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Residual(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.a = conv_bn(cin, cout, stride)
        self.b = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = None if stride == 1 and cin == cout else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = self.b(self.a(x))
        return F.relu(out + (x if self.skip is None else self.skip(x)))


class Backbone(nn.Module):
    def __init__(self, channels, blocks):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(3, channels[0] // 2, 2), conv_bn(channels[0] // 2, channels[0], 2))
        stages = []
        cin = channels[0]
        for i, (cout, n) in enumerate(zip(channels, blocks)):
            layers = [Residual(cin, cout, 1 if i == 0 else 2)]
            layers += [Residual(cout, cout, 1) for _ in range(n - 1)]
            stages.append(nn.Sequential(*layers))
            cin = cout
        self.stages = nn.ModuleList(stages)

    def forward(self, x):
        x = self.stem(x)
        out = []
        for stage in self.stages:
            x = stage(x)
            out.append(x)
        return out


class DBNet(nn.Module):
    def __init__(self, channels=(32, 64, 128, 192), blocks=(1, 2, 2, 1), fpn=64, k=50.0):
        super().__init__()
        self.k = k
        self.backbone = Backbone(channels, blocks)
        self.lateral = nn.ModuleList(nn.Conv2d(c, fpn, 1, bias=False) for c in channels)
        self.smooth = nn.ModuleList(nn.Sequential(nn.Conv2d(fpn, fpn // 4, 3, padding=1, bias=False), nn.BatchNorm2d(fpn // 4), nn.ReLU(inplace=True)) for _ in channels)
        self.prob = self._head(fpn)
        self.thresh = self._head(fpn)

    @staticmethod
    def _head(fpn):
        return nn.Sequential(
            nn.Conv2d(fpn, fpn // 4, 3, padding=1, bias=False), nn.BatchNorm2d(fpn // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fpn // 4, fpn // 4, 2, 2), nn.BatchNorm2d(fpn // 4), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(fpn // 4, 1, 2, 2),
        )

    def forward(self, x):
        features = self.backbone(x)
        lateral = [conv(f) for conv, f in zip(self.lateral, features)]
        for i in range(len(lateral) - 2, -1, -1):
            lateral[i] = lateral[i] + F.interpolate(lateral[i + 1], size=lateral[i].shape[-2:], mode="nearest")
        size = lateral[0].shape[-2:]
        fused = torch.cat([F.interpolate(s(p), size=size, mode="nearest") for s, p in zip(self.smooth, lateral)], dim=1)
        logits = self.prob(fused)
        if not self.training:
            return torch.sigmoid(logits)
        thresh = torch.sigmoid(self.thresh(fused))
        binary = torch.sigmoid(self.k * (torch.sigmoid(logits) - thresh))
        return logits, thresh, binary


def normalize(image):
    image = np.asarray(image, np.float32)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    return ((image[..., :3] - MEAN) / STD).transpose(2, 0, 1)


def _perimeter(poly):
    return float(np.linalg.norm(np.roll(poly, -1, axis=0) - poly, axis=1).sum())


def _segment_distance(xs, ys, a, b):
    d = b - a
    length = float(d @ d)
    if length < 1e-9:
        return np.hypot(xs - a[0], ys - a[1])
    t = np.clip(((xs - a[0]) * d[0] + (ys - a[1]) * d[1]) / length, 0, 1)
    return np.hypot(xs - (a[0] + t * d[0]), ys - (a[1] + t * d[1]))


NEXT = [1, 2, 3, 0]
PREV = [3, 0, 1, 2]


def _quad(quad):
    q = np.asarray(quad, np.float64).reshape(4, 2)
    if 0.5 * np.sum(q[:, 0] * q[NEXT, 1] - q[NEXT, 0] * q[:, 1]) < 0:
        q = q[::-1].copy()
    edges = q[NEXT] - q
    turns = edges[:, 0] * edges[NEXT, 1] - edges[:, 1] * edges[NEXT, 0]
    return q, edges, bool(np.all(turns >= -1e-9))


def _offset_quad(q, edges, lengths, distance):
    normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1) / np.maximum(lengths, 1e-9)[:, None]
    p = q + distance * normals
    pa, da = p[PREV], edges[PREV]
    det = -da[:, 0] * edges[:, 1] + da[:, 1] * edges[:, 0]
    rhs = p - pa
    t = (-rhs[:, 0] * edges[:, 1] + rhs[:, 1] * edges[:, 0]) / np.where(np.abs(det) < 1e-9, 1.0, det)
    return np.where((np.abs(det) < 1e-9)[:, None], p, pa + t[:, None] * da)


def _fill_quad(target, q, value):
    h, w = target.shape
    x0, x1 = max(int(np.floor(q[:, 0].min())), 0), min(int(np.ceil(q[:, 0].max())), w - 1)
    y0, y1 = max(int(np.floor(q[:, 1].min())), 0), min(int(np.ceil(q[:, 1].max())), h - 1)
    if x1 < x0 or y1 < y0:
        return
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    inside = np.ones(xs.shape, bool)
    for i in range(4):
        a, b = q[i], q[NEXT[i]]
        inside &= (b[0] - a[0]) * (ys - a[1]) - (b[1] - a[1]) * (xs - a[0]) >= 0
    target[y0:y1 + 1, x0:x1 + 1][inside] = value


def _shoelace(q):
    return 0.5 * float(np.sum(q[:, 0] * q[NEXT, 1] - q[NEXT, 0] * q[:, 1]))


def targets(shape, quads, ignore, shrink=0.4, thresh_min=0.3, thresh_max=0.7, min_size=3.0):
    h, w = shape
    gt = np.zeros((h, w), np.float32)
    mask = np.ones((h, w), np.float32)
    thresh = np.zeros((h, w), np.float32)
    thresh_mask = np.zeros((h, w), np.float32)
    for quad, skip in zip(quads, ignore):
        q, edges, convex = _quad(quad)
        if not convex:
            hull = convex_hull(quad)
            if len(hull) < 3:
                continue
            q = np.asarray(min_area_rect(hull), np.float64)
            q, edges, convex = _quad(q)
        lengths = np.hypot(edges[:, 0], edges[:, 1])
        side = min((lengths[0] + lengths[2]) / 2, (lengths[1] + lengths[3]) / 2)
        a = _shoelace(q)
        if skip or side < min_size or a <= 0:
            _fill_quad(mask, q, 0.0)
            continue
        distance = a * (1 - shrink**2) / max(float(lengths.sum()), 1e-6)
        inner = _offset_quad(q, edges, lengths, -distance)
        if _shoelace(inner) < 1:
            _fill_quad(mask, q, 0.0)
            continue
        _fill_quad(gt, inner, 1.0)
        outer = _offset_quad(q, edges, lengths, distance)
        _fill_quad(thresh_mask, outer, 1.0)
        x0 = max(int(np.floor(outer[:, 0].min())), 0)
        x1 = min(int(np.ceil(outer[:, 0].max())), w - 1)
        y0 = max(int(np.floor(outer[:, 1].min())), 0)
        y1 = min(int(np.ceil(outer[:, 1].max())), h - 1)
        if x1 < x0 or y1 < y0:
            continue
        ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float64)
        dist = np.min([_segment_distance(xs, ys, q[i], q[NEXT[i]]) for i in range(4)], axis=0)
        value = 1 - np.clip(dist / max(distance, 1e-6), 0, 1)
        region = thresh[y0:y1 + 1, x0:x1 + 1]
        np.maximum(region, value.astype(np.float32), out=region)
    thresh = thresh * (thresh_max - thresh_min) + thresh_min
    return gt, mask, thresh, thresh_mask


def loss(outputs, gt, mask, thresh_map, thresh_mask, alpha=1.0, beta=10.0, ratio=3.0, eps=1e-6):
    logits, thresh, binary = outputs
    logits, thresh, binary = logits[:, 0], thresh[:, 0], binary[:, 0]
    bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
    positive = gt * mask
    negative = (1 - gt) * mask
    n_pos = int(positive.sum())
    n_neg = int(min(negative.sum(), n_pos * ratio)) if n_pos else int(min(negative.sum(), 1000))
    neg_values = (bce * negative).flatten()
    hard = torch.topk(neg_values, n_neg).values.sum() if n_neg else neg_values.sum() * 0
    prob_loss = ((bce * positive).sum() + hard) / (n_pos + n_neg + eps)
    inter = (binary * gt * mask).sum()
    union = (binary * mask).sum() + (gt * mask).sum() + eps
    binary_loss = 1 - 2 * inter / union
    thresh_loss = (torch.abs(thresh - thresh_map) * thresh_mask).sum() / (thresh_mask.sum() + eps)
    total = prob_loss + alpha * binary_loss + beta * thresh_loss
    return total, {"prob": prob_loss.item(), "binary": binary_loss.item(), "thresh": thresh_loss.item()}


def boxes(prob, threshold=0.3, box_threshold=0.6, unclip=1.5, min_size=3.0, max_boxes=3000):
    bitmap = prob > threshold
    labels, count = label(bitmap, 8)
    if count == 0:
        return np.zeros((0, 4, 2), np.float32), np.zeros(0, np.float32)
    ys, xs = np.nonzero(labels)
    ids = labels[ys, xs]
    order = np.argsort(ids, kind="stable")
    ys, xs, ids = ys[order], xs[order], ids[order]
    starts = np.searchsorted(ids, np.arange(1, count + 1))
    ends = np.searchsorted(ids, np.arange(1, count + 1), side="right")
    quads, scores = [], []
    for s, e in zip(starts[:max_boxes], ends[:max_boxes]):
        if e - s < 4:
            continue
        px, py = xs[s:e], ys[s:e]
        score = float(prob[py, px].mean())
        if score < box_threshold:
            continue
        points = np.stack([px, py], axis=1).astype(np.float64)
        corners = np.concatenate([points, points + [1, 0], points + [0, 1], points + [1, 1]])
        rect = min_area_rect(convex_hull(corners))
        width, height = np.linalg.norm(rect[1] - rect[0]), np.linalg.norm(rect[3] - rect[0])
        if min(width, height) < min_size:
            continue
        distance = area(rect) * unclip / max(_perimeter(rect), 1e-6)
        grown = order_quad(min_area_rect(offset_convex(rect, distance)))
        quads.append(grown)
        scores.append(score)
    return np.array(quads, np.float32).reshape(-1, 4, 2), np.array(scores, np.float32)


def resize_for_detection(image, max_side=1280, multiple=32):
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    out_h = max(multiple, int(round(h * scale / multiple)) * multiple)
    out_w = max(multiple, int(round(w * scale / multiple)) * multiple)
    return resize(image, (out_h, out_w)), (out_w / w, out_h / h)


class Detector:
    def __init__(self, model, max_side=1280, threshold=0.3, box_threshold=0.6, unclip=1.5, min_size=3.0):
        self.model = model.eval()
        self.max_side = max_side
        self.threshold = threshold
        self.box_threshold = box_threshold
        self.unclip = unclip
        self.min_size = min_size

    @torch.no_grad()
    def probability(self, image):
        resized, (sx, sy) = resize_for_detection(np.asarray(image, np.float32), self.max_side)
        x = torch.from_numpy(normalize(resized))[None]
        return self.model(x)[0, 0].numpy(), (sx, sy)

    def __call__(self, image):
        prob, (sx, sy) = self.probability(image)
        quads, scores = boxes(prob, self.threshold, self.box_threshold, self.unclip, self.min_size)
        if len(quads):
            quads = quads / np.array([sx, sy], np.float32)
        return quads, scores
