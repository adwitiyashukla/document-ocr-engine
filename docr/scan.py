import numpy as np

from docr.geometry import apply_homography, area, convex_hull, fill_convex, homography, min_area_rect, order_quad, polygon_iou, quad_size, rotation, warp_perspective
from docr.imaging import box_mean, closing, dilate, erode, gaussian_blur, label, opening, resize, to_float, to_gray


def otsu(values, bins=256, value_range=None):
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer) and value_range is None:
        lo, hi = float(values.min()) - 0.5, float(values.max()) + 0.5
        bins = int(hi - lo)
    else:
        lo, hi = value_range if value_range else (float(values.min()), float(values.max()))
    values = values.astype(np.float64).ravel()
    if hi <= lo:
        return float(lo)
    hist, edges = np.histogram(values, bins=bins, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(hist).astype(np.float64)
    total = weight[-1]
    mean = np.cumsum(hist * centers)
    w0 = weight[:-1]
    w1 = total - w0
    valid = (w0 > 0) & (w1 > 0)
    m0 = np.where(valid, mean[:-1] / np.maximum(w0, 1), 0)
    m1 = np.where(valid, (mean[-1] - mean[:-1]) / np.maximum(w1, 1), 0)
    between = np.where(valid, w0 * w1 * (m0 - m1) ** 2, -1)
    return float(edges[1:][int(np.argmax(between))])


def sauvola(gray, window=25, k=0.2, r=128.0):
    gray = to_float(gray).astype(np.float64)
    mean = box_mean(gray, (window, window))
    sq = box_mean(gray * gray, (window, window))
    std = np.sqrt(np.maximum(sq - mean * mean, 0))
    return mean * (1 + k * (std / r - 1))


def binarize(gray, window=25, k=0.2):
    return to_float(gray) > sauvola(gray, window, k)


def flatten_shading(gray, size=None):
    gray = to_float(gray)
    h, w = gray.shape
    if size is None:
        size = max(15, (min(h, w) // 25) | 1)
    background = closing(gray, (size, size))
    background = box_mean(background, (size, size))
    return np.clip(gray / np.maximum(background, 1.0) * 255.0, 0, 255).astype(np.float32)


def paper_map(rgb):
    rgb = to_float(rgb)
    top = rgb.max(axis=2)
    low = rgb.min(axis=2)
    saturation = (top - low) / np.maximum(top, 1.0)
    return top / 255.0 - 0.8 * saturation


def douglas_peucker(points, epsilon):
    points = np.asarray(points, np.float64)
    if len(points) < 4:
        return points
    start = 0
    far = int(np.argmax(np.linalg.norm(points - points[0], axis=1)))
    loop = np.concatenate([points, points[:1]])

    def simplify(chain):
        keep = np.zeros(len(chain), bool)
        keep[0] = keep[-1] = True
        stack = [(0, len(chain) - 1)]
        while stack:
            i, j = stack.pop()
            if j <= i + 1:
                continue
            a, b = chain[i], chain[j]
            d = b - a
            norm = np.hypot(*d)
            seg = chain[i + 1:j] - a
            dist = np.abs(d[0] * seg[:, 1] - d[1] * seg[:, 0]) / norm if norm > 0 else np.linalg.norm(seg, axis=1)
            k = int(np.argmax(dist))
            if dist[k] > epsilon:
                keep[i + 1 + k] = True
                stack.append((i, i + 1 + k))
                stack.append((i + 1 + k, j))
        return chain[keep]

    first = simplify(loop[start:far + 1])
    second = simplify(loop[far:])
    return np.concatenate([first[:-1], second[:-1]])


def _fit_line(points):
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center)
    return center, vt[0]


def _intersect(line_a, line_b):
    (p, d), (q, e) = line_a, line_b
    det = d[0] * (-e[1]) - d[1] * (-e[0])
    if abs(det) < 1e-9:
        return None
    rhs = q - p
    t = (rhs[0] * (-e[1]) - rhs[1] * (-e[0])) / det
    return p + t * d


def fit_quad(hull, epsilon_ratio=0.01, max_angle=30.0):
    rect = min_area_rect(hull)
    diag = np.hypot(*(hull.max(axis=0) - hull.min(axis=0)))
    poly = douglas_peucker(hull, max(epsilon_ratio * diag, 1.0))
    if len(poly) < 3:
        return rect
    from docr.geometry import _orient

    poly = _orient(poly)
    rect = _orient(rect)
    sides = []
    for k in range(4):
        a, b = rect[k], rect[(k + 1) % 4]
        d = b - a
        n = np.array([d[1], -d[0]]) / max(np.hypot(*d), 1e-9)
        sides.append((a, d / max(np.hypot(*d), 1e-9), n))
    groups = [[] for _ in range(4)]
    cos_limit = np.cos(np.deg2rad(max_angle))
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        d = q - p
        length = np.hypot(*d)
        if length < 1e-6:
            continue
        n = np.array([d[1], -d[0]]) / length
        mid = (p + q) / 2
        scores = []
        for a, _, sn in sides:
            alignment = float(n @ sn)
            offset = abs(float((mid - a) @ sn))
            scores.append(alignment - offset / max(diag, 1e-9))
        k = int(np.argmax(scores))
        if float(n @ sides[k][2]) < cos_limit:
            continue
        steps = max(int(length // 2), 2)
        t = np.linspace(0, 1, steps)[:, None]
        groups[k].append(p + t * d)
    lines = []
    for k in range(4):
        if groups[k]:
            lines.append(_fit_line(np.concatenate(groups[k])))
        else:
            a, d, _ = sides[k]
            lines.append((a, d))
    corners = []
    for k in range(4):
        point = _intersect(lines[k - 1], lines[k])
        if point is None:
            return order_quad(rect)
        corners.append(point)
    quad = np.array(corners)
    if area(quad) < 0.5 * area(rect) or area(quad) > 2.0 * area(rect):
        return order_quad(rect)
    return order_quad(quad)


def fill_holes(mask):
    labels, count = label(~mask, 4)
    if count == 0:
        return mask
    border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
    keep = np.zeros(count + 1, bool)
    keep[border] = True
    keep[0] = True
    return mask | ~keep[labels]


def region_boundary(region):
    padded = np.pad(region, 1)
    return region & ~erode(padded, (3, 3))[1:-1, 1:-1]


def extreme_quad(points):
    rect = min_area_rect(points)
    ex = rect[1] - rect[0]
    ey = rect[3] - rect[0]
    width = max(np.hypot(*ex), 1e-9)
    height = max(np.hypot(*ey), 1e-9)
    rel = points - rect[0]
    u = rel @ (ex / width) / width
    v = rel @ (ey / height) / height
    corners = [np.argmin(u + v), np.argmax(u - v), np.argmax(u + v), np.argmax(v - u)]
    return points[corners]


def page_candidates(rgb, max_side=512):
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    sh, sw = max(int(round(h * scale)), 8), max(int(round(w * scale)), 8)
    small = gaussian_blur(resize(rgb, (sh, sw)), 1.5)
    paper = paper_map(small)
    mask = paper > otsu(paper)
    k = max(3, (min(sh, sw) // 40) | 1)
    mask = fill_holes(opening(closing(mask, (k, k)), (k, k)))
    labels, count = label(mask)
    out = []
    if count == 0:
        return out, scale, paper
    for index in range(1, count + 1):
        region = labels == index
        size = int(region.sum())
        if size < 0.05 * sh * sw:
            continue
        ys, xs = np.nonzero(region_boundary(region))
        hull = convex_hull(np.stack([xs, ys], axis=1).astype(np.float64))
        if len(hull) < 3:
            continue
        quad = extreme_quad(hull)
        solidity = size / max(area(hull), 1.0)
        fill = size / max(area(quad), 1.0)
        brightness = float(paper[region].mean())
        out.append({"quad": quad / scale, "area": size / (sh * sw), "solidity": solidity, "fill": fill, "brightness": brightness})
    return out, scale, paper


def find_page(rgb, max_side=512, full_area=0.92):
    rgb = np.asarray(rgb)
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)
    h, w = rgb.shape[:2]
    full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)
    candidates, _, _ = page_candidates(rgb, max_side)
    if not candidates:
        return full, 0.0
    best = max(candidates, key=lambda c: c["area"] * c["solidity"] * min(c["fill"], 1.0))
    quad = order_quad(best["quad"])
    if area(quad) > full_area * w * h:
        return full, float(best["solidity"])
    return quad, float(best["solidity"])


def find_page_from_text(rgb, quads, max_side=512, full_area=0.92):
    rgb = np.asarray(rgb)
    h, w = rgb.shape[:2]
    full = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64)
    quads = np.asarray(quads, np.float64).reshape(-1, 4, 2)
    if not len(quads):
        return full, 0.0
    scale = min(1.0, max_side / max(h, w))
    sh, sw = max(int(round(h * scale)), 8), max(int(round(w * scale)), 8)
    small = gaussian_blur(resize(rgb, (sh, sw)), 1.0)
    words = np.zeros((sh, sw), bool)
    for quad in quads * scale:
        fill_convex(words, quad, True)
    heights = np.array([quad_size(q)[1] for q in quads * scale])
    k = max(3, int(np.median(heights) * 1.5) | 1)
    near = dilate(words, (k, k)) & ~dilate(words, (3, 3))
    if near.sum() < 50:
        return full, 0.0
    chroma = small[..., :2] / (small.sum(axis=2) + 1e-6)[..., None]
    bright = small.max(axis=2)
    dist = np.linalg.norm(chroma - np.median(chroma[near], axis=0), axis=2)
    center = np.median(dist[near])
    spread = np.median(np.abs(dist[near] - center)) + 1e-3
    paper = (dist < center + 6 * spread + 0.01) & (bright > 0.45 * np.median(bright[near]))
    kk = max(3, (min(sh, sw) // 40) | 1)
    paper = fill_holes(opening(closing(paper | words, (kk, kk)), (kk, kk)))
    labels, count = label(paper)
    if count == 0:
        return full, 0.0
    hits = np.bincount(labels[words].ravel(), minlength=count + 1)
    hits[0] = 0
    region = labels == int(np.argmax(hits))
    ys, xs = np.nonzero(region_boundary(region))
    hull = convex_hull(np.stack([xs, ys], axis=1).astype(np.float64))
    if len(hull) < 3:
        return full, 0.0
    quad = order_quad(extreme_quad(hull)) / scale
    if area(quad) > full_area * w * h:
        return full, 1.0
    return quad, float(hits.max() / max(words.sum(), 1))


def clip_to_image(quad, width, height):
    from docr.geometry import clip_convex

    frame = np.array([[0, 0], [width, 0], [width, height], [0, height]], np.float64)
    return clip_convex(convex_hull(quad), frame)


def page_iou(predicted, target, width, height):
    a = clip_to_image(predicted, width, height)
    b = clip_to_image(target, width, height)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    return polygon_iou(a, b)


def rectify(rgb, quad, max_side=2000):
    quad = order_quad(quad)
    w, h = quad_size(quad)
    scale = min(1.0, max_side / max(w, h, 1))
    out_w, out_h = max(int(round(w * scale)), 8), max(int(round(h * scale)), 8)
    target = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], np.float64)
    matrix = homography(quad, target)
    return warp_perspective(rgb, matrix, (out_w, out_h), fill=255.0), matrix


def estimate_skew(gray, max_angle=5.0, step=0.1, max_points=60000, seed=0):
    ink = ~binarize(gray)
    ys, xs = np.nonzero(ink)
    if len(xs) < 50:
        return 0.0
    if len(xs) > max_points:
        pick = np.random.default_rng(seed).choice(len(xs), max_points, replace=False)
        xs, ys = xs[pick], ys[pick]
    angles = np.arange(-max_angle, max_angle + step / 2, step)
    t = np.deg2rad(angles)
    best, best_score = 0.0, -1.0
    height = gray.shape[0] + gray.shape[1]
    for a, rad in zip(angles, t):
        projected = ys * np.cos(rad) - xs * np.sin(rad)
        hist = np.bincount(np.clip(np.rint(projected + gray.shape[1]).astype(np.int64), 0, 2 * height), minlength=2 * height + 1)
        score = float(np.sum(np.diff(hist.astype(np.float64)) ** 2))
        if score > best_score:
            best, best_score = float(a), score
    return best


def deskew(image, angle):
    h, w = image.shape[:2]
    if abs(angle) < 1e-3:
        return to_float(image), np.eye(3)
    matrix = rotation(-angle, ((w - 1) / 2, (h - 1) / 2))
    return warp_perspective(image, matrix, (w, h), fill=255.0), matrix


def scan(rgb, find=True, max_side=2000):
    rgb = np.asarray(rgb)
    h, w = rgb.shape[:2]
    if find:
        quad, score = find_page(rgb)
    else:
        quad, score = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float64), 1.0
    page, matrix = rectify(rgb, quad, max_side)
    return {"page": page, "quad": quad, "matrix": matrix, "score": score}


def to_original(points, matrix):
    return apply_homography(np.linalg.inv(matrix), points)


def to_page(points, matrix):
    return apply_homography(matrix, points)


def clean(rgb):
    gray = to_gray(np.asarray(rgb))
    return flatten_shading(gray)
