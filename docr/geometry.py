import numpy as np

from docr.imaging import remap


def _normalizer(points):
    center = points.mean(axis=0)
    spread = np.sqrt(((points - center) ** 2).sum(axis=1)).mean()
    scale = np.sqrt(2) / spread if spread > 0 else 1.0
    return np.array([[scale, 0, -scale * center[0]], [0, scale, -scale * center[1]], [0, 0, 1]])


def apply_homography(matrix, points):
    points = np.asarray(points, np.float64)
    ones = np.ones(points.shape[:-1] + (1,))
    mapped = np.concatenate([points, ones], axis=-1) @ np.asarray(matrix, np.float64).T
    return mapped[..., :2] / mapped[..., 2:3]


def homography(src, dst):
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    norm_src = _normalizer(src)
    norm_dst = _normalizer(dst)
    s = apply_homography(norm_src, src)
    d = apply_homography(norm_dst, dst)
    n = len(s)
    a = np.zeros((2 * n, 9))
    x, y, u, v = s[:, 0], s[:, 1], d[:, 0], d[:, 1]
    a[0::2, 0], a[0::2, 1], a[0::2, 2] = -x, -y, -1
    a[0::2, 6], a[0::2, 7], a[0::2, 8] = u * x, u * y, u
    a[1::2, 3], a[1::2, 4], a[1::2, 5] = -x, -y, -1
    a[1::2, 6], a[1::2, 7], a[1::2, 8] = v * x, v * y, v
    _, _, vt = np.linalg.svd(a)
    matrix = np.linalg.inv(norm_dst) @ vt[-1].reshape(3, 3) @ norm_src
    return matrix / matrix[2, 2]


def warp_perspective(image, matrix, size, fill=0.0):
    width, height = size
    inverse = np.linalg.inv(np.asarray(matrix, np.float64))
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float64)
    denom = inverse[2, 0] * xs + inverse[2, 1] * ys + inverse[2, 2]
    valid = denom > 1e-12
    denom = np.where(valid, denom, 1.0)
    map_x = np.where(valid, (inverse[0, 0] * xs + inverse[0, 1] * ys + inverse[0, 2]) / denom, -1e6)
    map_y = np.where(valid, (inverse[1, 0] * xs + inverse[1, 1] * ys + inverse[1, 2]) / denom, -1e6)
    return remap(image, map_x, map_y, fill)


def rotation(angle_degrees, center):
    t = np.deg2rad(angle_degrees)
    c, s = np.cos(t), np.sin(t)
    cx, cy = center
    return np.array([[c, -s, cx - c * cx + s * cy], [s, c, cy - s * cx - c * cy], [0, 0, 1.0]])


def box_to_quad(box):
    x0, y0, x1, y1 = box
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)


def signed_area(poly):
    poly = np.asarray(poly, np.float64)
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def area(poly):
    return abs(signed_area(poly))


def convex_hull(points):
    points = np.unique(np.asarray(points, np.float64), axis=0)
    if len(points) <= 2:
        return points
    order = np.lexsort((points[:, 1], points[:, 0]))
    points = points[order]

    def half(seq):
        chain = []
        for p in seq:
            while len(chain) >= 2:
                o, a = chain[-2], chain[-1]
                if (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0]) > 0:
                    break
                chain.pop()
            chain.append(p)
        return chain

    lower = half(points)
    upper = half(points[::-1])
    hull = np.array(lower[:-1] + upper[:-1])
    if signed_area(hull) < 0:
        hull = hull[::-1]
    return hull


def min_area_rect(points):
    points = np.asarray(points, np.float64)
    hull = convex_hull(points)
    if len(hull) < 3:
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        return box_to_quad((x0, y0, x1, y1))
    edges = np.roll(hull, -1, axis=0) - hull
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    c, s = np.cos(angles), np.sin(angles)
    rx = hull[None, :, 0] * c[:, None] + hull[None, :, 1] * s[:, None]
    ry = -hull[None, :, 0] * s[:, None] + hull[None, :, 1] * c[:, None]
    areas = (rx.max(1) - rx.min(1)) * (ry.max(1) - ry.min(1))
    k = int(np.argmin(areas))
    x0, x1 = rx[k].min(), rx[k].max()
    y0, y1 = ry[k].min(), ry[k].max()
    corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    back = np.stack([corners[:, 0] * c[k] - corners[:, 1] * s[k], corners[:, 0] * s[k] + corners[:, 1] * c[k]], axis=1)
    return order_quad(back)


def order_quad(quad):
    quad = np.asarray(quad, np.float64)
    center = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
    quad = quad[np.argsort(angles)]
    start = int(np.argmin(quad[:, 0] + quad[:, 1]))
    return np.roll(quad, -start, axis=0)


def _orient(poly):
    poly = np.asarray(poly, np.float64)
    return poly if signed_area(poly) >= 0 else poly[::-1]


def clip_convex(subject, clip):
    output = [np.asarray(p, np.float64) for p in _orient(subject)]
    clip = _orient(clip)
    for i in range(len(clip)):
        if not output:
            break
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a
        current, output = output, []

        def side(p):
            return edge[0] * (p[1] - a[1]) - edge[1] * (p[0] - a[0])

        for j in range(len(current)):
            p, q = current[j], current[(j + 1) % len(current)]
            sp, sq = side(p), side(q)
            if sp >= 0:
                output.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                output.append(p + t * (q - p))
    return np.array(output) if len(output) >= 3 else np.zeros((0, 2))


def intersection_area(a, b):
    clipped = clip_convex(a, b)
    return area(clipped) if len(clipped) >= 3 else 0.0


def polygon_iou(a, b):
    a = convex_hull(a)
    b = convex_hull(b)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    inter = intersection_area(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def offset_convex(poly, distance):
    poly = _orient(poly)
    n = len(poly)
    edges = np.roll(poly, -1, axis=0) - poly
    lengths = np.linalg.norm(edges, axis=1)
    lengths = np.where(lengths > 0, lengths, 1.0)
    normals = np.stack([edges[:, 1], -edges[:, 0]], axis=1) / lengths[:, None]
    starts = poly + distance * normals
    out = np.zeros_like(poly)
    for i in range(n):
        p0, d0 = starts[i - 1], edges[i - 1]
        p1, d1 = starts[i], edges[i]
        det = d0[0] * (-d1[1]) - d0[1] * (-d1[0])
        if abs(det) < 1e-9:
            out[i] = p1
            continue
        rhs = p1 - p0
        t = (rhs[0] * (-d1[1]) - rhs[1] * (-d1[0])) / det
        out[i] = p0 + t * d0
    return out


def points_in_convex(points, poly):
    poly = _orient(poly)
    points = np.asarray(points, np.float64)
    inside = np.ones(points.shape[:-1], bool)
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        cross = (b[0] - a[0]) * (points[..., 1] - a[1]) - (b[1] - a[1]) * (points[..., 0] - a[0])
        inside &= cross >= 0
    return inside


def fill_convex(mask, poly, value=1):
    h, w = mask.shape[:2]
    poly = np.asarray(poly, np.float64)
    x0 = max(int(np.floor(poly[:, 0].min())), 0)
    x1 = min(int(np.ceil(poly[:, 0].max())), w - 1)
    y0 = max(int(np.floor(poly[:, 1].min())), 0)
    y1 = min(int(np.ceil(poly[:, 1].max())), h - 1)
    if x1 < x0 or y1 < y0:
        return mask
    ys, xs = np.mgrid[y0:y1 + 1, x0:x1 + 1]
    inside = points_in_convex(np.stack([xs, ys], axis=-1), poly)
    mask[y0:y1 + 1, x0:x1 + 1][inside] = value
    return mask


def quad_size(quad):
    quad = np.asarray(quad, np.float64)
    width = (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3])) / 2
    height = (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1])) / 2
    return width, height


def crop_quad(image, quad, height, max_width=None, min_width=8, fill=255.0):
    quad = np.asarray(quad, np.float64)
    w, h = quad_size(quad)
    width = max(int(round(height * w / max(h, 1e-6))), min_width)
    if max_width is not None:
        width = min(width, max_width)
    target = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float64)
    matrix = homography(quad, target)
    return warp_perspective(image, matrix, (width, height), fill)
