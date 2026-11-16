import numpy as np
import torch

from docr.geometry import quad_size
from docr.imaging import remap


def word_lines(quads, max_gap=2.5, max_offset=0.5, max_angle=15.0):
    quads = np.asarray(quads, np.float64).reshape(-1, 4, 2)
    n = len(quads)
    if n == 0:
        return []
    left = (quads[:, 0] + quads[:, 3]) / 2
    right = (quads[:, 1] + quads[:, 2]) / 2
    heights = np.array([quad_size(q)[1] for q in quads])
    direction = right - left
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
    normal = np.stack([-direction[:, 1], direction[:, 0]], axis=1)
    cos_limit = np.cos(np.deg2rad(max_angle))
    best_next = np.full(n, -1)
    best_cost = np.full(n, np.inf)
    for i in range(n):
        delta = left - right[i]
        along = delta @ direction[i]
        across = np.abs(delta @ normal[i])
        h = heights[i]
        ratio = heights / max(h, 1e-6)
        ok = (along > -0.5 * h) & (along < max_gap * h) & (across < max_offset * h) & (ratio > 0.6) & (ratio < 1.6) & (direction @ direction[i] > cos_limit)
        ok[i] = False
        if not ok.any():
            continue
        cost = np.where(ok, np.maximum(along, 0) + 3 * across, np.inf)
        j = int(np.argmin(cost))
        best_next[i], best_cost[i] = j, cost[j]
    best_prev = np.full(n, -1)
    prev_cost = np.full(n, np.inf)
    for i in range(n):
        j = best_next[i]
        if j >= 0 and best_cost[i] < prev_cost[j]:
            best_prev[j], prev_cost[j] = i, best_cost[i]
    lines = []
    for start in range(n):
        if best_prev[start] >= 0 and best_next[best_prev[start]] == start:
            continue
        chain = [start]
        seen = {start}
        while True:
            j = best_next[chain[-1]]
            if j < 0 or j in seen or best_prev[j] != chain[-1]:
                break
            chain.append(j)
            seen.add(j)
        lines.append(chain)
    return lines


def line_points(quads, lines, min_words=2, per_word=3):
    quads = np.asarray(quads, np.float64).reshape(-1, 4, 2)
    spans = []
    for chain in lines:
        if len(chain) < min_words:
            continue
        points = []
        for k in chain:
            q = quads[k]
            a, b = (q[0] + q[3]) / 2, (q[1] + q[2]) / 2
            for t in np.linspace(0.1, 0.9, per_word):
                points.append(a + t * (b - a))
        spans.append(np.array(points))
    return spans


def levenberg_marquardt(residual, params, iterations=60, tolerance=1e-10):
    params = params.clone()
    mu = 1e-3
    current = residual(params)
    cost = float(current @ current)
    for _ in range(iterations):
        jac = torch.autograd.functional.jacobian(residual, params, vectorize=True)
        a = jac.T @ jac
        g = jac.T @ current
        diag = torch.diagonal(a).clamp(min=1e-9)
        improved = False
        gain = 0.0
        while mu < 1e10:
            step = torch.linalg.solve(a + mu * torch.diag(diag), -g)
            trial = params + step
            new = residual(trial)
            new_cost = float(new @ new)
            if np.isfinite(new_cost) and new_cost < cost:
                gain = cost - new_cost
                params, current, cost = trial, new, new_cost
                mu = max(mu / 3, 1e-12)
                improved = True
                break
            mu *= 4
        if not improved or gain < tolerance * max(cost, 1e-12):
            break
    return params, cost


def _basis(x, v, degree_x, degree_v):
    terms = [x**a * v**b for a in range(1, degree_x + 1) for b in range(degree_v + 1)]
    return torch.stack(terms, dim=-1)


def fit(spans, degree_x=3, degree_v=2, iterations=40):
    points = np.concatenate(spans)
    x_ref = float(np.median(points[:, 0]))
    y_ref = float(np.median(points[:, 1]))
    scale = float(max(np.ptp(points[:, 0]), np.ptp(points[:, 1]), 1.0)) / 2
    xs = torch.tensor((points[:, 0] - x_ref) / scale, dtype=torch.float64)
    ys = torch.tensor((points[:, 1] - y_ref) / scale, dtype=torch.float64)
    index = torch.tensor(np.concatenate([np.full(len(s), i) for i, s in enumerate(spans)]))
    n_spans = len(spans)
    n_coef = degree_x * (degree_v + 1)
    v0 = torch.tensor([float(((s[:, 1] - y_ref) / scale).mean()) for s in spans], dtype=torch.float64)

    def residual(p):
        v = p[:n_spans][index]
        coef = p[n_spans:]
        return v + _basis(xs, v, degree_x, degree_v) @ coef - ys

    start = torch.cat([v0, torch.zeros(n_coef, dtype=torch.float64)])
    before = float((residual(start) ** 2).mean())
    params, cost = levenberg_marquardt(residual, start, iterations)
    after = cost / max(len(points), 1)
    v = params[:n_spans].numpy()
    return {
        "coef": params[n_spans:].detach(), "x_ref": x_ref, "y_ref": y_ref, "scale": scale, "v": v, "degree": (degree_x, degree_v),
        "before": before, "after": after, "x_range": (float(points[:, 0].min()), float(points[:, 0].max())),
    }


def surface(model, u, v):
    degree_x, degree_v = model["degree"]
    x0, x1 = model["x_range"]
    x = torch.tensor((np.clip(u, x0, x1) - model["x_ref"]) / model["scale"], dtype=torch.float64)
    w = torch.tensor((v - model["y_ref"]) / model["scale"], dtype=torch.float64)
    inside = w.clamp(float(np.min(model["v"])), float(np.max(model["v"])))
    with torch.no_grad():
        y = w + _basis(x, inside, degree_x, degree_v) @ model["coef"]
    return y.numpy() * model["scale"] + model["y_ref"]


def render(image, model, shape):
    h, w = shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float64)
    map_y = surface(model, xs.ravel(), ys.ravel()).reshape(h, w).astype(np.float32)
    return remap(image, xs.astype(np.float32), map_y, fill=255.0), (xs.astype(np.float32), map_y)


def dewarp(image, quads, min_lines=6, min_bend=0.15, degree_x=3, degree_v=2):
    image = np.asarray(image)
    lines = word_lines(quads)
    spans = line_points(quads, lines)
    if len(spans) < min_lines:
        return None
    model = fit(spans, degree_x, degree_v)
    height = float(np.median([quad_size(q)[1] for q in np.asarray(quads, np.float64).reshape(-1, 4, 2)]))
    if np.sqrt(model["before"]) * model["scale"] < min_bend * height:
        return None
    flat, maps = render(image, model, image.shape[:2])
    return {"image": flat, "maps": maps, "model": model, "lines": len(spans), "cost": (model["before"], model["after"])}


def to_flat(points, model, iterations=30):
    points = np.asarray(points, np.float64)
    shape = points.shape
    flat = points.reshape(-1, 2)
    u = flat[:, 0].copy()
    v = flat[:, 1].copy()
    for _ in range(iterations):
        y = surface(model, u, v)
        v = v - (y - flat[:, 1])
    return np.stack([u, v], axis=1).reshape(shape)
