import numpy as np


def to_float(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image.astype(np.float32)
    return image.astype(np.float32, copy=False)


def to_uint8(image):
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def to_gray(image):
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    gray = image[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    if image.dtype == np.uint8:
        return to_uint8(gray)
    return gray


def correlate1d(image, kernel, axis, mode="reflect"):
    image = to_float(image)
    kernel = np.asarray(kernel, np.float32)
    radius = len(kernel) // 2
    pad_width = [(0, 0)] * image.ndim
    pad_width[axis] = (radius, len(kernel) - 1 - radius)
    padded = np.pad(image, pad_width, mode=mode)
    out = np.zeros(image.shape, np.float32)
    n = image.shape[axis]
    index = [slice(None)] * image.ndim
    for i, weight in enumerate(kernel):
        if weight == 0:
            continue
        index[axis] = slice(i, i + n)
        out += weight * padded[tuple(index)]
    return out


def separable(image, kernel_y, kernel_x, mode="reflect"):
    return correlate1d(correlate1d(image, kernel_y, 0, mode), kernel_x, 1, mode)


def gaussian_kernel(sigma, size=None):
    if size is None:
        size = 2 * int(np.ceil(3 * sigma)) + 1
    x = np.arange(size, dtype=np.float64) - (size - 1) / 2
    kernel = np.exp(-(x**2) / (2 * sigma**2))
    return (kernel / kernel.sum()).astype(np.float32)


def gaussian_blur(image, sigma, size=None, mode="reflect"):
    if sigma <= 0:
        return to_float(image)
    kernel = gaussian_kernel(sigma, size)
    return separable(image, kernel, kernel, mode)


def sobel(image, mode="reflect"):
    gx = separable(image, [1, 2, 1], [-1, 0, 1], mode)
    gy = separable(image, [-1, 0, 1], [1, 2, 1], mode)
    return gx, gy


def integral(image):
    image = np.asarray(image, np.float64)
    out = np.zeros((image.shape[0] + 1, image.shape[1] + 1), np.float64)
    out[1:, 1:] = image.cumsum(0).cumsum(1)
    return out


def box_sum(image, size):
    kh, kw = size
    rh, rw = kh // 2, kw // 2
    padded = np.pad(np.asarray(image, np.float64), ((rh, kh - 1 - rh), (rw, kw - 1 - rw)), mode="reflect")
    table = integral(padded)
    h, w = image.shape
    return table[kh:kh + h, kw:kw + w] - table[:h, kw:kw + w] - table[kh:kh + h, :w] + table[:h, :w]


def box_mean(image, size):
    return box_sum(image, size) / (size[0] * size[1])


def _sliding_extreme(image, size, axis, reduce):
    if size <= 1:
        return image.copy()
    moved = np.moveaxis(image, axis, -1)
    n = moved.shape[-1]
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        fill = info.min if reduce is np.maximum else info.max
    else:
        fill = -np.inf if reduce is np.maximum else np.inf
    left = size // 2
    total = n + size - 1
    blocks = -(-total // size)
    padded = np.full(moved.shape[:-1] + (blocks * size,), fill, dtype=image.dtype)
    padded[..., left:left + n] = moved
    shaped = padded.reshape(moved.shape[:-1] + (blocks, size))
    prefix = reduce.accumulate(shaped, axis=-1).reshape(padded.shape)
    suffix = reduce.accumulate(shaped[..., ::-1], axis=-1)[..., ::-1].reshape(padded.shape)
    out = reduce(suffix[..., :n], prefix[..., size - 1:size - 1 + n])
    return np.moveaxis(out, -1, axis)


def _morph(image, size, reduce):
    image = np.asarray(image)
    is_bool = image.dtype == bool
    out = image.astype(np.uint8) if is_bool else image
    out = _sliding_extreme(out, size[0], 0, reduce)
    out = _sliding_extreme(out, size[1], 1, reduce)
    return out.astype(bool) if is_bool else out


def dilate(image, size):
    return _morph(image, size, np.maximum)


def erode(image, size):
    return _morph(image, size, np.minimum)


def opening(image, size):
    return dilate(erode(image, size), size)


def closing(image, size):
    return erode(dilate(image, size), size)


def _runs(mask):
    h, w = mask.shape
    padded = np.zeros((h, w + 2), np.int8)
    padded[:, 1:-1] = mask
    diff = np.diff(padded, axis=1)
    rows, starts = np.nonzero(diff == 1)
    _, ends = np.nonzero(diff == -1)
    return rows, starts, ends


def label(mask, connectivity=8):
    mask = np.asarray(mask, bool)
    h, w = mask.shape
    rows, starts, ends = _runs(mask)
    n = len(rows)
    out = np.zeros((h, w), np.int32)
    if n == 0:
        return out, 0
    stride = w + 2
    key_start = rows.astype(np.int64) * stride + starts
    key_end = rows.astype(np.int64) * stride + ends
    above = (rows.astype(np.int64) - 1) * stride
    if connectivity == 8:
        lo = np.searchsorted(key_end, above + starts, side="left")
        hi = np.searchsorted(key_start, above + ends, side="right")
    else:
        lo = np.searchsorted(key_end, above + starts, side="right")
        hi = np.searchsorted(key_start, above + ends - 1, side="right")
    counts = np.maximum(hi - lo, 0)
    counts[rows == 0] = 0
    total = int(counts.sum())
    labels = np.arange(n)
    if total:
        node_b = np.repeat(np.arange(n), counts)
        offsets = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        node_a = np.repeat(lo, counts) + offsets
        while True:
            la = labels[node_a]
            lb = labels[node_b]
            if np.array_equal(la, lb):
                break
            low = np.minimum(la, lb)
            np.minimum.at(labels, la, low)
            np.minimum.at(labels, lb, low)
            np.minimum.at(labels, node_a, low)
            np.minimum.at(labels, node_b, low)
            while True:
                jumped = labels[labels]
                if np.array_equal(jumped, labels):
                    break
                labels = jumped
    _, component = np.unique(labels, return_inverse=True)
    component = component.astype(np.int32) + 1
    lengths = ends - starts
    run_index = np.repeat(np.arange(n), lengths)
    cols = np.arange(int(lengths.sum())) - np.repeat(np.cumsum(lengths) - lengths, lengths) + np.repeat(starts, lengths)
    out[np.repeat(rows, lengths), cols] = component[run_index]
    return out, int(component.max())


def component_stats(labels, count):
    ys, xs = np.nonzero(labels)
    ids = labels[ys, xs]
    area = np.bincount(ids, minlength=count + 1)
    x0 = np.full(count + 1, np.iinfo(np.int32).max)
    y0 = np.full(count + 1, np.iinfo(np.int32).max)
    x1 = np.full(count + 1, -1)
    y1 = np.full(count + 1, -1)
    np.minimum.at(x0, ids, xs)
    np.minimum.at(y0, ids, ys)
    np.maximum.at(x1, ids, xs)
    np.maximum.at(y1, ids, ys)
    return area[1:], np.stack([x0, y0, x1 + 1, y1 + 1], axis=1)[1:]


def resize(image, shape):
    image = to_float(image)
    h, w = image.shape[:2]
    out_h, out_w = shape
    if (out_h, out_w) == (h, w):
        return image.copy()
    sy, sx = h / out_h, w / out_w
    if sy > 1 or sx > 1:
        image = gaussian_blur_xy(image, max(sy - 1, 0) / 2, max(sx - 1, 0) / 2)
    ys = np.clip((np.arange(out_h) + 0.5) * sy - 0.5, 0, h - 1)
    xs = np.clip((np.arange(out_w) + 0.5) * sx - 0.5, 0, w - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    wy = (ys - y0).astype(np.float32)
    wx = (xs - x0).astype(np.float32)
    if image.ndim == 3:
        wy = wy[:, None, None]
        wx = wx[None, :, None]
    else:
        wy = wy[:, None]
        wx = wx[None, :]
    top = image[y0][:, x0] * (1 - wx) + image[y0][:, x1] * wx
    bottom = image[y1][:, x0] * (1 - wx) + image[y1][:, x1] * wx
    return top * (1 - wy) + bottom * wy


def gaussian_blur_xy(image, sigma_y, sigma_x):
    out = to_float(image)
    if sigma_y > 0.05:
        out = correlate1d(out, gaussian_kernel(sigma_y), 0)
    if sigma_x > 0.05:
        out = correlate1d(out, gaussian_kernel(sigma_x), 1)
    return out


def remap(image, map_x, map_y, fill=0.0):
    image = to_float(image)
    h, w = image.shape[:2]
    channels = image.shape[2] if image.ndim == 3 else 1
    map_x = np.asarray(map_x, np.float32)
    map_y = np.asarray(map_y, np.float32)
    valid = (map_x >= -1) & (map_x < w) & (map_y >= -1) & (map_y < h)
    if valid.any():
        x_lo = max(int(np.floor(map_x[valid].min())), 0)
        x_hi = min(int(np.floor(map_x[valid].max())) + 2, w)
        y_lo = max(int(np.floor(map_y[valid].min())), 0)
        y_hi = min(int(np.floor(map_y[valid].max())) + 2, h)
    else:
        x_lo, x_hi, y_lo, y_hi = 0, 1, 0, 1
    window = image[y_lo:y_hi, x_lo:x_hi].reshape(y_hi - y_lo, x_hi - x_lo, channels)
    padded = np.full((y_hi - y_lo + 2, x_hi - x_lo + 2, channels), fill, np.float32)
    padded[1:-1, 1:-1] = window
    flat = padded.reshape(-1, channels)
    x0 = np.where(valid, np.floor(map_x), x_lo - 1).astype(np.int64)
    y0 = np.where(valid, np.floor(map_y), y_lo - 1).astype(np.int64)
    wx = np.where(valid, map_x - x0, 0).astype(np.float32)[..., None]
    wy = np.where(valid, map_y - y0, 0).astype(np.float32)[..., None]
    stride = x_hi - x_lo + 2
    base = (y0 - y_lo + 1) * stride + (x0 - x_lo + 1)
    top = flat[base] * (1 - wx) + flat[base + 1] * wx
    bottom = flat[base + stride] * (1 - wx) + flat[base + stride + 1] * wx
    out = top * (1 - wy) + bottom * wy
    return out[..., 0] if image.ndim == 2 else out
