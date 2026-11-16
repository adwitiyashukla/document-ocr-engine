import numpy as np

from docr.imaging import gaussian_blur, label, sobel, to_float


def gradients(gray, sigma=0.0):
    image = to_float(gray)
    if sigma > 0:
        image = gaussian_blur(image, sigma)
    gx, gy = sobel(image, mode="edge")
    return gx, gy, np.hypot(gx, gy)


def suppress(gx, gy, magnitude):
    h, w = magnitude.shape
    m = np.zeros((h + 2, w + 2), np.float32)
    m[1:-1, 1:-1] = magnitude
    ax = np.abs(gx)
    ay = np.abs(gy)
    horizontal = ay < ax * np.tan(np.pi / 8)
    vertical = ay > ax * np.tan(3 * np.pi / 8)
    diagonal = ~(horizontal | vertical)
    same_sign = (gx * gy) > 0

    def at(dy, dx):
        return m[1 + dy:h + 1 + dy, 1 + dx:w + 1 + dx]

    keep = np.zeros((h, w), bool)
    keep |= horizontal & (magnitude > at(0, -1)) & (magnitude >= at(0, 1))
    keep |= vertical & (magnitude > at(-1, 0)) & (magnitude >= at(1, 0))
    keep |= diagonal & same_sign & (magnitude > at(-1, -1)) & (magnitude > at(1, 1))
    keep |= diagonal & ~same_sign & (magnitude > at(-1, 1)) & (magnitude > at(1, -1))
    return keep


def canny(gray, low, high, sigma=0.0):
    gx, gy, magnitude = gradients(gray, sigma)
    keep = suppress(gx, gy, magnitude)
    weak = keep & (magnitude > low)
    strong = keep & (magnitude > high)
    labels, count = label(weak, 8)
    if count == 0:
        return np.zeros_like(weak)
    good = np.zeros(count + 1, bool)
    good[np.unique(labels[strong])] = True
    good[0] = False
    return good[labels]


def auto_canny(gray, sigma=1.0, low_quantile=0.7, high_quantile=0.9):
    gx, gy, magnitude = gradients(gray, sigma)
    keep = suppress(gx, gy, magnitude)
    values = magnitude[keep]
    if values.size == 0:
        return np.zeros_like(keep)
    low, high = np.quantile(values, [low_quantile, high_quantile])
    weak = keep & (magnitude > low)
    strong = keep & (magnitude > high)
    labels, count = label(weak, 8)
    if count == 0:
        return np.zeros_like(weak)
    good = np.zeros(count + 1, bool)
    good[np.unique(labels[strong])] = True
    good[0] = False
    return good[labels]


def hough_accumulator(edges, rho_step=1.0, theta_step=np.pi / 180, min_theta=0.0, max_theta=np.pi):
    h, w = edges.shape
    num_theta = int(round((max_theta - min_theta) / theta_step))
    thetas = min_theta + np.arange(num_theta) * theta_step
    num_rho = int(round(((w + h) * 2 + 1) / rho_step))
    ys, xs = np.nonzero(edges)
    accumulator = np.zeros((num_rho, num_theta), np.int64)
    if xs.size == 0:
        return accumulator, thetas, num_rho
    cos = np.cos(thetas) / rho_step
    sin = np.sin(thetas) / rho_step
    chunk = max(1, 4_000_000 // num_theta)
    flat = np.zeros(num_rho * num_theta, np.int64)
    columns = np.arange(num_theta)
    for start in range(0, xs.size, chunk):
        x = xs[start:start + chunk, None].astype(np.float64)
        y = ys[start:start + chunk, None].astype(np.float64)
        r = np.rint(x * cos + y * sin).astype(np.int64) + (num_rho - 1) // 2
        flat += np.bincount((r * num_theta + columns).ravel(), minlength=flat.size)
    accumulator = flat.reshape(num_rho, num_theta)
    return accumulator, thetas, num_rho


def hough_lines(edges, threshold, rho_step=1.0, theta_step=np.pi / 180, max_lines=None, min_theta=0.0, max_theta=np.pi):
    accumulator, thetas, num_rho = hough_accumulator(edges, rho_step, theta_step, min_theta, max_theta)
    a = np.pad(accumulator, 1)
    center = a[1:-1, 1:-1]
    peak = (
        (center > threshold)
        & (center > a[1:-1, :-2])
        & (center >= a[1:-1, 2:])
        & (center > a[:-2, 1:-1])
        & (center >= a[2:, 1:-1])
    )
    r_index, t_index = np.nonzero(peak)
    votes = center[r_index, t_index]
    order = np.lexsort((t_index * num_rho + r_index, -votes))
    if max_lines is not None:
        order = order[:max_lines]
    rho = (r_index[order] - (num_rho - 1) // 2) * rho_step
    return np.stack([rho, thetas[t_index[order]], votes[order]], axis=1)
