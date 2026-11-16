import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.wasm.min.mjs";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";

const SETTINGS = { maxSide: 1280, threshold: 0.3, boxThreshold: 0.6, unclip: 1.5, minSize: 3, height: 32, maxWidth: 512, batch: 64 };
const f32 = Math.fround;
const MEAN = [0.485, 0.456, 0.406].map((v) => f32(f32(v) * 255));
const STD = [0.229, 0.224, 0.225].map((v) => f32(f32(v) * 255));
const CHARS = Array.from({ length: 95 }, (_, i) => String.fromCharCode(32 + i)).join("");

function norm(x, y) {
  return Math.sqrt(x * x + y * y);
}

function pyRound(x) {
  const f = Math.floor(x);
  const d = x - f;
  if (d > 0.5) return f + 1;
  if (d < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}

function median(values) {
  const a = Float64Array.from(values).sort();
  const n = a.length;
  if (!n) return 0;
  return n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2;
}

function makeImage(w, h, c, data) {
  return { w, h, c, data: data || new Float32Array(w * h * c) };
}

function gaussianKernel(sigma) {
  const size = 2 * Math.ceil(3 * sigma) + 1;
  const k = new Float64Array(size);
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = i - (size - 1) / 2;
    k[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
    sum += k[i];
  }
  return Float32Array.from(k, (v) => v / sum);
}

function reflect(i, n) {
  if (n === 1) return 0;
  while (i < 0 || i >= n) {
    if (i < 0) i = -i;
    if (i >= n) i = 2 * n - 2 - i;
  }
  return i;
}

function correlate(img, kernel, axis) {
  const { w, h, c, data } = img;
  const out = new Float32Array(w * h * c);
  const size = kernel.length;
  const r = Math.floor(size / 2);
  const row = w * c;
  if (axis === 0) {
    for (let y = 0; y < h; y++) {
      const dst = y * row;
      for (let k = 0; k < size; k++) {
        const wgt = kernel[k];
        if (wgt === 0) continue;
        const src = reflect(y + k - r, h) * row;
        for (let i = 0; i < row; i++) out[dst + i] += f32(wgt * data[src + i]);
      }
    }
  } else {
    const padded = new Float32Array((w + size - 1) * c);
    for (let y = 0; y < h; y++) {
      const base = y * row;
      for (let x = 0; x < w + size - 1; x++) {
        const from = base + reflect(x - r, w) * c;
        for (let ch = 0; ch < c; ch++) padded[x * c + ch] = data[from + ch];
      }
      for (let k = 0; k < size; k++) {
        const wgt = kernel[k];
        if (wgt === 0) continue;
        const off = k * c;
        for (let i = 0; i < row; i++) out[base + i] += f32(wgt * padded[off + i]);
      }
    }
  }
  return makeImage(w, h, c, out);
}

function blur(img, sigmaY, sigmaX) {
  let out = img;
  if (sigmaY > 0.05) out = correlate(out, gaussianKernel(sigmaY), 0);
  if (sigmaX > 0.05) out = correlate(out, gaussianKernel(sigmaX), 1);
  return out;
}

function resize(img, outH, outW) {
  const { w, h, c } = img;
  if (outH === h && outW === w) return makeImage(w, h, c, Float32Array.from(img.data));
  const sy = h / outH;
  const sx = w / outW;
  const src = sy > 1 || sx > 1 ? blur(img, Math.max(sy - 1, 0) / 2, Math.max(sx - 1, 0) / 2) : img;
  const out = makeImage(outW, outH, c);
  const x0s = new Int32Array(outW), x1s = new Int32Array(outW), wxs = new Float32Array(outW), vxs = new Float32Array(outW);
  for (let j = 0; j < outW; j++) {
    const xx = Math.min(Math.max((j + 0.5) * sx - 0.5, 0), w - 1);
    x0s[j] = Math.floor(xx);
    x1s[j] = Math.min(x0s[j] + 1, w - 1);
    wxs[j] = xx - x0s[j];
    vxs[j] = 1 - wxs[j];
  }
  const s = src.data;
  for (let i = 0; i < outH; i++) {
    const yy = Math.min(Math.max((i + 0.5) * sy - 0.5, 0), h - 1);
    const y0 = Math.floor(yy);
    const y1 = Math.min(y0 + 1, h - 1);
    const wy = f32(yy - y0);
    const vy = f32(1 - wy);
    const r0 = y0 * w, r1 = y1 * w;
    for (let j = 0; j < outW; j++) {
      const wx = wxs[j], vx = vxs[j];
      const a0 = (r0 + x0s[j]) * c, b0 = (r0 + x1s[j]) * c, d0 = (r1 + x0s[j]) * c, e0 = (r1 + x1s[j]) * c;
      const o = (i * outW + j) * c;
      for (let ch = 0; ch < c; ch++) {
        const top = f32(f32(s[a0 + ch] * vx) + f32(s[b0 + ch] * wx));
        const bottom = f32(f32(s[d0 + ch] * vx) + f32(s[e0 + ch] * wx));
        out.data[o + ch] = f32(top * vy) + f32(bottom * wy);
      }
    }
  }
  return out;
}

function toGray(img) {
  const out = makeImage(img.w, img.h, 1);
  for (let i = 0; i < img.w * img.h; i++) {
    const v = Math.fround(img.data[i * 3] * Math.fround(0.299)) + Math.fround(img.data[i * 3 + 1] * Math.fround(0.587)) + Math.fround(img.data[i * 3 + 2] * Math.fround(0.114));
    out.data[i] = Math.min(255, Math.max(0, pyRound(Math.fround(v))));
  }
  return out;
}

function label(mask, w, h, connectivity) {
  const labels = new Int32Array(w * h);
  const parent = [0];
  const find = (a) => {
    while (parent[a] !== a) {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  };
  const union = (a, b) => {
    a = find(a);
    b = find(b);
    if (a !== b) {
      if (a < b) parent[b] = a;
      else parent[a] = b;
    }
  };
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) continue;
      const near = [];
      if (x > 0 && mask[i - 1]) near.push(labels[i - 1]);
      if (y > 0) {
        if (mask[i - w]) near.push(labels[i - w]);
        if (connectivity === 8) {
          if (x > 0 && mask[i - w - 1]) near.push(labels[i - w - 1]);
          if (x < w - 1 && mask[i - w + 1]) near.push(labels[i - w + 1]);
        }
      }
      if (!near.length) {
        parent.push(parent.length);
        labels[i] = parent.length - 1;
      } else {
        let m = near[0];
        for (const v of near) m = Math.min(m, find(v));
        labels[i] = m;
        for (const v of near) union(m, v);
      }
    }
  }
  const remap = new Int32Array(parent.length);
  let count = 0;
  for (let i = 0; i < w * h; i++) {
    if (!labels[i]) continue;
    const root = find(labels[i]);
    if (!remap[root]) remap[root] = ++count;
    labels[i] = remap[root];
  }
  return { labels, count };
}

function cross(o, a, b) {
  return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
}

function signedArea(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const p = poly[i];
    const q = poly[(i + 1) % poly.length];
    s += p[0] * q[1] - p[1] * q[0];
  }
  return s / 2;
}

function convexHull(points) {
  const seen = new Set();
  const pts = [];
  for (const p of points) {
    const key = p[0] + "," + p[1];
    if (!seen.has(key)) {
      seen.add(key);
      pts.push(p);
    }
  }
  if (pts.length <= 2) return pts;
  pts.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const half = (seq) => {
    const chain = [];
    for (const p of seq) {
      while (chain.length >= 2 && cross(chain[chain.length - 2], chain[chain.length - 1], p) <= 0) chain.pop();
      chain.push(p);
    }
    return chain;
  };
  const lower = half(pts);
  const upper = half(pts.slice().reverse());
  let hull = lower.slice(0, -1).concat(upper.slice(0, -1));
  if (signedArea(hull) < 0) hull = hull.reverse();
  return hull;
}

function orderQuad(quad) {
  const cx = quad.reduce((s, p) => s + p[0], 0) / quad.length;
  const cy = quad.reduce((s, p) => s + p[1], 0) / quad.length;
  const sorted = quad.slice().sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
  let start = 0;
  for (let i = 1; i < sorted.length; i++) if (sorted[i][0] + sorted[i][1] < sorted[start][0] + sorted[start][1]) start = i;
  return sorted.slice(start).concat(sorted.slice(0, start));
}

function minAreaRect(points) {
  const hull = convexHull(points);
  if (hull.length < 3) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of points) {
      x0 = Math.min(x0, p[0]); y0 = Math.min(y0, p[1]); x1 = Math.max(x1, p[0]); y1 = Math.max(y1, p[1]);
    }
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]];
  }
  let best = null;
  for (let i = 0; i < hull.length; i++) {
    const a = hull[i];
    const b = hull[(i + 1) % hull.length];
    const t = Math.atan2(b[1] - a[1], b[0] - a[0]);
    const c = Math.cos(t), s = Math.sin(t);
    let rx0 = Infinity, rx1 = -Infinity, ry0 = Infinity, ry1 = -Infinity;
    for (const p of hull) {
      const rx = p[0] * c + p[1] * s;
      const ry = -p[0] * s + p[1] * c;
      rx0 = Math.min(rx0, rx); rx1 = Math.max(rx1, rx); ry0 = Math.min(ry0, ry); ry1 = Math.max(ry1, ry);
    }
    const area = (rx1 - rx0) * (ry1 - ry0);
    if (!best || area < best.area) best = { area, c, s, rx0, rx1, ry0, ry1 };
  }
  const { c, s, rx0, rx1, ry0, ry1 } = best;
  const corners = [[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1]];
  return orderQuad(corners.map(([u, v]) => [u * c - v * s, u * s + v * c]));
}

function offsetConvex(poly, distance) {
  if (signedArea(poly) < 0) poly = poly.slice().reverse();
  const n = poly.length;
  const edges = poly.map((p, i) => [poly[(i + 1) % n][0] - p[0], poly[(i + 1) % n][1] - p[1]]);
  const starts = poly.map((p, i) => {
    const e = edges[i];
    const len = norm(e[0], e[1]) || 1;
    return [p[0] + distance * (e[1] / len), p[1] + distance * (-e[0] / len)];
  });
  return poly.map((p, i) => {
    const j = (i - 1 + n) % n;
    const p0 = starts[j], d0 = edges[j], p1 = starts[i], d1 = edges[i];
    const det = d0[0] * -d1[1] - d0[1] * -d1[0];
    if (Math.abs(det) < 1e-9) return p1;
    const rx = p1[0] - p0[0], ry = p1[1] - p0[1];
    const t = (rx * -d1[1] - ry * -d1[0]) / det;
    return [p0[0] + t * d0[0], p0[1] + t * d0[1]];
  });
}

function perimeter(poly) {
  let s = 0;
  for (let i = 0; i < poly.length; i++) s += norm(poly[(i + 1) % poly.length][0] - poly[i][0], poly[(i + 1) % poly.length][1] - poly[i][1]);
  return s;
}

function boxes(prob, w, h) {
  const bitmap = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) bitmap[i] = prob[i] > SETTINGS.threshold ? 1 : 0;
  const { labels, count } = label(bitmap, w, h, 8);
  const pixels = Array.from({ length: count + 1 }, () => []);
  for (let i = 0; i < w * h; i++) if (labels[i]) pixels[labels[i]].push(i);
  const quads = [];
  const scores = [];
  for (let k = 1; k <= Math.min(count, 3000); k++) {
    const list = pixels[k];
    if (list.length < 4) continue;
    let sum = 0;
    for (const i of list) sum += prob[i];
    const score = sum / list.length;
    if (score < SETTINGS.boxThreshold) continue;
    const corners = [];
    for (const i of list) {
      const x = i % w, y = (i - x) / w;
      corners.push([x, y], [x + 1, y], [x, y + 1], [x + 1, y + 1]);
    }
    const rect = minAreaRect(corners);
    const width = norm(rect[1][0] - rect[0][0], rect[1][1] - rect[0][1]);
    const height = norm(rect[3][0] - rect[0][0], rect[3][1] - rect[0][1]);
    if (Math.min(width, height) < SETTINGS.minSize) continue;
    const distance = (Math.abs(signedArea(rect)) * SETTINGS.unclip) / Math.max(perimeter(rect), 1e-6);
    quads.push(orderQuad(minAreaRect(offsetConvex(rect, distance))).map((p) => [f32(p[0]), f32(p[1])]));
    scores.push(score);
  }
  return { quads, scores };
}

function solve(a, b) {
  const n = b.length;
  for (let i = 0; i < n; i++) {
    let p = i;
    for (let r = i + 1; r < n; r++) if (Math.abs(a[r][i]) > Math.abs(a[p][i])) p = r;
    [a[i], a[p]] = [a[p], a[i]];
    [b[i], b[p]] = [b[p], b[i]];
    for (let r = i + 1; r < n; r++) {
      const f = a[r][i] / a[i][i];
      for (let c = i; c < n; c++) a[r][c] -= f * a[i][c];
      b[r] -= f * b[i];
    }
  }
  const x = new Array(n).fill(0);
  for (let i = n - 1; i >= 0; i--) {
    let s = b[i];
    for (let c = i + 1; c < n; c++) s -= a[i][c] * x[c];
    x[i] = s / a[i][i];
  }
  return x;
}

function homography(src, dst) {
  const a = [];
  const b = [];
  for (let i = 0; i < 4; i++) {
    const [x, y] = src[i];
    const [u, v] = dst[i];
    a.push([x, y, 1, 0, 0, 0, -u * x, -u * y]);
    b.push(u);
    a.push([0, 0, 0, x, y, 1, -v * x, -v * y]);
    b.push(v);
  }
  const m = solve(a, b);
  return [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], 1]];
}

function invert3(m) {
  const [[a, b, c], [d, e, f], [g, h, i]] = m;
  const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
  const det = a * A + b * B + c * C;
  return [
    [A / det, -(b * i - c * h) / det, (b * f - c * e) / det],
    [B / det, (a * i - c * g) / det, -(a * f - c * d) / det],
    [C / det, -(a * h - b * g) / det, (a * e - b * d) / det],
  ];
}

function warp(img, matrix, outW, outH, fill) {
  const inv = invert3(matrix);
  const { w, h, c, data } = img;
  const out = makeImage(outW, outH, c);
  const get = (x, y, ch) => (x >= 0 && x < w && y >= 0 && y < h ? data[(y * w + x) * c + ch] : fill);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const den = inv[2][0] * x + inv[2][1] * y + inv[2][2];
      const o = (y * outW + x) * c;
      let mx = -1e6, my = -1e6;
      if (den > 1e-12) {
        mx = Math.fround((inv[0][0] * x + inv[0][1] * y + inv[0][2]) / den);
        my = Math.fround((inv[1][0] * x + inv[1][1] * y + inv[1][2]) / den);
      }
      if (mx < -1 || mx >= w || my < -1 || my >= h) {
        for (let ch = 0; ch < c; ch++) out.data[o + ch] = fill;
        continue;
      }
      const x0 = Math.floor(mx), y0 = Math.floor(my);
      const wx = f32(mx - x0), wy = f32(my - y0);
      const vx = f32(1 - wx), vy = f32(1 - wy);
      for (let ch = 0; ch < c; ch++) {
        const top = f32(f32(get(x0, y0, ch) * vx) + f32(get(x0 + 1, y0, ch) * wx));
        const bottom = f32(f32(get(x0, y0 + 1, ch) * vx) + f32(get(x0 + 1, y0 + 1, ch) * wx));
        out.data[o + ch] = f32(top * vy) + f32(bottom * wy);
      }
    }
  }
  return out;
}

function quadSize(q) {
  const d = (a, b) => norm(a[0] - b[0], a[1] - b[1]);
  return [(d(q[1], q[0]) + d(q[2], q[3])) / 2, (d(q[3], q[0]) + d(q[2], q[1])) / 2];
}

function cropQuad(gray, quad, fill) {
  const [qw, qh] = quadSize(quad);
  const height = SETTINGS.height;
  const width = Math.min(Math.max(pyRound((height * qw) / Math.max(qh, 1e-6)), 8), SETTINGS.maxWidth);
  const target = [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]];
  return warp(gray, homography(quad, target), width, height, fill);
}

function prepareCrop(crop) {
  const width = Math.ceil(Math.min(Math.max(pyRound((crop.w * SETTINGS.height) / Math.max(crop.h, 1)), 16), SETTINGS.maxWidth) / 4) * 4;
  return resize(crop, SETTINGS.height, width);
}

function greedy(logits, t, k) {
  const ids = [];
  let prev = -1;
  let conf = 0;
  for (let s = 0; s < t; s++) {
    let best = 0;
    let max = -Infinity;
    let total = 0;
    for (let j = 0; j < k; j++) if (logits[s * k + j] > max) { max = logits[s * k + j]; best = j; }
    for (let j = 0; j < k; j++) total += Math.exp(logits[s * k + j] - max);
    conf += -Math.log(total);
    if (best !== prev && best !== 0) ids.push(best);
    prev = best;
  }
  return { text: ids.map((i) => CHARS[i - 1] || "").join(""), confidence: Math.exp(conf / t) };
}

async function read(session, crops) {
  const prepared = crops.map(prepareCrop);
  const order = prepared.map((p, i) => i).sort((a, b) => prepared[a].w - prepared[b].w || a - b);
  const texts = new Array(crops.length);
  const scores = new Array(crops.length);
  for (let start = 0; start < order.length; start += SETTINGS.batch) {
    const idx = order.slice(start, start + SETTINGS.batch);
    const width = Math.max(...idx.map((i) => prepared[i].w));
    const height = SETTINGS.height;
    const input = new Float32Array(idx.length * height * width);
    idx.forEach((i, b) => {
      const p = prepared[i];
      const tail = [];
      for (let y = 0; y < height; y++) tail.push(p.data[y * p.w + p.w - 2], p.data[y * p.w + p.w - 1]);
      const pad = median(tail);
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          const v = x < p.w ? p.data[y * p.w + x] : pad;
          input[(b * height + y) * width + x] = (Math.fround(v) - 127.5) / 127.5;
        }
      }
    });
    const result = await session.run({ image: new ort.Tensor("float32", input, [idx.length, 1, height, width]) });
    const logits = result.logits;
    const [, t, k] = logits.dims;
    idx.forEach((i, b) => {
      const out = greedy(logits.data.subarray(b * t * k, (b + 1) * t * k), t, k);
      texts[i] = out.text;
      scores[i] = out.confidence;
    });
  }
  return { texts, scores };
}

function wordLines(quads) {
  const n = quads.length;
  const left = quads.map((q) => [(q[0][0] + q[3][0]) / 2, (q[0][1] + q[3][1]) / 2]);
  const right = quads.map((q) => [(q[1][0] + q[2][0]) / 2, (q[1][1] + q[2][1]) / 2]);
  const heights = quads.map((q) => quadSize(q)[1]);
  const dir = quads.map((q, i) => {
    const dx = right[i][0] - left[i][0], dy = right[i][1] - left[i][1];
    const len = Math.max(Math.hypot(dx, dy), 1e-9);
    return [dx / len, dy / len];
  });
  const cosLimit = Math.cos((15 * Math.PI) / 180);
  const next = new Array(n).fill(-1);
  const cost = new Array(n).fill(Infinity);
  for (let i = 0; i < n; i++) {
    const d = dir[i];
    const h = heights[i];
    let best = -1, bestCost = Infinity;
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const dx = left[j][0] - right[i][0], dy = left[j][1] - right[i][1];
      const along = dx * d[0] + dy * d[1];
      const across = Math.abs(dx * -d[1] + dy * d[0]);
      const ratio = heights[j] / Math.max(h, 1e-6);
      const ok = along > -0.5 * h && along < 2.5 * h && across < 0.5 * h && ratio > 0.6 && ratio < 1.6 && dir[j][0] * d[0] + dir[j][1] * d[1] > cosLimit;
      if (!ok) continue;
      const c = Math.max(along, 0) + 3 * across;
      if (c < bestCost) { bestCost = c; best = j; }
    }
    next[i] = best;
    cost[i] = bestCost;
  }
  const prev = new Array(n).fill(-1);
  const prevCost = new Array(n).fill(Infinity);
  for (let i = 0; i < n; i++) {
    const j = next[i];
    if (j >= 0 && cost[i] < prevCost[j]) { prev[j] = i; prevCost[j] = cost[i]; }
  }
  const lines = [];
  for (let s = 0; s < n; s++) {
    if (prev[s] >= 0 && next[prev[s]] === s) continue;
    const chain = [s];
    const seen = new Set([s]);
    for (;;) {
      const j = next[chain[chain.length - 1]];
      if (j < 0 || seen.has(j) || prev[j] !== chain[chain.length - 1]) break;
      chain.push(j);
      seen.add(j);
    }
    lines.push(chain);
  }
  return lines;
}

export function readingLines(quads, texts) {
  if (!quads.length) return [];
  const rows = wordLines(quads).map((chain) => ({
    y: chain.reduce((s, k) => s + quads[k].reduce((t, p) => t + p[1], 0) / 4, 0) / chain.length,
    x: Math.min(...quads[chain[0]].map((p) => p[0])),
    text: chain.map((k) => texts[k]).join(" "),
    chain,
  }));
  const spans = quads.map((q) => Math.max(Math.max(...q.map((p) => p[1])) - Math.min(...q.map((p) => p[1])), 1));
  const height = median(spans);
  rows.sort((a, b) => a.y - b.y || a.x - b.x || (a.text < b.text ? -1 : a.text > b.text ? 1 : 0));
  const lines = [];
  for (const row of rows) {
    const last = lines[lines.length - 1];
    if (last && Math.abs(row.y - last.reduce((s, r) => s + r.y, 0) / last.length) < 0.5 * height) last.push(row);
    else lines.push([row]);
  }
  return lines.map((line) => line.slice().sort((a, b) => a.x - b.x).flatMap((r) => r.chain));
}

function sliding(mask, w, h, size, isMax) {
  const r = Math.floor(size / 2);
  const tmp = new Uint8Array(w * h);
  const out = new Uint8Array(w * h);
  const outside = isMax ? 0 : 1;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = isMax ? 0 : 1;
      for (let k = -r; k < size - r; k++) {
        const xx = x + k;
        const s = xx >= 0 && xx < w ? mask[y * w + xx] : outside;
        if (isMax ? s : !s) { v = isMax ? 1 : 0; break; }
      }
      tmp[y * w + x] = v;
    }
  }
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let v = isMax ? 0 : 1;
      for (let k = -r; k < size - r; k++) {
        const yy = y + k;
        const s = yy >= 0 && yy < h ? tmp[yy * w + x] : outside;
        if (isMax ? s : !s) { v = isMax ? 1 : 0; break; }
      }
      out[y * w + x] = v;
    }
  }
  return out;
}

function fillHoles(mask, w, h) {
  const inverse = mask.map((v) => (v ? 0 : 1));
  const { labels, count } = label(inverse, w, h, 4);
  const keep = new Uint8Array(count + 1);
  for (let x = 0; x < w; x++) { keep[labels[x]] = 1; keep[labels[(h - 1) * w + x]] = 1; }
  for (let y = 0; y < h; y++) { keep[labels[y * w]] = 1; keep[labels[y * w + w - 1]] = 1; }
  return mask.map((v, i) => (v || !keep[labels[i]] ? 1 : 0));
}

function fillConvex(mask, w, h, poly) {
  if (signedArea(poly) < 0) poly = poly.slice().reverse();
  const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
  const x0 = Math.max(Math.floor(Math.min(...xs)), 0), x1 = Math.min(Math.ceil(Math.max(...xs)), w - 1);
  const y0 = Math.max(Math.floor(Math.min(...ys)), 0), y1 = Math.min(Math.ceil(Math.max(...ys)), h - 1);
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      let inside = true;
      for (let i = 0; i < poly.length && inside; i++) {
        const a = poly[i], b = poly[(i + 1) % poly.length];
        if ((b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]) < 0) inside = false;
      }
      if (inside) mask[y * w + x] = 1;
    }
  }
}

function extremeQuad(points) {
  const rect = minAreaRect(points);
  const ex = [rect[1][0] - rect[0][0], rect[1][1] - rect[0][1]];
  const ey = [rect[3][0] - rect[0][0], rect[3][1] - rect[0][1]];
  const width = Math.max(Math.hypot(ex[0], ex[1]), 1e-9);
  const height = Math.max(Math.hypot(ey[0], ey[1]), 1e-9);
  const uv = points.map((p) => {
    const rx = p[0] - rect[0][0], ry = p[1] - rect[0][1];
    return [(rx * (ex[0] / width) + ry * (ex[1] / width)) / width, (rx * (ey[0] / height) + ry * (ey[1] / height)) / height];
  });
  const pick = (f, low) => {
    let best = 0;
    for (let i = 1; i < uv.length; i++) if (low ? f(uv[i]) < f(uv[best]) : f(uv[i]) > f(uv[best])) best = i;
    return points[best];
  };
  return [pick(([u, v]) => u + v, true), pick(([u, v]) => u - v, false), pick(([u, v]) => u + v, false), pick(([u, v]) => v - u, false)];
}

function findPage(rgb, quads) {
  const { w, h } = rgb;
  const full = [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]];
  if (!quads.length) return full;
  const scale = Math.min(1, 512 / Math.max(h, w));
  const sh = Math.max(pyRound(h * scale), 8), sw = Math.max(pyRound(w * scale), 8);
  const small = blur(resize(rgb, sh, sw), 1.0, 1.0);
  const words = new Uint8Array(sw * sh);
  const scaled = quads.map((q) => q.map((p) => [p[0] * scale, p[1] * scale]));
  for (const q of scaled) fillConvex(words, sw, sh, q);
  const k = Math.max(3, Math.trunc(median(scaled.map((q) => quadSize(q)[1])) * 1.5) | 1);
  const wide = sliding(words, sw, sh, k, true);
  const thin = sliding(words, sw, sh, 3, true);
  const near = [];
  for (let i = 0; i < sw * sh; i++) if (wide[i] && !thin[i]) near.push(i);
  if (near.length < 50) return full;
  const n = sw * sh;
  const cr = new Float64Array(n), cg = new Float64Array(n), bright = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const r = small.data[i * 3], g = small.data[i * 3 + 1], b = small.data[i * 3 + 2];
    const s = r + g + b + 1e-6;
    cr[i] = r / s;
    cg[i] = g / s;
    bright[i] = Math.max(r, g, b);
  }
  const mr = median(near.map((i) => cr[i])), mg = median(near.map((i) => cg[i]));
  const dist = new Float64Array(n);
  for (let i = 0; i < n; i++) dist[i] = Math.hypot(cr[i] - mr, cg[i] - mg);
  const center = median(near.map((i) => dist[i]));
  const spread = median(near.map((i) => Math.abs(dist[i] - center))) + 1e-3;
  const brightLimit = 0.45 * median(near.map((i) => bright[i]));
  let paper = new Uint8Array(n);
  for (let i = 0; i < n; i++) paper[i] = (dist[i] < center + 6 * spread + 0.01 && bright[i] > brightLimit) || words[i] ? 1 : 0;
  const kk = Math.max(3, Math.floor(Math.min(sh, sw) / 40) | 1);
  paper = sliding(sliding(paper, sw, sh, kk, true), sw, sh, kk, false);
  paper = sliding(sliding(paper, sw, sh, kk, false), sw, sh, kk, true);
  paper = fillHoles(paper, sw, sh);
  const { labels, count } = label(paper, sw, sh, 8);
  if (!count) return full;
  const hits = new Int32Array(count + 1);
  for (let i = 0; i < n; i++) if (words[i]) hits[labels[i]]++;
  hits[0] = 0;
  let best = 0;
  for (let i = 1; i <= count; i++) if (hits[i] > hits[best]) best = i;
  const boundary = [];
  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      if (labels[y * sw + x] !== best) continue;
      let edge = false;
      for (let dy = -1; dy <= 1 && !edge; dy++) {
        for (let dx = -1; dx <= 1 && !edge; dx++) {
          const xx = x + dx, yy = y + dy;
          if (xx < 0 || yy < 0 || xx >= sw || yy >= sh || labels[yy * sw + xx] !== best) edge = true;
        }
      }
      if (edge) boundary.push([x, y]);
    }
  }
  const hull = convexHull(boundary);
  if (hull.length < 3) return full;
  const quad = orderQuad(extremeQuad(hull)).map((p) => [p[0] / scale, p[1] / scale]);
  if (Math.abs(signedArea(quad)) > 0.92 * w * h) return full;
  return quad;
}

function rectify(rgb, quad) {
  quad = orderQuad(quad);
  const [qw, qh] = quadSize(quad);
  const scale = Math.min(1, 2000 / Math.max(qw, qh, 1));
  const outW = Math.max(pyRound(qw * scale), 8), outH = Math.max(pyRound(qh * scale), 8);
  const target = [[0, 0], [outW - 1, 0], [outW - 1, outH - 1], [0, outH - 1]];
  const matrix = homography(quad, target);
  return { page: warp(rgb, matrix, outW, outH, 255), matrix };
}

function project(matrix, p) {
  const d = matrix[2][0] * p[0] + matrix[2][1] * p[1] + matrix[2][2];
  return [(matrix[0][0] * p[0] + matrix[0][1] * p[1] + matrix[0][2]) / d, (matrix[1][0] * p[0] + matrix[1][1] * p[1] + matrix[1][2]) / d];
}

export class Engine {
  static async load(detectorUrl, readerUrl) {
    const options = { executionProviders: ["wasm"], graphOptimizationLevel: "all" };
    const engine = new Engine();
    engine.detector = await ort.InferenceSession.create(detectorUrl, options);
    engine.reader = await ort.InferenceSession.create(readerUrl, options);
    return engine;
  }

  async run(rgb, { page: findPageOn = true, onStage = () => {} } = {}) {
    const timing = {};
    onStage("detect");
    let t = performance.now();
    const scale = SETTINGS.maxSide / Math.max(rgb.h, rgb.w);
    const outH = Math.max(32, pyRound((rgb.h * scale) / 32) * 32);
    const outW = Math.max(32, pyRound((rgb.w * scale) / 32) * 32);
    const resized = resize(rgb, outH, outW);
    const input = new Float32Array(3 * outH * outW);
    for (let i = 0; i < outH * outW; i++) for (let c = 0; c < 3; c++) input[c * outH * outW + i] = f32(f32(resized.data[i * 3 + c] - MEAN[c]) / STD[c]);
    const result = await this.detector.run({ image: new ort.Tensor("float32", input, [1, 3, outH, outW]) });
    const found = boxes(result.prob.data, outW, outH);
    const sx = f32(outW / rgb.w), sy = f32(outH / rgb.h);
    const quads = found.quads.map((q) => q.map((p) => [f32(p[0] / sx), f32(p[1] / sy)]));
    timing.detect = performance.now() - t;
    onStage("read", quads.length);
    t = performance.now();
    const gray = toGray(rgb);
    const hist = new Int32Array(256);
    for (const v of gray.data) hist[v]++;
    let acc = 0, lo = -1, hi = -1;
    const total = gray.data.length;
    for (let v = 0; v < 256; v++) {
      acc += hist[v];
      if (lo < 0 && acc >= Math.floor((total - 1) / 2) + 1) lo = v;
      if (hi < 0 && acc >= Math.floor(total / 2) + 1) { hi = v; break; }
    }
    const fill = (lo + hi) / 2;
    const crops = quads.map((q) => cropQuad(gray, q, fill));
    const { texts, scores } = await read(this.reader, crops);
    timing.read = performance.now() - t;
    const lines = readingLines(quads, texts);
    const out = { quads, texts, scores, boxScores: found.scores, timing, lines, text: lines.map((line) => line.map((k) => texts[k]).join(" ")).join("\n") };
    if (findPageOn) {
      onStage("page");
      t = performance.now();
      out.pageQuad = findPage(rgb, quads);
      const { page, matrix } = rectify(rgb, out.pageQuad);
      out.page = page;
      out.pageWords = quads.map((q) => q.map((p) => project(matrix, p)));
      timing.page = performance.now() - t;
    }
    return out;
  }
}

export function imageFromRGBA(width, height, rgba) {
  const img = makeImage(width, height, 3);
  for (let i = 0; i < width * height; i++) {
    img.data[i * 3] = rgba[i * 4];
    img.data[i * 3 + 1] = rgba[i * 4 + 1];
    img.data[i * 3 + 2] = rgba[i * 4 + 2];
  }
  return img;
}

export function imageToRGBA(img) {
  const rgba = new Uint8ClampedArray(img.w * img.h * 4);
  for (let i = 0; i < img.w * img.h; i++) {
    for (let c = 0; c < 3; c++) rgba[i * 4 + c] = pyRound(img.data[i * img.c + c]);
    rgba[i * 4 + 3] = 255;
  }
  return rgba;
}
