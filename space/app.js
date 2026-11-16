const MODELS = { detector: "models/dbnet.onnx", reader: "models/crnn.onnx", bytes: 5351569 + 10041233 };
const MAX_SIDE = 2048;
const UNSURE = 0.8;
const YELLOW = "rgb(255, 230, 102)";
const PINK = "rgb(255, 158, 187)";

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const frame = $("frame");
const tip = $("tip");
const statusLine = $("status");
const textBox = $("text");
const copyButton = $("copy");
const saveButton = $("save");
const drop = $("drop");
const tabs = [...document.querySelectorAll('[role="tab"]')];
const examples = [...document.querySelectorAll(".example")];

const state = { view: "photo", photo: null, page: null, result: null, active: -1, ready: false, busy: false, pending: false, run: 0, failed: false };

const seconds = (ms) => (ms / 1000).toFixed(1);
const megabytes = (bytes) => (bytes / 1e6).toFixed(1);
const percent = (score) => Math.round(score * 100);

function setStatus(message, error = false) {
  statusLine.textContent = message;
  statusLine.classList.toggle("error", error);
}

const worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });

worker.onerror = (event) => {
  event.preventDefault();
  state.failed = true;
  setStatus("This browser could not start the OCR engine. A recent version of Chrome, Edge, Firefox or Safari will work.", true);
};

worker.onmessage = ({ data }) => {
  if (data.type === "progress") {
    setStatus(`Downloading the models: ${megabytes(Math.min(data.loaded, MODELS.bytes))} of ${megabytes(MODELS.bytes)} MB, only on the first visit`);
  } else if (data.type === "starting") {
    setStatus("Starting ONNX Runtime");
  } else if (data.type === "ready") {
    state.ready = true;
    if (state.photo) process();
    else setStatus("Ready. Pick an example or upload a photo.");
  } else if (data.type === "stage") {
    if (data.id === state.run && !state.pending) setStatus(data.stage === "detect" ? "Finding the words" : data.stage === "read" ? `Reading ${data.count} ${data.count === 1 ? "word" : "words"}` : "Flattening the page");
  } else if (data.type === "result") {
    finish(data);
  } else if (data.type === "error") {
    fail(data);
  }
};

worker.postMessage({ type: "load", detector: new URL(MODELS.detector, location.href).href, reader: new URL(MODELS.reader, location.href).href });

function process() {
  if (!state.ready || !state.photo || state.failed) return;
  if (state.busy) {
    state.pending = true;
    return;
  }
  state.busy = true;
  state.pending = false;
  state.run += 1;
  const { width, height } = state.photo;
  let rgba;
  try {
    rgba = state.photo.getContext("2d").getImageData(0, 0, width, height).data;
  } catch (error) {
    state.busy = false;
    setStatus(`This browser would not let the page read the image (${error.message}).`, true);
    return;
  }
  worker.postMessage({ type: "run", id: state.run, width, height, rgba }, [rgba.buffer]);
  frame.classList.add("busy");
  setStatus("Finding the words");
}

function finish(data) {
  state.busy = false;
  if (state.pending) {
    process();
    return;
  }
  frame.classList.remove("busy");
  state.result = data;
  state.page = toCanvas(data.page);
  state.active = -1;
  renderText();
  draw();
  updateControls();
  const t = data.timing;
  const total = t.detect + t.read + (t.page || 0);
  const n = data.texts.length;
  if (!n) setStatus(`No words found (${seconds(total)} s). A sharper photo with the text facing the camera works best.`);
  else setStatus(`Found ${n} ${n === 1 ? "word" : "words"} in ${seconds(total)} s: finding ${seconds(t.detect)} s, reading ${seconds(t.read)} s, flattening ${seconds(t.page)} s.`);
}

function fail(data) {
  if (data.during === "load") {
    state.failed = true;
    setStatus(`The models could not be loaded (${data.message}). Reload the page to try again.`, true);
    return;
  }
  state.busy = false;
  if (state.pending) {
    process();
    return;
  }
  frame.classList.remove("busy");
  setStatus(`Something went wrong while reading this image (${data.message}).`, true);
}

function toCanvas({ width, height, rgba }) {
  const out = document.createElement("canvas");
  out.width = width;
  out.height = height;
  out.getContext("2d").putImageData(new ImageData(rgba, width, height), 0, 0);
  return out;
}

async function loadImage(src) {
  const img = new Image();
  img.decoding = "async";
  img.crossOrigin = "anonymous";
  img.src = src;
  try {
    await img.decode();
  } catch {
    setStatus("That file is not an image this browser can open. A JPEG or PNG photo works.", true);
    return;
  }
  const scale = Math.min(1, MAX_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
  const photo = document.createElement("canvas");
  photo.width = Math.max(1, Math.round(img.naturalWidth * scale));
  photo.height = Math.max(1, Math.round(img.naturalHeight * scale));
  const pctx = photo.getContext("2d", { willReadFrequently: true });
  pctx.fillStyle = "#ffffff";
  pctx.fillRect(0, 0, photo.width, photo.height);
  pctx.imageSmoothingQuality = "high";
  pctx.drawImage(img, 0, 0, photo.width, photo.height);
  state.photo = photo;
  state.page = null;
  state.result = null;
  state.active = -1;
  state.view = "photo";
  renderText();
  draw();
  updateControls();
  if (state.ready) process();
}

async function loadFile(file) {
  for (const button of examples) button.setAttribute("aria-pressed", "false");
  const url = URL.createObjectURL(file);
  await loadImage(url);
  URL.revokeObjectURL(url);
}

function currentQuads() {
  if (!state.result) return null;
  return state.view === "page" ? state.result.pageWords : state.result.quads;
}

function trace(quad) {
  ctx.beginPath();
  ctx.moveTo(quad[0][0], quad[0][1]);
  for (let i = 1; i < quad.length; i++) ctx.lineTo(quad[i][0], quad[i][1]);
  ctx.closePath();
}

function draw() {
  const base = state.view === "page" && state.page ? state.page : state.photo;
  if (!base) return;
  if (canvas.width !== base.width || canvas.height !== base.height) {
    canvas.width = base.width;
    canvas.height = base.height;
  }
  ctx.globalCompositeOperation = "source-over";
  ctx.drawImage(base, 0, 0);
  const quads = currentQuads();
  if (quads) {
    ctx.globalCompositeOperation = "multiply";
    quads.forEach((quad, i) => {
      ctx.fillStyle = i === state.active ? PINK : YELLOW;
      trace(quad);
      ctx.fill();
    });
    ctx.globalCompositeOperation = "source-over";
  }
  const what = state.view === "page" ? "Flattened page" : "Photo";
  canvas.setAttribute("aria-label", quads ? `${what} with ${quads.length} words highlighted` : what);
}

function updateControls() {
  for (const tab of tabs) {
    const selected = tab.dataset.view === state.view;
    tab.setAttribute("aria-selected", String(selected));
    tab.tabIndex = selected ? 0 : -1;
  }
  $("tab-page").disabled = !state.page;
  saveButton.hidden = !(state.view === "page" && state.page);
  copyButton.disabled = !(state.result && state.result.texts.length);
}

function setView(view) {
  if (view === state.view || (view === "page" && !state.page)) return;
  state.view = view;
  state.active = -1;
  placeTip();
  draw();
  updateControls();
}

function renderText() {
  textBox.textContent = "";
  textBox.scrollTop = 0;
  const result = state.result;
  if (!result || !result.texts.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = result ? "No words were found on this image." : "The words will appear here in reading order.";
    textBox.append(empty);
    return;
  }
  for (const line of result.lines) {
    const row = document.createElement("div");
    row.className = "line";
    line.forEach((k, j) => {
      if (j) row.append(" ");
      const word = document.createElement("span");
      word.className = result.scores[k] < UNSURE ? "w unsure" : "w";
      word.dataset.i = String(k);
      word.textContent = result.texts[k];
      if (result.scores[k] < UNSURE) word.title = `Read with ${percent(result.scores[k])}% confidence`;
      row.append(word);
    });
    textBox.append(row);
  }
}

function inside(quad, x, y) {
  let sign = 0;
  for (let i = 0; i < quad.length; i++) {
    const a = quad[i];
    const b = quad[(i + 1) % quad.length];
    const cross = (b[0] - a[0]) * (y - a[1]) - (b[1] - a[1]) * (x - a[0]);
    if (cross !== 0) {
      if (sign && Math.sign(cross) !== sign) return false;
      sign = Math.sign(cross);
    }
  }
  return true;
}

function area(quad) {
  let s = 0;
  for (let i = 0; i < quad.length; i++) {
    const a = quad[i];
    const b = quad[(i + 1) % quad.length];
    s += a[0] * b[1] - a[1] * b[0];
  }
  return Math.abs(s / 2);
}

function setActive(index, source) {
  if (index === state.active) return;
  state.active = index;
  draw();
  for (const word of textBox.querySelectorAll(".w.active")) word.classList.remove("active");
  if (index >= 0) {
    const word = textBox.querySelector(`.w[data-i="${index}"]`);
    if (word) {
      word.classList.add("active");
      if (source === "image") reveal(word);
    }
  }
  placeTip();
}

function reveal(word) {
  if (textBox.scrollHeight <= textBox.clientHeight) return;
  const top = word.offsetTop;
  if (top < textBox.scrollTop || top + word.offsetHeight > textBox.scrollTop + textBox.clientHeight) textBox.scrollTop = top - textBox.clientHeight / 2;
}

function placeTip() {
  const quads = currentQuads();
  if (!quads || state.active < 0) {
    tip.hidden = true;
    return;
  }
  const quad = quads[state.active];
  const sx = canvas.clientWidth / canvas.width;
  const sy = canvas.clientHeight / canvas.height;
  const xs = quad.map((p) => p[0] * sx);
  const ys = quad.map((p) => p[1] * sy);
  const score = state.result.scores[state.active];
  tip.textContent = "";
  const word = document.createElement("span");
  word.className = "tip-word";
  word.textContent = state.result.texts[state.active] || " ";
  const sure = document.createElement("span");
  sure.className = "tip-score";
  sure.textContent = `${percent(score)}%`;
  tip.append(word, sure);
  tip.hidden = false;
  const half = tip.offsetWidth / 2;
  const x = Math.min(Math.max((Math.min(...xs) + Math.max(...xs)) / 2, half), Math.max(canvas.clientWidth - half, half));
  const above = Math.min(...ys) - tip.offsetHeight - 8 >= 0;
  tip.style.left = `${x}px`;
  tip.style.top = `${above ? Math.min(...ys) - 8 : Math.max(...ys) + 8}px`;
  tip.classList.toggle("below", !above);
}

function pointAt(event) {
  const quads = currentQuads();
  if (!quads) return;
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) * canvas.width) / rect.width;
  const y = ((event.clientY - rect.top) * canvas.height) / rect.height;
  let hit = -1;
  let best = Infinity;
  quads.forEach((quad, i) => {
    if (inside(quad, x, y)) {
      const a = area(quad);
      if (a < best) {
        best = a;
        hit = i;
      }
    }
  });
  setActive(hit, "image");
}

canvas.addEventListener("pointermove", pointAt);
canvas.addEventListener("pointerdown", pointAt);
canvas.addEventListener("pointerleave", (event) => {
  if (event.pointerType === "mouse") setActive(-1);
});

textBox.addEventListener("pointerover", (event) => {
  const word = event.target.closest(".w");
  if (word) setActive(Number(word.dataset.i), "text");
});
textBox.addEventListener("pointerleave", () => setActive(-1));

for (const tab of tabs) {
  tab.addEventListener("click", () => setView(tab.dataset.view));
  tab.addEventListener("keydown", (event) => {
    if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
    const next = tabs[(tabs.indexOf(tab) + (event.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
    if (next.disabled) return;
    setView(next.dataset.view);
    next.focus();
  });
}

for (const button of examples) {
  button.addEventListener("click", () => {
    for (const other of examples) other.setAttribute("aria-pressed", String(other === button));
    loadImage(button.dataset.src);
  });
}

$("file").addEventListener("change", (event) => {
  const file = event.target.files[0];
  if (file) loadFile(file);
  event.target.value = "";
});

copyButton.addEventListener("click", async () => {
  if (!state.result) return;
  const text = state.result.text;
  let copied = false;
  try {
    await navigator.clipboard.writeText(text);
    copied = true;
  } catch {
    const field = document.createElement("textarea");
    field.value = text;
    field.setAttribute("readonly", "");
    field.style.position = "fixed";
    field.style.opacity = "0";
    document.body.append(field);
    field.select();
    copied = document.execCommand("copy");
    field.remove();
  }
  copyButton.textContent = copied ? "Copied" : "Copy failed";
  setTimeout(() => {
    copyButton.textContent = "Copy text";
  }, 1800);
});

saveButton.addEventListener("click", () => {
  if (!state.page) return;
  state.page.toBlob((blob) => {
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "flattened-page.png";
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(link.href), 4000);
  }, "image/png");
});

const hasFiles = (event) => event.dataTransfer && [...event.dataTransfer.types].includes("Files");
let depth = 0;
window.addEventListener("dragenter", (event) => {
  if (!hasFiles(event)) return;
  depth += 1;
  drop.hidden = false;
});
window.addEventListener("dragleave", (event) => {
  if (!hasFiles(event)) return;
  depth = Math.max(0, depth - 1);
  if (!depth) drop.hidden = true;
});
window.addEventListener("dragover", (event) => {
  if (hasFiles(event)) event.preventDefault();
});
window.addEventListener("drop", (event) => {
  if (!hasFiles(event)) return;
  event.preventDefault();
  depth = 0;
  drop.hidden = true;
  const files = [...event.dataTransfer.files];
  const file = files.find((f) => f.type.startsWith("image/")) || files[0];
  if (file) loadFile(file);
});
window.addEventListener("paste", (event) => {
  const files = [...((event.clipboardData && event.clipboardData.files) || [])];
  const file = files.find((f) => f.type.startsWith("image/"));
  if (file) loadFile(file);
});
window.addEventListener("resize", () => placeTip());

loadImage(examples[0].dataset.src);
