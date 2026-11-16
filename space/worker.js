import { Engine, imageFromRGBA, imageToRGBA } from "./engine.js";

let engine = null;

async function download(url, report) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`downloading ${url.split("/").pop()} failed with HTTP ${response.status}`);
  const reader = response.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    report(loaded);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  return bytes;
}

async function load({ detector, reader }) {
  let done = 0;
  const files = [];
  for (const url of [detector, reader]) {
    const before = done;
    const bytes = await download(url, (loaded) => self.postMessage({ type: "progress", loaded: before + loaded }));
    done += bytes.length;
    files.push(bytes);
  }
  self.postMessage({ type: "starting" });
  engine = await Engine.load(files[0], files[1]);
  self.postMessage({ type: "ready" });
}

async function run({ id, width, height, rgba }) {
  const image = imageFromRGBA(width, height, rgba);
  const out = await engine.run(image, { onStage: (stage, count) => self.postMessage({ type: "stage", id, stage, count }) });
  const page = imageToRGBA(out.page);
  self.postMessage(
    {
      type: "result",
      id,
      quads: out.quads,
      texts: out.texts,
      scores: out.scores,
      lines: out.lines,
      text: out.text,
      pageQuad: out.pageQuad,
      pageWords: out.pageWords,
      page: { width: out.page.w, height: out.page.h, rgba: page },
      timing: out.timing,
    },
    [page.buffer],
  );
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") await load(data);
    else if (data.type === "run") await run(data);
  } catch (error) {
    self.postMessage({ type: "error", id: data.id, during: data.type, message: error && error.message ? error.message : String(error) });
  }
};
