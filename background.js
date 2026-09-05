import {
  AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection, RawImage, env,
} from "./vendor/transformers.js";
import { Model, ZERO, l2, fit, holdout, usable, counts, MIN_PER_CLASS } from "./model.js";

const MODEL = "Xenova/clip-vit-base-patch32";

env.allowLocalModels = false;   // weights come from HF on first run, then live in the browser cache
env.backends.onnx.wasm.wasmPaths = browser.runtime.getURL("vendor/");
// Extension pages get no COOP/COEP, so no SharedArrayBuffer, so no ORT threads.
// Left at the default it tries anyway and warns on every single load.
env.backends.onnx.wasm.numThreads = 1;

let engine;
const load = () => (engine ??= (async () => {
  const at = {};   // one line per decile per file, not one per chunk
  const progress_callback = x => {
    const d = (x.progress / 10) | 0;
    if (x.status === "progress" && at[x.file] !== d) console.log(`sieve: ${(at[x.file] = d) * 10}% ${x.file}`);
  };
  const [tok, proc, txt, vis] = await Promise.all([
    AutoTokenizer.from_pretrained(MODEL),
    AutoProcessor.from_pretrained(MODEL),
    CLIPTextModelWithProjection.from_pretrained(MODEL, { dtype: "q8", progress_callback }),
    CLIPVisionModelWithProjection.from_pretrained(MODEL, { dtype: "q8", progress_callback }),
  ]);
  console.log("sieve: clip ready");
  return { tok, proc, txt, vis };
})());

// The background page is persistent (MV2), so these survive navigation and even
// tab close -- which matters a lot on an imageboard, where the same file is
// reposted across threads all day.
const cache = { img: new Map(), txt: new Map() };
const keep = (m, k, v) => {
  if (m.size > 3000) m.clear();   // ponytail: whole-cache dump, LRU if it ever shows in a profile
  m.set(k, v);
  return v;
};

async function embedText(s) {
  s = (s || "").trim();
  if (!s) return ZERO;
  if (cache.txt.has(s)) return cache.txt.get(s);
  const { tok, txt } = await load();
  // CLIP's text tower stops at 77 tokens and forum posts routinely run longer,
  // so chunk and mean-pool rather than silently classify only the first sentence.
  const ids = tok(s).input_ids.tolist()[0].slice(1, -1);
  const parts = [];
  for (let i = 0; i < ids.length; i += 75) parts.push(tok.decode(ids.slice(i, i + 75)));
  const { text_embeds } = await txt(tok(parts, { padding: true, truncation: true }));
  return keep(cache.txt, s, l2(text_embeds.tolist().reduce((a, r) => a.map((x, i) => x + r[i]))));
}

async function embedImage(src) {
  if (!src) return ZERO;
  if (cache.img.has(src)) return cache.img.get(src);
  const { proc, vis } = await load();
  const { image_embeds } = await vis(await proc(await RawImage.read(src)));
  return keep(cache.img, src, l2(image_embeds.tolist()[0]));
}

// One ORT session, one caller at a time. Without this a fast scroll fires a
// dozen overlapping inferences and they fight over the same session.
let tail = Promise.resolve();
const embed = (text, img) => {
  const run = tail.then(async () => ({ txt: await embedText(text), img: await embedImage(img) }));
  tail = run.catch(() => {});
  return run;
};

// storage.local round-trips typed arrays differently depending on backend, so
// normalise rather than bet on which shape comes back.
const toF32 = v => v instanceof Float32Array ? v
  : Float32Array.from(Array.isArray(v) ? v : Object.values(v));

let model = new Model();
let labels = [];
let scored = 0, spent = 0;

// Exact recall, in front of the model. A post you explicitly marked is a stored
// fact, not a prediction -- it must stay hidden on reload even while the model
// is still warming up and even if the model would score it 0.2. Generalising to
// *other* posts is the model's job; remembering this one isn't.
const keyOf = (text, img) => `${img || ""}\n${(text || "").trim().slice(0, 200)}`;
let taught = new Map();
const reindex = () => (taught = new Map(labels.filter(l => l.key).map(l => [l.key, l.y])));

const booted = (async () => {
  const s = await browser.storage.local.get({ labels: [] });
  labels = s.labels.map(l => ({ ...l, img: toF32(l.img), txt: toF32(l.txt) }));
  reindex();
  if (labels.length) model = fit(labels);
  const c = counts(labels);
  console.log(`sieve: ${c.pos} hide / ${c.neg} keep loaded, filtering ${usable(labels) ? "on" : "off"}`);
})();

browser.runtime.onMessage.addListener(async msg => {
  await booted;
  switch (msg.type) {
    case "score": {
      const known = taught.get(keyOf(msg.text, msg.img));
      if (known !== undefined) return { p: known, ready: true, exact: true };

      const t = performance.now();
      const e = await embed(msg.text, msg.img);
      spent += performance.now() - t;
      // Inference is serialised, so this average is also the throughput ceiling.
      if (++scored % 25 === 1) console.log(`sieve: ${scored} scored, ${(spent / scored) | 0}ms avg`);
      // ready=false means the score is not yet meaningful and nothing should be
      // hidden on the strength of it. See usable() in model.js.
      return { p: model.score(e.img, e.txt), ready: usable(labels) };
    }
    case "label": {
      const e = await embed(msg.text, msg.img);
      const key = keyOf(msg.text, msg.img);
      labels = labels.filter(l => l.key !== key);   // re-labelling replaces, not stacks
      labels.push({ img: e.img, txt: e.txt, y: msg.y, key, ts: Date.now() });
      reindex();
      // 1537 params: refitting the whole log is milliseconds and avoids the
      // recency drift you get from applying single gradient steps forever.
      model = fit(labels);
      await browser.storage.local.set({ labels });
      const c = counts(labels);
      console.log(`sieve: label y=${msg.y} -> ${c.pos} hide / ${c.neg} keep`);
      return { ...c, ready: usable(labels), need: MIN_PER_CLASS };
    }
    case "stats":
      return { ...counts(labels), ready: usable(labels), need: MIN_PER_CLASS, holdout: holdout(labels) };
    case "export":
      return labels.map(l => ({ img: [...l.img], txt: [...l.txt], y: l.y, ts: l.ts }));
    case "reset":
      labels = [];
      model = new Model();
      await browser.storage.local.set({ labels });
      return { n: 0 };
  }
});

// Both directions need to be one click. Positives alone can't train this model,
// and in practice you spot false positives more often than you spot new hides.
const MENU = { "sieve-hide": "sieve: hide posts like this", "sieve-keep": "sieve: this post is fine" };
for (const [id, title] of Object.entries(MENU))
  browser.contextMenus.create({ id, title, contexts: ["page", "selection", "image", "link"] });

browser.contextMenus.onClicked.addListener((info, tab) =>
  browser.tabs.sendMessage(tab.id, { type: "teach", y: info.menuItemId === "sieve-hide" ? 1 : 0 }));
