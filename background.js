import {
  AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection, RawImage, env,
} from "./vendor/transformers.js";
import { Model, ZERO, l2, fit, holdout, usable, counts, MIN_PER_CLASS, SEEN_WEIGHT } from "./model.js";

const MODEL = "Xenova/clip-vit-base-patch32";

env.allowLocalModels = false;   // weights come from HF on first run, then live in the browser cache
env.backends.onnx.wasm.wasmPaths = browser.runtime.getURL("vendor/");
// Extension pages get no COOP/COEP, so no SharedArrayBuffer, so no ORT threads.
// Left at the default it tries anyway and warns on every single load.
env.backends.onnx.wasm.numThreads = 1;

// Tried in order, first one that loads *and* runs wins.
//
// wasm is first because it measured faster, which was not the expectation:
// 254-335ms per post against WebGPU's 358-563ms, at half the download (q8 vs the
// fp16 WebGPU needs). ViT-B/32 at batch-of-1 is dispatch-bound, so the GPU
// round-trip costs more than the work it saves. Flip these if batching ever
// lands -- that's the regime where WebGPU should win.
const BACKENDS = [
  { device: "wasm", dtype: "q8" },
  { device: "webgpu", dtype: "fp16" },
];

let engine;
const load = () => (engine ??= (async () => {
  const at = {};   // one line per decile per file, not one per chunk
  const progress_callback = x => {
    const d = (x.progress / 10) | 0;
    if (x.status === "progress" && at[x.file] !== d) console.log(`sieve: ${(at[x.file] = d) * 10}% ${x.file}`);
  };
  const [tok, proc] = await Promise.all([
    AutoTokenizer.from_pretrained(MODEL),
    AutoProcessor.from_pretrained(MODEL),
  ]);

  for (const cfg of BACKENDS) {
    if (cfg.device === "webgpu" && !navigator.gpu) continue;
    try {
      const [txt, vis] = await Promise.all([
        CLIPTextModelWithProjection.from_pretrained(MODEL, { ...cfg, progress_callback }),
        CLIPVisionModelWithProjection.from_pretrained(MODEL, { ...cfg, progress_callback }),
      ]);
      // Loading successfully doesn't prove the backend can run -- WebGPU tends
      // to fail on the first actual inference, not at construction. Warming up
      // here moves the fallback to startup instead of the first post, and eats
      // the session-init latency spike while we're at it.
      await vis(await proc(new RawImage(new Uint8ClampedArray(224 * 224 * 3), 224, 224, 3)));
      await txt(tok(["warmup"], { padding: true, truncation: true }));

      console.log(`sieve: clip ready on ${cfg.device}/${cfg.dtype}`);
      return { tok, proc, txt, vis };
    } catch (e) {
      console.warn(`sieve: ${cfg.device}/${cfg.dtype} unusable (${e.message}), trying next`);
    }
  }
  throw new Error("sieve: no working inference backend");
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
  // Timed separately because it's a network fetch plus a decode, and on a fast
  // backend it can easily cost more than the inference it feeds.
  const t = performance.now();
  const raw = await RawImage.read(src);
  fetched += performance.now() - t;
  const { image_embeds } = await vis(await proc(raw));
  return keep(cache.img, src, l2(image_embeds.tolist()[0]));
}

// One ORT session, one caller at a time. Without this a fast scroll fires a
// dozen overlapping inferences and they fight over the same session.
let tail = Promise.resolve();
const embed = (text, img) => {
  const run = tail.then(async () => {
    // Timed in here, not at the call site: a catalog page makes ~200 posts
    // visible at once and they all queue behind this lock, so measuring from the
    // caller reports mostly other posts' work.
    const t = performance.now();
    const out = { txt: await embedText(text), img: await embedImage(img) };
    out.ms = performance.now() - t;
    return out;
  });
  tail = run.catch(() => {});
  return run;
};

// storage.local round-trips typed arrays differently depending on backend, so
// normalise rather than bet on which shape comes back.
const toF32 = v => v instanceof Float32Array ? v
  : Float32Array.from(Array.isArray(v) ? v : Object.values(v));

let model = new Model();
let labels = [];
let scored = 0, spent = 0, queued = 0, fetched = 0;

// Exact recall, in front of the model. A post you explicitly marked is a stored
// fact, not a prediction -- it must stay hidden on reload even while the model
// is still warming up and even if the model would score it 0.2. Generalising to
// *other* posts is the model's job; remembering this one isn't.
//
// Only explicit labels go in here. A merely-seen post must stay scoreable, or it
// would freeze at whatever the model thought the first time it scrolled past.
const keyOf = (text, img) => `${img || ""}\n${(text || "").trim().slice(0, 200)}`;
let taught = new Map(), seenKeys = new Set();

function reindex() {
  taught = new Map();
  seenKeys = new Set();
  for (const l of labels) {
    if (!l.key) continue;
    if (l.src === "seen") seenKeys.add(l.key); else taught.set(l.key, l.y);
  }
}

// Bounded, because these accrue on their own: every scored post adds one, so a
// couple of catalog pages would otherwise put thousands through fit() on every
// click. Oldest out.
const SEEN_MAX = 300;
let threshold = 0.85;

function noteSeen(e, key) {
  if (taught.has(key) || seenKeys.has(key)) return;
  const seen = labels.filter(l => l.src === "seen");
  if (seen.length >= SEEN_MAX) {
    const drop = new Set(seen.slice(0, seen.length - SEEN_MAX + 1));
    labels = labels.filter(l => !drop.has(l));
  }
  labels.push({ img: e.img, txt: e.txt, y: 0, w: SEEN_WEIGHT, src: "seen", key, ts: Date.now() });
  seenKeys.add(key);
  soon();
}

// Refitting and writing storage on every scored post would be constant churn, so
// implicit labels settle in batches. Explicit clicks still commit immediately.
let timer = null;
const soon = () => {
  clearTimeout(timer);
  timer = setTimeout(commit, 5000);
};
async function commit() {
  clearTimeout(timer);
  model = fit(labels);
  await browser.storage.local.set({ labels });
}

const booted = (async () => {
  const s = await browser.storage.local.get({ labels: [], threshold: 0.85 });
  threshold = s.threshold;
  labels = s.labels.map(l => ({ ...l, img: toF32(l.img), txt: toF32(l.txt) }));
  reindex();
  if (labels.length) model = fit(labels);
  const c = counts(labels);
  console.log(`sieve: ${c.pos} hide / ${c.neg} keep (${c.taught} clicked) loaded,`
    + ` filtering ${usable(labels) ? "on" : "off"}`);
})();

browser.storage.onChanged.addListener(c => c.threshold && (threshold = c.threshold.newValue));

browser.runtime.onMessage.addListener(async msg => {
  await booted;
  switch (msg.type) {
    case "score": {
      const known = taught.get(keyOf(msg.text, msg.img));
      if (known !== undefined) return { p: known, ready: true, exact: true };

      const t = performance.now();
      const e = await embed(msg.text, msg.img);
      spent += e.ms;
      queued += performance.now() - t - e.ms;
      // Chatty for the first few: call 1 carries ONNX session warmup and is
      // wildly unrepresentative of steady state.
      if (++scored <= 5 || scored % 25 === 0)
        console.log(`sieve: ${scored} scored, ${(spent / scored) | 0}ms avg`
          + ` = ${(fetched / scored) | 0}ms image fetch + ${((spent - fetched) / scored) | 0}ms inference`
          + ` (queued ${(queued / scored) | 0}ms)`);
      // ready=false means the score is not yet meaningful and nothing should be
      // hidden on the strength of it. See usable() in model.js.
      const p = model.score(e.img, e.txt);
      // Only posts the model left alone. If it flagged one and you didn't
      // correct it, that's agreement -- recording a contradicting "fine" would
      // train against the very thing you asked it to catch.
      if (p <= threshold) noteSeen(e, keyOf(msg.text, msg.img));
      return { p, ready: usable(labels) };
    }
    case "label": {
      const e = await embed(msg.text, msg.img);
      const key = keyOf(msg.text, msg.img);
      // Replaces, not stacks -- including any weak "seen" entry for this post,
      // which a click supersedes outright.
      labels = labels.filter(l => l.key !== key);
      labels.push({ img: e.img, txt: e.txt, y: msg.y, src: msg.y ? "hide" : "keep", key, ts: Date.now() });
      reindex();
      await commit();
      const c = counts(labels);
      console.log(`sieve: ${msg.y ? "hide" : "keep"} -> ${c.pos} hide / ${c.neg} keep (${c.taught} clicked)`);
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
