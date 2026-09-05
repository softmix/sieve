import {
  AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection, RawImage, env,
} from "./vendor/transformers.js";
import { Model, ZERO, K, l2, fit, holdout, usable, counts, MIN_PER_CLASS, SEEN_WEIGHT } from "./model.js";

const MODEL = "Xenova/clip-vit-base-patch32";

env.allowLocalModels = false;   // weights come from HF on first run, then live in the browser cache
env.backends.onnx.wasm.wasmPaths = browser.runtime.getURL("vendor/");
// Extension pages get no COOP/COEP, so no SharedArrayBuffer, so no ORT threads.
// Left at the default it tries anyway and warns on every single load.
env.backends.onnx.wasm.numThreads = 1;

// Tried in order, first one that loads *and* runs wins.
//
// WebGPU is only worth it batched, and dramatically so. Measured ms per image
// for the vision tower:
//
//              batch 1   batch 4   batch 16   batch 32
//   wasm/q8      125       113       111        112      <- compute-bound, flat
//   webgpu/fp16  101        25         6.3        6.3    <- 101ms/call, flat
//
// The WebGPU *call* costs ~101ms whether it carries 1 image or 16, so at batch 1
// it loses to wasm outright and at batch 16 it wins ~18x. Everything downstream
// batches for that reason; see embedTexts/embedImages. fp16 is about twice the
// download of q8, which is the price.
const BACKENDS = [
  { device: "webgpu", dtype: "fp16" },
  { device: "wasm", dtype: "q8" },
];

let engine, backend = "loading";
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

      backend = `${cfg.device}/${cfg.dtype}`;
      console.log(`sieve: clip ready on ${backend}`);
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

// Everything below embeds in batches, because on WebGPU one call costs the same
// whether it carries 1 image or 16: measured 101ms/call flat, i.e. 101ms/img at
// batch 1 but 6.3ms/img at batch 16. WASM is compute-bound and flat at ~112ms/img
// either way, so batching costs it nothing.

async function embedTexts(texts) {
  const out = new Array(texts.length);
  const { tok, txt } = await load();

  // CLIP's text tower stops at 77 tokens and forum posts routinely run longer,
  // so chunk and mean-pool rather than silently classify only the first sentence.
  // Every chunk of every post goes into one flat batch; `owner` maps back.
  const chunks = [], owner = [];
  texts.forEach((raw, i) => {
    const s = (raw || "").trim();
    if (!s) return void (out[i] = ZERO);
    const hit = cache.txt.get(s);
    if (hit) return void (out[i] = hit);
    const ids = tok(s).input_ids.tolist()[0].slice(1, -1);
    if (!ids.length) return void (out[i] = ZERO);
    for (let j = 0; j < ids.length; j += 75) {
      chunks.push(tok.decode(ids.slice(j, j + 75)));
      owner.push(i);
    }
  });
  if (!chunks.length) return out;

  const { text_embeds } = await txt(tok(chunks, { padding: true, truncation: true }));
  const sums = new Map();
  text_embeds.tolist().forEach((r, k) => {
    const acc = sums.get(owner[k]);
    if (acc) r.forEach((x, d) => (acc[d] += x)); else sums.set(owner[k], [...r]);
  });
  for (const [i, acc] of sums) out[i] = keep(cache.txt, (texts[i] || "").trim(), l2(acc));
  return out;
}

async function embedImages(srcs) {
  const out = new Array(srcs.length);
  const raws = [], need = [];

  // Fetches run concurrently even though inference doesn't -- they're the one
  // part of this that genuinely parallelises.
  const t = performance.now();
  await Promise.all(srcs.map(async (src, i) => {
    if (!src) return void (out[i] = ZERO);
    const hit = cache.img.get(src);
    if (hit) return void (out[i] = hit);
    try {
      raws[i] = await RawImage.read(src);
      need.push(i);
    } catch {
      out[i] = ZERO;   // a dead thumbnail shouldn't take the whole batch down
    }
  }));
  fetched += performance.now() - t;
  if (!need.length) return out;

  const { proc, vis } = await load();
  const { image_embeds } = await vis(await proc(need.map(i => raws[i])));
  const rows = image_embeds.tolist();
  need.forEach((i, k) => (out[i] = keep(cache.img, srcs[i], l2(rows[k]))));
  return out;
}

// One ORT session, one batch at a time. Without this a fast scroll fires several
// overlapping inferences and they fight over the same session.
let tail = Promise.resolve();
const embed = items => {
  const run = tail.then(async () => {
    const t = performance.now();
    const txts = await embedTexts(items.map(i => i.text));
    const imgs = await embedImages(items.map(i => i.img));
    const ms = (performance.now() - t) / items.length;
    return items.map((_, i) => ({ txt: txts[i], img: imgs[i], ms }));
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

// keyOf() builds `${url}\n${text}` and the text may itself contain newlines, so
// split once, yielding the image url. The post's own link is stored separately
// as `url` -- don't conflate them, spreading this over a label used to clobber it.
const splitKey = key => {
  const i = key.indexOf("\n");
  return { img: key.slice(0, i), text: key.slice(i + 1) };
};

// Bounded, because these accrue on their own: every scored post adds one, so a
// couple of catalog pages would otherwise put thousands through fit() on every
// click. Oldest out. Explicit labels are never pruned.
let seenMax = 300;
let threshold = 0.85;

function noteSeen(e, key, url) {
  if (taught.has(key) || seenKeys.has(key)) return;
  const seen = labels.filter(l => l.src === "seen");
  if (seen.length >= seenMax) {
    // Drops all excess in one pass, so lowering the setting takes effect on the
    // next scored post rather than decaying one at a time.
    const drop = new Set(seen.slice(0, seen.length - seenMax + 1));
    labels = labels.filter(l => !drop.has(l));
  }
  labels.push({ img: e.img, txt: e.txt, y: 0, w: SEEN_WEIGHT, src: "seen", key, url, ts: Date.now() });
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
  const s = await browser.storage.local.get({ labels: [], threshold: 0.85, seenMax: 300 });
  threshold = s.threshold;
  seenMax = s.seenMax;
  labels = s.labels.map(l => ({ ...l, img: toF32(l.img), txt: toF32(l.txt) }));
  reindex();
  if (labels.length) model = fit(labels);
  const c = counts(labels);
  console.log(`sieve: ${c.pos} hide / ${c.neg} keep (${c.taught} clicked) loaded,`
    + ` filtering ${usable(labels) ? "on" : "off"}`);
})();

browser.storage.onChanged.addListener(c => {
  if (c.threshold) threshold = c.threshold.newValue;
  if (c.seenMax) seenMax = c.seenMax.newValue;
});

browser.runtime.onMessage.addListener(async msg => {
  await booted;
  switch (msg.type) {
    case "score": {
      const out = new Array(msg.items.length);
      const todo = [];
      msg.items.forEach((it, i) => {
        const known = taught.get(keyOf(it.text, it.img));
        if (known !== undefined) out[i] = { p: known, ready: true, exact: true };
        else todo.push(i);
      });

      if (todo.length) {
        const t = performance.now();
        const es = await embed(todo.map(i => msg.items[i]));
        queued += performance.now() - t - es[0].ms * todo.length;

        todo.forEach((i, k) => {
          const e = es[k];
          spent += e.ms;
          const p = model.score(e.img, e.txt);
          // Only posts the model left alone. If it flagged one and you didn't
          // correct it, that's agreement -- recording a contradicting "fine"
          // would train against the very thing you asked it to catch.
          if (p <= threshold) noteSeen(e, keyOf(msg.items[i].text, msg.items[i].img), msg.items[i].url);
          // ready=false means the score isn't meaningful yet and nothing should
          // be hidden on the strength of it. See usable() in model.js.
          out[i] = { p, ready: usable(labels) };

          // Per-batch first: the cumulative average is dragged up for a long
          // time by the warmup batches and hides where throughput settled.
          if (++scored <= 5 || scored % 25 === 0)
            console.log(`sieve: ${scored} scored, ${e.ms | 0}ms/post in this batch of ${todo.length}`
              + ` (${(spent / scored) | 0}ms cumulative, ${(fetched / scored) | 0}ms of it fetch)`);
        });
      }
      return out;
    }
    case "label": {
      const [e] = await embed([{ text: msg.text, img: msg.img }]);
      const key = keyOf(msg.text, msg.img);
      // Replaces, not stacks -- including any weak "seen" entry for this post,
      // which a click supersedes outright.
      labels = labels.filter(l => l.key !== key);
      labels.push({ img: e.img, txt: e.txt, y: msg.y, src: msg.y ? "hide" : "keep", key, url: msg.url, ts: Date.now() });
      reindex();
      await commit();
      const c = counts(labels);
      console.log(`sieve: ${msg.y ? "hide" : "keep"} -> ${c.pos} hide / ${c.neg} keep (${c.taught} clicked)`);
      return { ...c, ready: usable(labels), need: MIN_PER_CLASS };
    }
    case "stats":
      return {
        ...counts(labels), ready: usable(labels), need: MIN_PER_CLASS,
        holdout: holdout(labels), backend,
      };

    // Uncertainty sampling: the posts the model is least sure about are the ones
    // where a label teaches it the most. They're drawn from the seen pool, which
    // already carries embeddings, so labelling one costs no inference at all.
    case "closeCalls": {
      return labels
        .filter(l => l.src === "seen")
        .map(l => ({ key: l.key, url: l.url, p: model.score(l.img, l.txt) }))
        .sort((a, b) => Math.abs(a.p - 0.5) - Math.abs(b.p - 0.5))
        .slice(0, msg.n ?? 12)
        .map(s => ({ ...s, ...splitKey(s.key) }));
    }

    // Promote a seen post to an explicit label in place. Reuses its stored
    // embeddings rather than re-fetching and re-embedding the post.
    case "relabel": {
      const l = labels.find(x => x.key === msg.key);
      if (!l) return { gone: true };
      Object.assign(l, { y: msg.y, w: 1, src: msg.y ? "hide" : "keep", ts: Date.now() });
      reindex();
      await commit();
      return { ...counts(labels), ready: usable(labels), need: MIN_PER_CLASS };
    }
    case "export":
      // Everything needed to rebuild the model elsewhere, weights and origin
      // included -- not just the embeddings.
      return labels.map(l => ({
        img: [...l.img], txt: [...l.txt], y: l.y, w: l.w ?? 1, src: l.src, key: l.key, url: l.url, ts: l.ts,
      }));

    // Restoring a backup, or carrying labels from the dev profile into your real
    // browser. Merges by key so importing twice is a no-op rather than a
    // duplicate, and imported entries win.
    case "import": {
      const ok = l => l && (l.y === 0 || l.y === 1)
        && l.img?.length === K && l.txt?.length === K;
      const good = (Array.isArray(msg.labels) ? msg.labels : []).filter(ok);
      if (!good.length) return { added: 0, skipped: msg.labels?.length ?? 0 };

      const incoming = good.map(l => ({
        ...l, img: toF32(l.img), txt: toF32(l.txt), w: l.w ?? 1, ts: l.ts ?? Date.now(),
      }));
      const keys = new Set(incoming.map(l => l.key).filter(Boolean));
      labels = [...labels.filter(l => !l.key || !keys.has(l.key)), ...incoming];
      reindex();
      await commit();
      console.log(`sieve: imported ${incoming.length}`);
      return { added: incoming.length, skipped: (msg.labels?.length ?? 0) - incoming.length, ...counts(labels) };
    }

    case "reset":
      labels = [];
      model = new Model();
      reindex();   // or exact recall keeps hiding posts that no longer have labels
      await browser.storage.local.set({ labels });
      return { ok: true };
  }
});

// Both directions need to be one click. Positives alone can't train this model,
// and in practice you spot false positives more often than you spot new hides.
const MENU = { "sieve-hide": "sieve: hide posts like this", "sieve-keep": "sieve: this post is fine" };
for (const [id, title] of Object.entries(MENU))
  browser.contextMenus.create({ id, title, contexts: ["page", "selection", "image", "link"] });

browser.contextMenus.onClicked.addListener((info, tab) =>
  browser.tabs.sendMessage(tab.id, { type: "teach", y: info.menuItemId === "sieve-hide" ? 1 : 0 }));

// No popup on the browser action, so clicking it fires this. about:addons ->
// Preferences is too many steps for something you check while labelling.
browser.browserAction.onClicked.addListener(() => browser.runtime.openOptionsPage());
