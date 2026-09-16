import {
  AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection, RawImage, env,
} from "./vendor/transformers.js";
import { Model, ZERO, K, l2, fit, holdout, usable, counts, MIN_PER_CLASS, SEEN_WEIGHT } from "./model.js";

const MODEL = "Xenova/clip-vit-base-patch32";

env.allowLocalModels = false;   // weights come from HF on first run, then live in the browser cache
env.backends.onnx.wasm.wasmPaths = browser.runtime.getURL("vendor/");
// No COOP/COEP on extension pages, so no SharedArrayBuffer and no ORT threads.
env.backends.onnx.wasm.numThreads = 1;

// One backend, no fallback, deliberately. A label stores the embedding, not the
// post, so a vector is only comparable to vectors from the same weights on the
// same device -- measured, fp32 and q8 place the *same image* at cosine 0.86.
// A model fit on one geometry and fed the other is confidently wrong rather than
// merely worse, so falling back would corrupt the label store, not degrade it.
//
// ponytail: no WebGPU means no sieve at all. The way out, if that ever bites, is
// dtype fp32 everywhere (606 MB, runs on wasm too) plus re-embedding the labels.
const BACKEND = { device: "webgpu", dtype: "fp16" };

// Stamped on every label, so a future change to the line above is visible in the
// data instead of silently rewriting what the old vectors mean.
const EV = `${BACKEND.device}/${BACKEND.dtype}`;

let engine, backend = "loading";
const load = () => (engine ??= (async () => {
  const at = {};   // one line per decile per file, not one per chunk
  const progress_callback = x => {
    const d = (x.progress / 10) | 0;
    if (x.status === "progress" && at[x.file] !== d) console.log(`sieve: ${(at[x.file] = d) * 10}% ${x.file}`);
  };
  try {
    if (!navigator.gpu) throw new Error("no WebGPU in this browser");
    const [tok, proc] = await Promise.all([
      AutoTokenizer.from_pretrained(MODEL),
      AutoProcessor.from_pretrained(MODEL),
    ]);
    const [txt, vis] = await Promise.all([
      CLIPTextModelWithProjection.from_pretrained(MODEL, { ...BACKEND, progress_callback }),
      CLIPVisionModelWithProjection.from_pretrained(MODEL, { ...BACKEND, progress_callback }),
    ]);
    // WebGPU fails on first inference, not at construction, so exercise it here
    // to make a broken device show up at startup rather than mid-page.
    await vis(await proc(new RawImage(new Uint8ClampedArray(224 * 224 * 3), 224, 224, 3)));
    await txt(tok(["warmup"], { padding: true, truncation: true }));

    backend = EV;
    console.log(`sieve: clip ready on ${backend}`);
    return { tok, proc, txt, vis };
  } catch (e) {
    // Reported, not swallowed. This `try` covers a 303 MB download as well as
    // the device probe, and a dropped fetch used to land in the same `catch` as
    // a dead GPU -- which is how a network blip could silently switch geometry.
    backend = `unavailable — ${e.message}`;
    // Retry the next time something needs embedding -- a failed download or a
    // lost device is worth another go. Only reaches here for failures after the
    // first await; the synchronous no-WebGPU throw lands before `engine ??=`
    // assigns, so that one stays rejected, which is what you want on a machine
    // that has no GPU to find.
    engine = null;
    console.error(`sieve: ${backend}`);
    throw e;
  }
})());

// Persistent background page (MV2), so these survive navigation and tab close.
const cache = { img: new Map(), txt: new Map() };
const keep = (m, k, v) => {
  if (m.size > 3000) m.clear();   // ponytail: whole-cache dump, LRU if it ever shows in a profile
  m.set(k, v);
  return v;
};

async function embedTexts(texts) {
  const out = new Array(texts.length);
  const { tok, txt } = await load();

  // CLIP's text tower caps at 77 tokens and posts run longer, so chunk and
  // mean-pool. Every chunk of every post shares one batch; `owner` maps back.
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

  // Fetches parallelise even though inference doesn't.
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

// One ORT session, so one batch at a time.
let tail = Promise.resolve();

// A WebGPU device lost after startup -- sleep/wake, driver reset -- keeps the
// session object alive and fails every OrtRun from then on ("Buffer unmapped"
// on the output readback). The load-time probe can't see that, so drop the
// engine and run again: load() re-probes, and falls through to wasm if the
// device is really gone. Only ever one rebuild per working spell, so a
// permanently broken device fails fast instead of reloading on every batch.
let mayRebuild = true;
const embed = items => {
  const once = async () => {
    const t = performance.now();
    const txts = await embedTexts(items.map(i => i.text));
    const imgs = await embedImages(items.map(i => i.img));
    const ms = (performance.now() - t) / items.length;
    return items.map((_, i) => ({ txt: txts[i], img: imgs[i], ms }));
  };
  const run = tail.then(async () => {
    try {
      const out = await once();
      mayRebuild = true;
      return out;
    } catch (e) {
      if (!mayRebuild) throw e;
      mayRebuild = false;
      console.warn(`sieve: inference failed on ${backend} (${e.message}), rebuilding`);
      engine = null;
      backend = "loading";
      return once();
    }
  });
  tail = run.catch(() => {});
  return run;
};

// storage.local round-trips typed arrays differently per backend.
const toF32 = v => v instanceof Float32Array ? v
  : Float32Array.from(Array.isArray(v) ? v : Object.values(v));

let model = new Model();
let labels = [];
let scored = 0, spent = 0, queued = 0, fetched = 0;

// Exact recall, in front of the model: an explicitly marked post is a stored
// fact, so it stays hidden regardless of what the model currently thinks. Only
// explicit labels go in `taught` -- a merely-seen post must stay scoreable.
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

// Splits a key back into its parts. Yields the *image* url; a label's own `url`
// field is the post permalink, which is a different thing.
const splitKey = key => {
  const i = key.indexOf("\n");
  return { img: key.slice(0, i), text: key.slice(i + 1) };
};

// Vectors only mean anything against others from the same backend, so more than
// one entry here says part of the set was embedded elsewhere and is quietly
// wrong. "unknown" is a label from before stamping; it is almost certainly
// webgpu/fp16, but nothing recorded it, so it can't claim to be.
const evTally = () => labels.reduce((m, l) => ((m[l.ev ?? "unknown"] = (m[l.ev ?? "unknown"] ?? 0) + 1), m), {});
const evLine = ev => Object.entries(ev).map(([k, n]) => `${n} ${k}`).join(" + ");

// Bounded: every scored post adds one, and they all go through fit() on each
// click. Oldest out. Explicit labels are never pruned.
let seenMax = 300;
let threshold = 0.85;

function noteSeen(e, key, url) {
  if (taught.has(key) || seenKeys.has(key)) return;
  const seen = labels.filter(l => l.src === "seen");
  if (seen.length >= seenMax) {
    // All excess in one pass, so lowering the setting takes effect immediately.
    const drop = new Set(seen.slice(0, seen.length - seenMax + 1));
    labels = labels.filter(l => !drop.has(l));
  }
  labels.push({ img: e.img, txt: e.txt, y: 0, w: SEEN_WEIGHT, src: "seen", key, url, ev: EV, ts: Date.now() });
  seenKeys.add(key);
  soon();
}

// A hide leaves no label behind -- it's the model's own call, and feeding it back
// as evidence would just confirm whatever it already believes. So keep the last
// few in a ring for the options page to second-guess. Newest first, in memory
// only: browsing refills it, and nothing here is worth a storage write.
const RECENT_MAX = 60;
let recent = [];

function noteHidden(e, key, url) {
  if (taught.has(key) || recent.some(r => r.key === key)) return;
  recent.unshift({ img: e.img, txt: e.txt, key, url, ev: EV, ts: Date.now() });
  recent.length = Math.min(recent.length, RECENT_MAX);
  // The only window onto this list: it's memory-only, and "why is nothing in
  // recently hidden" is otherwise unanswerable without a debugger.
  console.log(`sieve: hid ${recent.length} so far, latest ${url || key.split("\n")[0] || "(no url)"}`);
}

// Implicit labels settle in batches; explicit clicks commit immediately.
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
  const ev = evTally();
  if (Object.keys(ev).length > 1)
    console.warn(`sieve: labels span ${evLine(ev)} — vectors from different backends are not comparable`);
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
          const key = keyOf(msg.items[i].text, msg.items[i].img);
          const ready = usable(labels);
          // Only posts the model left alone become labels: if it flagged one and
          // you didn't correct it, a contradicting "fine" would train against
          // the catch. The flagged ones go in the ring instead.
          if (p <= threshold) noteSeen(e, key, msg.items[i].url);
          else if (ready) noteHidden(e, key, msg.items[i].url);
          out[i] = { p, ready };

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
      // Replaces rather than stacks, including any weak "seen" entry.
      labels = labels.filter(l => l.key !== key);
      labels.push({ img: e.img, txt: e.txt, y: msg.y, src: msg.y ? "hide" : "keep", key, url: msg.url, ev: EV, ts: Date.now() });
      reindex();
      await commit();
      const c = counts(labels);
      console.log(`sieve: ${msg.y ? "hide" : "keep"} -> ${c.pos} hide / ${c.neg} keep (${c.taught} clicked)`);
      return { ...c, ready: usable(labels), need: MIN_PER_CLASS };
    }
    case "stats":
      return {
        ...counts(labels), ready: usable(labels), need: MIN_PER_CLASS,
        holdout: holdout(labels), backend, evs: evTally(),
      };

    // Uncertainty sampling: whichever posts sit nearest 0.5. Drawn from the seen
    // pool, so they already carry embeddings and cost no inference to label.
    case "closeCalls": {
      return labels
        .filter(l => l.src === "seen")
        .map(l => ({ key: l.key, url: l.url, p: model.score(l.img, l.txt) }))
        .sort((a, b) => Math.abs(a.p - 0.5) - Math.abs(b.p - 0.5))
        .slice(0, msg.n ?? 12)
        .map(s => ({ ...s, ...splitKey(s.key) }));
    }

    // Scored now rather than at hide time, so the list reflects what the model
    // thinks after whatever you've already corrected.
    case "recentHidden":
      return recent
        .filter(r => !taught.has(r.key))
        .slice(0, msg.n ?? 12)
        .map(r => ({ key: r.key, url: r.url, p: model.score(r.img, r.txt), ...splitKey(r.key) }));

    // Promote a post in place, reusing its stored embeddings.
    case "relabel": {
      // A close call is already a label; a hidden post is only in the ring, so
      // it has to be added rather than amended.
      let l = labels.find(x => x.key === msg.key);
      if (!l) {
        l = recent.find(x => x.key === msg.key);
        if (!l) return { gone: true };
        labels.push(l);
      }
      Object.assign(l, { y: msg.y, w: 1, src: msg.y ? "hide" : "keep", ts: Date.now() });
      reindex();
      await commit();
      return { ...counts(labels), ready: usable(labels), need: MIN_PER_CLASS };
    }
    case "export":
      // Everything needed to rebuild the model elsewhere.
      return labels.map(l => ({
        img: [...l.img], txt: [...l.txt], y: l.y, w: l.w ?? 1, src: l.src, key: l.key, url: l.url, ev: l.ev, ts: l.ts,
      }));

    // Merges by key, so importing the same file twice is a no-op.
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
      recent = [];
      model = new Model();
      reindex();   // else exact recall keeps hiding posts whose labels are gone
      await browser.storage.local.set({ labels });
      return { ok: true };
  }
});

const MENU = { "sieve-hide": "sieve: hide posts like this", "sieve-keep": "sieve: this post is fine" };
for (const [id, title] of Object.entries(MENU))
  browser.contextMenus.create({ id, title, contexts: ["page", "selection", "image", "link"] });

browser.contextMenus.onClicked.addListener((info, tab) =>
  browser.tabs.sendMessage(tab.id, { type: "teach", y: info.menuItemId === "sieve-hide" ? 1 : 0 }));

// No popup on the browser action, so clicking it fires this.
browser.browserAction.onClicked.addListener(() => browser.runtime.openOptionsPage());
