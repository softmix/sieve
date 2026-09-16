import {
  AutoTokenizer, AutoProcessor, CLIPTextModelWithProjection,
  CLIPVisionModelWithProjection, RawImage, env,
} from "./vendor/transformers.js";
import {
  Model, Ambient, ZERO, K, l2, toF32, feats, fit, holdout, usable, counts, score, identOf,
  mapVectors, placeNew, hasMode, MODES,
  MIN_PER_CLASS, MIN_AMBIENT, REFIT_EVERY, PANIC_RATE, PANIC_WINDOW,
} from "./model.js";

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
  const bytes = new Array(srcs.length);
  const raws = [], need = [];

  // Fetches parallelise even though inference doesn't.
  const t = performance.now();
  await Promise.all(srcs.map(async (src, i) => {
    if (!src) return void (out[i] = ZERO);
    const hit = cache.img.get(src);
    if (hit) return void (out[i] = hit);
    try {
      // fetch + fromBlob rather than RawImage.read, which does exactly this and
      // then drops the encoded bytes. The archive wants them: 4chan deletes a
      // thread's images within days, and a training map full of dead thumbnails
      // is a training map you can't use. One fetch feeds both.
      const blob = await (await fetch(src)).blob();
      bytes[i] = new Uint8Array(await blob.arrayBuffer());
      raws[i] = await RawImage.fromBlob(blob);
      need.push(i);
    } catch {
      out[i] = ZERO;   // a dead thumbnail shouldn't take the whole batch down
    }
  }));
  fetched += performance.now() - t;
  if (!need.length) return { out, bytes };

  const { proc, vis } = await load();
  const { image_embeds } = await vis(await proc(need.map(i => raws[i])));
  const rows = image_embeds.tolist();
  need.forEach((i, k) => (out[i] = keep(cache.img, srcs[i], l2(rows[k]))));
  return { out, bytes };
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
    const { out: imgs, bytes } = await embedImages(items.map(i => i.img));
    const ms = (performance.now() - t) / items.length;
    return items.map((_, i) => ({ txt: txts[i], img: imgs[i], thumb: bytes[i], ms }));
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

let model = new Model();
let ambient = new Ambient();
let labels = [];
let scored = 0, spent = 0, queued = 0, fetched = 0;

// Exact recall, in front of the model: an explicitly marked post is a stored
// fact, so it stays hidden regardless of what the model currently thinks.
const keyOf = (text, img) => `${img || ""}\n${(text || "").trim().slice(0, 200)}`;
let taught = new Map(), taughtIds = new Map();

function reindex() {
  taught = new Map();
  taughtIds = new Map();
  for (const l of labels) {
    if (l.key) taught.set(l.key, l.y);
    // Archive identity too, so a decided post can be recognised in the archive
    // without matching its truncated text against a key built from full text.
    const it = identOf(l.url);
    if (it) taughtIds.set(it.id, l.y);
  }
}

// ---- the archive ---------------------------------------------------------
//
// Every post seen, hidden or not, kept so the map has something to draw. It is
// deliberately *not* a training set -- fit() never reads it. Feeding a hide back
// as evidence would only confirm what the model already believes, and a store
// the fitter cannot see makes that structural instead of a rule to remember.
//
// Eviction is cheap for the same reason the old seen-pool's wasn't: the learning
// was banked into `ambient` at insert time, so losing a record costs the ability
// to look at it and nothing else.
//
// Vectors and thumbnails get their own storage keys so an insert is an O(1)
// write. Only the small index is rewritten, and that's debounced.
let arc = [];                      // index: metadata only, no vectors
let arcById = new Map();
let arcNextId = 1;
let unwritten = new Map();         // id -> {img, txt, thumb} not yet persisted

// Rolling window behind usable()'s panic guard. A model hiding essentially the
// whole page is broken rather than strict, and without this that state is
// absorbing -- see the comment on PANIC_RATE.
let hideRing = [], hideCount = 0;
const noteOutcome = hid => {
  hideRing.push(hid);
  if (hid) hideCount++;
  if (hideRing.length > PANIC_WINDOW && hideRing.shift()) hideCount--;
};
const hideRate = () => (hideRing.length >= PANIC_WINDOW ? hideCount / PANIC_WINDOW : null);
const ready = () => usable(labels, ambient, hideRate());

// Vectors only mean anything against others from the same backend, so more than
// one entry here says part of the set was embedded elsewhere and is quietly
// wrong. "unknown" is a label from before stamping; it is almost certainly
// webgpu/fp16, but nothing recorded it, so it can't claim to be.
const evTally = () => labels.reduce((m, l) => ((m[l.ev ?? "unknown"] = (m[l.ev ?? "unknown"] ?? 0) + 1), m), {});
const evLine = ev => Object.entries(ev).map(([k, n]) => `${n} ${k}`).join(" + ");

let threshold = 0.85;

// Insert = archive + nudge, once per post. Not once per page view: the archive's
// own dedupe is the ambient dedupe, so revisiting a catalog costs nothing.
function remember(e, item, hidden) {
  const it = identOf(item.url);
  if (!it) return;                    // reddit, or no permalink: no archive, no nudge
  if (arcById.has(it.id)) return;

  const entry = {
    n: arcNextId++, id: it.id, board: it.board, thread: it.thread, url: item.url,
    text: (item.text || "").trim().slice(0, 300), img: item.img,
    ts: Date.now(), ev: EV, xy: null,
  };
  arc.push(entry);
  arcById.set(it.id, entry);
  unwritten.set(entry.n, { img: e.img, txt: e.txt, thumb: e.thumb });
  arcVec?.set(entry.n, { img: e.img, txt: e.txt });

  // One permanent step, and only when the post was not actually hidden: pushing
  // down something you asked it to catch would train against the catch. The gate
  // is "was hidden", not "scored high" -- with filtering off nothing is hidden,
  // so everything is a legitimate negative, and that is what lets a saturated
  // model climb back out instead of staying dead until Reset.
  if (!hidden) {
    const f = feats(e.img, e.txt);
    ambient.nudge(f, model.z(f));
    // Ambient drifts the combined score between clicks and only a refit puts the
    // label weights back in step. Clicks are far too rare to rely on.
    if (ambient.n % REFIT_EVERY === 0) refit();
  }
  soon();
}

const refit = () => (model = fit(labels, ambient));

let timer = null;
const soon = () => {
  clearTimeout(timer);
  timer = setTimeout(flush, 5000);
};

// Archive, ambient and coordinates settle in batches. Losing a few sightings to
// a crash costs nothing the model hasn't already absorbed, and rewriting the
// index per post would not scale past a few hundred entries.
async function flush() {
  clearTimeout(timer);
  const w = { arc, arcNextId, ambient: ambient.toJSON() };
  for (const [n, v] of unwritten) {
    w[`v${n}`] = { img: v.img, txt: v.txt };     // written once, never rewritten
    if (v.thumb) w[`t${n}`] = v.thumb;
  }
  unwritten.clear();
  await browser.storage.local.set(w);
}

// Explicit clicks are worth an immediate write.
async function commit() {
  refit();
  await browser.storage.local.set({ labels });
}

// A stored centering mean is only meaningful to the mapVectors() that produced
// it. Bump this whenever that function's shape changes, or new posts get placed
// against an origin that means something else.
const LAYOUT_V = 3;

// Which posts are on which map. Tested on the vector rather than on `e.img`,
// because a thumbnail that failed to fetch leaves a url behind and a ZERO
// embedding. Everything stays archived and still nudges whatever its modalities;
// this only decides what gets drawn.
const eligible = (vs, mode) => arc.filter(e => {
  const v = vs.get(e.n);
  return v && hasMode(v, mode);
});

// Archive vectors, loaded on demand rather than at boot. Nothing on the browsing
// path needs them -- dedupe and pruning run off the index -- so the cost lands on
// opening a view instead of on every browser start. Kept in memory afterwards,
// which is what the persistent MV2 background page is for.
let arcVec = null;
async function vectors() {
  if (arcVec) return arcVec;
  const got = await browser.storage.local.get(arc.map(e => `v${e.n}`));
  arcVec = new Map();
  for (const e of arc) {
    const v = got[`v${e.n}`];
    if (v) arcVec.set(e.n, { img: toF32(v.img), txt: toF32(v.txt) });
  }
  for (const [n, v] of unwritten) arcVec.set(n, { img: v.img, txt: v.txt });
  return arcVec;
}

// One-time upgrade. The old seen-pool was both the archive and the negative
// class at once, which is exactly why it had to stay capped at 300. Splitting
// them means its records move to the archive and its *evidence* is replayed as
// ambient -- the new semantics applied to old data, so the upgrade doesn't
// quietly knock the negative class out of a model you spent weeks training.
function migrate(old) {
  const seen = old.filter(l => l.src === "seen");
  if (!seen.length) return old;
  const kept = old.filter(l => l.src !== "seen");

  // Replay against the explicit-label fit, which is roughly the model that was
  // in force when each was recorded.
  model = kept.length ? fit(kept, ambient) : new Model();
  for (const l of seen) {
    const it = identOf(l.url);
    if (it && !arcById.has(it.id)) {
      // The old key packed the image url and the text together.
      const i = (l.key ?? "").indexOf("\n");
      const entry = {
        n: arcNextId++, id: it.id, board: it.board, thread: it.thread, url: l.url,
        text: i < 0 ? "" : l.key.slice(i + 1), img: i < 0 ? null : l.key.slice(0, i),
        ts: l.ts ?? Date.now(), ev: l.ev ?? "unknown", xy: null,
      };
      arc.push(entry);
      arcById.set(it.id, entry);
      // No thumbnail bytes for these -- the fetch that would have kept them
      // happened before there was anywhere to put them. They fall back to the
      // url and go dark when 4chan deletes the thread, which pruning removes.
      unwritten.set(entry.n, { img: l.img, txt: l.txt });
    }
    const f = feats(l.img, l.txt);
    ambient.nudge(f, model.z(f));
  }
  console.log(`sieve: migrated ${seen.length} seen labels into the archive and ambient`);
  return kept;
}

const booted = (async () => {
  const s = await browser.storage.local.get({
    labels: [], threshold: 0.85, arc: [], arcNextId: 1, ambient: null,
  });
  threshold = s.threshold;
  arc = s.arc;
  arcNextId = s.arcNextId;
  arcById = new Map(arc.map(e => [e.id, e]));
  if (s.ambient) ambient = Ambient.from({ ...s.ambient, w: toF32(s.ambient.w) });

  const loaded = s.labels.map(l => ({ ...l, img: toF32(l.img), txt: toF32(l.txt) }));
  labels = migrate(loaded);
  reindex();
  refit();
  if (labels.length !== loaded.length) {
    await browser.storage.local.set({ labels });
    await flush();
  }

  const c = counts(labels);
  console.log(`sieve: ${c.pos} hide / ${c.neg} keep clicked, ${ambient.n} seen,`
    + ` ${arc.length} archived, filtering ${ready() ? "on" : "off"}`);
  const ev = evTally();
  if (Object.keys(ev).length > 1)
    console.warn(`sieve: labels span ${evLine(ev)} — vectors from different backends are not comparable`);
})();

browser.storage.onChanged.addListener(c => {
  if (c.threshold) threshold = c.threshold.newValue;
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
          const p = score(model, ambient, e.img, e.txt);
          const on = ready();
          const hidden = on && p > threshold;
          noteOutcome(hidden);
          remember(e, msg.items[i], hidden);
          out[i] = { p, ready: on };

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
      console.log(`sieve: ${msg.y ? "hide" : "keep"} -> ${c.pos} hide / ${c.neg} keep`);
      return { ...c, ready: ready(), need: MIN_PER_CLASS };
    }
    case "stats":
      return {
        ...counts(labels), ready: ready(), need: MIN_PER_CLASS,
        seen: ambient.n, needSeen: MIN_AMBIENT, archived: arc.length,
        panic: hideRate() >= PANIC_RATE,
        holdout: holdout(labels, ambient), backend, evs: evTally(),
      };

    // Two views of the one archive; the map is the third. Uncertainty sampling
    // by proximity to 0.5, or everything the filter would collapse right now.
    case "closeCalls":
    case "recentHidden": {
      const vs = await vectors();
      const near = msg.type === "closeCalls";
      return arc
        .filter(e => !taughtIds.has(e.id) && vs.has(e.n))
        .map(e => ({ e, p: score(model, ambient, vs.get(e.n).img, vs.get(e.n).txt) }))
        .filter(({ p }) => near || p > threshold)
        .sort((a, b) => near ? Math.abs(a.p - 0.5) - Math.abs(b.p - 0.5) : b.p - a.p)
        .slice(0, msg.n ?? 12)
        .map(({ e, p }) => ({ key: e.id, url: e.url, img: e.img, text: e.text, p }));
    }

    // Promote an archive entry to a label, reusing its stored embeddings.
    case "relabel": {
      const e = arcById.get(msg.key);
      const v = e && (await vectors()).get(e.n);
      if (!v) return { gone: true };
      const key = keyOf(e.text, e.img);
      labels = labels.filter(l => l.key !== key);
      labels.push({
        img: v.img, txt: v.txt, y: msg.y, src: msg.y ? "hide" : "keep",
        key, url: e.url, ev: e.ev, ts: Date.now(),
      });
      reindex();
      await commit();
      return { ...counts(labels), ready: ready(), need: MIN_PER_CLASS };
    }

    // Scores and label state for the map. Only this, not the vectors: the map is
    // an extension page and can read storage.local itself, which beats pushing
    // 12 MB through the message channel. flush() first so nothing it needs is
    // still sitting in `unwritten`.
    case "mapState": {
      await flush();
      const vs = await vectors();
      const on = ready();
      const p = {}, mark = {}, mods = {};
      for (const e of arc) {
        const v = vs.get(e.n);
        if (!v) continue;
        // Which maps this post can appear on. The page filters per mode rather
        // than asking again every time you switch.
        const maps = MODES.filter(m => hasMode(v, m));
        if (!maps.length) continue;
        mods[e.n] = maps;
        p[e.n] = score(model, ambient, v.img, v.txt);
        const y = taughtIds.get(e.id);
        if (y !== undefined) mark[e.n] = y;
      }
      return { p, mark, mods, threshold, ready: on };
    }

    // Coordinates for everything archived. New posts are placed against the
    // stored layout rather than triggering a refit, which is what makes the map
    // the same map every time you open it -- spatial memory is most of what a
    // training tool is for, and a layout that rearranges itself has none.
    case "coords": {
      const mode = msg.mode ?? "both";
      const s = await browser.storage.local.get({ layoutMu: {}, layoutV: 0 });
      const vs = await vectors();
      const rows = eligible(vs, mode);
      const stale = s.layoutV !== LAYOUT_V;
      const mu = stale ? null : s.layoutMu?.[mode];
      // Nothing usable laid out for this mode yet: only the map has UMAP, so it
      // does the first one. A version bump drops every stored coordinate rather
      // than placing newcomers against an origin that means something else.
      if (stale) for (const e of arc) e.xy = null;
      if (!mu || !rows.some(e => e.xy?.[mode])) return { needLayout: true };

      const { vecs } = mapVectors(rows.map(e => vs.get(e.n)), mode, toF32(mu));
      const placed = [], todo = [];
      rows.forEach((e, i) => (e.xy?.[mode] ? placed : todo).push({ e, v: vecs[i] }));
      if (todo.length) {
        const ref = placed.map(p => ({ v: p.v, xy: p.e.xy[mode] }));
        for (const t of todo) t.e.xy = { ...t.e.xy, [mode]: placeNew(ref, t.v) };
        await flush();
        console.log(`sieve: placed ${todo.length} new posts on the ${mode} layout`);
      }
      return {
        xy: Object.fromEntries(rows.filter(e => e.xy?.[mode]).map(e => [e.n, e.xy[mode]])),
        placed: todo.length,
      };
    }

    // A fresh layout from the map, which is the only place UMAP lives. `mu` comes
    // with it because placement has to centre newcomers on the same mean, and
    // each mode keeps its own.
    case "saveLayout": {
      const mode = msg.mode ?? "both";
      const byN = new Map(arc.map(e => [e.n, e]));
      for (const [n, xy] of Object.entries(msg.xy)) {
        const e = byN.get(+n);
        if (e) e.xy = { ...e.xy, [mode]: xy };
      }
      const { layoutMu } = await browser.storage.local.get({ layoutMu: {} });
      await browser.storage.local.set({
        layoutMu: { ...layoutMu, [mode]: msg.mu }, layoutV: LAYOUT_V,
      });
      await flush();
      console.log(`sieve: laid out ${Object.keys(msg.xy).length} posts in ${mode} mode`);
      return { ok: true };
    }

    // "Open all the linux threads" -- the thing a lasso is for. Content scripts
    // can't reach browser.tabs at all, and tabs.create needs no permission of
    // its own, so a message is the whole of it.
    case "openTabs": {
      // A generous lasso over a dense region can hold hundreds of threads, and
      // there's no undo for opening them.
      const urls = (msg.urls ?? []).slice(0, 40);
      for (const url of urls) await browser.tabs.create({ url, active: false });
      console.log(`sieve: opened ${urls.length} threads`);
      return { opened: urls.length, capped: (msg.urls?.length ?? 0) > urls.length };
    }

    // Expired threads, pruned from the catalog's own membership list rather than
    // by asking the server about 3000 posts. Runs on catalog visit, so the map
    // can render purely from cached data.
    case "prune": {
      const live = new Set(msg.threads);
      // 4chan's catalog search re-renders #threads with only the matches, and a
      // snapshot taken after that would delete the entire board. The content
      // script only sends its first scan; this is the second line of defence.
      if (live.size < 20) return { skipped: true };
      const drop = arc.filter(e =>
        e.board === msg.board && !live.has(e.thread) && !taughtIds.has(e.id));
      if (!drop.length) return { dropped: 0 };

      const gone = new Set(drop.map(e => e.id));
      arc = arc.filter(e => !gone.has(e.id));
      for (const e of drop) {
        arcById.delete(e.id);
        arcVec?.delete(e.n);
        unwritten.delete(e.n);
      }
      await browser.storage.local.remove(drop.flatMap(e => [`v${e.n}`, `t${e.n}`]));
      await flush();
      console.log(`sieve: pruned ${drop.length} expired posts from /${msg.board}/`);
      return { dropped: drop.length };
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

    case "reset": {
      // Ambient is the only thing here that can't be undone any other way, so
      // this has to clear it too or a Reset leaves the model half-trained by
      // evidence whose records are gone.
      const keys = arc.flatMap(e => [`v${e.n}`, `t${e.n}`]);
      labels = [];
      arc = [];
      arcById = new Map();
      arcVec = null;
      unwritten.clear();
      arcNextId = 1;
      hideRing = [];
      hideCount = 0;
      model = new Model();
      ambient = new Ambient();
      reindex();   // else exact recall keeps hiding posts whose labels are gone
      await browser.storage.local.remove(keys);
      await browser.storage.local.set({ labels, arc, arcNextId, ambient: ambient.toJSON() });
      return { ok: true };
    }
  }
});

const MENU = { "sieve-hide": "sieve: hide posts like this", "sieve-keep": "sieve: this post is fine" };
for (const [id, title] of Object.entries(MENU))
  browser.contextMenus.create({ id, title, contexts: ["page", "selection", "image", "link"] });

browser.contextMenus.onClicked.addListener((info, tab) =>
  browser.tabs.sendMessage(tab.id, { type: "teach", y: info.menuItemId === "sieve-hide" ? 1 : 0 }));

// No popup on the browser action, so clicking it fires this.
browser.browserAction.onClicked.addListener(() => browser.runtime.openOptionsPage());
