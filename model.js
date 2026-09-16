// Logistic regression over [img | txt | img*txt] in shared CLIP space.
// No browser APIs here: this file runs under `node --test`.

export const K = 512;          // CLIP ViT-B/32 projection width
export const D = 3 * K;
const XS = Math.sqrt(K);

export const ZERO = new Float32Array(K);

// storage.local and runtime messaging both round-trip typed arrays differently
// depending on the path, and a Float32Array can come back as a plain object with
// numeric keys. Lives here because the map page needs it as much as the
// background does.
export const toF32 = v => v instanceof Float32Array ? v
  : Float32Array.from(Array.isArray(v) ? v : Object.values(v));

export const l2 = v => {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return Float32Array.from(v, x => x / n);
};

export function feats(img, txt) {
  const f = new Float32Array(D);
  for (let i = 0; i < K; i++) {
    f[i] = img[i];
    f[K + i] = txt[i];
    // Interaction term: the only block that can express "this image *with* this
    // text". Both inputs are unit vectors, so their product has ~1/sqrt(K) the
    // norm of either marginal; without XS its gradient is ~22x smaller and it
    // never gets a vote.
    f[2 * K + i] = img[i] * txt[i] * XS;
  }
  return f;
}

export class Model {
  constructor(w = new Float32Array(D), b = 0) {
    this.w = w;
    this.b = b;
  }

  static from({ w, b }) {
    return new Model(Float32Array.from(w), b);
  }

  toJSON() {
    return { w: Array.from(this.w), b: this.b };
  }

  // Public because scoring sums this with the ambient term below, and computing
  // feats() once for both matters more than hiding a field.
  z(f) {
    let z = this.b;
    for (let i = 0; i < D; i++) z += this.w[i] * f[i];
    return z;
  }

  // ZERO for a missing modality drops that block and the interaction, degrading
  // to text-only (or image-only) rather than scoring against garbage.
  score(img, txt) {
    return sig(this.z(feats(img, txt)));
  }

  // zOff is the ambient term's contribution, held fixed. These weights are then
  // a *correction* on top of it rather than a competing opinion -- see fit().
  learn(img, txt, y, lr = 0.5, decay = 1e-4, zOff = 0) {
    const f = feats(img, txt);
    const e = sig(this.z(f) + zOff) - y;
    for (let i = 0; i < D; i++) this.w[i] -= lr * (e * f[i] + decay * this.w[i]);
    this.b -= lr * e;
    return e;
  }
}

export const sig = z => 1 / (1 + Math.exp(-z));

// Archive identity. Not a model concern, and it lives here anyway because it
// needs a test and this is the only file `node --test` can reach -- the version
// that didn't have one silently collapsed every reply in a thread onto the OP.
//
// 4chan gives the same post several URL shapes: the catalog links /g/thread/123,
// the board index and thread pages link /g/thread/123/some-slug#p456, and a
// thread's own OP links #p123. Keying on board/thread/post folds those together
// so the OP is archived -- and nudged -- once rather than once per surface.
// Anything else (reddit) has no identity here on purpose: no archive, no nudge.
const IDENT = /boards\.4chan\.org\/([^/]+)\/thread\/(\d+)(?:\/[^#]*)?(?:#p(\d+))?/;

export const identOf = url => {
  const m = url ? IDENT.exec(url) : null;
  return m ? { board: m[1], thread: +m[2], id: `${m[1]}/${m[2]}/${m[3] ?? m[2]}` } : null;
};

// One permanent step per post you scrolled past without hiding. It lives
// outside the fitted weights because fit() rebuilds those from scratch on every
// click and would erase anything learned online. A sum of two linear terms is
// still linear: this is one model whose halves are maintained by two different
// processes, one refit and one accumulated.
export const AMBIENT_LR = 0.02;

// A wall to stand behind the argument. The gradient bound in nudge() says this
// can't run away; ambient is also the only irreversible thing in the extension
// -- never refit, never evicted, cleared only by Reset -- so a bug that beats
// the argument costs the whole label set. Pinned by the sweep in test.js.
export const AMBIENT_CAP = 6;

// Ambient drifts the combined score between clicks, and only a refit puts the
// label weights back in step with it. Clicks alone are too rare -- you can
// browse a whole board without one -- so refit on a sighting count too. Cheap,
// now that labels are deliberate clicks only: a few hundred, milliseconds.
export const REFIT_EVERY = 50;

export class Ambient {
  constructor(w = new Float32Array(D), b = 0, n = 0) {
    this.w = w;
    this.b = b;
    this.n = n;   // sightings; the negative-class evidence usable() counts
  }

  static from({ w, b, n }) {
    return new Ambient(Float32Array.from(w), b ?? 0, n ?? 0);
  }

  toJSON() {
    return { w: Array.from(this.w), b: this.b, n: this.n };
  }

  z(f) {
    let z = this.b;
    for (let i = 0; i < D; i++) z += this.w[i] * f[i];
    return z;
  }

  // y is always 0, and the error is taken against the *combined* score rather
  // than this term alone. That's what makes thousands of one-class updates safe:
  // once a region reads as keep, sig() of a very negative z is ~0 and further
  // sightings move nothing. Self-extinguishing, not self-reinforcing.
  //
  // No decay, deliberately. Decay would shrink the weights on every sighting
  // that no longer moves them, so a mark would fade out exactly when the model
  // had settled -- the opposite of permanent.
  nudge(f, zFit, lr = AMBIENT_LR) {
    const e = sig(this.z(f) + zFit);
    for (let i = 0; i < D; i++) this.w[i] -= lr * e * f[i];
    this.b -= lr * e;
    this.n++;
    this.#cap();
  }

  #cap() {
    let n2 = this.b * this.b;
    for (let i = 0; i < D; i++) n2 += this.w[i] * this.w[i];
    if (n2 <= AMBIENT_CAP * AMBIENT_CAP) return;
    const s = AMBIENT_CAP / Math.sqrt(n2);
    for (let i = 0; i < D; i++) this.w[i] *= s;
    this.b *= s;
  }
}

// The score everything outside this file should use: feats() once, both terms.
export function score(fit, amb, img, txt) {
  const f = feats(img, txt);
  return sig(fit.z(f) + amb.z(f));
}

export const MIN_PER_CLASS = 3;

// Sightings that stand in for the negative class. Browsing alone should get you
// to a working filter: one catalog clears this, so it guards a fresh profile
// with three angry clicks and nothing else rather than asking for a chore.
export const MIN_AMBIENT = 50;

// Saturation is an *absorbing* state, and that is the whole reason this exists.
// A one-class fit scores everything ~1.00; everything then sits above the
// threshold; ambient only nudges what's below it, so nothing pushes back and
// the extension is dead until Reset -- which throws away every label you
// trained. Reachable in ordinary use: click ✗ often and ✓ rarely and the label
// set goes all-positive while MIN_AMBIENT keeps filtering switched on.
//
// A model hiding essentially the entire page is broken, not strict. Stop
// filtering, which puts every post back under the threshold and lets ambient
// pull the model out on its own.
export const PANIC_RATE = 0.9;
export const PANIC_WINDOW = 50;

export const counts = labels => {
  let pos = 0, neg = 0;
  for (const l of labels) l.y ? pos++ : neg++;
  return { pos, neg };
};

// One-class training is degenerate, not just inaccurate: nothing counteracts the
// bias, so it saturates and scores everything ~1.00. Callers must not filter
// until this passes. Ambient sightings satisfy the negative side -- they are
// real negative evidence, they just aren't stored as labels.
export const usable = (labels, ambient, hideRate = null) => {
  if (hideRate !== null && hideRate >= PANIC_RATE) return false;
  const { pos, neg } = counts(labels);
  return pos >= MIN_PER_CLASS && (neg >= MIN_PER_CLASS || (ambient?.n ?? 0) >= MIN_AMBIENT);
};

// Refit from scratch; online learn() alone drifts toward whatever was clicked
// most recently.
//
// epochs/lr/decay are tuned against the 0.85 threshold, not against accuracy.
// decay=1e-3, or an annealed rate, or epochs=30 all keep ranking perfect while
// squashing scores toward 0.5 so nothing crosses the threshold and the filter
// silently does nothing. The "scores actually clear the default threshold" test
// pins this; watch that rather than accuracy if you change them.
//
// Fitted *on top of* the ambient term, which is the only way the two stay
// calibrated. CLIP embeddings are anisotropic, so ambient's accumulated weight
// points largely along the cone that every post projects onto -- left to itself
// it drags true hides down with everything else and the filter quietly dies.
// Passing its contribution in as a fixed offset makes these weights the residual
// instead: labels win where they exist, ambient generalises where they don't.
// The sweep in test.js fails at 7% without this.
//
// ponytail: O(labels * epochs) per click. Cheap now that labels are only
// deliberate clicks -- a few hundred, not the old 300 seen entries on top.
export function fit(labels, ambient = new Ambient(), { epochs = 200, lr = 0.5, decay = 1e-4 } = {}) {
  const m = new Model();
  const idx = labels.map((_, i) => i);
  // Constant through the fit, so pay for it once rather than per epoch.
  const off = labels.map(l => ambient.z(feats(l.img, l.txt)));
  // Class weights over sample weight rather than count. Without this a few hides
  // lose to the pile of keeps and the model converges on hiding nothing. Every
  // label now carries w=1 -- ambient evidence is not stored here any more -- but
  // imported sets may not, so the weighting stays.
  let wpos = 0, wneg = 0;
  for (const l of labels) l.y ? (wpos += l.w ?? 1) : (wneg += l.w ?? 1);
  const total = wpos + wneg;
  const wt = l => (l.w ?? 1) * total / (2 * ((l.y ? wpos : wneg) || 1));
  let seed = 1;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let e = 0; e < epochs; e++) {
    for (let i = idx.length - 1; i > 0; i--) {
      const j = (rnd() * (i + 1)) | 0;
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    for (const i of idx)
      m.learn(labels[i].img, labels[i].txt, labels[i].y, lr * wt(labels[i]), decay, off[i]);
  }
  return m;
}

// The vector the map clusters on: both modalities, equally weighted, centered.
//
// Not feats(). That interaction block earns its keep as a discriminative lift,
// but as a *distance* it's a 4th-order term with no interpretation, and it would
// only add noise to a neighbourhood.
//
// Centering is not optional. CLIP embeddings sit in a narrow cone -- two
// unrelated images still have cosine ~0.8 -- so without it every neighbour list
// is dominated by the mean direction and the layout is mush. See the negative
// test in test.js, which asserts exactly that failure.
// Pass the `mu` from a previous call to place new items onto an existing
// layout. The centering mean has to be the one the layout was built with, or a
// newcomer is measured from a different origin than its neighbours were.
export function mapVectors(items, mu0 = null) {
  const has = items.map(it => it.txt.some(x => x !== 0));
  const mu = mu0 ?? new Float32Array(2 * K);
  if (!mu0) {
    let nt = 0;
    for (let k = 0; k < items.length; k++) {
      const it = items[k];
      for (let i = 0; i < K; i++) mu[i] += it.img[i];
      if (!has[k]) continue;
      nt++;
      for (let i = 0; i < K; i++) mu[K + i] += it.txt[i];
    }
    for (let i = 0; i < K; i++) mu[i] /= items.length || 1;
    for (let i = K; i < 2 * K; i++) mu[i] /= nt || 1;
  }

  const vecs = items.map((it, k) => {
    const v = new Float32Array(2 * K);
    for (let i = 0; i < K; i++) v[i] = it.img[i] - mu[i];
    // A textless post gets the mean text vector, so after centering its text
    // block is zero and it compares on image alone. Leaving ZERO there instead
    // hands every one of them the same -mu block, and they cluster together for
    // the single thing they have in common: having no text.
    if (has[k]) for (let i = 0; i < K; i++) v[K + i] = it.txt[i] - mu[K + i];
    return l2(v);
  });
  return { mu, vecs };
}

// Out-of-sample placement: where does a newly archived post go on an existing
// layout?
//
// umap-js's own transform() cannot answer this. It needs the rpForest and
// searchGraph that fit() builds in memory, and neither survives a reload -- so
// restoring a layout from stored coordinates would still mean refitting every
// session, and the map would rearrange itself every time you opened it.
//
// What transform() does *before* its optimisation pass is initTransform: drop
// the point at the weighted average of its neighbours' existing coordinates.
// That needs nothing but the vectors and coordinates already on disk, and for
// placing one post among thousands of known ones the refinement isn't worth
// rebuilding the whole layout to get.
export function placeNew(placed, vec, k = 8) {
  if (!placed.length) return [0, 0];
  const best = [];
  for (const p of placed) {
    let c = 0;
    for (let i = 0; i < vec.length; i++) c += vec[i] * p.v[i];
    if (best.length < k || c > best[best.length - 1].c) {
      best.push({ c, xy: p.xy });
      best.sort((a, b) => b.c - a.c);
      if (best.length > k) best.pop();
    }
  }
  // Weight by rank, not by raw cosine. Even centered, neighbours of a CLIP point
  // sit at similar similarities, so raw weights would be near-uniform and every
  // newcomer would land in the middle of its neighbourhood's bounding box.
  // Subtracting the k-th best makes the weights carry the ordering instead.
  const floor = best[best.length - 1].c;
  let wsum = 0, x = 0, y = 0;
  for (const b of best) {
    const w = b.c - floor + 1e-6;
    wsum += w;
    x += w * b.xy[0];
    y += w * b.xy[1];
  }
  return [x / wsum, y / wsum];
}

// Every nth label held out, refit on the rest. Shown in the options page.
//
// Scored with the ambient term included, because that's what the extension
// actually does. It also finally measures something: this used to be every 5th
// of a list dominated by 300 weak "seen" entries, and now every label in it is a
// deliberate click.
export function holdout(labels, ambient = new Ambient(), frac = 0.2, opts) {
  const test = labels.filter((_, i) => i % Math.round(1 / frac) === 0);
  const train = labels.filter((_, i) => i % Math.round(1 / frac) !== 0);
  if (!test.length || !train.length) return null;
  const m = fit(train, ambient, opts);
  const wrong = test.filter(l => (score(m, ambient, l.img, l.txt) > 0.5 ? 1 : 0) !== l.y).length;
  return { n: test.length, acc: 1 - wrong / test.length };
}
