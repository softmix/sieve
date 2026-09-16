// Logistic regression over [img | txt | img*txt] in shared CLIP space.
// No browser APIs here: this file runs under `node --test`.

export const K = 512;          // CLIP ViT-B/32 projection width
export const D = 3 * K;
const XS = Math.sqrt(K);

export const ZERO = new Float32Array(K);

// A Float32Array can come back from storage or a message as a plain object with
// numeric keys, depending on the path it took.
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

  // zOff is the ambient term's contribution, held fixed, so these weights come
  // out as a correction on top of it rather than a competing opinion. See fit().
  learn(img, txt, y, lr = 0.5, decay = 1e-4, zOff = 0) {
    const f = feats(img, txt);
    const e = sig(this.z(f) + zOff) - y;
    for (let i = 0; i < D; i++) this.w[i] -= lr * (e * f[i] + decay * this.w[i]);
    this.b -= lr * e;
    return e;
  }
}

export const sig = z => 1 / (1 + Math.exp(-z));

// 4chan gives one post several URL shapes -- /g/thread/123 from the catalog,
// /g/thread/123/some-slug#p456 from the index, a bare #p123 from the thread
// itself -- so identity is board/thread/post, or the same post is archived and
// nudged once per surface. Anything without a match here (reddit) gets neither.
const IDENT = /boards\.4chan\.org\/([^/]+)\/thread\/(\d+)(?:\/[^#]*)?(?:#p(\d+))?/;

export const identOf = url => {
  const m = url ? IDENT.exec(url) : null;
  return m ? { board: m[1], thread: +m[2], id: `${m[1]}/${m[2]}/${m[3] ?? m[2]}` } : null;
};

// One permanent step per post scrolled past without hiding, kept outside the
// fitted weights because fit() rebuilds those from scratch on every click.
export const AMBIENT_LR = 0.02;

// Ambient is never refit and never evicted, so a bug beating the gradient bound
// in nudge() costs the whole label set. Pinned by the sweep in test.js.
export const AMBIENT_CAP = 6;

// Ambient drifts the combined score between clicks and only a refit puts the
// label weights back in step. Clicks are far too rare to rely on -- a whole
// board can go by without one.
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

  // The error is the *combined* score, not this term alone, which is what makes
  // thousands of one-class updates safe: once a region reads keep the gradient
  // is ~0 and further sightings move nothing.
  //
  // No decay. It would shrink the weights on exactly those vanished-gradient
  // sightings, fading a mark out once the model settles.
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

export function score(fit, amb, img, txt) {
  const f = feats(img, txt);
  return sig(fit.z(f) + amb.z(f));
}

export const MIN_PER_CLASS = 3;

// One catalog clears this, so it only guards a fresh profile with three angry
// clicks and no other evidence.
export const MIN_AMBIENT = 50;

// Without this, saturation is an absorbing state: a one-class fit scores
// everything ~1.00, everything is therefore hidden, ambient only nudges what
// isn't, nothing pushes back, and Reset is the only exit. Reachable by clicking
// ✗ often and ✓ rarely. Hiding the whole page is broken rather than strict, so
// stop filtering and let ambient climb back out.
export const PANIC_RATE = 0.9;
export const PANIC_WINDOW = 50;

export const counts = labels => {
  let pos = 0, neg = 0;
  for (const l of labels) l.y ? pos++ : neg++;
  return { pos, neg };
};

// One-class training is degenerate, not merely inaccurate: nothing counteracts
// the bias, so it saturates at ~1.00 on everything. Callers must not filter
// until this passes. Ambient sightings count as the negative side.
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
// Fitted on top of ambient, which is the only way the two stay calibrated. CLIP
// embeddings are anisotropic, so ambient accumulates along the cone that every
// post projects onto; without its contribution passed in as a fixed offset here,
// it drags true hides down with everything else and the filter dies quietly.
export function fit(labels, ambient = new Ambient(), { epochs = 200, lr = 0.5, decay = 1e-4 } = {}) {
  const m = new Model();
  const idx = labels.map((_, i) => i);
  const off = labels.map(l => ambient.z(feats(l.img, l.txt)));   // fixed through the fit
  // Class-balanced, or a few hides lose to the pile of keeps and the model
  // converges on hiding nothing. By weight rather than count for imported sets.
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

export const MODES = ["both", "image", "text"];

// A post lacking the modality being clustered on has a zero block after
// centering, which leaves it systematically less similar to everything that has
// one, so it forms its own region however the hole is filled. Admitting only the
// posts that have a mode's modality is what keeps that from happening.
export const hasMode = (v, mode) => {
  const i = v.img.some(x => x !== 0), t = v.txt.some(x => x !== 0);
  return mode === "image" ? i : mode === "text" ? t : i && t;
};

// Callers must filter by hasMode() first, or the gap comes straight back.
//
// Not feats(): the interaction block is a discriminative lift, and as a distance
// it's a 4th-order term with no interpretation that only adds noise.
//
// Centering is not optional. CLIP embeddings sit in a narrow cone, unrelated
// images still at cosine ~0.8, so without it every neighbour list is dominated
// by the mean direction and the layout is mush.
//
// Pass a `mu` back in to place items onto an existing layout: the mean has to be
// the one that layout was built with, or a newcomer is measured from a different
// origin than its neighbours.
export function mapVectors(items, mode = "both", mu0 = null) {
  const useI = mode !== "text", useT = mode !== "image";
  const W = (useI ? K : 0) + (useT ? K : 0);

  const mu = mu0 ?? new Float32Array(W);
  if (!mu0) {
    for (const it of items) {
      let o = 0;
      if (useI) { for (let i = 0; i < K; i++) mu[i] += it.img[i]; o = K; }
      if (useT) for (let i = 0; i < K; i++) mu[o + i] += it.txt[i];
    }
    for (let i = 0; i < W; i++) mu[i] /= items.length || 1;
  }

  const vecs = items.map(it => {
    const v = new Float32Array(W);
    let o = 0;
    if (useI) { for (let i = 0; i < K; i++) v[i] = it.img[i] - mu[i]; o = K; }
    if (useT) for (let i = 0; i < K; i++) v[o + i] = it.txt[i] - mu[o + i];
    return l2(v);
  });
  return { mu, vecs };
}

// Ray casting, for the map's lasso.
export function inside(px, py, poly) {
  let hit = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    // Half-open on one end only, so a vertex on the ray isn't counted twice.
    if ((yi > py) !== (yj > py) && px < ((xj - xi) * (py - yi)) / (yj - yi) + xi) hit = !hit;
  }
  return hit;
}

// Where a newly archived post goes on an existing layout.
//
// Not umap-js's transform(): it needs the rpForest and searchGraph that fit()
// builds in memory, neither of which survives a reload, so using it would mean
// refitting every session and a map that rearranges itself on every open. This
// is its initTransform step -- the weighted average of a point's neighbours'
// coordinates -- which needs only what's already on disk.
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
  // Weight by rank, not raw cosine. Even centered, a CLIP point's neighbours sit
  // at similar similarities, so raw weights are near-uniform and every newcomer
  // lands in the middle of its neighbourhood's bounding box.
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

// Every nth label held out, refit on the rest, scored with ambient included
// because that's what the extension does. Shown in the options page.
export function holdout(labels, ambient = new Ambient(), frac = 0.2, opts) {
  const test = labels.filter((_, i) => i % Math.round(1 / frac) === 0);
  const train = labels.filter((_, i) => i % Math.round(1 / frac) !== 0);
  if (!test.length || !train.length) return null;
  const m = fit(train, ambient, opts);
  const wrong = test.filter(l => (score(m, ambient, l.img, l.txt) > 0.5 ? 1 : 0) !== l.y).length;
  return { n: test.length, acc: 1 - wrong / test.length };
}
