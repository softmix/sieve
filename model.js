// Logistic regression over [img | txt | img*txt] in shared CLIP space.
// No browser APIs here: this file runs under `node --test`.

export const K = 512;          // CLIP ViT-B/32 projection width
export const D = 3 * K;
const XS = Math.sqrt(K);

export const ZERO = new Float32Array(K);

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
