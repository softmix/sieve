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

  #z(f) {
    let z = this.b;
    for (let i = 0; i < D; i++) z += this.w[i] * f[i];
    return z;
  }

  // ZERO for a missing modality drops that block and the interaction, degrading
  // to text-only (or image-only) rather than scoring against garbage.
  score(img, txt) {
    return 1 / (1 + Math.exp(-this.#z(feats(img, txt))));
  }

  learn(img, txt, y, lr = 0.5, decay = 1e-4) {
    const f = feats(img, txt);
    const e = 1 / (1 + Math.exp(-this.#z(f))) - y;
    for (let i = 0; i < D; i++) this.w[i] -= lr * (e * f[i] + decay * this.w[i]);
    this.b -= lr * e;
    return e;
  }
}

export const MIN_PER_CLASS = 3;

// Weight of an implicit "scrolled past without hiding" label, vs 1 for a click.
export const SEEN_WEIGHT = 0.15;

export const counts = labels => {
  let pos = 0, neg = 0, taught = 0;
  for (const l of labels) {
    if (l.y) pos++; else neg++;
    if (l.src !== "seen") taught++;
  }
  return { pos, neg, taught };
};

// One-class training is degenerate, not just inaccurate: nothing counteracts the
// bias, so it saturates and scores everything ~1.00. Callers must not filter
// until this passes.
export const usable = labels => {
  const { pos, neg } = counts(labels);
  return pos >= MIN_PER_CLASS && neg >= MIN_PER_CLASS;
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
// ponytail: O(labels * epochs) per click, ~340ms at 320 labels, ~1s at 1000.
// Warm-start from the current weights if that gets annoying.
export function fit(labels, { epochs = 200, lr = 0.5, decay = 1e-4 } = {}) {
  const m = new Model();
  const idx = labels.map((_, i) => i);
  // Class weights over sample weight, not count, so low-weight "seen" labels
  // dilute correctly. Without this a few hides lose to the pile of keeps and the
  // model converges on hiding nothing.
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
    for (const i of idx) m.learn(labels[i].img, labels[i].txt, labels[i].y, lr * wt(labels[i]), decay);
  }
  return m;
}

// Every nth label held out, refit on the rest. Shown in the options page.
export function holdout(labels, frac = 0.2, opts) {
  const test = labels.filter((_, i) => i % Math.round(1 / frac) === 0);
  const train = labels.filter((_, i) => i % Math.round(1 / frac) !== 0);
  if (!test.length || !train.length) return null;
  const m = fit(train, opts);
  const wrong = test.filter(l => (m.score(l.img, l.txt) > 0.5 ? 1 : 0) !== l.y).length;
  return { n: test.length, acc: 1 - wrong / test.length };
}
