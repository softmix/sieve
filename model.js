// Logistic regression over [img | txt | img*txt] in shared CLIP space.
// No browser APIs in this file on purpose: it runs under `node --test`.

export const K = 512;          // CLIP ViT-B/32 projection width
export const D = 3 * K;
const XS = Math.sqrt(K);       // see feats()

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
    // The interaction block is what lets the model flag a post whose image and
    // text are each individually fine. Both inputs are unit vectors, so the
    // elementwise product has ~1/sqrt(K) the norm of either marginal block --
    // without XS its gradient is ~22x smaller and it never gets a say.
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

  // Pass ZERO for a missing modality: that block and the interaction both drop
  // out, and the model degrades to plain text-only (or image-only) logistic
  // regression rather than scoring against garbage.
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

export const counts = labels => {
  const pos = labels.reduce((n, l) => n + l.y, 0);
  return { pos, neg: labels.length - pos };
};

// Trained on one class only, this model is not merely inaccurate, it's
// degenerate: every gradient step pushes the bias the same way and nothing
// pushes back, so it saturates and scores everything ~1.00. Refuse to filter
// until both classes exist rather than hiding the entire page.
export const usable = labels => {
  const { pos, neg } = counts(labels);
  return pos >= MIN_PER_CLASS && neg >= MIN_PER_CLASS;
};

// Refit from scratch over the whole label log. Online learn() drifts toward
// whatever you clicked most recently; this doesn't, and it's cheap enough
// (1537 params) to just re-run whenever labels change.
// ponytail: cost is O(labels * epochs) on every click -- ~200ms at 240 labels,
// ~1s at 1000. Warm-start from the current weights if that ever gets annoying.
export function fit(labels, { epochs = 100, lr = 0.5, decay = 1e-3 } = {}) {
  const m = new Model();
  const idx = labels.map((_, i) => i);
  // Balanced class weights. A filter exists for rare things, so a handful of
  // "hide" labels must not be drowned out by everything marked fine -- without
  // this the model converges on never hiding anything as soon as the keeps pile up.
  const { pos, neg } = counts(labels);
  const wt = y => (y ? labels.length / (2 * (pos || 1)) : labels.length / (2 * (neg || 1)));
  let seed = 1;
  const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  for (let e = 0; e < epochs; e++) {
    for (let i = idx.length - 1; i > 0; i--) {
      const j = (rnd() * (i + 1)) | 0;
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    // Annealed step. With a flat rate this hasn't converged by the last epoch,
    // so the order you happened to click labels in shifts scores by up to 0.17 --
    // enough to flip a post either side of the threshold. Annealing takes that
    // to 0.002.
    const step = lr / (1 + e);
    for (const i of idx) m.learn(labels[i].img, labels[i].txt, labels[i].y, step * wt(labels[i].y), decay);
  }
  return m;
}

// Leave-nothing-out is useless on 1537 params, so hold out a slice by hash of
// position. Reported in the options page so you can see if it's actually working.
export function holdout(labels, frac = 0.2, opts) {
  const test = labels.filter((_, i) => i % Math.round(1 / frac) === 0);
  const train = labels.filter((_, i) => i % Math.round(1 / frac) !== 0);
  if (!test.length || !train.length) return null;
  const m = fit(train, opts);
  const wrong = test.filter(l => (m.score(l.img, l.txt) > 0.5 ? 1 : 0) !== l.y).length;
  return { n: test.length, acc: 1 - wrong / test.length };
}
