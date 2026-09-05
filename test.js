import test from "node:test";
import assert from "node:assert/strict";
import { K, Model, ZERO, l2, fit, holdout, usable, counts, SEEN_WEIGHT } from "./model.js";

// Synthetic stand-ins for CLIP embeddings. jit() nudges one toward a fresh
// direction so train and test never see the same vector. `a` is relative to the
// unit signal, so the perturbation must be normalised too or it swamps it.
const unit = s => l2(Array.from({ length: K }, (_, i) => Math.sin(s * (i + 1) * 12.9898)));
const jit = (v, k, a = 0.3) => {
  const n = unit(k * 3.7 + 0.5);
  return l2(Array.from(v, (x, i) => x + a * n[i]));
};

// Real CLIP embeddings are anisotropic: they sit in a narrow cone, so unrelated
// images still have cosine ~0.8. Near-orthogonal test vectors would make every
// test here easier than reality.
const CONE = unit(42);
const emb = s => l2(Array.from(unit(s), (x, i) => x + 2 * CONE[i]));

const [IA, IB, TA, TB] = [emb(1), emb(2), emb(3), emb(4)];

// Two conjunctions that disagree: image A is only bad with text A, image B only
// with text B, crossed pairings fine. The real shape of "the combination is the
// problem", and not expressible without the interaction block.
const PAIRS = [[IA, TA, 1], [IB, TB, 1], [IA, TB, 0], [IB, TA, 0]];

const train = (m, hook) => {
  for (let k = 0; k < 300; k++)
    for (const [i, t, y] of PAIRS) {
      m.learn(jit(i, k), jit(t, k + 991), y);
      hook?.(m);
    }
  return m;
};

const worstErr = (m, seeds = [5000, 5001, 5002, 5003]) =>
  Math.max(...seeds.flatMap(k => PAIRS.map(([i, t, y]) => Math.abs(m.score(jit(i, k), jit(t, k + 991)) - y))));

test("l2 returns a unit vector", () => {
  const v = l2([3, 4, ...new Array(K - 2).fill(0)]);
  assert.ok(Math.abs(Math.hypot(...v) - 1) < 1e-6);
});

test("learns a combination that neither modality predicts alone", () => {
  const err = worstErr(train(new Model()));
  assert.ok(err < 0.3, `worst holdout error ${err.toFixed(3)}`);
});

test("without the interaction block the same task is unlearnable", () => {
  // Zeroing w[2K..] each step is exactly a plain [img | txt] concat model.
  // Summing the four constraints puts c(IA)+c(TA)+c(IB)+c(TB) both above and
  // below 2*threshold, so no weighting satisfies them.
  const err = worstErr(train(new Model(), m => m.w.fill(0, 2 * K)));
  assert.ok(err > 0.4, `concat-only should fail but got ${err.toFixed(3)}`);
});

test("a missing image degrades to text-only rather than scoring noise", () => {
  const m = new Model();
  for (let k = 0; k < 200; k++) {
    m.learn(ZERO, jit(TA, k), 1);
    m.learn(ZERO, jit(TB, k), 0);
  }
  assert.ok(m.score(ZERO, jit(TA, 9001)) > 0.8);
  assert.ok(m.score(ZERO, jit(TB, 9001)) < 0.2);
});

test("fit refits from the log and is order-independent", () => {
  const labels = [];
  for (let k = 0; k < 60; k++)
    for (const [i, t, y] of PAIRS) labels.push({ img: jit(i, k), txt: jit(t, k + 991), y });

  assert.ok(worstErr(fit(labels)) < 0.3);

  // Same labels reversed must hide the same posts. Not bit-identical weights --
  // the shuffle visits them differently -- just the same verdicts.
  const [a, b] = [fit(labels), fit([...labels].reverse())];
  const drift = Math.max(...PAIRS.map(([i, t]) =>
    Math.abs(a.score(jit(i, 7001), jit(t, 7992)) - b.score(jit(i, 7001), jit(t, 7992)))));
  assert.ok(drift < 0.05, `label order shifted predictions by ${drift.toFixed(3)}`);
});

test("one-class labels are rejected instead of hiding everything", () => {
  // Guards: hide two posts, reload, and the whole page is gone at 1.00, because
  // nothing counteracts the bias when every label says y=1.
  const onlyHides = [0, 1, 2, 3].map(k => ({ img: jit(IA, k), txt: jit(TA, k + 991), y: 1 }));
  assert.equal(usable(onlyHides), false);
  assert.deepEqual(counts(onlyHides), { pos: 4, neg: 0, taught: 4 });

  const m = fit(onlyHides);
  assert.ok(m.score(jit(IB, 7), jit(TB, 7)) > 0.9, "a one-class fit really does saturate");

  const mixed = [...onlyHides, ...[0, 1, 2].map(k => ({ img: jit(IB, k), txt: jit(TB, k + 991), y: 0 }))];
  assert.equal(usable(mixed), true);
});

test("rare positives survive a pile of negatives", () => {
  // Without class weighting, the cheapest fit for 4 hides against 200 keeps is
  // to hide nothing.
  const labels = [];
  for (let k = 0; k < 4; k++) labels.push({ img: jit(IA, k), txt: jit(TA, k + 991), y: 1 });
  for (let k = 0; k < 200; k++) labels.push({ img: jit(IB, k), txt: jit(TB, k + 991), y: 0 });
  const m = fit(labels);
  assert.ok(m.score(jit(IA, 8001), jit(TA, 8992)) > 0.5, "hide class was drowned out");
  assert.ok(m.score(jit(IB, 8001), jit(TB, 8992)) < 0.5);
});

test("a pile of weak 'seen' labels cannot outvote a few explicit hides", () => {
  // Implicit negatives only work if they stay quiet enough not to swamp clicks.
  const labels = [];
  for (let k = 0; k < 4; k++) labels.push({ img: jit(IA, k), txt: jit(TA, k + 991), y: 1, src: "hide" });
  for (let k = 0; k < 300; k++)
    labels.push({ img: jit(IB, k), txt: jit(TB, k + 991), y: 0, w: SEEN_WEIGHT, src: "seen" });

  assert.deepEqual(counts(labels), { pos: 4, neg: 300, taught: 4 });
  assert.equal(usable(labels), true, "seen labels should satisfy the negative class");

  const m = fit(labels);
  assert.ok(m.score(jit(IA, 8001), jit(TA, 8992)) > 0.85, "explicit hides got drowned out");
  assert.ok(m.score(jit(IB, 8001), jit(TB, 8992)) < 0.5);
});

test("scores actually clear the default threshold, not just rank correctly", () => {
  // Invisible to accuracy: a model can separate the classes perfectly while
  // squashing every score toward 0.5, so nothing crosses 0.85 and the filter
  // does nothing. Ranking isn't enough, the numbers must land at the threshold.
  const labels = [];
  for (let k = 0; k < 5; k++)
    for (const [i, t, y] of PAIRS) labels.push({ img: jit(i, k), txt: jit(t, k + 991), y });

  const m = fit(labels);
  const held = [];
  for (let k = 9000; k < 9030; k++)
    for (const [i, t, y] of PAIRS) held.push({ p: m.score(jit(i, k), jit(t, k + 991)), y });

  const hides = held.filter(l => l.y).map(l => l.p);
  const over = hides.filter(p => p > 0.85).length / hides.length;
  assert.ok(over > 0.8, `only ${(over * 100) | 0}% of true hides clear 0.85 at ${labels.length} labels`);
  assert.ok(held.filter(l => !l.y).every(l => l.p < 0.85), "a true keep crossed the threshold");
});

test("holdout reports accuracy on data it did not train on", () => {
  const labels = [];
  for (let k = 0; k < 60; k++)
    for (const [i, t, y] of PAIRS) labels.push({ img: jit(i, k), txt: jit(t, k + 991), y });
  const h = holdout(labels);
  assert.ok(h.n > 0 && h.acc > 0.9, JSON.stringify(h));
  assert.equal(holdout([]), null);
});
