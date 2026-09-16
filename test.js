import test from "node:test";
import assert from "node:assert/strict";
import {
  K, Model, ZERO, l2, feats, fit, holdout, usable, counts, score,
  Ambient, AMBIENT_CAP, MIN_AMBIENT, PANIC_RATE, REFIT_EVERY,
} from "./model.js";

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
  assert.equal(usable(onlyHides, new Ambient()), false);
  assert.deepEqual(counts(onlyHides), { pos: 4, neg: 0 });

  const m = fit(onlyHides);
  assert.ok(m.score(jit(IB, 7), jit(TB, 7)) > 0.9, "a one-class fit really does saturate");

  const mixed = [...onlyHides, ...[0, 1, 2].map(k => ({ img: jit(IB, k), txt: jit(TB, k + 991), y: 0 }))];
  assert.equal(usable(mixed, new Ambient()), true);
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

// A browsing session. Mostly posts unrelated to anything labelled, with a real
// keep every third, which is roughly what a board looks like.
//
// `gated` mirrors background.js: a nudge is skipped for a post that was actually
// *hidden*, because pushing that down would train against the thing you asked it
// to catch. Note the gate is on hidden, not on high-scoring -- with filtering
// off nothing is hidden, so everything is a legitimate negative. That is what
// makes saturation recoverable instead of absorbing.
const browse = (m, amb, n, { seed = 0, gated = true } = {}) => {
  for (let k = 0; k < n; k++) {
    const s = seed + k;
    const [i, t] = s % 3 === 0
      ? [jit(IA, s), jit(TB, s + 991)]                       // a crossed pair: a true keep
      : [emb(100 + (s % 997) * 0.37), emb(5000 + (s % 991) * 0.41)];   // unrelated traffic
    if (gated && score(m, amb, i, t) > 0.85) continue;
    amb.nudge(feats(i, t), m.z(feats(i, t)));
  }
};

const clearing = (m, amb) => {
  const hides = [];
  for (let k = 9000; k < 9020; k++)
    for (const [i, t, y] of PAIRS)
      if (y) hides.push(score(m, amb, jit(i, k), jit(t, k + 991)));
  return hides.filter(p => p > 0.85).length / hides.length;
};

test("ambient sightings satisfy the negative class", () => {
  const hides = [0, 1, 2].map(k => ({ img: jit(IA, k), txt: jit(TA, k + 991), y: 1 }));
  const amb = new Ambient();
  assert.equal(usable(hides, amb), false, "three hides and no evidence of anything else");
  amb.n = MIN_AMBIENT;
  assert.equal(usable(hides, amb), true, "browsing alone should unlock filtering");
});

test("ambient browsing never squashes true hides below the threshold", () => {
  // The failure fit() already warns about -- ranking stays perfect while every
  // score drifts toward 0.5, so nothing clears 0.85 and the filter silently does
  // nothing. Ambient pushes in exactly that direction and is never refit away,
  // so it must be swept: the damage is cumulative and a single N would miss it.
  const labels = [];
  for (let k = 0; k < 5; k++)
    for (const [i, t, y] of PAIRS) labels.push({ img: jit(i, k), txt: jit(t, k + 991), y });
  const amb = new Ambient();
  let m = fit(labels, amb);

  // Refit on the sighting count as production does; without that this measures a
  // state the extension is never actually in.
  const session = (n, seed) => {
    for (let k = 0; k < n; k += REFIT_EVERY) {
      browse(m, amb, Math.min(REFIT_EVERY, n - k), { seed: seed + k });
      m = fit(labels, amb);
    }
  };

  let seen = 0;
  for (const n of [100, 1000, 10000]) {
    session(n - seen, seen);
    seen = n;
    const over = clearing(m, amb);
    assert.ok(over > 0.8, `after ${n} sightings only ${(over * 100) | 0}% of true hides clear 0.85`);
  }

  let n2 = amb.b * amb.b;
  for (const x of amb.w) n2 += x * x;
  assert.ok(Math.sqrt(n2) <= AMBIENT_CAP + 1e-6, `cap breached at ${Math.sqrt(n2).toFixed(2)}`);
});

test("a saturated model is recoverable rather than permanently dead", () => {
  // Without the panic guard this is terminal: a one-class fit hides everything,
  // every post is therefore hidden, the gate blocks every nudge, and nothing can
  // ever push back. Reset is the only exit and it costs the whole label set.
  const onlyHides = [0, 1, 2, 3].map(k => ({ img: jit(IA, k), txt: jit(TA, k + 991), y: 1 }));
  const m = fit(onlyHides);
  const amb = new Ambient();
  amb.n = MIN_AMBIENT;

  assert.ok(score(m, amb, jit(IB, 7), jit(TB, 7)) > 0.9, "the fit really does saturate");
  assert.equal(usable(onlyHides, amb), true, "and filtering is switched on");
  assert.equal(usable(onlyHides, amb, PANIC_RATE), false, "hiding the whole page must stop filtering");

  // Filtering off means nothing is hidden, so every post nudges again.
  browse(m, amb, 3000, { gated: false });
  assert.ok(score(m, amb, jit(IB, 8001), jit(TB, 8992)) < 0.85, "never climbed back out");
});

test("ambient nudges self-extinguish once a region reads as keep", () => {
  // The bound that makes thousands of one-class updates safe: the error is the
  // *combined* score, so a region that already reads keep stops attracting
  // weight. Take the error from ambient's own z instead and it never settles --
  // it walks to the cap and drags the hides down with it.
  const onlyHides = [0, 1, 2, 3].map(k => ({ img: jit(IA, k), txt: jit(TA, k + 991), y: 1 }));
  const m = fit(onlyHides);          // saturated, so this region starts at ~1.00
  const amb = new Ambient();
  const f = feats(jit(IB, 1), jit(TB, 992));
  const norm = () => { let s = amb.b * amb.b; for (const x of amb.w) s += x * x; return Math.sqrt(s); };

  const run = n => { for (let k = 0; k < n; k++) amb.nudge(f, m.z(f)); return norm(); };
  const a = norm(), b = run(200), c = run(200);

  assert.ok(b - a > 0.05, `nothing moved while the region still read hide: ${(b - a).toFixed(3)}`);
  assert.ok(c - b < (b - a) / 2,
    `growth did not decay: ${(b - a).toFixed(3)} then ${(c - b).toFixed(3)}`);
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
