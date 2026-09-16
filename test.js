import test from "node:test";
import assert from "node:assert/strict";
import {
  K, Model, ZERO, l2, feats, fit, holdout, usable, counts, score, mapVectors, placeNew, identOf,
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

// Three topics sitting in the same cone, the way a board's posts actually do.
const topics = () => {
  const out = [];
  for (let c = 0; c < 3; c++)
    for (let k = 0; k < 12; k++)
      out.push({ img: jit(emb(10 + c), k, 0.25), txt: jit(emb(20 + c), k + 77, 0.25), topic: c });
  return out;
};

const cos = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };

// Mean cosine within a topic minus mean cosine across topics. The bigger this
// is, the more a neighbour-based layout has to work with.
const separation = (vs, items) => {
  let wi = 0, wn = 0, bi = 0, bn = 0;
  for (let a = 0; a < vs.length; a++)
    for (let b = a + 1; b < vs.length; b++) {
      const c = cos(vs[a], vs[b]);
      if (items[a].topic === items[b].topic) { wi += c; wn++; } else { bi += c; bn++; }
    }
  return wi / wn - bi / bn;
};

test("map vectors separate topics that raw embeddings do not", () => {
  const items = topics();
  const raw = items.map(it => l2([...it.img, ...it.txt]));
  const sep = separation(mapVectors(items).vecs, items);
  const rawSep = separation(raw, items);
  assert.ok(sep > rawSep * 2,
    `centering barely helped: ${rawSep.toFixed(3)} -> ${sep.toFixed(3)}`);
  assert.ok(sep > 0.2, `topics are not separable enough to lay out: ${sep.toFixed(3)}`);
});

test("without centering the cone swamps the topics", () => {
  // The same failure test.js's header warns about, made load-bearing: uncentered,
  // everything is ~0.8 to everything and a neighbour list is noise.
  const items = topics();
  const raw = items.map(it => l2([...it.img, ...it.txt]));
  let lo = 1;
  for (let a = 0; a < raw.length; a++)
    for (let b = a + 1; b < raw.length; b++) lo = Math.min(lo, cos(raw[a], raw[b]));
  assert.ok(lo > 0.6, `uncentered vectors should all be crowded together, floor was ${lo.toFixed(3)}`);
});

test("textless posts do not cluster together for having no text", () => {
  const items = topics();
  // One per topic loses its text, so if they end up neighbours it can only be
  // because of the hole rather than because of what they are.
  const mute = [0, 12, 24];
  for (const i of mute) items[i].txt = ZERO;
  const vs = mapVectors(items).vecs;

  for (const i of mute) {
    const own = vs.map((v, j) => ({ j, c: cos(vs[i], v) }))
      .filter(x => x.j !== i)
      .sort((a, b) => b.c - a.c)[0];
    assert.equal(items[own.j].topic, items[i].topic,
      `a textless post's nearest neighbour was topic ${items[own.j].topic}, not its own`);
  }
});

test("a new post lands among its own topic, not in the middle", () => {
  // The layout has to survive a reload, so newcomers are placed against stored
  // coordinates rather than by refitting. If this drifts toward the centroid of
  // everything, the map slowly turns into a blob and nobody notices until the
  // spatial memory it exists to build has already rotted.
  const items = topics();
  const { mu, vecs } = mapVectors(items);

  // Stand-in layout: each topic parked in its own corner.
  const corners = [[-10, -10], [10, -10], [0, 10]];
  const placed = items.map((it, i) => ({ v: vecs[i], xy: corners[it.topic] }));

  for (let c = 0; c < 3; c++) {
    const fresh = { img: jit(emb(10 + c), 500, 0.25), txt: jit(emb(20 + c), 577, 0.25) };
    // Same mu the layout was built with -- that's what the second argument is for.
    const { vecs: [v] } = mapVectors([fresh], mu);
    const [x, y] = placeNew(placed, v);
    const d = Math.hypot(x - corners[c][0], y - corners[c][1]);
    const other = Math.min(...corners.filter((_, j) => j !== c)
      .map(([ax, ay]) => Math.hypot(x - ax, y - ay)));
    assert.ok(d < other, `topic ${c} landed ${d.toFixed(1)} from home, ${other.toFixed(1)} from a neighbour`);
    assert.ok(d < 4, `topic ${c} drifted ${d.toFixed(1)} toward the middle`);
  }
});

test("placing against an empty layout does not explode", () => {
  const { vecs } = mapVectors(topics());
  assert.deepEqual(placeNew([], vecs[0]), [0, 0]);
});

test("every URL shape for one post resolves to one identity", () => {
  // The bug this pins cost a whole verification round-trip: a thread's URL
  // carries a slug, so /thread/123/some-slug#p456 never reached the #p group and
  // every reply in the thread collapsed onto the OP. 425 posts scored, 150
  // archived, and nothing in the log said why.
  const op = "g/12345/12345";
  for (const [url, want] of [
    ["https://boards.4chan.org/g/thread/12345", op],                        // catalog
    ["https://boards.4chan.org/g/thread/12345#p12345", op],                 // index, OP
    ["https://boards.4chan.org/g/thread/12345/sqt-stupid-questions", op],   // catalog, slugged
    ["https://boards.4chan.org/g/thread/12345/sqt-stupid-questions#p12345", op],
    ["https://boards.4chan.org/g/thread/12345/sqt#p12350", "g/12345/12350"],  // a reply
    ["https://boards.4chan.org/g/thread/12345#p12350", "g/12345/12350"],
  ]) assert.equal(identOf(url)?.id, want, url);

  assert.equal(identOf("https://boards.4chan.org/g/thread/12345/sqt#p12350").thread, 12345);
  assert.equal(identOf("https://boards.4chan.org/vg/thread/9#p9").board, "vg");

  // No identity anywhere else, which is what keeps reddit out of the archive --
  // and therefore out of ambient.
  for (const u of ["https://old.reddit.com/r/g/comments/abc/x/", "", null, undefined])
    assert.equal(identOf(u), null, String(u));
});

test("holdout reports accuracy on data it did not train on", () => {
  const labels = [];
  for (let k = 0; k < 60; k++)
    for (const [i, t, y] of PAIRS) labels.push({ img: jit(i, k), txt: jit(t, k + 991), y });
  const h = holdout(labels);
  assert.ok(h.n > 0 && h.acc > 0.9, JSON.stringify(h));
  assert.equal(holdout([]), null);
});
