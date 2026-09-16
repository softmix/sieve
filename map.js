// The archive, laid out by UMAP and coloured by score.
//
// Two jobs, and only one of them is colour's. Hot regions are the undesired
// clusters and the warm fringe around them is where a label moves the model
// most -- that's the training half. Finding posts you'd otherwise have missed is
// done by the *layout*: bump order scatters a topic across the catalog and this
// gathers it. The score gradient contributes nothing to that, deliberately --
// the model is trained on hide/keep, not on interest, so almost everything you'd
// want to read scores low along with everything you wouldn't.
//
// No k-means. UMAP's own normalisation handles the anisotropy (it subtracts each
// point's nearest-neighbour distance, which is a local de-coning), and with no
// per-cluster actions and no names there's nothing left for a hard partition to
// do that the layout doesn't already do better.
import { mapVectors, toF32 } from "./model.js";

// The UMD bundle assigns a *namespace* to the global, so the constructor sits
// one level down. Resolved rather than assumed, because `new UMAP()` on the
// namespace fails with "UMAP is not a constructor", which doesn't point here.
const Umap = globalThis.UMAP?.UMAP ?? globalThis.UMAP;

const $ = id => document.getElementById(id);
const say = s => ($("status").textContent = s);
const send = m => browser.runtime.sendMessage(m);

// Same generator as fit()'s, so a re-layout of unchanged data lands identically.
const lcg = (s = 1) => () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;

const cv = $("map"), ctx = cv.getContext("2d");
let pts = [];                              // {n, id, url, text, img, p, mark, hidden, x, y}
let view = { x: 0, y: 0, k: 1 };
let hover = null, pinned = null;

// ---- colour --------------------------------------------------------------

// A two-stop ramp rather than a hue sweep. Sweeping 220deg to 10deg looks
// prettier and ranks worse -- it runs through green and yellow, which read as
// categories rather than as an ordering. This drops luminance as it goes, so it
// survives greyscale and colour blindness.
const A = [110, 150, 210], B = [190, 45, 40];
const fill = p => {
  const t = p < 0 ? 0 : p > 1 ? 1 : p;
  return `rgb(${A[0] + (B[0] - A[0]) * t | 0} ${A[1] + (B[1] - A[1]) * t | 0} ${A[2] + (B[2] - A[2]) * t | 0})`;
};

// Same two as the badge in content.css, so a ring means what it means on a post.
const RING = { 1: "#c0392b", 0: "#27803d" };
// Canvas has no `currentColor`, so the focus ring has to pick a side.
const INK = matchMedia("(prefers-color-scheme: dark)").matches ? "#fff" : "#111";

// ---- layout --------------------------------------------------------------

const toScreen = q => [(q.x - view.x) * view.k + cv.width / 2, (q.y - view.y) * view.k + cv.height / 2];

function fitView() {
  if (!pts.length) return;
  const xs = pts.map(q => q.x), ys = pts.map(q => q.y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  view.x = (x0 + x1) / 2;
  view.y = (y0 + y1) / 2;
  view.k = 0.9 * Math.min(cv.width / (x1 - x0 || 1), cv.height / (y1 - y0 || 1));
}

function resize() {
  const r = $("wrap").getBoundingClientRect();
  const d = devicePixelRatio || 1;
  cv.width = Math.max(1, r.width * d);
  cv.height = Math.max(1, r.height * d);
  draw();
}

function draw() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  const d = devicePixelRatio || 1;
  const r = 3.2 * d;
  const focused = pinned ?? hover;

  for (const q of pts) {
    const [sx, sy] = toScreen(q);
    if (sx < -r || sy < -r || sx > cv.width + r || sy > cv.height + r) continue;

    ctx.beginPath();
    ctx.arc(sx, sy, q === focused ? r * 2 : r, 0, Math.PI * 2);
    ctx.fillStyle = fill(q.p);
    ctx.fill();

    // Rings are categorical on top of the continuous fill: what you decided,
    // and what the filter is collapsing right now.
    const ring = RING[q.mark] ?? (q.hidden ? "#000" : null);
    if (!ring && q !== focused) continue;
    ctx.lineWidth = (q === focused ? 2.5 : 1.4) * d;
    ctx.strokeStyle = q === focused ? INK : ring;
    ctx.globalAlpha = q === focused ? 1 : 0.55;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }
}

// ---- inspection ----------------------------------------------------------

const nearest = (mx, my) => {
  const d = devicePixelRatio || 1;
  let best = null, bd = (14 * d) ** 2;
  for (const q of pts) {
    const [sx, sy] = toScreen(q);
    const dd = (sx - mx * d) ** 2 + (sy - my * d) ** 2;
    if (dd < bd) { bd = dd; best = q; }
  }
  return best;
};

let blobUrl = null;
async function showFocus(q) {
  const box = $("focus");
  if (!q) {
    box.className = "empty";
    box.textContent = "Hover a point to inspect it; click to pin it here.";
    return;
  }
  box.className = "";
  box.replaceChildren();

  const img = document.createElement("img");
  img.alt = "";
  // Stored bytes first -- 4chan deletes a thread's images within days, and the
  // whole reason they're kept is that an old region of the map stays inspectable.
  const got = await browser.storage.local.get(`t${q.n}`);
  const raw = got[`t${q.n}`];
  if (blobUrl) URL.revokeObjectURL(blobUrl);
  blobUrl = null;
  if (raw) {
    const bytes = raw instanceof Uint8Array ? raw : Uint8Array.from(Object.values(raw));
    img.src = blobUrl = URL.createObjectURL(new Blob([bytes], { type: "image/jpeg" }));
  } else if (q.img) {
    img.src = q.img;
  }

  const score = document.createElement("div");
  score.id = "fscore";
  score.textContent = `${q.p.toFixed(2)}${q.hidden ? " — hidden" : ""}`
    + (q.mark === 1 ? " — you hid this" : q.mark === 0 ? " — you kept this" : "");

  const text = document.createElement("div");
  text.id = "ftext";
  text.textContent = q.text || "(no text)";

  const marks = document.createElement("div");
  marks.className = "marks";
  for (const [y, glyph, cls] of [[0, "✓ fine", "keep"], [1, "✗ hide", "hide"]]) {
    const b = document.createElement("button");
    b.className = cls;
    b.textContent = glyph;
    b.onclick = async () => {
      const r = await send({ type: "relabel", key: q.id, y });
      if (r.gone) return say("that post is no longer in the archive");
      q.mark = y;
      say(`${r.pos} hide / ${r.neg} fine`);
      draw();
      showFocus(q);
    };
    marks.append(b);
  }

  const a = document.createElement("a");
  if (q.url) Object.assign(a, { href: q.url, target: "_blank", rel: "noreferrer", textContent: "open the thread" });

  box.append(img, score, text, marks, a);
}

// ---- interaction ---------------------------------------------------------

cv.onmousemove = e => {
  if (drag) {
    const d = devicePixelRatio || 1;
    view.x -= (e.clientX - drag.x) * d / view.k;
    view.y -= (e.clientY - drag.y) * d / view.k;
    drag = { x: e.clientX, y: e.clientY, moved: true };
    return draw();
  }
  const r = cv.getBoundingClientRect();
  const q = nearest(e.clientX - r.left, e.clientY - r.top);
  if (q === hover) return;
  hover = q;
  draw();
  if (!pinned) showFocus(q);
};

let drag = null;
cv.onmousedown = e => { drag = { x: e.clientX, y: e.clientY }; cv.classList.add("drag"); };
addEventListener("mouseup", () => {
  // A drag that never moved is a click, and a click pins whatever is under it.
  if (drag && !drag.moved) { pinned = hover; showFocus(pinned); draw(); }
  drag = null;
  cv.classList.remove("drag");
});

cv.onwheel = e => {
  e.preventDefault();
  const r = cv.getBoundingClientRect();
  const d = devicePixelRatio || 1;
  const [mx, my] = [(e.clientX - r.left) * d, (e.clientY - r.top) * d];
  // Keep whatever is under the pointer fixed while the scale changes.
  const before = [(mx - cv.width / 2) / view.k + view.x, (my - cv.height / 2) / view.k + view.y];
  view.k *= Math.exp(-e.deltaY * 0.0015);
  view.x = before[0] - (mx - cv.width / 2) / view.k;
  view.y = before[1] - (my - cv.height / 2) / view.k;
  draw();
};

addEventListener("resize", resize);

// ---- boot ----------------------------------------------------------------

async function layout(items) {
  // Checked here so boot()'s catch puts it in the header rather than a console
  // nobody has open.
  if (typeof Umap !== "function")
    throw new Error("vendor/umap.js exposed no UMAP constructor — did umap-js change its bundle?");
  const n = items.length;
  const umap = new Umap({
    // The default 15 is 10% of a small archive, which makes the local manifold
    // estimate noise. Lower, and read the plot as suggestive rather than proof.
    nNeighbors: Math.max(2, Math.min(8, n - 1)),
    minDist: 0.15,
    nComponents: 2,
    random: lcg(1),
  });
  const vecs = mapVectors(items).map(v => Array.from(v));
  return umap.fitAsync(vecs, e => {
    if (e % 25 === 0) say(`laying out ${n} posts — ${((e / umap.getNEpochs()) * 100) | 0}%`);
  });
}

async function boot() {
  say("reading the archive…");
  const st = await send({ type: "mapState" });
  const { arc } = await browser.storage.local.get({ arc: [] });
  const live = arc.filter(e => st.p[e.n] !== undefined);

  if (live.length < 5)
    return say("Not much archived yet — browse a board with the extension on and come back.");

  const got = await browser.storage.local.get(live.map(e => `v${e.n}`));
  const items = live.map(e => {
    const v = got[`v${e.n}`];
    return { img: toF32(v.img), txt: toF32(v.txt) };
  });

  const xy = await layout(items);
  pts = live.map((e, i) => ({
    n: e.n, id: e.id, url: e.url, text: e.text, img: e.img,
    p: st.p[e.n], mark: st.mark[e.n], hidden: st.ready && st.p[e.n] > st.threshold,
    x: xy[i][0], y: xy[i][1],
  }));

  const hid = pts.filter(q => q.hidden).length;
  say(`${pts.length} posts, ${hid} currently filtered`
    + (st.ready ? "" : " — filtering is off"));
  resize();
  fitView();
  draw();
}

boot().catch(e => say(`failed: ${e.message}`));
