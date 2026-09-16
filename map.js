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
import { mapVectors, toF32, identOf, inside } from "./model.js";

// Same page, embedded over a catalog by the content script. The difference is
// only emphasis: the posts on the board you're looking at stay lit and the rest
// of your history dims behind them, so you see where today sits in it.
const OVERLAY = location.hash === "#overlay";

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
  // In overlay mode, frame this board rather than the whole archive -- but fall
  // back to everything if none of today's threads are on the map yet.
  const on = here ? pts.filter(q => q.here) : pts;
  const use = on.length > 1 ? on : pts;
  if (!use.length) return;
  const xs = use.map(q => q.x), ys = use.map(q => q.y);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  view.x = (x0 + x1) / 2;
  view.y = (y0 + y1) / 2;
  // Capped, or a board whose threads all landed in one spot zooms to absurdity.
  view.k = Math.min(60, 0.9 * Math.min(cv.width / (x1 - x0 || 1), cv.height / (y1 - y0 || 1)));
}

function resize() {
  const r = $("wrap").getBoundingClientRect();
  const d = devicePixelRatio || 1;
  cv.width = Math.max(1, r.width * d);
  cv.height = Math.max(1, r.height * d);
  draw();
}

let here = null;    // ids of the posts on the page underneath, in overlay mode

// Shift-drag rings a region. A lasso rather than a rectangle because the point
// is to take the blob you can see, and blobs aren't rectangles -- that
// flexibility is also the reason there's no k-means partition to select from.
let lasso = null;
let sel = new Set();
const SEL = "#00a0ff";

// Posts, not threads: a lasso over a thread's replies selects the same thread
// many times, and opening it once is what you meant.
const threads = () => [...new Set([...sel].map(q => {
  const it = identOf(q.url);
  return it && `https://boards.4chan.org/${it.board}/thread/${it.thread}`;
}).filter(Boolean))];

function showSel() {
  const t = threads().length;
  $("selbar").hidden = !sel.size;
  $("selcount").textContent = `${sel.size} selected — ${t} thread${t === 1 ? "" : "s"}`;
  $("opentabs").textContent = `open ${t} in tabs`;
  $("opentabs").disabled = !t;
}

function draw() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  const d = devicePixelRatio || 1;
  const r = 3.2 * d;
  const focused = pinned ?? hover;

  for (const q of pts) {
    const [sx, sy] = toScreen(q);
    if (sx < -r || sy < -r || sx > cv.width + r || sy > cv.height + r) continue;

    ctx.globalAlpha = here && !q.here ? 0.18 : 1;
    ctx.beginPath();
    ctx.arc(sx, sy, q === focused ? r * 2 : r, 0, Math.PI * 2);
    ctx.fillStyle = fill(q.p);
    ctx.fill();
    ctx.globalAlpha = 1;

    // Rings are categorical on top of the continuous fill: what you decided,
    // and what the filter is collapsing right now.
    const ring = sel.has(q) ? SEL : RING[q.mark] ?? (q.hidden ? "#000" : null);
    if (!ring && q !== focused) continue;
    ctx.lineWidth = (q === focused || sel.has(q) ? 2.5 : 1.4) * d;
    ctx.strokeStyle = q === focused ? INK : ring;
    ctx.globalAlpha = q === focused || sel.has(q) ? 1 : 0.55;
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  if (lasso?.length > 1) {
    ctx.beginPath();
    ctx.moveTo(lasso[0][0], lasso[0][1]);
    for (const [x, y] of lasso.slice(1)) ctx.lineTo(x, y);
    ctx.closePath();
    ctx.strokeStyle = SEL;
    ctx.lineWidth = 1.5 * d;
    ctx.setLineDash([5 * d, 4 * d]);
    ctx.stroke();
    ctx.setLineDash([]);
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

const canvasXY = e => {
  const r = cv.getBoundingClientRect();
  const d = devicePixelRatio || 1;
  return [(e.clientX - r.left) * d, (e.clientY - r.top) * d];
};

cv.onmousemove = e => {
  if (lasso) {
    lasso.push(canvasXY(e));
    return draw();
  }
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
cv.onmousedown = e => {
  if (e.shiftKey) { lasso = [canvasXY(e)]; return; }
  drag = { x: e.clientX, y: e.clientY };
  cv.classList.add("drag");
};

addEventListener("mouseup", () => {
  if (lasso) {
    // A shift-click with no drag means "clear", which beats a modifier nobody
    // would guess.
    sel = new Set(lasso.length < 3 ? [] : pts.filter(q => inside(...toScreen(q), lasso)));
    lasso = null;
    showSel();
    return draw();
  }
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

let live = [], items = [];

// A full UMAP fit. Deliberately manual after the first one: placement drifts
// slowly, and you notice it exactly when a new region looks wrong -- which is
// the right moment to be offered the button rather than having the map
// rearranged under you on every open.
async function relayout() {
  if (typeof Umap !== "function")
    throw new Error("vendor/umap.js exposed no UMAP constructor — did umap-js change its bundle?");

  const n = items.length;
  const { mu, vecs } = mapVectors(items);
  const umap = new Umap({
    // The default 15 is 10% of a small archive, which makes the local manifold
    // estimate noise. Lower, and read the plot as suggestive rather than proof.
    nNeighbors: Math.max(2, Math.min(8, n - 1)),
    minDist: 0.15,
    nComponents: 2,
    random: lcg(1),
  });
  const coords = await umap.fitAsync(vecs.map(v => Array.from(v)), e => {
    if (e % 25 === 0) say(`laying out ${n} posts — ${((e / umap.getNEpochs()) * 100) | 0}%`);
  });

  const xy = Object.fromEntries(live.map((e, i) => [e.n, coords[i]]));
  await send({ type: "saveLayout", xy, mu: Array.from(mu) });
  return xy;
}

function settle(xy, note) {
  pts = live
    .filter(e => xy[e.n])
    .map(e => ({
      n: e.n, id: e.id, url: e.url, text: e.text, img: e.img,
      p: st.p[e.n], mark: st.mark[e.n], hidden: st.ready && st.p[e.n] > st.threshold,
      x: xy[e.n][0], y: xy[e.n][1], here: here ? here.has(e.id) : false,
    }));
  const hid = pts.filter(q => q.hidden).length;
  say(`${pts.length} posts, ${hid} currently filtered`
    + (st.offMap ? `, ${st.offMap} text-only left off` : "")
    + (st.ready ? "" : " — filtering is off") + (note ? ` — ${note}` : ""));
  resize();
  fitView();
  draw();
}

// The content script sends the board's post links once the frame has loaded.
// It may arrive either side of boot() finishing, so apply whatever is ready.
addEventListener("message", e => {
  if (e.data?.sieve !== "here") return;
  here = new Set(e.data.urls.map(u => identOf(u)?.id).filter(Boolean));
  $("lg-here").hidden = false;
  if (!pts.length) return;
  for (const q of pts) q.here = here.has(q.id);
  fitView();
  draw();
});

let st = null;
async function boot() {
  say("reading the archive…");
  st = await send({ type: "mapState" });
  const { arc } = await browser.storage.local.get({ arc: [] });
  live = arc.filter(e => st.p[e.n] !== undefined);

  if (live.length < 5)
    return say("Not much archived yet — browse a board with the extension on and come back.");

  const got = await browser.storage.local.get(live.map(e => `v${e.n}`));
  items = live.map(e => {
    const v = got[`v${e.n}`];
    return { img: toF32(v.img), txt: toF32(v.txt) };
  });

  const c = await send({ type: "coords" });
  const xy = c.needLayout ? await relayout() : c.xy;
  settle(xy, c.placed ? `${c.placed} newly placed` : "");

  $("opentabs").onclick = async () => {
    const urls = threads();
    $("opentabs").disabled = true;
    const r = await send({ type: "openTabs", urls });
    say(r.capped ? `opened ${r.opened} — capped, lasso fewer` : `opened ${r.opened} threads`);
    $("opentabs").disabled = false;
  };
  $("clearsel").onclick = () => { sel = new Set(); showSel(); draw(); };

  $("refit").hidden = false;
  $("refit").onclick = async () => {
    $("refit").disabled = true;
    try { settle(await relayout()); } finally { $("refit").disabled = false; }
  };
}

if (OVERLAY) {
  const close = () => parent.postMessage({ sieve: "close" }, "*");
  $("close").hidden = false;
  $("close").onclick = close;
  addEventListener("keydown", e => e.key === "Escape" && close());
}

boot().catch(e => say(`failed: ${e.message}`));
