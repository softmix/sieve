/* global SITES */
const site = SITES.find(s =>
  s.host === location.hostname && (!s.path || s.path.test(location.pathname)));
if (site) run();

async function run() {
  let { threshold } = await browser.storage.local.get({ threshold: 0.85 });
  browser.storage.onChanged.addListener(c => c.threshold && (threshold = c.threshold.newValue));

  // Right-click still works anywhere in a post; the buttons are for bulk work.
  let target = null;
  addEventListener("contextmenu", e => (target = e.target.closest?.(site.post)), true);
  browser.runtime.onMessage.addListener(msg => {
    if (msg.type === "teach" && target) teach(target, msg.y);
  });

  // Every post gets scored eventually, nearest-to-viewport first. Skipping
  // offscreen posts would be cheaper but you'd then see each one flash into view
  // before being hidden, which defeats the point.
  const pending = new Set();
  const seen = new WeakSet();

  const scan = () => {
    const before = pending.size;
    for (const p of document.querySelectorAll(site.post))
      if (!seen.has(p)) {
        seen.add(p);
        set(p, {});     // draw the badge immediately so "not scored yet" is visible
        pending.add(p);
      }
    if (pending.size > before) pump();
  };

  const distance = p => {
    const r = p.getBoundingClientRect();
    return r.bottom < 0 ? -r.bottom : r.top > innerHeight ? r.top - innerHeight : 0;
  };

  // On WebGPU a batch of 16 costs the same wall-clock as a batch of 1, so this
  // is close to free. Kept modest rather than whole-page because the queue
  // re-sorts between batches -- that's what lets it follow your scrolling.
  const BATCH = 16;

  let pumping = false;
  async function pump() {
    if (pumping) return;
    pumping = true;
    while (pending.size) {
      // ponytail: sorts all pending each round, so O(n^2 log n) to drain a page.
      // At ~200 posts that's noise beside one batch, and re-reading position
      // every round means it tracks scrolling for free -- a priority assigned at
      // enqueue time goes stale the moment you move.
      const batch = [...pending].sort((a, b) => distance(a) - distance(b)).slice(0, BATCH);
      for (const p of batch) pending.delete(p);
      await classify(batch);
    }
    pumping = false;
  }

  // ---- badge -------------------------------------------------------------

  const state = new WeakMap();

  function badge(post) {
    let el = post.querySelector(":scope > .sieve-tag");
    if (el) return el;

    el = document.createElement("span");
    el.className = "sieve-tag";
    el.innerHTML = '<span class="sieve-p"></span>';

    // Peek. Without this the only way to see a hidden post is ✓, which asserts
    // "this is fine" -- you'd have to label a post before you could look at it,
    // and peeking at correctly-hidden posts would poison the label set.
    el.querySelector(".sieve-p").onclick = e => {
      e.preventDefault();
      e.stopPropagation();
      set(post, { peek: !state.get(post)?.peek });
    };
    for (const [y, glyph, title] of [[0, "✓", "this post is fine"], [1, "✗", "hide posts like this"]]) {
      const b = document.createElement("button");
      b.textContent = glyph;
      b.title = title;
      // Posts are usually inside links on catalog pages; without this, marking
      // one navigates away.
      b.onclick = e => { e.preventDefault(); e.stopPropagation(); teach(post, y); };
      el.append(b);
    }
    post.prepend(el);
    return el;
  }

  function set(post, patch) {
    const s = Object.assign(state.get(post) ?? {}, patch);
    state.set(post, s);

    const hide = s.mark === "hide" || (s.mark !== "keep" && !!s.auto);
    const el = badge(post);
    // Must be absent, not empty -- the stylesheet keys off [data-mark] existing.
    if (s.mark) el.dataset.mark = s.mark; else delete el.dataset.mark;
    el.querySelector(".sieve-p").textContent =
      hide ? `hidden ${s.peek ? "▾" : "▸"}` :   // click to peek, without labelling it
      s.mark === "keep" ? "kept" :
      s.p == null ? "…" : s.p.toFixed(2);       // … means not scored yet
    if (s.tally) el.title = s.tally;
    if (s.p != null) post.dataset.sieve = s.p.toFixed(2);

    post.classList.toggle("sieve-hidden", hide && !s.peek);
  }

  // ---- scoring and teaching ----------------------------------------------

  async function classify(posts) {
    // web-ext reloads the background page on every source edit while content
    // scripts from the old generation keep running, so a dead-channel error here
    // is routine during development. Leave the posts alone rather than throwing.
    const res = await browser.runtime.sendMessage({
      type: "score",
      items: posts.map(p => ({ text: site.text(p), img: site.image(p) })),
    }).catch(() => null);
    if (!res) return;

    posts.forEach((post, i) => {
      const r = res[i];
      if (!r) return;
      // exact: you marked this precise post, so it's a stored fact and warmup
      // doesn't apply. Otherwise the score only acts once both classes exist.
      if (r.exact) return set(post, { p: r.p, mark: r.p ? "hide" : "keep" });
      set(post, { p: r.p, auto: r.ready && r.p > threshold });
    });
  }

  async function teach(post, y) {
    set(post, { mark: y ? "hide" : "keep" });   // optimistic, so bulk clicking feels instant

    const res = await browser.runtime.sendMessage({
      type: "label", y, text: site.text(post), img: site.image(post),
    }).catch(() => null);
    if (!res) return;

    set(post, {
      tally: `${res.pos} hide / ${res.neg} keep`
        + (res.ready ? "" : ` — filtering starts at ${res.need} of each`),
    });
  }

  // Last: scan() reaches set() and the badge helpers, which are `const` and so
  // are still in the temporal dead zone anywhere above this line.
  new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
  scan();
}
