/* global SITES */
const site = SITES.find(s =>
  s.host === location.hostname && (!s.path || s.path.test(location.pathname)));
if (site) run();

async function run() {
  // hiding=false still scores and badges everything, it just doesn't collapse.
  let { threshold, hiding } = await browser.storage.local.get({ threshold: 0.85, hiding: true });
  browser.storage.onChanged.addListener(c => {
    if (c.threshold) threshold = c.threshold.newValue;
    if (c.hiding) hiding = c.hiding.newValue;
  });

  // Right-click works anywhere in a post; the badge buttons are for bulk work.
  let target = null;
  addEventListener("contextmenu", e => (target = e.target.closest?.(site.post)), true);
  browser.runtime.onMessage.addListener(msg => {
    if (msg.type === "teach" && target) teach(target, msg.y);
  });

  // Every post gets scored, nearest-to-viewport first. Skipping offscreen posts
  // would be cheaper but they'd then flash into view before being hidden.
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

  // Modest rather than whole-page: the queue re-sorts between batches, which is
  // what lets it follow scrolling.
  const BATCH = 16;

  let pumping = false;
  async function pump() {
    if (pumping) return;
    pumping = true;
    while (pending.size) {
      // ponytail: re-sorts all pending each round, O(n^2 log n) to drain a page.
      // Negligible beside one batch, and re-reading position each round is what
      // tracks scrolling -- a priority set at enqueue time goes stale.
      const batch = [...pending].sort((a, b) => distance(a) - distance(b)).slice(0, BATCH);
      for (const p of batch) pending.delete(p);
      await classify(batch);
    }
    pumping = false;
  }

  // ---- badge -------------------------------------------------------------

  const state = new WeakMap();

  // The element the badge lives in and that collapses when hidden. Sites box
  // their posts in an inner element; the outer one puts the badge outside it.
  const host = post => site.mount?.(post) ?? post;

  function badge(post) {
    const into = host(post);
    let el = into.querySelector(":scope > .sieve-tag");
    if (el) return el;

    el = document.createElement("span");
    el.className = "sieve-tag";
    if (site.block) el.dataset.block = "";
    if (site.side) el.dataset.side = site.side;
    el.innerHTML = '<span class="sieve-p"></span>';

    // Peek: reveals without labelling. ✓ asserts "this is fine", so using it to
    // look at a hidden post would poison the label set.
    const p = el.querySelector(".sieve-p");
    p.addEventListener("pointerdown", e => {
      e.preventDefault();
      e.stopImmediatePropagation();
      set(post, { peek: !state.get(post)?.peek });
    }, true);
    for (const ev of ["mousedown", "mouseup", "click", "auxclick"]) {
      p.addEventListener(ev, e => { e.preventDefault(); e.stopImmediatePropagation(); }, true);
    }
    for (const [y, glyph, title] of [[0, "✓", "this post is fine"], [1, "✗", "hide posts like this"]]) {
      const b = document.createElement("button");
      b.textContent = glyph;
      b.title = title;
      // Catalog previews are wrapped in an <a> and other extensions bind their
      // own handlers, so a plain onclick loses the race and navigates instead.
      // Act on pointerdown and swallow every later event in the sequence.
      b.addEventListener("pointerdown", e => {
        e.preventDefault();
        e.stopImmediatePropagation();
        teach(post, y);
      }, true);
      for (const ev of ["mousedown", "mouseup", "click", "auxclick"]) {
        b.addEventListener(ev, e => { e.preventDefault(); e.stopImmediatePropagation(); }, true);
      }
      el.append(b);
    }
    into.prepend(el);
    return el;
  }

  function set(post, patch) {
    const s = Object.assign(state.get(post) ?? {}, patch);
    state.set(post, s);

    const hide = s.mark === "hide" || (s.mark !== "keep" && !!s.auto);
    const el = badge(post);
    // Must be absent, not empty -- the stylesheet keys off [data-mark] existing.
    if (s.mark) el.dataset.mark = s.mark; else delete el.dataset.mark;
    // With hiding off, a would-be-hidden post reads as revealed rather than
    // claiming to be hidden while plainly visible.
    const shown = s.peek || !hiding;
    el.querySelector(".sieve-p").textContent =
      hide ? `hidden ${shown ? "▾" : "▸"}` :    // click to peek, without labelling it
      s.mark === "keep" ? "kept" :
      s.p == null ? "…" : s.p.toFixed(2);       // … means not scored yet
    if (s.tally) el.title = s.tally;
    if (s.p != null) post.dataset.sieve = s.p.toFixed(2);

    host(post).classList.toggle("sieve-hidden", hide && !shown);
  }

  // ---- scoring and teaching ----------------------------------------------

  async function classify(posts) {
    // Dead channel is routine in development: web-ext reloads the background
    // page while old-generation content scripts keep running.
    const res = await browser.runtime.sendMessage({
      type: "score",
      items: posts.map(p => ({ text: site.text(p), img: site.image(p), url: site.link?.(p) ?? null })),
    }).catch(() => null);
    if (!res) return;

    posts.forEach((post, i) => {
      const r = res[i];
      if (!r) return;
      // exact = you marked this precise post, so warmup doesn't apply.
      if (r.exact) return set(post, { p: r.p, mark: r.p ? "hide" : "keep" });
      set(post, { p: r.p, auto: hiding && r.ready && r.p > threshold });
    });
  }

  async function teach(post, y) {
    set(post, { mark: y ? "hide" : "keep" });   // optimistic, so bulk clicking feels instant

    const res = await browser.runtime.sendMessage({
      type: "label", y, text: site.text(post), img: site.image(post),
      // Stored so the options page can link back to the post itself.
      url: site.link?.(post) ?? null,
    }).catch(() => null);
    if (!res) return;

    set(post, {
      tally: `${res.pos} hide / ${res.neg} keep`
        + (res.ready ? "" : ` — filtering starts at ${res.need} of each`),
    });
  }

  // Last: scan() reaches set() and the badge helpers, which are `const` and so
  // are in the temporal dead zone anywhere above this line.
  new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
  scan();
}
