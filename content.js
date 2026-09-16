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

  // Delegated rather than per-button, because 4chan rebuilds post markup (inline
  // quotes, its own hide control) and a badge that round-trips through innerHTML
  // survives as markup with its handlers gone. badge() then finds it and hands it
  // back, so those controls are dead for good -- and a dead <button> inside
  // 4chan's delform is a submit button, which is how clicking ✓ became a POST to
  // the delete endpoint. A document listener can't be lost that way.
  // Capture, so the site's own click handlers don't get there first.
  addEventListener("click", e => {
    const b = e.target.closest?.(".sieve-tag > *");
    const post = b && e.target.closest(site.post);
    if (!post) return;
    e.preventDefault();
    e.stopPropagation();
    // Peek reveals without labelling. ✓ asserts "this post is fine", so using it
    // to look at a hidden post would poison the label set.
    if (b.dataset.sieveY === undefined) set(post, { peek: !state.get(post)?.peek });
    else teach(post, +b.dataset.sieveY);
  }, true);

  // Every post gets scored, nearest-to-viewport first. Skipping offscreen posts
  // would be cheaper but they'd then flash into view before being hidden.
  const pending = new Set();
  const seen = new WeakSet();

  // Anything the archive holds for this board that isn't on the catalog has
  // 404'd. Snapshot the *first* substantial render and only that one: 4chan's
  // catalog search re-renders #threads with just the matches, and a snapshot
  // taken after you've typed in it would look like the whole board had expired.
  // The size floor also covers the catalog not having rendered yet at
  // document_idle, and the background refuses implausibly small sets anyway.
  let pruned = false;
  const prune = posts => {
    if (pruned || !site.catalog || posts.length < 20) return;
    pruned = true;
    const threads = posts
      .map(p => +(/\/thread\/(\d+)/.exec(site.link?.(p) ?? "")?.[1] ?? 0))
      .filter(Boolean);
    const board = /^\/([^/]+)\//.exec(location.pathname)?.[1];
    if (board && threads.length)
      browser.runtime.sendMessage({ type: "prune", board, threads }).catch(() => {});
  };

  // The map, over the catalog. An iframe of the extension's own map page rather
  // than a second renderer here: the inspect panel, labelling and re-layout come
  // with it, and there's only one thing to keep working.
  const ORIGIN = new URL(browser.runtime.getURL("map.html")).origin;
  let frame = null;

  function toggleMap() {
    if (frame) { frame.remove(); frame = null; return; }
    frame = document.createElement("iframe");
    frame.id = "sieve-overlay";
    frame.src = browser.runtime.getURL("map.html") + "#overlay";
    frame.onload = () => frame?.contentWindow.postMessage({
      sieve: "here",
      urls: [...document.querySelectorAll(site.post)].map(p => site.link?.(p)).filter(Boolean),
    }, ORIGIN);
    document.body.append(frame);
  }

  // The frame can't remove itself, so its close button asks.
  addEventListener("message", e => {
    if (e.origin === ORIGIN && e.data?.sieve === "close" && frame) toggleMap();
  });

  // Inserted *before* the thread container rather than inside it: 4chan rebuilds
  // that container on its own sort and filter, and anything within goes with it.
  const mapLink = posts => {
    if (!site.catalog || document.getElementById("sieve-open")) return;
    const box = posts[0]?.parentElement;
    if (!box?.parentElement) return;
    const a = document.createElement("a");
    a.id = "sieve-open";
    a.textContent = "▦ sieve map";
    a.title = "the whole archive, laid out — this board's threads highlighted";
    a.onclick = toggleMap;
    box.parentElement.insertBefore(a, box);
  };

  const scan = () => {
    const before = pending.size;
    const posts = [...document.querySelectorAll(site.post)];
    prune(posts);
    mapLink(posts);
    for (const p of posts)
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

    for (const [y, glyph, title] of [[0, "✓", "this post is fine"], [1, "✗", "hide posts like this"]]) {
      const b = document.createElement("button");
      // Default is type=submit, and 4chan wraps every post in a form.
      b.type = "button";
      b.textContent = glyph;
      b.title = title;
      b.dataset.sieveY = y;
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
      tally: `${res.pos} hide / ${res.neg} fine`
        + (res.ready ? "" : ` — filtering starts at ${res.need} hides`),
    });
  }

  // Last: scan() reaches set() and the badge helpers, which are `const` and so
  // are in the temporal dead zone anywhere above this line.
  new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
  scan();
}
