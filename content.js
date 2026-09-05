/* global SITES */
const site = SITES[location.hostname];
if (site) run();

async function run() {
  let { threshold } = await browser.storage.local.get({ threshold: 0.85 });
  browser.storage.onChanged.addListener(c => c.threshold && (threshold = c.threshold.newValue));

  // Right-clicking is the whole teaching UI, in both directions. Cheaper and far
  // less fragile than injecting buttons into every post on every site.
  let target = null;
  addEventListener("contextmenu", e => (target = e.target.closest?.(site.post)), true);
  browser.runtime.onMessage.addListener(msg => {
    if (msg.type === "teach" && target) teach(target, msg.y);
  });

  // Classify just before a post scrolls in, not on page load -- a long thread is
  // hundreds of posts and each image is an inference.
  const io = new IntersectionObserver(es => {
    for (const e of es) if (e.isIntersecting) { io.unobserve(e.target); classify(e.target); }
  }, { rootMargin: "300px" });

  const seen = new WeakSet();
  const scan = () => {
    for (const p of document.querySelectorAll(site.post))
      if (!seen.has(p)) { seen.add(p); io.observe(p); }
  };
  new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
  scan();

  async function classify(post) {
    // web-ext reloads the background page on every source edit while content
    // scripts from the old generation keep running, so a dead-channel error here
    // is routine during development. Leave the post alone rather than throwing
    // once per post on the page.
    const res = await browser.runtime.sendMessage({
      type: "score", text: site.text(post), img: site.image(post),
    }).catch(() => null);
    if (!res) return;
    post.dataset.sieve = res.p.toFixed(2);   // every post carries its score, for tuning the threshold
    // Until both classes have a few labels the score means nothing, so record it
    // for tuning but don't act on it. res.exact bypasses that: you marked this
    // exact post, so it hides regardless of what the model currently thinks.
    if (res.exact) { if (res.p) collapse(post, 1, true); return; }
    if (res.ready && res.p > threshold) collapse(post, res.p);
  }

  const bar = (post, text) => {
    post.querySelector(":scope > .sieve-bar")?.remove();
    const el = document.createElement("div");
    el.className = "sieve-bar";
    el.textContent = text;
    post.prepend(el);
    return el;
  };

  function collapse(post, p, exact) {
    if (post.classList.contains("sieve-hidden")) return;
    const label = exact ? "hidden — you marked this one" : `hidden ${p.toFixed(2)} — click if this is fine`;
    bar(post, label).onclick = () => teach(post, 0);
    post.classList.add("sieve-hidden");
  }

  async function teach(post, y) {
    const res = await browser.runtime.sendMessage({
      type: "label", y, text: site.text(post), img: site.image(post),
    }).catch(() => null);
    if (!res) return;

    // Feedback matters here: marking a visible post "fine" otherwise looks like
    // nothing happened, and the tally is what tells you how far you still are
    // from the point where filtering switches on.
    const tally = `${res.pos} hide / ${res.neg} keep`
      + (res.ready ? "" : `, filtering starts at ${res.need} of each`);

    if (y) {
      post.classList.add("sieve-hidden");
      bar(post, `hidden — ${tally}`).onclick = () => teach(post, 0);
      return;
    }
    post.classList.remove("sieve-hidden");
    const el = bar(post, `marked fine — ${tally}`);
    setTimeout(() => el.remove(), 3000);
  }
}
