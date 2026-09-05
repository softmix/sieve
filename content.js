/* global SITES */
const site = SITES[location.hostname];
if (site) run();

async function run() {
  let { threshold } = await browser.storage.local.get({ threshold: 0.85 });
  browser.storage.onChanged.addListener(c => c.threshold && (threshold = c.threshold.newValue));

  // Right-clicking is the whole "this is a bad post" UI. Cheaper and far less
  // fragile than injecting a button into every post on every site.
  let target = null;
  addEventListener("contextmenu", e => (target = e.target.closest?.(site.post)), true);
  browser.runtime.onMessage.addListener(msg => {
    if (msg.type === "teach" && target) teach(target, 1);
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
    if (res.p > threshold) collapse(post, res.p);
  }

  function collapse(post, p) {
    if (post.querySelector(":scope > .sieve-bar")) return;
    const bar = document.createElement("div");
    bar.className = "sieve-bar";
    bar.textContent = `hidden ${p.toFixed(2)} — show`;
    bar.onclick = () => teach(post, 0);
    post.prepend(bar);
    post.classList.add("sieve-hidden");
  }

  async function teach(post, y) {
    await browser.runtime.sendMessage({
      type: "label", y, text: site.text(post), img: site.image(post),
    });
    if (y) return collapse(post, 1);
    post.classList.remove("sieve-hidden");
    post.querySelector(":scope > .sieve-bar")?.remove();
  }
}
