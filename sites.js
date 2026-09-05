// Per-site DOM adapters; only the scraping differs, the model is shared.
//
// Matched in order, first hit wins, so narrower `path` entries go above the
// catch-all for the same host. A new host also needs adding to manifest.json's
// content_scripts.matches -- nothing enforces that pairing, so check it first if
// a site does nothing at all.
//
// mount:  element the badge goes in, and that collapses when hidden. Should be
//         the site's own boxed element, else the badge sits outside the post.
//         Defaults to `post`.
// block:  give the badge its own line instead of floating it.
// side:   "left" to float left.
// image:  the thumbnail, deliberately -- already decoded in the page, and CLIP
//         resizes to 224px anyway.
// eslint-disable-next-line no-unused-vars
const SITES = [
  {
    // Client-rendered thread previews; shares no DOM with the board index.
    host: "boards.4chan.org",
    path: /^\/[^/]+\/catalog/,
    post: ".thread",
    block: true,   // no header row to sit beside, and narrow images sit next to a float
    text: p => p.querySelector(".teaser")?.innerText ?? "",
    image: p => p.querySelector("img.thumb")?.src ?? null,
    link: p => p.querySelector("a[href*='/thread/']")?.href ?? null,
  },
  {
    // Board index and thread pages, both server-rendered.
    host: "boards.4chan.org",
    post: ".postContainer",
    mount: p => p.querySelector(".post"),
    text: p => p.querySelector(".postMessage")?.innerText ?? "",
    image: p => p.querySelector(".fileThumb img")?.src ?? null,
    link: p => p.querySelector("a[href*='/thread/']")?.href ?? null,
  },
  {
    host: "old.reddit.com",
    post: ".thing",
    mount: p => p.querySelector(".entry"),
    side: "left",   // the entry column is wide; floated right lands nowhere near the post
    text: p => [".title", ".md"].map(s => p.querySelector(s)?.innerText ?? "").join("\n").trim(),
    image: p => p.querySelector(".thumbnail img")?.src ?? null,
    link: p => (p.dataset.permalink ? location.origin + p.dataset.permalink : null)
      ?? p.querySelector("a.comments")?.href ?? null,
  },
];
