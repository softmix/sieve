// Per-site DOM adapters. The model is shared across every site and every page
// type -- only the scraping differs, and everything downstream sees the same
// (text, imageUrl).
//
// `mount` picks the element the badge is injected into and that collapses when a
// post is hidden -- point it at the site's own visually-boxed element, or the
// badge ends up outside the post rather than inside it. Defaults to `post`.
//
// Matched in order, first hit wins, so put narrower `path` entries above the
// catch-all for the same host. Adding a host also means adding it to
// manifest.json's content_scripts.matches; nothing enforces that pairing, so if
// a new site does nothing at all, check there first.
//
// image() returns the thumbnail on purpose. It's already decoded in the page so
// the fetch is free, and CLIP resizes to 224px anyway. Point it at the full
// image if classification turns out to need the detail.
// eslint-disable-next-line no-unused-vars
const SITES = [
  {
    // Catalog is client-rendered from a `var catalog` blob into a DOM that
    // shares nothing with the board index: thread previews, not posts.
    host: "boards.4chan.org",
    path: /^\/[^/]+\/catalog/,
    post: ".thread",
    // A thread preview has no header row to sit beside, and a floated badge ends
    // up left of a narrow image rather than above it. Give it its own line.
    block: true,
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
    // Reddit's entry column is wide, so floating right puts the badge miles from
    // the post it belongs to.
    side: "left",
    text: p => [".title", ".md"].map(s => p.querySelector(s)?.innerText ?? "").join("\n").trim(),
    image: p => p.querySelector(".thumbnail img")?.src ?? null,
    link: p => (p.dataset.permalink ? location.origin + p.dataset.permalink : null)
      ?? p.querySelector("a.comments")?.href ?? null,
  },
];
