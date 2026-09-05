// Per-site DOM adapters. The model is shared across every site -- only the
// scraping differs, and everything downstream sees the same (text, imageUrl).
//
// Adding a site means a key here AND a matching entry in manifest.json's
// content_scripts.matches. Nothing enforces that; if a site does nothing, that's
// the first thing to check.
//
// image() returns the thumbnail on purpose. It's already decoded in the page so
// the fetch is free, and CLIP resizes to 224px anyway. Point it at the full
// image if classification turns out to need the detail.
// eslint-disable-next-line no-unused-vars
const SITES = {
  "boards.4chan.org": {
    post: ".postContainer",
    text: p => p.querySelector(".postMessage")?.innerText ?? "",
    image: p => p.querySelector(".fileThumb img")?.src ?? null,
  },

  "old.reddit.com": {
    post: ".thing",
    text: p => [".title", ".md"].map(s => p.querySelector(s)?.innerText ?? "").join("\n").trim(),
    image: p => p.querySelector(".thumbnail img")?.src ?? null,
  },
};
