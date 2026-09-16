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
// catalog: this page lists *every* live thread on the board, so anything the
//         archive holds for this board that isn't here has 404'd. That's how
//         expired posts get pruned without asking the server about each one.
// nav:    places the map link. Takes a factory rather than an element, because a
//         page can carry more than one anchor and each wants its own. Returns
//         {found, added}; nothing found and the caller floats the link over the
//         page instead. Does the insertion itself because the right anchor is a
//         per-site judgement, and it must be idempotent -- it runs on every scan,
//         which is what puts the link back when something wipes it.
// block:  give the badge its own line instead of floating it.
// side:   "left" to float left.
// image:  the thumbnail, deliberately -- already decoded in the page, and CLIP
//         resizes to 224px anyway.

// Shared by every 4chan page, because the map is a global tool and the board
// index in 4chan X's json-index mode is a different entry from the catalog.
const FOURCHAN_NAV = make => {
  // The path, not the host: the top link points at 4chan.org and the bottom one
  // at 4channel.org. Not the link text either, which is the part that changes
  // when they restyle. More appear as a page fills in, so this has to stay
  // correct when called again with a longer list.
  const ads = [...document.querySelectorAll('a[href*="/advertise"]')];

  // Appended to the anchor's *parent* rather than placed after the anchor: the
  // brackets around each link are text nodes either side of it, so inserting
  // straight after lands between them -- "[Advertise on 4chan [sieve map]]".
  //
  // Counted per parent rather than checked with a boolean, so two anchors
  // sharing one container get one link each rather than the second being
  // mistaken for already-done.
  let added = 0;
  for (const box of new Set(ads.map(a => a.parentElement).filter(Boolean))) {
    const want = ads.filter(a => a.parentElement === box).length;
    const have = box.querySelectorAll(":scope > .sieve-open").length;
    for (let i = have; i < want; i++) { box.append(" ", make()); added++; }
  }
  return { found: ads.length, added };
};

// eslint-disable-next-line no-unused-vars
const SITES = [
  {
    // Client-rendered thread previews; shares no DOM with the board index.
    host: "boards.4chan.org",
    path: /^\/[^/]+\/catalog/,
    post: ".thread",
    catalog: true,
    nav: FOURCHAN_NAV,
    block: true,   // no header row to sit beside, and narrow images sit next to a float
    text: p => p.querySelector(".teaser")?.innerText ?? "",
    image: p => p.querySelector("img.thumb")?.src ?? null,
    link: p => p.querySelector("a[href*='/thread/']")?.href ?? null,
  },
  {
    // Board index and thread pages, both server-rendered.
    host: "boards.4chan.org",
    post: ".postContainer",
    nav: FOURCHAN_NAV,
    mount: p => p.querySelector(".post"),
    text: p => p.querySelector(".postMessage")?.innerText ?? "",
    image: p => p.querySelector(".fileThumb img")?.src ?? null,
    // The post's own "No." link. On the index it's `thread/N#pM`; inside a thread
    // it's a bare `#pM` that .href resolves against the thread URL -- both land on
    // the post. Matching /thread/ anywhere instead picks up a quotelink from the
    // post body, which points at somebody else's thread.
    link: p => p.querySelector("a[title='Link to this post']")?.href ?? null,
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
