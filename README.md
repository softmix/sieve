# sieve

Hides posts you'd hide. Learns from clicks, not from a word list — and scores the
image and the text *together*, so it can flag a post whose picture and whose
words are each individually unremarkable.

CLIP embeds both into the same 512-d space; a logistic regression runs over
`[img | txt | img⊙txt]`. That third block is the point. See the test named
"without the interaction block the same task is unlearnable" — a plain
`[img | txt]` concat provably cannot express two conjunctions that disagree
("image A is only bad with text A, image B only with text B"), and the test
fails at chance without it.

Everything runs locally. Nothing leaves the browser except the one-time model
download from Hugging Face.

## Use

Right-click a post → "sieve: hide posts like this" or "sieve: this post is
fine". That's the whole interface. Posts scoring above the threshold collapse to
a one-line bar; clicking the bar marks it fine and expands it. Every post carries
its score in `data-sieve`, so inspect a few to pick a threshold in the options
page.

**You have to teach it both directions.** Filtering stays off until there are at
least 3 labels of each class, and this is not politeness — logistic regression
trained on one class is degenerate. Every gradient step pushes the bias the same
way with nothing pushing back, so it saturates and scores *everything* ~1.00.
Feed it four "hide" clicks and no "fine" clicks and it will hide the entire page,
including a post with a USB logo and no relevant text. `usable()` in `model.js`
refuses to filter until both classes exist; the options page says so plainly.

Labels are also class-balanced when fitting, so a handful of hides doesn't get
drowned out once the keeps pile up.

Each post carries a badge: the score, or `…` if it hasn't been scored yet, plus
✓ / ✗ buttons for bulk labelling. On a hidden post the score becomes a `▸` peek
toggle — clicking it reveals the post *without* labelling it, so you can check a
hide before deciding. ✓ and ✗ are the only things that write a label.

The first page load downloads ~40 MB of CLIP weights into the browser cache.
After that it's local and cached; repeated images are cached by URL, which on an
imageboard is most of them.

Measured on a 4chan catalog: **~170–330 ms of compute per post** (WASM,
single-threaded). Inference is serialised behind one ORT session, so on a catalog
page — where ~200 posts become visible at once — the queue wait dominates at
2–3 s and takes about a minute to drain. Thread and index pages don't have this
problem. If it becomes annoying, the fix is to drop queued work for posts that
have scrolled away rather than to make inference faster.

## Dev

```
npm test          # the model, no browser needed
npm run dev       # web-ext run against a scratch Firefox profile
```

`npm run dev` prints extension `console.log` and JS errors straight to the
terminal, which is worth more than it sounds — see below. Editing a source file
reloads the extension automatically; content-script changes also need a tab
reload.

### Things that cost time to work out

- **`--args=--no-remote` is mandatory.** Without it, Windows Firefox hands the
  launch to your already-running instance and exits, web-ext's debugger
  connection is refused, and you get a stray tab in your real browser.
- **Logging to the terminal needs `--pref=devtools.console.stdout.chrome=true`**
  (both prefs are already in the `dev` script). Without it you're stuck with the
  Browser Console GUI.
- **Don't use `--keep-profile-changes`/`--firefox-profile`** unless the profile
  was made by Firefox itself (`firefox -CreateProfile`). Pointed at an empty
  directory, Firefox exits before initialising and you get the same refused
  connection as the `--no-remote` failure, which is a confusing coincidence.
  The cost of skipping it is re-downloading CLIP once per browser start.
- **`transformers.web.min.js` is not standalone.** It imports
  `onnxruntime-common` and `onnxruntime-web/webgpu` as bare specifiers, expecting
  a bundler. An import map would fix that with no build step, except it has to be
  an inline `<script>` and the extension CSP blocks inline scripts — so
  `vendor.mjs` rewrites the two specifiers as it copies. It throws if they ever
  stop matching, so a transformers upgrade fails loudly.
- **MV2 on purpose.** A persistent background page keeps CLIP and the embedding
  caches in memory. Under MV3 the event page gets killed and reloads ~40 MB of
  weights.
- Model *weights* are fetched from Hugging Face at run time and that's fine —
  they're data. The ONNX *runtime* is vendored, because remotely-hosted code is
  what gets an add-on rejected.
- **CLIP embeddings are anisotropic.** They sit in a narrow cone, so two
  unrelated images still have cosine ~0.8. The synthetic vectors in `test.js`
  reproduce this deliberately — with near-orthogonal vectors every test here is
  easier than reality, and the one-class blow-up in particular measures 0.86
  instead of the 0.98 you actually get.
- `fit()` anneals the learning rate. With a flat rate it hasn't converged by the
  last epoch, and the order you happened to click labels in shifts scores by up
  to 0.17 — enough to flip a post across the threshold.

## Adding a site

An entry in `sites.js` (`post` selector, `text(post)`, `image(post)`) **and** a
matching pattern in `manifest.json`'s `content_scripts.matches`. Nothing enforces
that pairing; if a new site does nothing at all, that's the first thing to check.
The model is shared across every site — only the scraping differs.

`image()` returns the thumbnail deliberately: it's already decoded in the page so
the fetch is free, and CLIP resizes to 224px anyway. Point it at the full image
if classification turns out to need the detail.

## Retraining

Every label stores its two embeddings, so refitting never re-runs CLIP. The
background page already refits the whole log on each new label — 1537 parameters,
milliseconds — which avoids the recency drift of pure online learning. The
options page reports holdout accuracy.

If you want to try a different head, "export labels" gives you JSON of
`{img, txt, y}` ready for `sklearn.linear_model.LogisticRegression` on the same
1536-d concat. Paste `coef_`/`intercept_` back into `w`/`b`.

The obvious upgrade if accuracy stalls with plenty of labels: `img⊙txt` is a
*diagonal* bilinear form, pairing dimension i only with dimension i. A small MLP
head over the same input learns interactions it can't reach.

## Release

Signed unlisted, self-hosted off a GitHub release, same as imagetabs/unfucker —
but with node instead of make/jq/sponge, which aren't installed on either the
Windows or WSL side here.

```
source .env       # same export WEB_EXT_API_KEY / WEB_EXT_API_SECRET file as the others
npm run sign      # web-ext sign --channel=unlisted
npm run release   # rewrites updates.json, requires a clean tree, gh release create
```

`web-ext` reads the credentials from the environment, so `.env` has to be
sourced rather than just present — `npm` won't pick it up on its own.
