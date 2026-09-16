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

Each post carries a badge: its score, or `…` if it hasn't been scored yet, plus
✓ / ✗ buttons. Right-clicking a post does the same via the context menu. Posts
scoring above the threshold collapse, leaving the badge, and the score becomes a
`▸` toggle that peeks at the post *without* labelling it — ✓ and ✗ are the only
things that write a label. The score is also on every post in `data-sieve`.

**You mostly only have to mark the bad ones.** Every post you scroll past without
hiding leaves one permanent gradient step behind — `Ambient`, a second weight
vector that `fit()` never touches. A sighting leaves a mark and then gets out of
the training set, so ✓ stays a deliberate signal rather than a chore.

That split is the whole shape of the thing. `labels` holds only deliberate
clicks: small, balanced, high-signal, so a refit stays in the tens of
milliseconds forever and `holdout()` measures something real. Ambient holds
everything else, permanently, at no fitting cost at all.

It only runs for posts that weren't actually hidden — pushing down something you
asked it to catch would train against the catch. Note the gate is *was hidden*,
not *scored high*: with filtering off nothing is hidden, so everything is a
legitimate negative, and that is precisely what lets a broken model climb back
out (see below).

Two things keep it from running away. The error is taken against the **combined**
score, so once a region reads as keep the gradient is ~0 and further sightings
move nothing — self-extinguishing, not self-reinforcing. And `fit()` is handed
ambient's contribution as a fixed offset, so the label weights are a *residual*
on top of it rather than a competing opinion. That second part is not optional:
CLIP embeddings are anisotropic, so ambient accumulates largely along the cone
that every post projects onto, and without the offset it drags true hides down
with everything else and the filter dies quietly. The sweep in `test.js` fails at
7% of hides clearing 0.85 without it.

Filtering stays off until 3 hides and 50 sightings, and that is not politeness —
logistic regression trained on one class is degenerate. Every gradient step
pushes the bias the same way with nothing pushing back, so it saturates and
scores *everything* ~1.00. Four "hide" clicks and no negatives will hide the
entire page, including a post with a USB logo and no relevant text.

Worse, saturation used to be an **absorbing** state: everything hidden means
nothing gets nudged means nothing can push back, and the only exit is a Reset
that costs every label you trained. It's reachable in ordinary use — click ✗
often and ✓ rarely. `PANIC_RATE` switches filtering off when the model is hiding
essentially the whole page, which puts every post back under the threshold and
lets ambient pull it out on its own.

## The archive and the map

Everything seen — hidden or not, with its thumbnail bytes — goes into an
`archive` that `fit()` never reads. Feeding a hide back as evidence would only
confirm what the model already believes, and a store the fitter cannot see makes
that structural rather than a rule someone has to remember.

Eviction is therefore cheap, which it wasn't when one pool did both jobs: the
learning was banked into ambient at insert time, so losing a record costs the
ability to *look* at it and nothing else. Expired threads are pruned from the
catalog's own membership list rather than by asking the server about each of
thousands of posts. Thumbnail bytes are kept because 4chan deletes a thread's
images within days and a training map full of dead thumbnails is not one.

The map (`map.html`) lays that archive out with UMAP and colours it by score.
Two jobs, and only one of them is colour's:

- **training** — hot regions are the undesired clusters, and the warm fringe
  around them is where a label moves the model most. Uncertainty sampling laid
  out in space instead of as a flat twelve-item list.
- **finding** — done by the *layout*, not by any signal. Bump order scatters a
  topic across the catalog and this gathers it. The score gradient contributes
  nothing here, deliberately: the model is trained on hide/keep, not on interest,
  so the posts you want to read score low along with the ones you don't.

There is no k-means. UMAP subtracts each point's nearest-neighbour distance when
it builds its graph, which is a local de-coning and exactly the medicine CLIP's
anisotropy needs; k-means would want the global centering hack instead. With no
per-cluster actions and no names, a hard partition has nothing left to do that
the layout doesn't already do better. Shift-drag lassos a region and opens its
threads — by thread, not by post, since a lasso over replies picks one thread
many times.

The same page opens as a full-screen iframe over the catalog, with that board's
threads lit and the rest of your history dimmed behind them. An iframe rather
than a second renderer in the content script: the inspect panel, the labelling
and the re-layout button all come along, and there is one thing to keep working.

`mapVectors()` centers before laying out, and clusters on the raw embeddings
rather than on `feats()` — the interaction block earns its keep as a
discriminative lift, but it's a 4th-order term with no meaning as a distance.

**The map has three modes: image + text, image only, text only.** They cost no
extra inference at all; both embeddings are already stored per post, and a mode
only changes which blocks go into the layout. Each keeps its own membership and
its own persisted layout, so switching is one UMAP fit the first time and free
after.

Modes exist because a missing modality is not a neutral one. Its block is zero
after centering, so a post lacking it is systematically *less* similar to
everything that has it and drifts into its own region — and imputing the gap
only moves the problem: filling in missing text made the text-only posts behave
and produced a cluster of image-only ones instead. A mode admits only the posts
that have its modality, so there is no hole to cluster on and the means need no
special-casing. On one real archive: 888 posts, 876 with text, 290 with an
image, 278 with both — and it was that gap of 12 that showed up as a blob.

Layouts persist. umap-js's `transform()` can't place a new point without the
`rpForest` and `searchGraph` that `fit()` builds in memory, neither of which
survives a reload — so `placeNew()` does what `initTransform` does instead, which
is the weighted average of a point's neighbours' existing coordinates. Weighted
by *rank*, not raw cosine: even centered, a CLIP point's neighbours sit at
similar similarities, so raw weights are near-uniform and every newcomer drifts
to the middle until the map is a blob again. Re-laying out is a button, because
the drift is slow and you notice it exactly when a new region looks wrong.

The first page load downloads CLIP into the browser cache — 303 MB of fp16
weights, 176 MB vision plus 127 MB text. After that it's local; repeated images
are cached by URL, which on an imageboard is most of them.

Every post gets scored, nearest-to-the-viewport first, re-evaluated after each
one so it follows your scrolling. Skipping offscreen posts would be cheaper but
you'd then watch each one flash into view before being hidden.

Posts are embedded **16 at a time**, which is the single biggest thing about this
pipeline's speed. Vision-tower cost per image:

|              | batch 1 | batch 4 | batch 16 |
|--------------|---------|---------|----------|
| wasm / q8    | 125 ms  | 113 ms  | 111 ms   |
| webgpu/ fp16 | 101 ms  | 25 ms   | **6.3 ms** |

A WebGPU call costs ~101 ms whether it carries 1 image or 16 — it's dispatch-
bound, and the compute was free all along. So unbatched WebGPU actually *loses*
to WASM, which is compute-bound and flat. Batched, it wins by ~18x.

End to end on a 4chan catalog that's ~250–330 ms per post unbatched, against
**~25 ms per post** batched on WebGPU, of which ~1 ms is image fetch.

WASM is **not** kept as a fallback, and that is the important part. A label
stores the embedding, not the post, so a vector is only comparable to vectors
from the same weights on the same device. Measured on three images, fp32 and q8
place the *same image* at cosine 0.86–0.97, and q8 inflates every image×text
similarity by ~0.05 — enough that one of the three changes which caption it
matches best. A model fit on one geometry and fed the other is confidently wrong
rather than merely worse, so a silent fallback corrupts the label set instead of
degrading it. If WebGPU is unavailable sieve stops and says so in the options
page, and every label carries the backend that embedded it so a mix is visible.

There is no cheaper portable dtype to retreat to. fp16 refuses to initialise
outside WebGPU at all (`InsertedPrecisionFreeCast_… node_args.end() was false`);
the dtypes that run on both are fp32 at 606 MB and q8 at 154 MB, and switching to
either invalidates every stored vector.

Batch size is deliberately 16 rather than the whole page: the queue re-sorts
between batches, and that's what lets it follow your scrolling.

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
- **The dev profile lives at `~/.sieve-ffprofile`, outside this directory, and
  must stay there.** web-ext watches the source tree to hot-reload; Firefox
  writes to its profile constantly. A profile in here means an extension reload
  every few seconds — which re-initialises CLIP and kills content scripts before
  they can score anything, so the page just looks broken. `--watch-ignored` did
  not reliably exclude it.
- **`--firefox-profile` needs a path separator in it.** web-ext decides
  path-vs-profile-name by looking for one, so `.ffprofile` becomes `-P .ffprofile`
  (a name), Firefox opens the profile manager, and the debugger connection is
  refused — exactly the same symptom as the `--no-remote` failure above, which
  makes the two easy to confuse. All of this is settled in `web-ext-config.mjs`.
  It matters because web-ext's default is a throwaway profile per run, which
  silently wipes every label you've clicked.
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
  instead of the 0.98 you actually get. This is also why the map centers before
  laying out, and why ambient needs `fit()` to know about it.
- **umap-js's UMD bundle assigns a namespace, not the class.** `window.UMAP` is
  `{ UMAP }`, so `new UMAP(...)` fails with "UMAP is not a constructor" and the
  message points nowhere near the cause.
- **A 4chan thread URL carries a slug.** The resolved permalink is
  `/g/thread/123/some-slug#p456`, so a regex expecting `#p` straight after the id
  still matches, with the fragment group empty, and folds every reply in the
  thread onto the OP — costing them both their archive entry and their ambient
  nudge, silently. `identOf()` lives in `model.js` only because that is the file
  `node --test` can reach.
- **A missing modality is not a zero vector.** Leaving `ZERO` in one block hands
  every post with that gap the same `-mu` block, and they cluster on the hole.
  Worse, averaging a block over the whole set while only some posts contribute
  shrinks the mean, under-applies the centering, and degrades *everyone's*
  neighbourhood. Both halves are pinned in `test.js`.
- **`fit()`'s epochs/lr/decay are tuned against the threshold, not against
  accuracy**, and the difference is not academic. At `decay=1e-3`, or with an
  annealed learning rate, or at 30 epochs instead of 200, the model still ranks
  perfectly — class means 0.83 vs 0.17 — while squashing every score toward 0.5
  so that almost nothing crosses 0.85 and the filter silently does nothing at
  all. Accuracy tests cannot see this. There's a test that asserts against the
  threshold instead; watch that one if you touch those numbers.

## Adding a site

An entry in `sites.js` (`post` selector, `text(post)`, `image(post)`) **and** a
matching pattern in `manifest.json`'s `content_scripts.matches`. Nothing enforces
that pairing; if a new site does nothing at all, that's the first thing to check.
The model is shared across every site — only the scraping differs.

`catalog: true` means the page lists *every* live thread on the board, which is
what makes pruning possible. The archive is 4chan-only, and `identOf()` is that
rule: no identity means no archive entry, and since insert is the ambient
trigger, it also means no nudge. So reddit contributes only deliberate clicks and
never silently shifts a /g/-tuned model with its very different content.

`image()` returns the thumbnail deliberately: it's already decoded in the page so
the fetch is free, and CLIP resizes to 224px anyway. Point it at the full image
if classification turns out to need the detail.

## Options page

Click the sieve button in the toolbar to open it (about:addons → Preferences also
works, but is more clicks than you want while labelling).

Status (including which backend actually loaded), the hide threshold, export,
reset, a link to the map — and **close calls**.

Close calls is uncertainty sampling: the posts whose score sits nearest 0.5,
which are the ones where a label moves the model most. Close calls and
"currently hidden" are now two views of the one archive rather than two separate
rolling windows — the latter used to be sixty entries in memory that died with
the browser. The map is the third view. Labelling from any of them reuses the
stored embeddings and costs no inference at all.

## Retraining

Every label stores its two embeddings, so refitting never re-runs CLIP. The
background page refits the whole log on each new label — 1537 parameters,
milliseconds — which avoids the recency drift of pure online learning, and also
every `REFIT_EVERY` sightings, because ambient drifts the combined score between
clicks and only a refit puts the label weights back in step. Clicks alone are far
too rare: you can browse a whole board without one.

The options page reports holdout accuracy, and it finally means something: it
used to be every fifth of a list dominated by 300 weak seen-entries, and now
every label in it is a deliberate click.

If you want to try a different head, "export labels" gives you JSON of
`{img, txt, y}` ready for `sklearn.linear_model.LogisticRegression` on the same
1536-d concat. Paste `coef_`/`intercept_` back into `w`/`b`. Note it won't carry
the ambient term, which is where most of your browsing actually lives.

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
