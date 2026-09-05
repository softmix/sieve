const $ = id => document.getElementById(id);

browser.storage.local.get({ threshold: 0.85 }).then(({ threshold }) => {
  $("threshold").value = threshold;
  $("tv").value = threshold;
});

$("threshold").oninput = e => {
  $("tv").value = e.target.value;
  browser.storage.local.set({ threshold: +e.target.value });
};

async function stats() {
  const s = await browser.runtime.sendMessage({ type: "stats" });
  // holdout is null until there are enough labels to split -- say so rather than
  // printing a number that means nothing.
  const h = s.holdout
    ? `holdout accuracy: ${(s.holdout.acc * 100).toFixed(0)}% on ${s.holdout.n} held-out labels`
    : "holdout accuracy: not enough labels yet";
  // Say plainly when filtering is off. Trained on one class this model saturates
  // and scores everything ~1.00, so it stays disabled rather than hiding the page.
  const state = s.ready ? "filtering active"
    : `filtering off — needs ${s.need} of each class (have ${s.pos} hide, ${s.neg} keep)`;
  // "seen" are posts you scrolled past without hiding: weak evidence, gathered
  // automatically, and the reason you don't have to click ✓ constantly.
  const seen = s.pos + s.neg - s.taught;
  $("stats").textContent =
    `${s.taught} clicked (${s.pos} hide, ${s.neg - seen} fine) + ${seen} seen\n${state}\n${h}`;
}
stats();

$("export").onclick = async () => {
  const labels = await browser.runtime.sendMessage({ type: "export" });
  const url = URL.createObjectURL(new Blob([JSON.stringify(labels)], { type: "application/json" }));
  // <a download> rather than browser.downloads, which would cost a permission
  // for something the DOM already does.
  Object.assign(document.createElement("a"), { href: url, download: "sieve-labels.json" }).click();
};

$("reset").onclick = async () => {
  if (!confirm("Delete every label and reset the model?")) return;
  await browser.runtime.sendMessage({ type: "reset" });
  stats();
};
