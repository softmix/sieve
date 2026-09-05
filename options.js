const $ = id => document.getElementById(id);
const send = msg => browser.runtime.sendMessage(msg);

// ---- settings ------------------------------------------------------------

browser.storage.local.get({ threshold: 0.85, seenMax: 300, hiding: true }).then(s => {
  $("threshold").value = s.threshold;
  $("tv").value = s.threshold;
  $("seenMax").value = s.seenMax;
  $("hiding").checked = s.hiding;
});

$("hiding").onchange = e => browser.storage.local.set({ hiding: e.target.checked });

$("threshold").oninput = e => {
  $("tv").value = e.target.value;
  browser.storage.local.set({ threshold: +e.target.value });
};

$("seenMax").onchange = e => {
  const n = Math.max(20, Math.min(3000, +e.target.value || 300));
  e.target.value = n;
  browser.storage.local.set({ seenMax: n });
};

// ---- status --------------------------------------------------------------

async function stats() {
  const s = await send({ type: "stats" });
  const seen = s.pos + s.neg - s.taught;
  const h = s.holdout
    ? `holdout accuracy ${(s.holdout.acc * 100).toFixed(0)}% on ${s.holdout.n} held-out labels`
    : "holdout accuracy: not enough labels yet";
  const state = s.ready
    ? "filtering active"
    : `filtering OFF — needs ${s.need} of each class (have ${s.pos} hide, ${s.neg} keep)`;

  $("stats").textContent =
    `${state}\n${s.taught} clicked (${s.pos} hide, ${s.neg - seen} fine) + ${seen} seen\n`
    + `${h}\nrunning on ${s.backend}`;
}

// ---- close calls ---------------------------------------------------------

async function calls() {
  const list = await send({ type: "closeCalls", n: 12 });
  const ul = $("calls");
  ul.replaceChildren();

  if (!list.length) {
    ul.append(Object.assign(document.createElement("li"), {
      className: "empty",
      textContent: "Nothing yet — browse a page with the extension on and posts will collect here.",
    }));
    return;
  }

  for (const c of list) {
    const li = document.createElement("li");

    const img = document.createElement("img");
    // c.img is the thumbnail, c.url the post. Labels predating url have none.
    if (c.img) img.src = c.img;
    img.alt = "";
    img.loading = "lazy";

    const txt = document.createElement("div");
    txt.className = "txt";
    txt.textContent = c.text || "(no text)";

    const a = document.createElement("a");
    a.className = "post";
    a.href = c.url || c.img || "#";
    a.target = "_blank";
    a.rel = "noreferrer";
    a.append(img, txt);

    const num = document.createElement("span");
    num.className = "num";
    num.textContent = c.p.toFixed(2);

    li.append(a, num);
    for (const [y, glyph, title] of [[0, "✓", "this post is fine"], [1, "✗", "hide posts like this"]]) {
      const b = document.createElement("button");
      b.textContent = glyph;
      b.title = title;
      b.onclick = async () => {
        li.remove();               // optimistic, so these are quick to click through
        await send({ type: "relabel", key: c.key, y });
        stats();
      };
      li.append(b);
    }
    ul.append(li);
  }
}

$("more").onclick = calls;

// ---- data ----------------------------------------------------------------

$("export").onclick = async () => {
  const labels = await send({ type: "export" });
  const url = URL.createObjectURL(new Blob([JSON.stringify(labels)], { type: "application/json" }));
  // <a download> rather than browser.downloads, which costs a permission.
  Object.assign(document.createElement("a"), { href: url, download: "sieve-labels.json" }).click();
};

$("import").onclick = () => $("file").click();

$("file").onchange = async e => {
  const file = e.target.files[0];
  if (!file) return;
  e.target.value = "";                 // so picking the same file twice re-fires
  let parsed;
  try {
    parsed = JSON.parse(await file.text());
  } catch {
    return void ($("io").textContent = "That file isn't valid JSON.");
  }
  const r = await send({ type: "import", labels: parsed });
  $("io").textContent = r.added
    ? `Imported ${r.added} labels${r.skipped ? `, skipped ${r.skipped} malformed` : ""}.`
    : "Nothing importable in that file.";
  stats();
  calls();
};

$("reset").onclick = async () => {
  if (!confirm("Delete every label and reset the model?")) return;
  await send({ type: "reset" });
  stats();
  calls();
};

stats();
calls();
