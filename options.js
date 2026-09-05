const $ = id => document.getElementById(id);
const send = msg => browser.runtime.sendMessage(msg);

// ---- settings ------------------------------------------------------------

browser.storage.local.get({ threshold: 0.85, seenMax: 300 }).then(s => {
  $("threshold").value = s.threshold;
  $("tv").value = s.threshold;
  $("seenMax").value = s.seenMax;
});

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
  // holdout is null until there's enough to split -- say so rather than print a
  // number that means nothing.
  const h = s.holdout
    ? `holdout accuracy ${(s.holdout.acc * 100).toFixed(0)}% on ${s.holdout.n} held-out labels`
    : "holdout accuracy: not enough labels yet";
  // Trained on one class this model saturates and scores everything ~1.00, so it
  // stays off rather than hiding the whole page. Be explicit about that.
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
    if (c.url) img.src = c.url;
    img.alt = "";

    const txt = document.createElement("div");
    txt.className = "txt";
    txt.textContent = c.text || "(no text)";

    const num = document.createElement("span");
    num.className = "num";
    num.textContent = c.p.toFixed(2);

    li.append(img, txt, num);
    for (const [y, glyph, title] of [[0, "✓", "this post is fine"], [1, "✗", "hide posts like this"]]) {
      const b = document.createElement("button");
      b.textContent = glyph;
      b.title = title;
      b.onclick = async () => {
        li.remove();               // optimistic; these are quick to click through
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
  // <a download> rather than browser.downloads, which would cost a permission
  // for something the DOM already does.
  Object.assign(document.createElement("a"), { href: url, download: "sieve-labels.json" }).click();
};

$("reset").onclick = async () => {
  if (!confirm("Delete every label and reset the model?")) return;
  await send({ type: "reset" });
  stats();
  calls();
};

stats();
calls();
