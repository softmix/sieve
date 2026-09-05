// Replaces the make/jq/sponge release target from the other extensions -- same
// steps, but node is installed on both Windows and WSL and those aren't.
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, copyFileSync, readdirSync } from "node:fs";

const sh = (cmd, args) => execFileSync(cmd, args, { encoding: "utf8", stdio: "pipe" }).trim();
const run = (cmd, args) => execFileSync(cmd, args, { stdio: "inherit" });

const { name, version, browser_specific_settings: bss } = JSON.parse(readFileSync("manifest.json"));
const id = bss.gecko.id;

// Everything checkable, checked before anything is written. The first version of
// this wrote updates.json and *then* demanded a clean tree, so any actual version
// bump failed on its own write -- it could only pass when it had nothing to do.
if (sh("git", ["status", "--porcelain"])) throw new Error("git state not clean");
if (!sh("git", ["remote"])) throw new Error("no git remote -- create the GitHub repo first");

const xpi = readdirSync("web-ext-artifacts").find(f => f.endsWith(".xpi"));
if (!xpi) throw new Error("no signed xpi -- run `source .env && npm run sign` first");

// Firefox fetches updates.json from raw.githubusercontent at the URL baked into
// the manifest, so it has to be committed and pushed, not merely written. Both
// that and the release asset are fetched unauthenticated: the repo must be public
// or auto-update silently stops working.
const link = `https://github.com/softmix/${name}/releases/download/${version}/${name}.xpi`;
writeFileSync("updates.json",
  JSON.stringify({ addons: { [id]: { updates: [{ version, update_link: link }] } } }, null, 2) + "\n");

if (sh("git", ["status", "--porcelain", "updates.json"])) {
  run("git", ["add", "updates.json"]);
  run("git", ["commit", "-m", version]);
}
run("git", ["push", "-u", "origin", "HEAD"]);

copyFileSync(`web-ext-artifacts/${xpi}`, `${name}.xpi`);
run("gh", ["release", "create", version, `${name}.xpi`, "-t", version, "-n", version]);
