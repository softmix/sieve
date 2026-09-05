// Bumps updates.json, pushes, and cuts the GitHub release.
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, copyFileSync, readdirSync } from "node:fs";

const sh = (cmd, args) => execFileSync(cmd, args, { encoding: "utf8", stdio: "pipe" }).trim();
const run = (cmd, args) => execFileSync(cmd, args, { stdio: "inherit" });

const { name, version, browser_specific_settings: bss } = JSON.parse(readFileSync("manifest.json"));
const id = bss.gecko.id;

// Checked before anything is written: writing updates.json first would dirty the
// tree and fail this check on every real version bump.
if (sh("git", ["status", "--porcelain"])) throw new Error("git state not clean");
if (!sh("git", ["remote"])) throw new Error("no git remote -- create the GitHub repo first");

// Matched on version: signing leaves older artifacts behind, and picking one
// would ship the wrong build under the right tag.
const xpi = readdirSync("web-ext-artifacts").find(f => f.endsWith(`-${version}.xpi`));
if (!xpi) throw new Error(`no signed xpi for ${version} -- run \`source .env && npm run sign\` first`);

// Firefox fetches updates.json from the URL in the manifest, so it must be
// pushed, not merely written. Both it and the release asset are fetched
// unauthenticated: the repo must be public or auto-update stops working.
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
