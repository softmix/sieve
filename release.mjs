// Replaces the make/jq/sponge release target from the other extensions -- same
// steps, but node is already installed on both Windows and WSL and those aren't.
import { execFileSync } from "node:child_process";
import { readFileSync, writeFileSync, copyFileSync, readdirSync } from "node:fs";

const sh = (cmd, args) => execFileSync(cmd, args, { encoding: "utf8", stdio: "pipe" }).trim();

const { name, version, browser_specific_settings: bss } = JSON.parse(readFileSync("manifest.json"));
const id = bss.gecko.id;
const link = `https://github.com/softmix/${name}/releases/download/${version}/${name}.xpi`;

const updates = { addons: { [id]: { updates: [{ version, update_link: link }] } } };
writeFileSync("updates.json", JSON.stringify(updates, null, 2) + "\n");

if (sh("git", ["status", "--porcelain"])) throw new Error("git state not clean");

const xpi = readdirSync("web-ext-artifacts").find(f => f.endsWith(".xpi"));
if (!xpi) throw new Error("no signed xpi -- run `npm run sign` first");
copyFileSync(`web-ext-artifacts/${xpi}`, `${name}.xpi`);

execFileSync("gh", ["release", "create", version, `${name}.xpi`, "-t", version, "-n", version],
  { stdio: "inherit" });
