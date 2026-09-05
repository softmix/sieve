import { homedir } from "node:os";
import { join } from "node:path";
import { mkdirSync } from "node:fs";

// The dev profile lives OUTSIDE the extension directory, and that is the whole
// point. web-ext watches the source tree for changes to hot-reload the
// extension; Firefox writes to its profile constantly. Put the profile in here
// and you get a reload loop every few seconds -- which re-downloads nothing but
// does re-initialise CLIP, kill every content script before it can score, and
// reset every badge on the page. --watch-ignored did not reliably exclude it.
//
// It has to persist somewhere, though: web-ext's default is a throwaway profile
// per run, so every restart silently wipes browser.storage.local, i.e. every
// label you have clicked.
const profile = join(homedir(), ".sieve-ffprofile");
mkdirSync(profile, { recursive: true });

export default {
  verbose: true,   // required: without it web-ext does not relay Firefox's stdout,
                   // so the console.stdout prefs below produce nothing

  // Top level so `build` and `sign` can't drift apart. web-ext already drops
  // node_modules, web-ext-artifacts, *.xpi and .git; these are the rest of the
  // repo that has no business inside the packaged add-on. `npm run build` then
  // `npm run manifest` prints what actually shipped.
  ignoreFiles: [
    "*.md",
    "package.json",
    "package-lock.json",
    "test.js",
    "vendor.mjs",
    "release.mjs",
    "web-ext-config.mjs",
    "updates.json",
    ".env",
    ".gitignore",
  ],

  run: {
    firefox: process.env.SIEVE_FIREFOX ?? "C:/Program Files (x86)/Mozilla Firefox/firefox.exe",
    firefoxProfile: profile,
    profileCreateIfMissing: true,
    keepProfileChanges: true,
    // Without --no-remote, Windows Firefox hands the launch to your existing
    // instance and exits; web-ext then reports a refused debugger connection.
    args: ["--no-remote"],
    pref: [
      "devtools.console.stdout.chrome=true",
      "devtools.console.stdout.content=true",
    ],
  },
};
