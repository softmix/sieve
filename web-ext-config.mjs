import { homedir } from "node:os";
import { join } from "node:path";
import { mkdirSync } from "node:fs";

// The dev profile must live OUTSIDE the extension directory: web-ext watches the
// source tree to hot-reload, Firefox writes to its profile constantly, and the
// two together give a reload loop every few seconds. --watch-ignored did not
// reliably exclude it. It has to persist somewhere, though -- web-ext's default
// is a throwaway profile per run, which wipes every label on each restart.
const profile = join(homedir(), ".sieve-ffprofile");
mkdirSync(profile, { recursive: true });

export default {
  verbose: true,   // required, or web-ext doesn't relay Firefox's stdout at all
                   // and the console.stdout prefs below produce nothing

  // Top level so `build` and `sign` can't drift apart. web-ext already drops
  // node_modules, web-ext-artifacts, *.xpi and .git.
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
    // Without this, Windows Firefox hands the launch to an existing instance and
    // exits; web-ext then reports a refused debugger connection.
    args: ["--no-remote"],
    pref: [
      "devtools.console.stdout.chrome=true",
      "devtools.console.stdout.content=true",
    ],
  },
};
