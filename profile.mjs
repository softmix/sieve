// Creates the dev profile directory.
//
// web-ext makes a throwaway profile per run, which means every restart wipes
// browser.storage.local -- i.e. every label you've clicked, plus the ~40MB CLIP
// download. --keep-profile-changes fixes that, but web-ext decides whether
// --firefox-profile is a *path* or a named profile by testing whether the
// directory exists, so if it doesn't it guesses "name" and fails with
// "cannot be resolved to a profile path". Creating it first settles that.
import { mkdirSync } from "node:fs";

mkdirSync(".ffprofile", { recursive: true });
