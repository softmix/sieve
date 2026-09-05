// Copies the ONNX runtime out of node_modules so nothing is fetched from a CDN
// at run time. Remote *code* is what gets an add-on rejected; model weights are
// data and stay remote (first run downloads them into the browser cache).
import { copyFileSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from "node:fs";

const FILES = [
  // The "bundle" variant inlines the wasm *loader*; only the .wasm below is
  // fetched at run time, from env.backends.onnx.wasm.wasmPaths.
  ["onnxruntime-web/dist/ort.webgpu.bundle.min.mjs", "ort.webgpu.mjs"],
  ["onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs", "ort-wasm-simd-threaded.asyncify.mjs"],
  ["onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm", "ort-wasm-simd-threaded.asyncify.wasm"],
];

// transformers.web.min.js leaves onnxruntime as bare specifiers for a bundler to
// resolve. An import map would fix it without a build step, but it has to be an
// inline <script> and the extension CSP blocks those -- so rewrite the two
// specifiers here instead of pulling in a bundler for it.
const REMAP = {
  "onnxruntime-common": "./onnxruntime-common/index.js",
  "onnxruntime-web/webgpu": "./ort.webgpu.mjs",
};

mkdirSync("vendor/onnxruntime-common", { recursive: true });
for (const [from, to] of FILES) copyFileSync(`node_modules/${from}`, `vendor/${to}`);

let src = readFileSync("node_modules/@huggingface/transformers/dist/transformers.web.min.js", "utf8");
for (const [bare, rel] of Object.entries(REMAP)) {
  // Fail loudly on upgrade rather than shipping a module that won't load.
  if (!src.includes(`"${bare}"`)) throw new Error(`no import of "${bare}" to remap -- did transformers change?`);
  src = src.replaceAll(`"${bare}"`, `"${rel}"`);
}
writeFileSync("vendor/transformers.js", src);

// Multi-file ESM package -- index.js imports ./tensor.js and friends, so the
// whole tree has to come along.
const common = "node_modules/onnxruntime-common/dist/esm";
const js = readdirSync(common).filter(f => f.endsWith(".js"));
for (const f of js) copyFileSync(`${common}/${f}`, `vendor/onnxruntime-common/${f}`);

console.log(`vendored ${FILES.length + 1} files + ${js.length} onnxruntime-common modules`);
