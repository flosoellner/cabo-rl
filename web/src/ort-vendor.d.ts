// The vendored onnxruntime-web build (public/ort/ort.wasm.min.mjs) has no
// co-located .d.ts - it's copied straight from the package's dist/ folder,
// not resolved through node_modules typings. Untyped as `any` is fine
// here: netAgent.ts is the only caller, and it's a thin wrapper around a
// handful of well-known calls (InferenceSession.create, session.run, new
// Tensor(...), env.wasm.wasmPaths).
declare module "*.mjs" {
  const mod: any;
  export = mod;
}
