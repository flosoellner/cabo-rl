// Minimal offline app-shell cache for the Cabo PWA.
// Cache-first: once installed, the app loads and plays with no network at all.
// Bump CACHE_NAME on any deploy that changes shipped files so clients pick up the update.
const CACHE_NAME = "cabo-shell-v1";
const SHELL_FILES = [
  "./",
  "./index.html",
  "./style.css",
  "./dist/main.js",
  "./dist/app.js",
  "./dist/engine.js",
  "./dist/storage.js",
  "./public/manifest.json",
  "./public/icons/icon-192.png",
  "./public/icons/icon-512.png",
  "./public/icons/apple-touch-icon.png",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(SHELL_FILES)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((names) => Promise.all(names.filter((n) => n !== CACHE_NAME).map((n) => caches.delete(n))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(
    caches.match(event.request).then((cached) => {
      if (cached) return cached;
      return fetch(event.request)
        .then((response) => {
          const copy = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, copy));
          return response;
        })
        .catch(() => cached);
    })
  );
});
