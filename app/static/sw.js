// App-shell cache only; API calls always go to the network.
const SHELL = "shell-v3";
const FILES = ["/", "/index.html", "/styles.css", "/app.js", "/icon.svg", "/manifest.json", "/bg.js", "/motion.js"];
self.addEventListener("install", (e) => e.waitUntil(caches.open(SHELL).then((c) => c.addAll(FILES))));
self.addEventListener("activate", (e) => e.waitUntil(caches.keys().then((ks) =>
  Promise.all(ks.filter((k) => k !== SHELL).map((k) => caches.delete(k))))));
self.addEventListener("fetch", (e) => {
  if (new URL(e.request.url).pathname.startsWith("/api/")) return;
  e.respondWith(fetch(e.request).catch(() => caches.match(e.request)));
});
