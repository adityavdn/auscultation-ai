"use strict";
// Smooth scroll (Lenis) driven by GSAP's ticker, plus scroll reveals. Content stays readable without it.
(() => {
  const { gsap, ScrollTrigger, Lenis } = window;
  if (!gsap || !ScrollTrigger) return;
  gsap.registerPlugin(ScrollTrigger);

  if (Lenis) {
    const lenis = new Lenis({
      lerp: 0.1,
      anchors: { offset: -64 },
      prevent: (node) => node.closest && node.closest(".chat-log, select, .frame"),
    });
    lenis.on("scroll", ScrollTrigger.update);
    gsap.ticker.add((time) => lenis.raf(time * 1000));
    gsap.ticker.lagSmoothing(0);
    window.__lenis = lenis;
  }

  const mm = gsap.matchMedia();
  mm.add("(prefers-reduced-motion: no-preference)", () => {
    const ease = "expo.out";
    gsap.timeline({ defaults: { ease, duration: 1.2 } })
      .from(".hero .eyebrow", { y: 16, autoAlpha: 0 })
      .from(".hero h1 .line", { yPercent: 110, stagger: 0.12 }, "-=0.95")
      .from(".hero .lead", { y: 22, autoAlpha: 0 }, "-=0.9")
      .from(".hero .cta", { y: 16, autoAlpha: 0 }, "-=0.9")
      .from(".hero .stats > *", { y: 12, autoAlpha: 0, stagger: 0.05 }, "-=0.9");

    gsap.to(".hero-inner", {
      yPercent: -14, autoAlpha: 0.15, ease: "none",
      scrollTrigger: { trigger: ".hero", start: "top top", end: "bottom top", scrub: true },
    });

    gsap.utils.toArray("[data-reveal]").forEach((node) => gsap.from(node, {
      y: 36, autoAlpha: 0, duration: 1.1, ease, scrollTrigger: { trigger: node, start: "top 88%", once: true },
    }));

    const items = ".step, .stat, .stage, .app-card";
    gsap.set(items, { autoAlpha: 0, y: 32 });
    ScrollTrigger.batch(items, {
      start: "top 90%", once: true,
      onEnter: (batch) => gsap.to(batch, { autoAlpha: 1, y: 0, duration: 1, ease, stagger: 0.08 }),
    });

    gsap.utils.toArray("[data-count]").forEach((node) => {
      const end = parseFloat(node.dataset.count), dec = Number(node.dataset.dec || 0), state = { v: 0 };
      gsap.to(state, {
        v: end, duration: 1.8, ease,
        scrollTrigger: { trigger: node, start: "top 92%", once: true },
        onUpdate: () => { node.textContent = state.v.toFixed(dec); },
      });
    });
  });
})();
