"use strict";
// Background: a heart and a pair of lungs drawn as DNA-like double helices, twisting on the GPU.
// Decorative only. Without WebGL (or three.js) the CSS gradient behind it remains.
(() => {
  const canvas = document.getElementById("bg");
  if (!window.THREE || !canvas) return;
  const THREE = window.THREE;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const small = innerWidth < 760;
  const STEP = small ? 0.05 : 0.03; // sample spacing along each strand (world units)

  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: false, powerPreference: "high-performance" });
  } catch { return; }
  renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);

  // ---------- geometry: helices along polylines ----------
  const buf = { center: [], nrm: [], bin: [], ang: [], side: [], rad: [], size: [], color: [] };
  const v = (x, y, z) => new THREE.Vector3(x, y, z);

  function resample(points, closed) {
    const pts = closed ? [...points, points[0]] : points;
    const out = [pts[0].clone()];
    let carry = 0;
    for (let i = 1; i < pts.length; i++) {
      const a = pts[i - 1], b = pts[i], len = a.distanceTo(b);
      let d = STEP - carry;
      while (d <= len) { out.push(a.clone().lerp(b, d / len)); d += STEP; }
      carry = len - (d - STEP);
    }
    return out;
  }

  // Parallel-transport frames keep the twist smooth along curved paths.
  function helix(points, { radius, twist, c1, c2, closed = false, rungEvery = 7, phase = 0 }) {
    const p = resample(points, closed);
    if (p.length < 3) return;
    let t = p[1].clone().sub(p[0]).normalize();
    let n = Math.abs(t.z) < 0.9 ? v(0, 0, 1).cross(t).normalize() : v(1, 0, 0).cross(t).normalize();
    const col1 = new THREE.Color(c1), col2 = new THREE.Color(c2);
    for (let i = 0; i < p.length; i++) {
      const next = p[Math.min(i + 1, p.length - 1)], prev = p[Math.max(i - 1, 0)];
      const tn = next.clone().sub(prev).normalize();
      const axis = t.clone().cross(tn);
      if (axis.lengthSq() > 1e-10) n.applyAxisAngle(axis.normalize(), Math.acos(Math.min(1, Math.max(-1, t.dot(tn)))));
      t = tn;
      const b = t.clone().cross(n).normalize();
      const a = phase + i * STEP * twist;
      const push = (side, size, col) => {
        buf.center.push(p[i].x, p[i].y, p[i].z); buf.nrm.push(n.x, n.y, n.z); buf.bin.push(b.x, b.y, b.z);
        buf.ang.push(a); buf.side.push(side); buf.rad.push(radius); buf.size.push(size); buf.color.push(col.r, col.g, col.b);
      };
      push(1, 1, col1);
      push(-1, 1, col2);
      if (i % rungEvery === 0) {
        const rung = col1.clone().lerp(col2, 0.5).multiplyScalar(0.55);
        for (const s of [-0.6, -0.2, 0.2, 0.6]) push(s, 0.55, rung);
      }
    }
  }

  // Heart: the classic heart curve, stacked in depth for volume, plus the aortic arch.
  const heartRing = (z, k) => {
    const pts = [];
    for (let i = 0; i < 160; i++) {
      const t = (i / 160) * Math.PI * 2;
      const x = 16 * Math.sin(t) ** 3;
      const y = 13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t);
      pts.push(v((x / 17) * k, (y / 17) * k, z));
    }
    return pts;
  };
  const HEART = { c1: 0xff4d61, c2: 0xff9a7a };
  for (const z of [-0.42, -0.21, 0, 0.21, 0.42]) {
    const k = Math.sqrt(1 - (z / 0.55) ** 2);
    helix(heartRing(z, k), { radius: 0.05, twist: 9, closed: true, ...HEART, phase: z * 6 });
  }
  const arch = [];
  for (let i = 0; i <= 40; i++) {
    const t = i / 40;
    arch.push(v(0.12 - Math.sin(t * Math.PI) * 0.15 - t * 0.55, 0.62 + Math.sin(t * Math.PI) * 0.62, 0));
  }
  helix(arch, { radius: 0.07, twist: 8, ...HEART });
  const heartPointCount = buf.center.length / 3; // heart dots come first; split off below so it can beat

  // Lungs: two lobes stacked in depth (the left one with a cardiac notch) and a branching airway tree.
  const lobe = (cx, dir, z, k) => {
    const pts = [];
    for (let i = 0; i < 180; i++) {
      const t = (i / 180) * Math.PI * 2;
      let x = Math.cos(t) * 1.25 * (1 + 0.18 * Math.sin(t));
      const y = Math.sin(t) * 2.2 + (Math.sin(t) < 0 ? 0.25 * Math.sin(t) : 0);
      const medial = Math.cos(t) * dir < 0; // the inner side faces the heart
      if (medial) x *= 0.62 + 0.18 * Math.abs(Math.sin(t));
      if (medial && dir < 0 && Math.sin(t) < 0.1 && Math.sin(t) > -0.9) x *= 0.75; // cardiac notch (left lung)
      pts.push(v(cx + x * k * dir, y * k, z));
    }
    return pts;
  };
  const LUNG = { c1: 0x37e3c8, c2: 0x6aa8ff };
  for (const z of [-0.6, -0.3, 0, 0.3, 0.6]) {
    const k = Math.sqrt(1 - (z / 0.8) ** 2);
    helix(lobe(2.15, 1, z, k), { radius: 0.045, twist: 8, closed: true, ...LUNG, phase: z * 5 });
    helix(lobe(-2.15, -1, z, k), { radius: 0.045, twist: 8, closed: true, ...LUNG, phase: z * 5 + 1 });
  }
  let seed = 7;
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const branch = (from, to, depth, radius) => {
    helix([from, to], { radius, twist: 10, ...LUNG, rungEvery: 6 });
    if (depth === 0) return;
    const dir = to.clone().sub(from);
    for (const turn of [-1, 1]) {
      const d = dir.clone().multiplyScalar(0.62).applyAxisAngle(v(0, 0, 1), turn * (0.45 + rand() * 0.35));
      d.z += (rand() - 0.5) * 0.5;
      branch(to, to.clone().add(d), depth - 1, radius * 0.8);
    }
  };
  helix([v(0, 3.2, 0), v(0, 1.75, 0)], { radius: 0.09, twist: 9, ...LUNG });
  branch(v(0, 1.75, 0), v(1.2, 0.95, 0.1), small ? 2 : 3, 0.075);
  branch(v(0, 1.75, 0), v(-1.2, 0.95, -0.1), small ? 2 : 3, 0.075);

  // ---------- material: dots placed on the GPU so the helices twist cheaply ----------
  const geo = new THREE.BufferGeometry();
  const attr = (name, arr, size) => geo.setAttribute(name, new THREE.Float32BufferAttribute(arr, size));
  attr("position", buf.center, 3); attr("nrm", buf.nrm, 3); attr("bin", buf.bin, 3); attr("ang", buf.ang, 1);
  attr("side", buf.side, 1); attr("rad", buf.rad, 1); attr("dotSize", buf.size, 1); attr("color", buf.color, 3);
  const uniforms = { uTime: { value: 0 }, uPR: { value: renderer.getPixelRatio() }, uAlpha: { value: 1 }, uSize: { value: small ? 72 : 78 } };
  const material = new THREE.ShaderMaterial({
    uniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, vertexColors: true,
    vertexShader: `
      attribute vec3 nrm; attribute vec3 bin; attribute float ang; attribute float side; attribute float rad; attribute float dotSize;
      uniform float uTime; uniform float uPR; uniform float uSize;
      varying vec3 vColor; varying float vShade;
      void main() {
        float a = ang + uTime * 1.6;
        vec3 p = position + side * rad * (nrm * cos(a) + bin * sin(a));
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        gl_PointSize = uSize * dotSize * uPR / -mv.z;
        vShade = 0.55 + 0.45 * sin(a) * sign(side);
        vColor = color;
        gl_Position = projectionMatrix * mv;
      }`,
    fragmentShader: `
      uniform float uAlpha; varying vec3 vColor; varying float vShade;
      void main() {
        float d = length(gl_PointCoord - 0.5);
        float m = smoothstep(0.5, 0.0, d);
        float k = m * vShade * uAlpha;
        gl_FragColor = vec4(vColor * k, k);
      }`,
  });
  const all = new THREE.Points(geo, material);
  const organs = new THREE.Group();
  organs.add(all);
  scene.add(organs);

  // ---------- layout, motion ----------
  function resize() {
    const w = innerWidth, h = innerHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.position.set(0, 0, w < 760 ? 11.8 : 9.2);
    camera.updateProjectionMatrix();
  }
  resize();
  addEventListener("resize", resize);

  let mx = 0, my = 0, rx = 0, ry = 0, scroll = 0;
  addEventListener("pointermove", (e) => { mx = e.clientX / innerWidth - 0.5; my = e.clientY / innerHeight - 0.5; }, { passive: true });

  const beat = (t) => { // lub-dub at 72 bpm
    const ph = (t % 0.833) / 0.833;
    return Math.exp(-(((ph - 0.04) / 0.05) ** 2)) + 0.55 * Math.exp(-(((ph - 0.24) / 0.05) ** 2));
  };
  let last = 0, clock = 0, frame = 0;
  function tick(now) {
    const dt = last ? Math.min((now - last) / 1000, 1 / 20) : 1 / 60;
    last = now;
    clock += dt;
    const target = Math.min(1.4, scrollY / innerHeight);
    scroll += (target - scroll) * (1 - Math.exp(-dt / 0.25));
    rx += (my * 0.25 - rx) * (1 - Math.exp(-dt / 0.6));
    ry += (mx * 0.5 - ry) * (1 - Math.exp(-dt / 0.6));
    uniforms.uTime.value = clock;
    const breathe = 1 + 0.03 * Math.sin((clock / 4.6) * Math.PI * 2);
    const pulse = 1 + 0.045 * beat(clock);
    organs.rotation.set(rx + 0.05, ry + clock * 0.06 + scroll * 0.9, 0);
    organs.scale.setScalar(breathe * (1 - scroll * 0.18));
    organs.position.set(scroll * (innerWidth < 760 ? 0 : 2.2), scroll * 0.6, 0);
    heart.scale.setScalar(pulse);
    uniforms.uAlpha.value = 0.78 - Math.min(scroll, 1) * 0.43;
    renderer.render(scene, camera);
    frame = requestAnimationFrame(tick);
  }

  // The heart sits low and left of centre, in front of the lungs, as in the chest.
  const heart = new THREE.Group();
  {
    const n = heartPointCount;
    const hg = new THREE.BufferGeometry();
    for (const name of Object.keys(geo.attributes)) {
      const a = geo.attributes[name];
      hg.setAttribute(name, new THREE.Float32BufferAttribute(a.array.slice(0, n * a.itemSize), a.itemSize));
      geo.setAttribute(name, new THREE.Float32BufferAttribute(a.array.slice(n * a.itemSize), a.itemSize));
    }
    heart.add(new THREE.Points(hg, material));
    heart.position.set(-0.55, -0.85, 0.9);
    heart.rotation.set(0.15, -0.35, -0.42);
    heart.scale.setScalar(1);
    organs.add(heart);
  }

  if (still) { uniforms.uTime.value = 0.6; organs.rotation.set(0.05, 0.25, 0); renderer.render(scene, camera); return; }
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { cancelAnimationFrame(frame); frame = 0; } else if (!frame) { last = 0; frame = requestAnimationFrame(tick); }
  });
  frame = requestAnimationFrame(tick);
})();
