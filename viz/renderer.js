// nbody-viz — the browser half: layers 3 and 4 of the graphics stack.
//
// The browser supplies the window and the swapchain (layers 1 and 2). This
// file supplies the GPU API calls and the rendering model, and the wasm module
// supplies the physics. Nothing is fetched from a CDN and nothing is bundled.
//
// The whole field is one instanced draw call: a four-vertex quad, drawn once
// per particle, positioned from a buffer of (x, y, mass, heat) copied straight
// out of wasm memory. Per-frame CPU work is one bufferSubData and one draw.

const BASE = 0, SIMD = 1;
const MODE = { base: 0, simd: 1, stacked: 2 };
const MODE_NAME = ["base", "simd", "stacked"];

// --- palette ------------------------------------------------------------
// Additive blending only ever adds light, so the ground has to be dark for
// anything to read. Overlapping discs sum toward white on their own, which is
// where the accretion glow comes from — no per-particle colour logic.
const CLEAR = [0x05 / 255, 0x07 / 255, 0x0d / 255, 1.0];
const TINT = [0xff / 255, 0xd9 / 255, 0xb0 / 255];

// World-space half-extent of the view at zoom 0. The seeded disk has radius 1.
const VIEW_HALF_EXTENT = 1.4;
// Disc radius in world units at mass 1, before the sqrt(mass) scaling.
const BASE_RADIUS = 0.008;

// Wall clock offered to the physics each frame, in seconds — what is left of a
// 60 Hz frame after the draw and the browser's own work. Under `stacked` the
// two worlds split it, which is what lets the faster kernel run more ticks and
// pull its simulated clock ahead.
//
// This budget and the default n are chosen together. Measured at n = 1000,
// simd finishes the ten ticks RFC §2.4 allows and base does not, so simd holds
// ~35 real seconds per orbit while base takes ~87. Below n = 750 neither
// kernel is stressed and the panels run in lockstep; above n = 1400 both are
// starved and the whole thing crawls.
const FRAME_BUDGET = 0.014;

const VERT = `#version 300 es
layout(location = 0) in vec2 a_corner;    // quad corner, in [-1, 1]
layout(location = 1) in vec4 a_particle;  // x, y, mass, heat

uniform vec2  u_scale;   // world -> clip, corrected for this viewport's aspect
uniform vec2  u_centre;  // world-space point at the middle of the viewport
uniform float u_radius;  // disc radius at mass 1

out vec2 v_corner;

void main() {
    v_corner = a_corner;
    // Mass maps to area in 2D, so radius goes as its square root.
    float r = u_radius * sqrt(a_particle.z);
    vec2 world = a_particle.xy + a_corner * r;
    gl_Position = vec4((world - u_centre) * u_scale, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;

in vec2 v_corner;
uniform vec3 u_tint;
out vec4 fragColor;

void main() {
    // Distance from the quad's centre turns the quad into a disc, and the
    // smooth falloff antialiases its edge without any multisampling.
    float d = length(v_corner);
    float a = smoothstep(1.0, 0.0, d);
    a *= a;  // tighten the core so dense regions read as bright, not flat
    fragColor = vec4(u_tint * a, 1.0);
}`;

// --- configuration, carried in the URL -----------------------------------
// The address bar is the record of the run on screen: a link reproduces it
// exactly. This is the page's version of what bench/main.zig does when it
// prints the seed and full config above every table.

const DEFAULTS = {
    n: 1000, seed: 0xc0ffee, preset: 0, merging: 1, mode: MODE.stacked,
    // Slider positions, both exponents: speed = 10^speed, extent = 1.4 * 2^zoom.
    speed: 0, zoom: 0,
};

function readConfig() {
    const q = new URLSearchParams(location.search);
    const int = (key, lo, hi) => {
        const raw = q.get(key);
        if (raw === null) return DEFAULTS[key];
        const v = Number(raw.trim());            // Number() accepts 0x-prefixed hex
        if (!Number.isInteger(v) || v < lo || v > hi) return DEFAULTS[key];
        return v;
    };
    const real = (key, lo, hi) => {
        const raw = q.get(key);
        if (raw === null) return DEFAULTS[key];
        const v = Number(raw);
        if (!Number.isFinite(v) || v < lo || v > hi) return DEFAULTS[key];
        return v;
    };
    return {
        n: int("n", 1, 65536),
        seed: int("seed", 0, 0xffffffff),
        preset: int("preset", 0, 1),
        merging: int("merging", 0, 1),
        mode: int("mode", 0, 2),
        speed: real("speed", -1.5, 0),
        zoom: real("zoom", -1.2, 1.6),
    };
}

function writeConfig(cfg) {
    const q = new URLSearchParams();
    q.set("n", cfg.n);
    q.set("seed", "0x" + cfg.seed.toString(16).toUpperCase());
    q.set("preset", cfg.preset);
    q.set("merging", cfg.merging);
    q.set("mode", cfg.mode);
    q.set("speed", cfg.speed.toFixed(2));
    q.set("zoom", cfg.zoom.toFixed(2));
    history.replaceState(null, "", "?" + q.toString());
}

// --- GL plumbing ---------------------------------------------------------

function fail(message) {
    const el = document.getElementById("error");
    el.textContent = message;
    el.style.display = "grid";
    throw new Error(message);
}

function compile(gl, type, source) {
    const sh = gl.createShader(type);
    gl.shaderSource(sh, source);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        fail("shader failed to compile:\n" + gl.getShaderInfoLog(sh));
    }
    return sh;
}

function buildProgram(gl) {
    const prog = gl.createProgram();
    gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
        fail("program failed to link:\n" + gl.getProgramInfoLog(prog));
    }
    return prog;
}

// One VAO and one instance buffer per world. The quad is shared: it is the
// same four corners for every particle in both panels.
function buildPanel(gl, quad, maxParticles, floatsPerParticle) {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);

    gl.bindBuffer(gl.ARRAY_BUFFER, quad);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);

    const instances = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, instances);
    gl.bufferData(gl.ARRAY_BUFFER, maxParticles * floatsPerParticle * 4, gl.DYNAMIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);
    // The divisor is what makes this one draw call instead of n of them:
    // attribute 1 advances once per instance rather than once per vertex.
    gl.vertexAttribDivisor(1, 1);

    gl.bindVertexArray(null);
    return { vao, instances };
}

// --- boot ----------------------------------------------------------------

const canvas = document.getElementById("gl");
const gl = canvas.getContext("webgl2", { alpha: false, antialias: false });
if (!gl) fail("This page needs WebGL2, which this browser did not provide.");

let wasm;
try {
    const response = await fetch("./nbody.wasm");
    const { instance } = await WebAssembly.instantiateStreaming(response, {
        env: { now: () => performance.now() },
    });
    wasm = instance.exports;
} catch (e) {
    fail("Could not load nbody.wasm: " + e.message +
         "\n\nServe this directory over HTTP rather than opening the file directly.");
}

const FLOATS = wasm.floatsPerParticle();
const LANES = wasm.laneCount();

const program = buildProgram(gl);
const u = {
    scale: gl.getUniformLocation(program, "u_scale"),
    centre: gl.getUniformLocation(program, "u_centre"),
    radius: gl.getUniformLocation(program, "u_radius"),
    tint: gl.getUniformLocation(program, "u_tint"),
};

const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

gl.clearColor(...CLEAR);
gl.disable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
// Additive: overlapping particles sum their light and saturate to white.
gl.blendFunc(gl.ONE, gl.ONE);

let cfg = readConfig();
let panels = null;      // [base, simd], allocated to the current n
let panelCapacity = 0;

function restart() {
    writeConfig(cfg);
    if (!wasm.start(cfg.n, cfg.seed, cfg.preset, cfg.merging, cfg.mode)) {
        fail("The simulation rejected that configuration.");
    }
    if (panelCapacity < cfg.n) {
        // Release the previous pair before allocating a wider one, or every
        // restart at a larger n strands a VAO and a buffer on the GPU.
        if (panels) for (const p of panels) {
            gl.deleteVertexArray(p.vao);
            gl.deleteBuffer(p.instances);
        }
        panels = [
            buildPanel(gl, quad, cfg.n, FLOATS),
            buildPanel(gl, quad, cfg.n, FLOATS),
        ];
        panelCapacity = cfg.n;
    }
    document.body.dataset.mode = MODE_NAME[cfg.mode];
    syncControls();
    for (const el of document.querySelectorAll('[data-f="lanes"]')) el.textContent = LANES;
}

// --- the frame -----------------------------------------------------------

function resize() {
    // Without the device-pixel-ratio scaling the canvas is upscaled from CSS
    // pixels and every disc edge is soft on a retina display.
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
        canvas.width = w;
        canvas.height = h;
    }
}

// A fresh view each frame rather than a cached one. Growing the wasm heap
// detaches every existing Float32Array over memory.buffer; constructing the
// view per frame costs nothing (it is a window onto the same bytes, not a
// copy) and sidesteps the detachment entirely.
function particleView(which) {
    const n = wasm.particleCount(which);
    if (n === 0) return null;
    return new Float32Array(wasm.memory.buffer, wasm.renderBufferOffset(which), n * FLOATS);
}

function drawPanel(which, x, y, w, h) {
    const data = particleView(which);
    if (!data) return;

    gl.viewport(x, y, w, h);

    // Correct for this viewport's own aspect, so a half-width panel in stacked
    // mode does not squash the disk.
    const extent = VIEW_HALF_EXTENT * Math.pow(2, cfg.zoom);
    const aspect = w / h;
    const halfX = aspect >= 1 ? extent * aspect : extent;
    const halfY = aspect >= 1 ? extent : extent / aspect;

    gl.uniform2f(u.scale, 1 / halfX, 1 / halfY);
    gl.uniform2f(u.centre, 0, 0);
    gl.uniform1f(u.radius, BASE_RADIUS);
    gl.uniform3f(u.tint, ...TINT);

    const panel = panels[which];
    gl.bindVertexArray(panel.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, panel.instances);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, data.length / FLOATS);
}

function draw() {
    gl.useProgram(program);
    gl.clear(gl.COLOR_BUFFER_BIT);

    const w = canvas.width, h = canvas.height;
    if (cfg.mode === MODE.stacked) {
        const half = Math.floor(w / 2);
        drawPanel(BASE, 0, 0, half, h);
        drawPanel(SIMD, half, 0, w - half, h);
    } else {
        drawPanel(cfg.mode === MODE.base ? BASE : SIMD, 0, 0, w, h);
    }
}

const hud = {
    [BASE]: document.getElementById("hud-base"),
    [SIMD]: document.getElementById("hud-simd"),
};

function magnitude(ns) {
    if (ns <= 0) return "—";
    if (ns >= 1e6) return (ns / 1e6).toFixed(2) + " M";
    if (ns >= 1e3) return (ns / 1e3).toFixed(1) + " k";
    return Math.round(ns).toString();
}

function updateHud(which) {
    if (!wasm.running(which)) return;
    const el = hud[which];
    const rt = wasm.realtimeFactor(which);
    el.querySelector('[data-f="count"]').textContent = wasm.particleCount(which);
    el.querySelector('[data-f="clock"]').textContent = wasm.clockSeconds(which).toFixed(1) + "s";
    el.querySelector('[data-f="ns"]').textContent = magnitude(wasm.nsPerTick(which));
    el.querySelector('[data-f="rt"]').textContent = rt > 0 ? rt.toFixed(2) : "—";
    const stats = el.querySelector(".stats");
    stats.classList.toggle("slow", rt > 0 && rt < 1);
    stats.classList.toggle("fast", rt >= 1);
}

// --- the loop ------------------------------------------------------------

let running = false;
let last = performance.now();

function frame(t) {
    // Cap the reported frame time so a backgrounded tab does not return with a
    // multi-second frame.
    const elapsed = Math.min((t - last) / 1000, 0.25);
    last = t;

    resize();
    if (running) {
        // Under stacked the two worlds share one frame, so each gets half the
        // budget. The faster kernel fits more ticks into its half and its
        // simulated clock pulls ahead — the comparison, made visible.
        const share = cfg.mode === MODE.stacked ? FRAME_BUDGET / 2 : FRAME_BUDGET;
        wasm.advance(elapsed * Math.pow(10, cfg.speed), share);
    }
    draw();
    updateHud(BASE);
    updateHud(SIMD);

    requestAnimationFrame(frame);
}

// --- controls ------------------------------------------------------------

const form = document.getElementById("controls");
const runButton = document.getElementById("run");

function setRunning(next) {
    running = next;
    runButton.textContent = running ? "pause" : "start";
    runButton.dataset.state = running ? "running" : "paused";
    // Reset the frame clock so a long pause is not charged to the first frame.
    last = performance.now();
}

function syncControls() {
    form.n.value = cfg.n;
    form.seed.value = "0x" + cfg.seed.toString(16).toUpperCase();
    form.preset.value = cfg.preset;
    form.mode.value = cfg.mode;
    form.merging.checked = cfg.merging === 1;
    form.speed.value = cfg.speed;
    form.zoom.value = cfg.zoom;
    form.speedout.value = Math.pow(10, cfg.speed).toFixed(2) + "×";
    form.zoomout.value = (VIEW_HALF_EXTENT * Math.pow(2, cfg.zoom)).toFixed(2) + " R";
}

form.addEventListener("submit", (e) => {
    e.preventDefault();
    const seed = Number(form.seed.value.trim());
    cfg = {
        ...cfg,
        n: Math.min(65536, Math.max(1, Math.round(Number(form.n.value) || DEFAULTS.n))),
        seed: Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff ? seed : DEFAULTS.seed,
        preset: Number(form.preset.value),
        mode: Number(form.mode.value),
        merging: form.merging.checked ? 1 : 0,
    };
    restart();
});

// Selects and the checkbox reseed immediately; the text and number fields wait
// for the button, so a half-typed value is never applied.
for (const name of ["preset", "mode", "merging"]) {
    form[name].addEventListener("change", () => form.requestSubmit());
}

// The sliders are view and pacing settings, not physics: they take effect
// without reseeding, so you can slow a run down while watching it.
for (const name of ["speed", "zoom"]) {
    form[name].addEventListener("input", () => {
        cfg[name] = Number(form[name].value);
        syncControls();
        writeConfig(cfg);
    });
}

runButton.addEventListener("click", () => setRunning(!running));

// --- help ----------------------------------------------------------------
// The captions live once, in the markup, next to the control they describe.
// The overlay collects them so they can be read all at once rather than one
// hover at a time.

const help = document.getElementById("help");

for (const label of form.querySelectorAll("label")) {
    const cap = label.querySelector(".cap");
    const field = label.querySelector("input, select");
    if (!cap || !field) continue;
    const dt = document.createElement("dt");
    dt.textContent = field.name;
    const dd = document.createElement("dd");
    dd.textContent = cap.textContent;
    document.getElementById("help-list").append(dt, dd);
}

const setHelp = (open) => document.body.classList.toggle("help", open);

document.getElementById("help-toggle")
    .addEventListener("click", () => setHelp(!document.body.classList.contains("help")));
help.addEventListener("click", () => setHelp(false));

addEventListener("keydown", (e) => {
    if (e.key === "Escape") return setHelp(false);
    // Space must not steal the key while a field has focus, or typing a seed
    // would toggle the simulation.
    if (e.code === "Space" && e.target === document.body) {
        e.preventDefault();
        setRunning(!running);
    }
});

restart();
setRunning(false);   // seeded and drawn, waiting for you to start it
requestAnimationFrame(frame);
