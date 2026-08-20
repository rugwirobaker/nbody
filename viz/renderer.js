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
// where the accretion glow comes from.
const CLEAR = [0x05 / 255, 0x07 / 255, 0x0d / 255, 1.0];

// The temperature ramp. A merge banks the kinetic energy it destroys into
// `heat` (RFC Step 10), which then decays, so a body flares and cools back.
//
// Warm reads as hot and the ramp runs through white, which keeps the two
// signals apart: additive blending already turns a crowd of cold particles
// white, so white means crowded and orange means hot. Ramping toward blue
// instead — the direction a real blackbody moves — would put heat and density
// on the same axis and make them impossible to tell apart. A straight line
// from blue to orange passes through a dusty pink, so white is a third stop
// rather than an accident of the interpolation.
const TINT_COLD = [0xbc / 255, 0xd0 / 255, 0xee / 255]; // inert dust
const TINT_MID = [0xff / 255, 0xfa / 255, 0xf0 / 255]; // the crossover
const TINT_HOT = [0xff / 255, 0x8a / 255, 0x3c / 255]; // just merged

// Specific heat (heat/mass) that reads as fully hot, and how much brighter a
// fully hot body burns than a cold one.
//
// Set from the temperature bodies actually reach. Sampled across 34,159 merge
// events at the demo's config, the body coming out of a merge holds a median
// heat/mass of 0.029, a 90th percentile of 0.102, and a 99th of 0.262. A hot
// point of 0.06 therefore puts a typical merge at white and a strong one at
// full orange, which is what the ramp is for.
//
// It also sets how long a flare lasts, and that is the part worth keeping in
// mind. Heat decays at a fixed fractional rate, so what the eye reads as the
// fade is the time a body spends *above* this threshold: from a median merge
// that is around 660 ticks, and from a 90th-percentile one around 1,900.
// Raising the hot point shortens both.
const HOT_POINT = 0.06;
const HEAT_GAIN = 2.5;

// World-space half-extent of the view at zoom 0. The seeded disk has radius 1.
//
// 2.8 rather than the disk's own size, because the disk does not keep it. Each
// merge leaks a little energy into the system (RFC-001 Step 10), so the
// survivors drift outward for as long as you watch: half of them are past 2.4
// by ten seconds and past 7 by a hundred. 2.8 frames the accretion phase, which
// is the part worth watching, and the zoom range below follows the rest.
const VIEW_HALF_EXTENT = 2.8;

// How much larger than life bodies are drawn.
//
// True radii are far below a pixel: k = 5e-4 puts a mass-1 body at 0.14 px and
// the largest body a run produces at about 3 px, so drawn at life size the
// whole field is specks and an accreted giant looks like the dust around it.
//
// The exaggeration is the same for every body, which is what makes it usable:
// relative sizes stay exact, and so does relative closeness — two discs at the
// same overlap are equally near to merging whatever their masses. That is the
// property a fixed threshold could not offer at any scale, since it overstated
// a speck by 7x and a giant by 69x.
//
// Discs touch at 4x the true merge distance. The help sheet says so.
const DISPLAY_SCALE = 16.0;

// Smallest radius a body may be drawn at, in device pixels.
//
// Bodies are drawn at the radius the physics merges them on (RFC-002 §6), and
// that radius is small: k = 5e-4 puts a mass-1 body at 0.14 px and a mass-100
// body at 1.4 px. Without a floor the field would be invisible. Below the floor
// the drawn disc overstates the body, which is the very thing RFC-002 removes —
// but only among bodies whose true size is under a pixel, where the picture
// cannot be honest either way and there is nothing to predict. Above it the
// disc is exact, which covers the large pairs whose merges you can now see
// coming.
//
// With the scale above, only genuine dust lands here — anything that has merged
// even once is drawn at its own size.
const MIN_RADIUS_PX = 1.5;

// Wall clock offered to the physics each frame, in seconds.
//
// Alone on screen a world gets a budget large enough to never engage, so it
// runs the ticks the frame owes and the frame takes as long as it takes. A
// kernel that cannot finish inside a refresh interval then produces a low
// frame rate, which is how the same comparison reads in a native window. The
// figure is a liveness guard and nothing else: without one, a large n would
// stop the page answering clicks.
//
// Under `stacked` both worlds share one frame, so neither may spend it all;
// they split a 60 Hz frame's usable remainder and each publishes at its own
// rate instead.
//
// Pictures per second, measured over the first two seconds of a run:
//
//                 alone            stacked
//        n     base    simd      base    simd
//     1000       57      60        24      60
//     1500       24      59        12      28
//     2000       14      33         5      15
//
// The two modes peak at different n, since stacked halves the budget: 1000
// pairs a choppy base against a smooth simd, and 1500 is where a solo base
// collapses to 24 while a solo simd still holds the display's ceiling.
const SOLO_BUDGET = 0.5;
const STACKED_BUDGET = 0.014;

const VERT = `#version 300 es
layout(location = 0) in vec2 a_corner;    // quad corner, in [-1, 1]
layout(location = 1) in vec4 a_particle;  // x, y, mass, heat

uniform vec2  u_scale;      // world -> clip, corrected for this viewport's aspect
uniform vec2  u_centre;     // world-space point at the middle of the viewport
uniform float u_radius;     // k, from the physics: a body's radius is k*sqrt(m)
uniform float u_min_radius; // floor in world units, so dust stays visible
uniform vec3  u_cold;       // colour of a body with no heat in it
uniform vec3  u_mid;        // colour halfway up the ramp
uniform vec3  u_hot;        // colour at the hot point
uniform float u_hot_point;  // heat/mass that reads as fully hot
uniform float u_gain;       // extra brightness a fully hot body emits

out vec2 v_corner;
out vec3 v_colour;

void main() {
    v_corner = a_corner;

    // Mass maps to area in 2D, so radius goes as its square root. This is the
    // same r(m) the merge rule tests against, so two discs touching on screen
    // means a merge this tick.
    float r = max(u_radius * sqrt(a_particle.z), u_min_radius);
    vec2 world = a_particle.xy + a_corner * r;
    gl_Position = vec4((world - u_centre) * u_scale, 0.0, 1.0);

    // Colour and brightness both come from temperature, since both are things
    // temperature does to a radiating body. Mass stays out of it: it is
    // already in the radius, and the light a body puts out is brightness times
    // area, so feeding mass to both would count it twice and let the largest
    // body drown the field.
    //
    // Temperature is heat per unit mass rather than heat. Heat pools when
    // bodies merge, which leaves it heavy-tailed — a 90th percentile of 0.29
    // against a maximum of 7.13 in the same frame. Dividing by mass gives the
    // intensive quantity, which stays inside a range a ramp can use.
    float temperature = a_particle.w / max(a_particle.z, 1e-6);
    float t = clamp(temperature / u_hot_point, 0.0, 1.0);

    vec3 hue = t < 0.5 ? mix(u_cold, u_mid, t * 2.0)
                       : mix(u_mid, u_hot, t * 2.0 - 1.0);

    // Per particle rather than per pixel: all four corners carry the same
    // colour, so the rasterizer interpolates between equal values.
    v_colour = hue * (1.0 + u_gain * t);
}`;

const FRAG = `#version 300 es
precision highp float;

in vec2 v_corner;
in vec3 v_colour;
out vec4 fragColor;

void main() {
    // Distance from the quad's centre turns the quad into a disc, and the
    // smooth falloff antialiases its edge without any multisampling.
    float d = length(v_corner);
    float a = smoothstep(1.0, 0.0, d);
    a *= a;  // tighten the core so dense regions read as bright, not flat
    // A hot body's colour runs past white at the centre and comes back into
    // range down the falloff, so it reads as a blown-out core inside a
    // coloured skirt — which is what a bright thing looks like.
    fragColor = vec4(v_colour * a, 1.0);
}`;

// --- configuration, carried in the URL -----------------------------------
// The address bar is the record of the run on screen: a link reproduces it
// exactly. This is the page's version of what bench/main.zig does when it
// prints the seed and full config above every table.

// The page opens in stacked, where n = 1000 separates the panels most clearly:
// base publishes at ~24 fps against simd's 60 (see the table above).
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
        zoom: real("zoom", -6.0, 4.5),
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
// Imported rather than duplicated: a copy of k here could drift out of
// agreement with the simulation (RFC-002 §6).
const MERGE_RADIUS_SCALE = wasm.mergeRadiusScale();

const program = buildProgram(gl);
const u = {
    scale: gl.getUniformLocation(program, "u_scale"),
    centre: gl.getUniformLocation(program, "u_centre"),
    radius: gl.getUniformLocation(program, "u_radius"),
    minRadius: gl.getUniformLocation(program, "u_min_radius"),
    cold: gl.getUniformLocation(program, "u_cold"),
    mid: gl.getUniformLocation(program, "u_mid"),
    hot: gl.getUniformLocation(program, "u_hot"),
    hotPoint: gl.getUniformLocation(program, "u_hot_point"),
    gain: gl.getUniformLocation(program, "u_gain"),
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

// Pictures per second, per panel, and the serial of the one on the GPU.
//
// Each published picture carries the same fixed amount of physics — RFC §2.4's
// ten-tick frame — so the rate at which they arrive is throughput. It is the
// reference implementation's FPS in another spelling, and it is the symptom
// rather than the measurement: ns/tick stays the reported metric (§2.5 rule 2).
const RATE_WINDOW = 0.5;
const rate = [freshRate(), freshRate()];
const published = [-1, -1];

function freshRate() {
    return { updates: 0, since: 0, fps: 0 };
}

function sampleRate(which, t) {
    const r = rate[which];
    const elapsed = (t - r.since) / 1000;
    if (elapsed < RATE_WINDOW) return;
    const updates = wasm.renderUpdates(which);
    r.fps = (updates - r.updates) / elapsed;
    r.updates = updates;
    r.since = t;
}

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
    const t = performance.now();
    for (const which of [BASE, SIMD]) {
        rate[which] = { updates: wasm.renderUpdates(which), since: t, fps: 0 };
        published[which] = -1;
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
    gl.uniform1f(u.radius, MERGE_RADIUS_SCALE * DISPLAY_SCALE);
    // The floor is a pixel count, so it converts through the current zoom.
    gl.uniform1f(u.minRadius, (MIN_RADIUS_PX * halfY * 2.0) / h);
    gl.uniform3f(u.cold, ...TINT_COLD);
    gl.uniform3f(u.mid, ...TINT_MID);
    gl.uniform3f(u.hot, ...TINT_HOT);
    gl.uniform1f(u.hotPoint, HOT_POINT);
    gl.uniform1f(u.gain, HEAT_GAIN);

    // The panel is redrawn every frame, from whatever picture its kernel last
    // finished. Uploading only when a new one exists is what puts the deficit
    // on screen: a starved kernel repeats a picture instead of showing a
    // smaller step of motion.
    const serial = wasm.renderUpdates(which);
    const panel = panels[which];
    gl.bindVertexArray(panel.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, panel.instances);
    if (serial !== published[which]) {
        gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
        published[which] = serial;
    }
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
    el.querySelector('[data-f="fps"]').textContent = rate[which].fps.toFixed(0);
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
        // Alone, a world may take the whole frame and let the frame rate fall.
        // Under stacked the two share one, so each gets half and publishes at
        // its own rate instead.
        const share = cfg.mode === MODE.stacked ? STACKED_BUDGET / 2 : SOLO_BUDGET;
        wasm.advance(elapsed * Math.pow(10, cfg.speed), share);
        sampleRate(BASE, t);
        sampleRate(SIMD, t);
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
    // Reset the frame clock and the rate windows, so a long pause is charged
    // neither to the first frame nor to the first rate figure after it.
    last = performance.now();
    for (const which of [BASE, SIMD]) {
        rate[which].updates = wasm.renderUpdates(which);
        rate[which].since = last;
    }
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
