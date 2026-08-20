# A crash introduction to graphics

Written for someone who can read the physics in this repository and has never
written a renderer. It explains what a GPU actually is asked to do, what the
objects in every graphics API mean, and how `nbody-viz` maps onto them. Every
concrete example is code that ships in `viz/renderer.js`.

The companion documents are [`RFC-001.md`](RFC-001.md) for the simulation and
[`disassembly.md`](disassembly.md) for the CPU kernels.

---

## 1. What happens to a pixel

A screen is an array of colours that the display controller reads out sixty or
more times a second. Everything below exists to fill that array.

The GPU fills it by running one program over a very wide batch, twice, with a
fixed-function stage in between:

```
  your data        your program       hardware           your program
  ---------        ------------       --------           ------------

  vertices    -->  vertex shader -->  rasterizer    -->  fragment shader  -->  pixels
                   once per           finds the pixels   once per
                   vertex             a triangle covers  covered pixel
```

**The vertex shader** runs once per vertex and answers one question: where on
the screen does this point land? Its output is a position in *clip space*, a
coordinate system where the visible region runs from −1 to +1 on each axis.
Whatever else it computes rides along to the next stage.

**The rasterizer** is fixed hardware. It takes the three corners of a triangle
and determines the set of pixels inside it, interpolating the vertex shader's
extra outputs across that set. This is the step you configure rather than
program.

**The fragment shader** runs once per covered pixel and answers one question:
what colour is this pixel? It reads the interpolated values the rasterizer
handed it and writes a colour.

Both programs run on thousands of shader cores at once, which is the whole
reason a GPU exists. Their scale is what the four layers below are organised
around: feeding one batch of work is cheap, and issuing a thousand small
batches is expensive.

Triangles are the only surface primitive the hardware knows. A quad is two
triangles. A disc is a quad whose fragment shader fades the corners away, which
is exactly how the particles in this demo are drawn (§6).

---

## 2. The four layers

Getting a pixel from your data onto a screen crosses four boundaries. Each is
owned by a different piece of software, and every graphics stack — a game
engine, a browser, a terminal emulator — has all four.

| Layer | What it is | Who supplies it in `nbody-viz` |
| --- | --- | --- |
| 1. Window | A rectangle of screen owned by the OS compositor, plus the event stream that goes with it | The browser |
| 2. Swapchain | The images that get shown in that rectangle, and the timing of the swap | The browser |
| 3. GPU API | The interface that turns your data and programs into GPU work | WebGL2, called from `viz/renderer.js` |
| 4. Renderer | Your decisions: what to draw, in what order, with which shaders | `viz/renderer.js` |

**Layer 1** is Cocoa on macOS, Wayland or X11 on Linux, Win32 on Windows. It
gives you a handle, a size in physical pixels, and keyboard and mouse events.
A native application talks to it through the OS API directly or through a
portable window library such as GLFW or SDL. In a browser tab the `<canvas>`
element is the window: `canvas.clientWidth` and `devicePixelRatio` are the size
in physical pixels, and `addEventListener` is the event stream.

**Layer 2** is the handoff between drawing and showing. You draw into an
off-screen image; when the frame is finished, that image becomes the one the
compositor displays and you get a different one to draw into. The rotation
prevents the display from reading an image while you are still writing to it,
and pacing the rotation to the display's refresh is what "vsync" means. A
native application creates the swapchain explicitly and chooses how many images
it holds. In a browser, `requestAnimationFrame` is the swapchain interface: the
callback runs once per display refresh, and the canvas contents are presented
when it returns.

**Layer 3** is where your data becomes GPU work. It creates the GPU-side
objects (§3), compiles your shaders, and submits commands. Metal, Vulkan,
Direct3D 12, OpenGL, WebGL2, and WebGPU are all layer 3. The API depends on
layer 1 in exactly one place: to present, it needs a *surface* derived from the
window handle. Everything else — buffers, textures, shaders, compute — works
without a window at all, which is why a compute job or an offline renderer runs
headless.

**Layer 4** is the part that is yours. It decides what a frame consists of, in
what order things are drawn, which shaders run, how the camera maps world
coordinates onto clip space, and how the data reaches the GPU. Two renderers
over the same layer 3 can look nothing alike. `viz/renderer.js` is 450 lines
and its rendering model is one sentence: clear the screen, then draw one
instanced quad per particle with additive blending. A game engine's renderer is
hundreds of thousands of lines because it has to handle materials, shadows,
transparency ordering, and a scene it did not write.

---

## 3. The objects a GPU API gives you

These names carry across Metal, Vulkan, D3D12, and WebGPU nearly unchanged, so
learning them once is learning all four.

**Device.** Your handle to one GPU, and the factory that creates everything
else. The GPU API creates it, by asking the driver: `requestAdapter` in WebGPU,
`MTLCreateSystemDefaultDevice` in Metal, `vkCreateDevice` in Vulkan,
`canvas.getContext("webgl2")` here. The OS owns the window and loads the
driver; the API, through the driver, produces the device. A machine with two GPUs can
hand you two devices, and objects created on one are unusable on the other.

**Queue.** Where work is submitted for execution. The GPU runs commands
asynchronously: submitting returns immediately, and the work completes later.

**Command buffer.** A recorded list of GPU commands — bind this pipeline, set
this uniform, draw these vertices — built on the CPU and submitted to the queue
as a unit. Recording is deliberately separate from execution, so the list can
be built on one thread while the GPU executes an earlier one.

**Buffer.** A block of GPU memory holding an array of bytes whose layout you
choose. Vertex positions, per-instance data, matrices, arbitrary structs, the
inputs and outputs of a compute shader: all buffers. The GPU sees a flat range
of memory; the meaning comes from how a shader reads it. In this demo the two
buffers are four quad corners (§5) and the packed `(x, y, mass, heat)` array
copied out of wasm memory each frame.

**Texture.** A block of GPU memory holding an image: elements addressed by 2D
or 3D coordinates, in a format the hardware understands (`rgba8`, `r32f`,
depth formats), with dedicated hardware for reading them. That hardware is the
difference that matters. A texture read takes *fractional* coordinates and returns a blend of the
neighbouring elements, in dedicated hardware, along with mipmapping, wrapping,
and a memory layout arranged so that pixels adjacent on screen sit close
together in memory instead of following row order. Textures are also
what a shader can *write* to as a render target, which is how a frame is drawn
into something other than the screen.

The division of labour follows from that. A buffer holds data whose meaning is
yours; a texture holds data indexed by position and read with filtering.
Colour maps, height fields, glyph atlases, and the render target itself are
textures. Vertex data, instance data, and simulation state are buffers.

**Shader.** A program compiled for the GPU, running once per item in a batch,
across thousands of cores. Written in GLSL, MSL, HLSL, or WGSL, compiled either
ahead of time or by the driver at load. The three kinds are vertex, fragment,
and compute — the last one runs over an arbitrary index range with no
rasterizer involved, which is how a GPU does physics rather than pictures.

**Pipeline.** The complete configuration of one pass through the hardware: the
vertex and fragment shader pair, the layout of the vertex data, the blend
equation, the depth and stencil settings, and the format of the output. Modern
APIs make you build this as one immutable object up front, because the driver
compiles the whole configuration into a hardware state that would otherwise
have to be assembled at draw time. Switching pipelines mid-frame costs real
time, so renderers sort their draws to switch as rarely as possible. WebGL2
predates this design and sets the same state through individual calls —
`useProgram`, `blendFunc`, `enable(BLEND)` — which is the loose form of the
same thing.

**Surface and swapchain.** A surface is the GPU API's handle to a region of
screen owned by the window system, created from the window's native handle. The
window is the OS object; the surface is the GPU-side object derived from it,
and it carries what the GPU needs to know — pixel format, colour space, size,
which presentation modes are available. The swapchain is the ring of textures
allocated for that surface. The browser creates and manages both, which is why
`viz/renderer.js` contains no code for either.

---

## 4. A frame, end to end

The structure below is what `viz/renderer.js` does in `frame()`, and it is the
structure of every real-time renderer.

**Setup, once at load.** Compile the shaders and link them into a program.
Create the buffers. Upload anything static. Configure the state that does not
change per frame:

```js
gl.clearColor(...CLEAR);
gl.disable(gl.DEPTH_TEST);
gl.enable(gl.BLEND);
gl.blendFunc(gl.ONE, gl.ONE);
```

**Per frame:**

1. **Advance the simulation.** Here that means calling into wasm with the
   elapsed wall clock; RFC §2.4's fixed-timestep accumulator lives on the Zig
   side, in `viz/timestep.zig`.
2. **Upload what changed.** One `bufferSubData` per panel, copying the packed
   particle array. The buffer was allocated once at its full size; each upload
   overwrites its contents. This demo skips the upload when the physics has no
   new picture to show, which is what makes a kernel that cannot keep up look
   like one.
3. **Clear.** `gl.clear(gl.COLOR_BUFFER_BIT)` resets the frame to the
   background colour. Skipping it leaves the previous frame underneath, which
   is how trail effects are built deliberately.
4. **Set the state and the uniforms.** Uniforms are the values constant across
   a whole draw call — here the camera transform, the disc radius, and the
   tint.
5. **Draw.** One call per panel.
6. **Return.** The browser presents the canvas.

The cost model is worth internalising: uploads and draw calls are CPU work paid
per call, and shader invocations are GPU work paid per vertex and per pixel. A
renderer is fast when the CPU issues few, large batches. This is the reasoning
behind everything in §5.

---

## 5. One draw call for the whole field

The naive way to draw a thousand particles is a thousand draw calls, each
setting a position and drawing a quad. Each call costs CPU time in the driver
regardless of how little it draws, so the frame becomes bounded by the number
of calls rather than by the pixels.

**Instancing** fixes this. You supply the geometry once, and a second buffer
holding the per-instance data, and tell the API how fast each attribute
advances:

```js
gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);   // the quad's corners
gl.vertexAttribPointer(1, 4, gl.FLOAT, false, 0, 0);   // (x, y, mass, heat)
gl.vertexAttribDivisor(1, 1);   // advance once per instance, not per vertex
```

`vertexAttribDivisor(1, 1)` is the whole mechanism. Attribute 0 has the default
divisor of 0, so the vertex shader walks its four corners for every instance.
Attribute 1 has divisor 1, so it advances one element per instance. Then:

```js
gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n);
```

One call, and `4 × n` vertex shader invocations — four corners for each of the
`n` particles. The per-frame CPU work for the field is one buffer upload and
one draw, whatever `n` is.
The vertex shader reads the instance data as an ordinary attribute:

```glsl
float r = u_radius * sqrt(a_particle.z);      // mass maps to area in 2D
vec2 world = a_particle.xy + a_corner * r;
gl_Position = vec4((world - u_centre) * u_scale, 0.0, 1.0);
```

Those three lines are the camera as well as the geometry. `u_centre` is the
world point at the middle of the viewport and `u_scale` converts world units to
clip space, aspect-corrected per viewport so the half-width panels in `stacked`
mode do not squash the disk. A 2D camera is one multiply and one subtract; a 3D
camera is the same idea as a 4×4 matrix.

Instancing is also how a GPU-accelerated terminal draws text. Each cell is a
quad; the per-instance data is a grid position, a colour, and the coordinates
of a glyph inside an atlas texture; one draw call covers the screen. The
rendering model of a terminal and the rendering model of this demo are the same
shape, differing in where the fragment colour comes from.

---

## 6. Making a quad look like a glowing disc

The geometry is a square. The fragment shader turns it into a disc:

```glsl
float d = length(v_corner);          // 0 at the centre, 1 at the edge midpoints
float a = smoothstep(1.0, 0.0, d);   // 1 at the centre, fading to 0 at d = 1
a *= a;                              // tighten the core
fragColor = vec4(u_tint * a, 1.0);
```

`v_corner` is the quad corner the vertex shader passed through, interpolated
across the quad by the rasterizer, so each pixel receives its own position
within the square. `smoothstep` gives a soft edge, which antialiases the disc
for free — the alternative is multisampling, which costs memory bandwidth on
every pixel of the frame.

The glow comes from the blend mode. `blendFunc(ONE, ONE)` makes each fragment
*add* to what is already in the framebuffer, so overlapping particles sum their
light and dense regions saturate toward white on their own. This is why the
background is near-black (`#05070d`): additive blending only ever adds, so a
dark ground is what lets anything read. It is also why depth testing is
disabled — addition is commutative, so the draw order of transparent additive
sprites does not matter.

The whole visual identity of the demo is those four lines plus two constants.
No texture is loaded and no image file ships.

---

## 7. What a game engine adds

A renderer draws what it is told. An engine is the machinery that decides what
to tell it, and the difference is almost entirely layer 4 and above:

| An engine also has | Why `nbody-viz` does not need it |
| --- | --- |
| Scene graph and culling | The scene is one flat array, and all of it is on screen |
| Material system | One shader, one palette |
| Asset pipeline | No meshes, no images, no fonts |
| Transparency sorting | Additive blending is order-independent |
| Shadows, lighting, post-processing | 2D points that emit their own light |
| Animation, skinning, physics engine | The physics *is* the application |
| Audio, input mapping, scripting, editor | Six HTML controls |

Layers 1 through 3 are the same in both cases. An engine wraps them in
portability shims, and it is common to write those shims for three GPU APIs so
one codebase runs on Metal, Vulkan, and D3D12. That portability work is the
reason engines look large from outside; the actual drawing at the bottom is
recognisably §4.

---

## 8. Choosing a layer 3

This is the decision the RFC originally answered with raylib and that
`nbody-viz` answers with WebGL2.

| API | Reaches | Shape |
| --- | --- | --- |
| **WebGL2** | Every current browser | OpenGL ES 3.0 in JavaScript. Global state set through individual calls, driver-managed memory, no explicit synchronisation. |
| **WebGPU** | Recent browsers, and native through Dawn or wgpu | Explicit pipelines, bind groups, and command encoders. Includes compute shaders. Reads like a modern native API. |
| **Metal / Vulkan / D3D12** | One platform each | Explicit everything, including memory allocation and synchronisation. Maximum control. |
| **raylib, SDL_gpu, sokol** | Many platforms via one of the above | Portability wrappers. Reach the browser by compiling through emscripten. |

`nbody-viz` chose WebGL2 for reasons specific to this project:

- **The browser is the widest distribution there is.** A link runs the demo on
  every platform, with no install and no signing.
- **The library was already renderer-free and I/O-free**, so it compiles to
  `wasm32-freestanding` unmodified. The result is 25 KB, with no emscripten in
  the toolchain and nothing in `build.zig.zon`.
- **`+simd128` gives the wasm target a real 4-wide vector**, the same width as
  NEON, so the page runs the actual Part 3 kernel rather than an emulation of
  it.
- **WebGL2 is small enough to hold in your head.** The renderer is one file
  with no dependencies, which suits a project whose subject is the CPU kernel.

WebGPU is deferred to the project that ports Phase A to GPU compute, since
compute shaders are the reason to reach for it. Two things will carry over from
this renderer: the shader mathematics of §6 and the instance-buffer model of
§5. The API model is its own subject — pipelines become immutable objects, bind
groups replace individual uniform calls, and command encoders replace
immediate-mode state — so WebGL2 fluency is worth counting as fluency in
layer 4 rather than in any particular layer 3.

---

## 9. Where to look in the code

| File | Layer | What is in it |
| --- | --- | --- |
| `viz/index.html` | 1 | The canvas, the controls, the HUD. All text is HTML over the canvas rather than drawn in GL, which avoids needing a glyph atlas. |
| `viz/renderer.js` | 3, 4 | Shaders, buffers, the instanced draw, the camera, the frame loop. |
| `viz/main.zig` | — | The wasm module: exports, the two worlds, the demo constants. |
| `viz/world.zig` | — | One running simulation of either kernel, its timing, and its packed render buffer. |
| `viz/timestep.zig` | — | RFC §2.4's accumulator and its ten-tick clamp. Host-testable, so `zig build test` covers it. |
| `build.zig` | — | The `viz` step: the fixed `wasm32-freestanding+simd128` target query, `entry = .disabled`, `rdynamic = true`, `strip = true`. |

Read them in that order and the four layers appear in sequence.
