# voxel_engine

A small, fast Vulkan 1.3 voxel renderer in Rust (`ash` + `winit`), built as a
library with a raylib-shaped polling API. Written to power
[project_watt_cubed](../project_watt_cubed), usable by any voxel game.

```rust
use voxel_engine::{run, Config, Camera3D, Color, Key, Vec3, WarpParams};

fn main() {
    voxel_engine::run(Config::default(), move |eng| {
        if eng.is_key_pressed(Key::Escape) || eng.should_close() {
            return false;
        }
        let cam = Camera3D { position: Vec3::new(0.0, 10.0, 20.0),
                             target: Vec3::ZERO, up: Vec3::Y, fovy: 70.0,
                             warp: WarpParams::IDENTITY };
        let mut f = eng.begin_frame(Color::SKYBLUE);
        {
            let mut f3 = f.begin_3d(&cam);
            f3.draw_cube(Vec3::ZERO, Vec3::splat(2.0), Color::RED);
        }
        f.draw_text("hello", 16, 16, 20, Color::RAYWHITE);
        true
    });
}
```

## What it does

- **Frame loop**: `run(config, |eng| ...)` — a per-frame callback with polled
  input (edge + held keys, drainable char queue, raw mouse deltas with cursor
  capture), delta time, and an optional frame cap. Winit 0.30 underneath.
- **Meshes**: 16-byte unlit vertices (`pos f32x3 + color u8x4`), u32 indices.
  Uploads land in device-local memory suballocated from 64 MiB blocks (one
  `vkAllocateMemory` per block, not per mesh); on unified-memory GPUs (Apple
  Silicon) uploads are direct memcpys, elsewhere a staging copy recorded into
  the same frame. Meshes uploaded mid-update are drawable the same frame;
  frees are deferred until the GPU is provably done.
- **3D**: reversed-Z infinite-far projection (D32, `GREATER_OR_EQUAL`),
  negative-viewport y-flip (GL winding parity), automatic frustum culling
  against per-mesh AABBs, immediate cubes and wire cubes.
- **2D overlay**: rects, lines, text from an embedded public-domain 8x8 font
  (no asset files), alpha-blended over the 3D pass in call order.
- **Settings at runtime**: borderless fullscreen, vsync (FIFO/MAILBOX/
  IMMEDIATE), MSAA 1–8x with resolve, resizable window — all applied lazily at
  the next frame boundary.
- **Correctness**: synchronization2 + dynamic rendering, per-swapchain-image
  present semaphores, dynamic viewport/scissor (pipelines never rebuild on
  resize), validation layer + debug messenger in debug builds
  (`VOXEL_ENGINE_VALIDATION=0/1` to override).

## Building

- `cargo run --release --bin demo` — spinning demo scene (F fullscreen,
  V vsync, M MSAA cycle, Esc quit).
- Shaders are Slang (`shaders/`), compiled by `build.rs` with `slangc` and
  embedded into the binary; checked-in SPIR-V under `shaders_spv/` is used as
  a fallback when `slangc` is not installed.
- `nix develop` — dev shell with Rust, slangc, the Vulkan loader/tools and
  validation layers. `nix run` builds and runs the demo.
- macOS: install MoltenVK + the Vulkan loader (`brew install molten-vk
  vulkan-loader`); the engine finds Homebrew's loader automatically.
- `cargo test` — CPU-side unit tests (font atlas, input semantics, frustum &
  projection math, allocator free-list).

For performance consider installing mold and creating this file.
```
cat ~/.cargo/config.toml
[target.x86_64-unknown-linux-gnu]
linker = "clang"
rustflags = ["-C", "target-cpu=native", "-C", "link-arg=-fuse-ld=mold"]
```
