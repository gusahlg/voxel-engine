# Repository Guidelines

## Project Structure & Module Organization

This is a Rust 2024 Vulkan renderer experiment using `ash` and `winit`.
Application entry is [src/main.rs](/home/gusahlg/repos/voxel-engine/src/main.rs), which creates the event loop and `App`. Vulkan code lives under `src/vk/`, with renderer state in `src/vk/renderer/`. Device selection and logical-device setup are in `src/vk/renderer/device/`, frame synchronization and command buffers in `src/vk/renderer/frame/`, and pipeline/shader helpers in `src/vk/renderer/rendering/`. Slang shader sources are in `shaders/`; compiled SPIR-V outputs are written to `shaders_spv/` by `build.rs`.

## Build, Test, and Development Commands

- `nix develop`: enter the intended Linux dev shell with Rust, Slang `slangc`, Vulkan loader/tools, Wayland, and X11 dependencies.
- `cargo check`: type-check the project and run `build.rs`, including shader compilation.
- `cargo build`: build the `vk_rust_renderer` binary.
- `cargo run`: build and launch the renderer window locally.
- `cargo test`: run unit/integration tests when present.
- `cargo fmt`: format Rust code before committing.
- `vulkaninfo --summary`: verify that Vulkan is visible on the host when runtime initialization fails.

## Coding Style & Naming Conventions

Use standard `rustfmt` formatting with 4-space indentation. Follow Rust naming conventions: modules and functions in `snake_case`, types in `PascalCase`, constants in `SCREAMING_SNAKE_CASE`. Keep unsafe Vulkan calls narrow and close to the resource they create, with explicit error messages on `expect`. Prefer existing module boundaries over adding new top-level modules. Shader files should use the `name.stage.slang` pattern, for example `tri.vert.slang` and `tri.frag.slang`.

## Testing Guidelines

There are no tests in the current tree. Add focused unit tests beside Rust modules with `#[cfg(test)]` where logic is host-testable, and integration tests under `tests/` only when they do not require a display or GPU. For renderer behavior, at minimum run `cargo check` and, when hardware/display access is available, `cargo run`.

## Commit & Pull Request Guidelines

Recent commits use short, imperative or descriptive summaries such as `Changed to Vulkan 1.3 features` and `Restructered command_buffer logic`. Keep the first line concise and specific to one change. Pull requests should include a short description, commands run, any Vulkan/platform assumptions, and screenshots or notes for visible rendering changes. Link related issues when available.

## Agent-Specific Instructions

Do not commit generated build artifacts unless they are intentionally tracked. Avoid unrelated refactors while touching Vulkan setup code, since resource lifetime and initialization order are tightly coupled.
