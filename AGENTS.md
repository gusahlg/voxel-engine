# Repository Guidelines

## Project Structure & Module Organization

This is a Rust 2024 Vulkan renderer library using `ash` and `winit`. The public
API starts in `src/lib.rs`, the runnable example is `src/bin/demo.rs`, and Vulkan
implementation modules live under `src/vk/`. Slang sources are in `shaders/`;
`build.rs` compiles them via the workspace member `crates/slang-build`
(`voxel_slang_build`) into Cargo's output directory and uses `shaders_spv/`
only as a checked-in fallback. Refresh that fallback explicitly as documented
in the README. The game crate can depend on `voxel_slang_build` as a
build-dependency to compile its own Slang with the same pinned `slangc` and
fallback rules.

## Build, Test, and Development Commands

- `nix develop`: enter the intended Linux dev shell with Rust, Slang `slangc`, Vulkan loader/tools, Wayland, and X11 dependencies.
- `cargo check`: type-check the project and run `build.rs`, including shader compilation.
- `cargo build`: build the library and demo.
- `cargo run --bin demo`: build and launch the renderer window locally.
- `cargo test`: run the host-side unit and integration tests.
- `cargo fmt`: format Rust code before committing.
- `vulkaninfo --summary`: verify that Vulkan is visible on the host when runtime initialization fails.

## Coding Style & Naming Conventions

Use standard `rustfmt` formatting with 4-space indentation. Follow Rust naming conventions: modules and functions in `snake_case`, types in `PascalCase`, constants in `SCREAMING_SNAKE_CASE`. Keep unsafe Vulkan calls narrow and close to the resource they create, with explicit error messages on `expect`. Prefer existing module boundaries over adding new top-level modules. Shader files should use the `name.stage.slang` pattern, for example `tri.vert.slang` and `tri.frag.slang`.

## Testing Guidelines

Add focused unit tests beside Rust modules where logic is host-testable, and
integration tests under `tests/` when they do not require a display or GPU. Run
`cargo test --all-targets`; shader changes also require
`cargo test --test shader_validation`. Exercise renderer changes with validation
layers when Vulkan hardware and a display are available.

## Commit & Pull Request Guidelines

Recent commits use short, imperative or descriptive summaries such as `Changed to Vulkan 1.3 features` and `Restructered command_buffer logic`. Keep the first line concise and specific to one change. Pull requests should include a short description, commands run, any Vulkan/platform assumptions, and screenshots or notes for visible rendering changes. Link related issues when available.

## Agent-Specific Instructions

Do not commit generated build artifacts unless they are intentionally tracked. Avoid unrelated refactors while touching Vulkan setup code, since resource lifetime and initialization order are tightly coupled.
