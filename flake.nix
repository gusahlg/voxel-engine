{
  description = "voxel_engine — a small, fast Vulkan 1.3 voxel renderer (library + demo)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        isLinux = pkgs.stdenv.isLinux;

        # ash dlopens the Vulkan loader; winit dlopens the windowing stack.
        runtimeLibs = with pkgs; [ vulkan-loader libxkbcommon ]
          ++ pkgs.lib.optionals isLinux [
            wayland
            libx11
            libxcursor
            libxrandr
            libxi
          ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            rustc
            cargo
            rustfmt
            clippy
            shader-slang
            vulkan-headers
            vulkan-tools
            vulkan-validation-layers
          ] ++ runtimeLibs;

          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;
          VK_LAYER_PATH =
            "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";
        };

        packages.default = pkgs.rustPlatform.buildRustPackage {
          pname = "voxel_engine-demo";
          version = "0.1.0";
          src = ./.;
          cargoLock.lockFile = ./Cargo.lock;

          nativeBuildInputs = [ pkgs.shader-slang ];

          postFixup = pkgs.lib.optionalString isLinux ''
            patchelf --set-rpath "${pkgs.lib.makeLibraryPath runtimeLibs}" $out/bin/demo || true
          '';

          meta.mainProgram = "demo";
        };
      });
}
