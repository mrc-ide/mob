{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    flake-parts.url = "github:hercules-ci/flake-parts";
    reside.url = "github:plietar/reside.nix";
    reside.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = inputs: inputs.flake-parts.lib.mkFlake { inherit inputs; } ({ lib, ... }: {
    systems = [ "x86_64-linux" ];
    perSystem = { config, pkgs, system, self', ... }:
      let
        cudaPackages = pkgs.cudaPackages_12_8;
      in
      {
        _module.args.pkgs = import inputs.nixpkgs {
          inherit system;
          config.allowUnfree = true;
          config.nvidia.acceptLicense = true;
          overlays = [ inputs.reside.overlays.default ];
        };

        # Using CUDA is quite a mess. You need both a "userspace driver" and a
        # "kernel driver", and their versions need to match.
        #
        # Usually, both of these are managed by the native package manager which
        # tries to keep them consistent. They may get out of sync if the CUDA
        # version is upgraded but the machine is not rebooted: the userspace
        # driver is the old version but the kernel is still running the older
        # one. You can fix that either by rebooting or unloading and re-loading
        # the kernel.
        #
        # When using Nix-built programs on non-NixOS this becomes its own special
        # kind of mess.  We don't really want to use the system provided
        # "userspace driver", as it would have been linked to the system libc
        # etc... The better option is to use the Nix-built userspace driver, but
        # you need to make sure that matches the right version.
        #
        # nixGL does that by reading /proc/driver/nvidia/version at build-time,
        # which is impure. Instead we hardcode a list of versions we support,
        # build all of these, and pick the right one when creating the shell.
        packages.selectDriver =
          let
            buildDriver = version: attrs: (pkgs.linuxPackages.nvidiaPackages.mkDriver ({
              useSettings = false;
              usePersistenced = false;
              inherit version;
            } // attrs)).override {
              libsOnly = true;
            };

            cudaDrivers =
              lib.mapAttrs buildDriver {
                "575.57.08" = { sha256_64bit = "sha256-KqcB2sGAp7IKbleMzNkB3tjUTlfWBYDwj50o3R//xvI="; };
                "575.51.02" = { sha256_64bit = "sha256-XZ0N8ISmoAC8p28DrGHk/YN1rJsInJ2dZNL8O+Tuaa0="; };
              };
          in

          pkgs.writeShellApplication {
            name = "select-driver";
            runtimeInputs = [ pkgs.jq ];
            text = ''
              if [[ -f /sys/module/nvidia/version ]]; then
                version="$(cat /sys/module/nvidia/version)"
                if jq --exit-status \
                  --raw-output \
                  --arg version "$version" \
                  'to_entries | .[] | select((.key | split(".")[:2]) == ($version | split(".")[:2])) | .value' \
                  "${pkgs.writers.writeJSON "drivers.json" cudaDrivers}"
                then
                  echo >&2 "Using CUDA driver version: $version"
                else
                  echo >&2 "Unsupported kernel CUDA driver version"
                  exit 1
                fi
              else
                echo >&2 "Could not detect kernel CUDA driver version"
                exit 1
              fi
            '';
            passthru.shellHook = ''
              if driver="$(${lib.getExe self'.packages.selectDriver})"; then
                export LD_LIBRARY_PATH="$driver/lib"
              fi
            '';
          };

        packages.mob = pkgs.rPackages.buildRPackage {
          name = "mob";
          src = ./.;
          buildInputs = [
            pkgs.R
            cudaPackages.cudatoolkit
            cudaPackages.cuda_cudart
            cudaPackages.cuda_cccl
          ];
          propagatedBuildInputs = [
            pkgs.rPackages.Rcpp
            pkgs.rPackages.cli
            pkgs.rPackages.dplyr
            pkgs.rPackages.dust
            pkgs.rPackages.patrick
            pkgs.rPackages.rcmdcheck
            pkgs.rPackages.testthat
            pkgs.rPackages.withr
          ];
        };
        packages.default = self'.packages.mob;

        devShells.benchmark = pkgs.mkShell {
          buildInputs = [
            (pkgs.rWrapper.override {
              packages = [
                self'.packages.mob
                pkgs.rPackages.ggplot2
                pkgs.rPackages.purrr
                pkgs.rPackages.tidyr
                pkgs.rPackages.ggbeeswarm
                pkgs.rPackages.ggpubr
              ];
            })
          ];
          shellHook = self'.packages.selectDriver.shellHook;
        };

        # This shell is used to run the testsuite in buildkite
        devShells.ci = pkgs.mkShell {
          buildInputs = [
            cudaPackages.cudatoolkit
            cudaPackages.cuda_cudart
            cudaPackages.cuda_cccl
            (pkgs.rWrapper.override {
              packages = [
                pkgs.rPackages.Rcpp
                pkgs.rPackages.cli
                pkgs.rPackages.dplyr
                pkgs.rPackages.dust
                pkgs.rPackages.patrick
                pkgs.rPackages.rcmdcheck
                pkgs.rPackages.testthat
                pkgs.rPackages.withr

                # benchmark stuff
                pkgs.rPackages.ggplot2
                pkgs.rPackages.purrr
                pkgs.rPackages.tidyr
                pkgs.rPackages.ggbeeswarm
                pkgs.rPackages.ggpubr
              ];
            })
          ];
          shellHook = self'.packages.selectDriver.shellHook;
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [
            cudaPackages.cudatoolkit
            cudaPackages.cuda_cudart
            cudaPackages.cuda_cccl
            cudaPackages.nsight_compute
            pkgs.gdb
            pkgs.eog
            pkgs.pprof
            pkgs.gperftools
            pkgs.go

            (pkgs.radianWrapper.override {
              wrapR = true;
              packages = [
                pkgs.rPackages.bench
                pkgs.rPackages.devtools
                pkgs.rPackages.dplyr
                pkgs.rPackages.dust
                pkgs.rPackages.gganimate
                pkgs.rPackages.ggbeeswarm
                pkgs.rPackages.ggpubr
                pkgs.rPackages.gifski
                pkgs.rPackages.individual
                pkgs.rPackages.patrick
                pkgs.rPackages.pkgdepends
                pkgs.rPackages.proffer
                pkgs.rPackages.terra
                pkgs.rPackages.tidyterra
                pkgs.rPackages.tidyverse
                pkgs.rPackages.wrswoR
              ];
            })
          ];

          R_REMOTES_UPGRADE = "never";

          shellHook = self'.packages.selectDriver.shellHook + ''
            export R_LIBS_USER=$(${lib.getExe pkgs.git} rev-parse --show-toplevel)/.libs
            mkdir -p $R_LIBS_USER
          '';
        };
      };
  });
}
