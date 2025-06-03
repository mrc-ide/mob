{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
    reside.url = "github:plietar/reside.nix";
    reside.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, reside }:
    let
      pkgs = import nixpkgs {
        system = "x86_64-linux";
        config.allowUnfree = true;
        config.nvidia.acceptLicense = true;
        overlays = [ reside.overlays.default ];
      };
      inherit (pkgs) lib;

      cudaPackages = pkgs.cudaPackages_12_8;
      mkShell = pkgs.mkShell.override { stdenv = cudaPackages.backendStdenv; };

      buildDriver = version: attrs: (pkgs.linuxPackages.nvidiaPackages.mkDriver ({
        useSettings = false;
        usePersistenced = false;
        inherit version;
      } // attrs)).override {
        libsOnly = true;
      };

      # cat /sys/module/nvidia/version
      drivers = lib.mapAttrs buildDriver {
        "575.51.02" = { sha256_64bit = "sha256-XZ0N8ISmoAC8p28DrGHk/YN1rJsInJ2dZNL8O+Tuaa0="; };
        "570.124.06" = {
          url = "https://us.download.nvidia.com/tesla/570.124.06/NVIDIA-Linux-x86_64-570.124.06.run";
          sha256_64bit = "sha256-GBjJBlfRflEN6foDI4X/fpkGPoSOkBy0Y27nHIszkxM=";
        };
      };

      select-driver = pkgs.writeShellApplication {
        name = "select-driver";
        runtimeInputs = [ pkgs.jq ];
        text = ''
          jq --raw-output \
            --rawfile version /sys/module/nvidia/version \
            'to_entries | .[] | select((.key | split(".")[:2]) == ($version | split(".")[:2])) | .value' \
            "${pkgs.writers.writeJSON "drivers.json" drivers}"
        '';
      };
    in
    {
      devShells.x86_64-linux.default = mkShell {
        hardeningDisable = [ "all" ];
        buildInputs = [
          select-driver
          cudaPackages.cudatoolkit
          cudaPackages.cuda_cudart
          cudaPackages.cuda_cccl
          cudaPackages.nsight_compute
          pkgs.gdb
          pkgs.cmake
          pkgs.eog
          pkgs.pprof
          pkgs.gperftools

          (pkgs.radianWrapper.override {
            wrapR = true;
            packages = [
              pkgs.rPackages.bench
              pkgs.rPackages.devtools
              pkgs.rPackages.dplyr
              pkgs.rPackages.dust
              pkgs.rPackages.individual
              pkgs.rPackages.gganimate
              pkgs.rPackages.ggbeeswarm
              pkgs.rPackages.ggpubr
              pkgs.rPackages.gifski
              pkgs.rPackages.pkgdepends
              pkgs.rPackages.proffer
              pkgs.rPackages.terra
              pkgs.rPackages.tidyterra
              pkgs.rPackages.tidyverse
              pkgs.rPackages.patrick
            ];
          })
        ];

        shellHook = ''
          export R_REMOTES_UPGRADE=never
          export R_LIBS_USER=$(${lib.getExe pkgs.git} rev-parse --show-toplevel)/.libs
          mkdir -p $R_LIBS_USER
          export LD_LIBRARY_PATH="$(${lib.getExe select-driver})/lib"
        '';
      };
    };
}
