{
  description = "main";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
      ];
    in
    flake-utils.lib.eachSystem supportedSystems (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python313;
        roarm_sdk = python.pkgs.buildPythonPackage rec {
          pname = "roarm_sdk";
          version = "0.1.0";
          
          pyproject = true;
          build-system = with python.pkgs; [ setuptools ];
        
          src = python.pkgs.fetchPypi {
            inherit pname version;
            sha256 = "sha256-Y0yTonKQBzX0QYQft9aiJvEMMl3pwXLVhOfPbJVdggA=";
          };
        
          propagatedBuildInputs = with python.pkgs; [
            pyserial
            requests
            simplejson
            pytest
            flake8
          ];
        
          pythonImportsCheck = [ "roarm_sdk" ];
        };
        pythonEnv = python.withPackages (
          ps: [
            ps.mypy
            ps.keyboard
            ps.websockets
            ps.flask
            ps.opencv4
            roarm_sdk
          ]
        );
      in
      {
        packages.default = pkgs.writeShellApplication {
          name = "main";
          runtimeInputs = [ pythonEnv ];
          text = ''
            exec python ${./main.py} "$@"
          '';
        };

        devShells.default = pkgs.mkShell {
          buildInputs = [ pythonEnv ];
        };

        formatter = nixpkgs.legacyPackages.${system}.nixfmt;
      }
    );
}
