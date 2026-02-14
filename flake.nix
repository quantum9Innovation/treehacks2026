{
  description = "pre-commit checks";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixpkgs-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    pre-commit.url = "github:cachix/git-hooks.nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      pre-commit,
    }:
    let
      supportedSystems = [
        "x86_64-linux"
      ];
    in
    flake-utils.lib.eachSystem supportedSystems (
      system:
      let
        pkgs = import nixpkgs { inherit system; };
      in
      {
        checks = {
          pre-commit-check = pre-commit.lib.${system}.run {
            src = ./.;
            hooks = {
              autoflake.enable = true;
              check-builtin-literals.enable = true;
              check-docstring-first.enable = true;
              check-python.enable = true;
              isort.enable = false; # conflicts with formatting
              mypy.enable = false; # no access to libraries
              name-tests-test.enable = true;
              pyright.enable = true;
              python-debug-statements.enable = true;
              pyupgrade.enable = true;
              ruff.enable = true;
              ruff-format.enable = true;
            };
          };
        };

        devShells.default = pkgs.mkShell {
          buildInputs = self.checks.${system}.pre-commit-check.enabledPackages;
          inherit (self.checks.${system}.pre-commit-check) shellHook;
        };

        formatter = nixpkgs.legacyPackages.${system}.nixfmt;
      }
    );
}
