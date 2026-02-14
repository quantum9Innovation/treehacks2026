import Lake
open Lake

package «roarm» where
  version := "1.0.0"

@[default_target]
lean_lib «Roarm» where
  srcDir := "."
  roots := #[`Math, `Types, `Trajectories, `Utils]

@[default_target]
lean_exe «roarm» where
  root := `Main
 