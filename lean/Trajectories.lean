import Math
import Types
import Utils
open Nat

-- constants
def π : Float := 3.1415926536
def limit : Float := 490
def floor : Millimeters := 200
def maxGrip : Float := 90

-- lemniscate
namespace Lemniscate

def radius : Millimeters := 300
def nSamples : Nat := 50
def parametricBounds : Bounds := {min := 0, max := 2 * π}
def tSamples : List Float := Math.linspace parametricBounds nSamples
def lemniscatePlane : Traceable :=
  Math.lemniscate radius
def planeSamples : List Point2D := tSamples.map lemniscatePlane

end Lemniscate

-- lorenz attractor
namespace Lorenz

def σ : Float := 10
def ρ : Float := 28
def β : Float := 8 / 3

def attractor : Point3D → Derivative3D := Math.lorenz σ ρ β
def rk4Samples : Nat := 100000
def nSamples : Nat := 600
def selectSamples : List Nat := List.map (fun x => x.toUInt64.toNat) (Math.linspace {min := 0, max := Float.ofNat (nSamples - 1)} nSamples)
def stepSize : Float := 0.01
def solver : Point3D → Point3D := Math.rk4 stepSize attractor
def y0 : Point3D := {x := 10, y := 20, z := 30}
def solution : List Point3D :=
  let rec loop (i : Nat) (acc : List Point3D) (curr : Point3D) : List Point3D :=
      match i with
      | 0 => acc.reverse
      | succ m => loop m (curr :: acc) (solver curr)

  loop rk4Samples [y0] (solver y0)
  
def curveSamples : List Point3D := Utils.enumFilterMap (fun (i : Nat) (x : Point3D) => if selectSamples.contains i then some x else none) solution
def scaling : Float := 4.0
def pointSamples : List Point3D := curveSamples.map (fun p => scaling * p)

end Lorenz
