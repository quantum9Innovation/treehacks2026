import Math
import Types
import Utils
open Nat

-- constants
def limit : Float := 490
def floor : Millimeters := 75
def maxGrip : Float := 90

-- lemniscate
namespace Lemniscate

def radius : Millimeters := 350
def squish : Float := 0.4
def nSamples : Nat := 150
def parametricBounds : Bounds := {min := 0, max := 2 * Math.π}
def tSamples : List Float := Math.linspace parametricBounds nSamples
def lemniscatePlane : Traceable2D :=
  Math.lemniscate radius squish
def planeSamples : List Point2D := Utils.dualize <| tSamples.map lemniscatePlane

end Lemniscate

-- sine wave
namespace Wave

def wavelength : Millimeters := 150
def amplitude : Millimeters := 100
def nSamples : Nat := 200
def parametricBounds : Bounds := {min := -1 * wavelength, max := 1 * wavelength}
def tSamples : List Float := Math.linspace parametricBounds nSamples
def wavePlane : Traceable2D :=
  Math.wave amplitude wavelength
def planeSamples : List Point2D := Utils.dualize <| tSamples.map wavePlane

end Wave

-- spring
namespace Spring

def radius : Millimeters := 250
def amplitude : Millimeters := 400
def frequency : Hertz := 1.5
def phaseShift : Float := 0
def nSamples : Nat := 200
def parametricBounds : Bounds := {min := 0, max := 9 / frequency}
def tSamples : List Float := Math.linspace parametricBounds nSamples
def springPlane : Traceable2D :=
  Math.spring radius amplitude frequency phaseShift
def planeSamples : List Point2D := Utils.dualize <| tSamples.map springPlane

end Spring

-- lorenz attractor
namespace Lorenz

def σ : Float := 10
def ρ : Float := 28
def β : Float := 8 / 3

def attractor : Point3D → Derivative3D := Math.lorenz σ ρ β
def rk4Samples : Nat := 100000
def nSamples : Nat := 600
def selectSamples : List Nat := List.map (λ x => x.toUInt64.toNat) (Math.linspace {min := 0, max := Float.ofNat (nSamples - 1)} nSamples)
def stepSize : Float := 0.01
def solver : Point3D → Point3D := Math.rk4 stepSize attractor
def y0 : Point3D := {x := 10, y := 20, z := 30}
def solution : List Point3D :=
  let rec loop (i : Nat) (acc : List Point3D) (curr : Point3D) : List Point3D :=
      match i with
      | 0 => acc.reverse
      | succ m => loop m (curr :: acc) (solver curr)

  loop rk4Samples [y0] (solver y0)
  
def curveSamples : List Point3D := Utils.enumFilterMap (λ (i : Nat) (x : Point3D) => if selectSamples.contains i then some x else none) solution
def scaling : Float := 4.0
def pointSamples : List Point3D := Utils.dualize <| curveSamples.map (λ p => scaling * p)

end Lorenz

-- rossler
namespace Rossler

def a : Float := 0.2
def b : Float := 0.2
def c : Float := 5.7

def attractor : Point3D → Derivative3D := Math.lorenz a b c 
def rk4Samples : Nat := 100000
def nSamples : Nat := 600
def selectSamples : List Nat := List.map (λ x => x.toUInt64.toNat) (Math.linspace {min := 0, max := Float.ofNat (nSamples - 1)} nSamples)
def stepSize : Float := 0.01
def solver : Point3D → Point3D := Math.rk4 stepSize attractor
def y0 : Point3D := {x := 5, y := 6, z := 7}
def solution : List Point3D :=
  let rec loop (i : Nat) (acc : List Point3D) (curr : Point3D) : List Point3D :=
      match i with
      | 0 => acc.reverse
      | succ m => loop m (curr :: acc) (solver curr)

  loop rk4Samples [y0] (solver y0)
  
def curveSamples : List Point3D := Utils.enumFilterMap (λ (i : Nat) (x : Point3D) => if selectSamples.contains i then some x else none) solution
def scaling : Float := 10.0
def pointSamples : List Point3D := Utils.dualize <| curveSamples.map (λ p => scaling * p)

end Rossler

-- helix
namespace Helix

def radius : Millimeters := 100
def turns : Nat := 4
def height : Millimeters := 400
def samplesPerTurn : Nat := 100
def nSamples : Nat := turns * samplesPerTurn
def parametricBounds : Bounds := {min := 0, max := Float.ofNat turns}
def tSamples : List Float := Math.linspace parametricBounds nSamples
def helixPlane : Traceable3D :=
  Math.helix radius height turns
def pointSamples : List Point3D := Utils.dualize <| tSamples.map helixPlane

end Helix
