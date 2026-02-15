import Types

namespace Math

def π : Float := 3.1415926536

def linspace (bounds : Bounds) (n : Nat) : List Float :=
  if n == 0 then 
    [bounds.max] 
  else
    let step := (bounds.max - bounds.min) / n.toFloat
    List.range (n + 1) |>.map (λ i => bounds.min + i.toFloat * step)
    
def rk4 [VectorLike Float β] (h : Float) (y' : β → β) (y : β) : β :=
  let dy := y' y
  let k1 := dy
  let k2 := y' (y + (h / 2) * k1)
  let k3 := y' (y + (h / 2) * k2)
  let k4 := y' (y + h * k3)
  y + h / 6 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

def lemniscate (a h t : Float) : Point2D :=
  {x := a * cost / denom, y := h * a * sint * cost / denom}
  where
    cost : Float := Float.cos t
    sint : Float := Float.sin t
    denom : Float := 1 + sint^2
    
def wave (a μ t : Float) : Point2D :=
  {x := t, y := a * Float.sin (t * (2 * π) / μ)}
  
def spring (x A ω ϕ t : Float) : Point2D :=
  {x := x, y := A * Float.cos (ω * t / (2 * π) - ϕ)}

def lorenz (σ ρ β: Float) (vec : Point3D) : Derivative3D :=
  {x := σ * (vec.y - vec.x), y := vec.x * (ρ - vec.z) - vec.y, z := vec.x * vec.y - β * vec.z}
  
def rossler (a b c: Float) (vec : Point3D) : Derivative3D :=
  {x := -vec.y - vec.z, y := vec.x + a * vec.y, z := b + vec.z * (vec.x - c)}
  
def helix (radius height : Float) (turns : Nat) (t : Float) : Point3D :=
  let overTurns := 1 / (Float.ofNat turns)
  {x := radius * Float.cos (t * (2 * π) * overTurns), y := radius * Float.sin (t * (2 * π) * overTurns), z := height * t * overTurns}

end Math
