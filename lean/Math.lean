import Types

namespace Math

def linspace (bounds : Bounds) (n : Nat) : List Float :=
  if n == 0 then 
    [bounds.max] 
  else
    let step := (bounds.max - bounds.min) / n.toFloat
    List.range (n + 1) |>.map (fun i => bounds.min + i.toFloat * step)

def lemniscate (a : Float) (t : Float) : Point2D :=
  (a * cost / denom, a * sint * cost / denom)
  where
    cost : Float := Float.cos t
    sint : Float := Float.sin t
    denom : Float := 1 + sint^2

end Math
