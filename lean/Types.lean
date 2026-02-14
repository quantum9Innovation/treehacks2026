abbrev Coordinate : Type := Float
abbrev Millimeters : Type := Float
abbrev Degrees : Type := Float

structure Point2D where
  x : Coordinate
  y : Coordinate

structure Point3D where
  x : Coordinate
  y : Coordinate
  z : Coordinate

def Point2D.add (u v : Point2D) : Point2D :=
  let args : List Point2D := [u, v]
  let addElem (f : Point2D → Float) : Float := (args.map f).sum
  {x := addElem (·.x), y := addElem (·.y)}

def Point3D.add (u v : Point3D) : Point3D :=
  let args : List Point3D := [u, v]
  let addElem (f : Point3D → Float) : Float := (args.map f).sum
  {x := addElem (·.x), y := addElem (·.y), z := addElem (·.z)}

def Point2D.scale (k : Float) (u : Point2D) : Point2D :=
  let scaleElem (f : Point2D → Float) : Float := k * f u
  {x := scaleElem (·.x), y := scaleElem (·.y)}

def Point3D.scale (k : Float) (u : Point3D) : Point3D :=
  let scaleElem (f : Point3D → Float) : Float := k * f u
  {x := scaleElem (·.x), y := scaleElem (·.y), z := scaleElem (·.z)}

instance : Add Point2D where
  add := Point2D.add

instance : HMul Float Point2D Point2D where
  hMul := Point2D.scale

instance : Add Point3D where
  add := Point3D.add

instance : HMul Float Point3D Point3D where
  hMul := Point3D.scale
 
-- define an interface for arbitrary vector spaces 
class VectorLike (α : Type) (β : Type) extends HMul α β β, Add β
-- automatically instantiate types as vector-like
instance [HMul α β β] [Add β] : VectorLike α β := {}

abbrev Derivative3D : Type := Point3D
def Traceable : Type := Float → Point2D

structure Bounds where
  min : Float
  max : Float
