abbrev Coordinate : Type := Float
abbrev Millimeters : Type := Float
abbrev Degrees : Type := Float
def Point2D : Type := Coordinate × Coordinate
def Traceable : Type := Float → Point2D

structure Bounds where
  min : Float
  max : Float
