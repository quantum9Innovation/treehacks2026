import Lean
import Math
import Types
import Trajectories

open Lean Json System

-- Directories
def trajectoryDir : String := "../control/trajectories/"

-- Subtypes
def CoordinateX : Type := {x : Float // (-limit ≤ x && x ≤ limit) = true}
def CoordinateY : Type := {y : Float // (-limit ≤ y && y ≤ limit) = true}
def CoordinateZ : Type := {z : Float // (0.0 ≤ z && z ≤ limit) = true}
def GripAngle : Type := {angle : Float // (0.0 ≤ angle && angle ≤ maxGrip) = true}

-- Instances
instance : ToString CoordinateX := ⟨(toString ·.val)⟩
instance : ToString CoordinateY := ⟨(toString ·.val)⟩
instance : ToString CoordinateZ := ⟨(toString ·.val)⟩
instance : ToString GripAngle  := ⟨(toString ·.val)⟩

-- Constructors with corresponding proof strategy
def cX (f : Millimeters) (h : (-limit ≤ f && f ≤ limit) = true := by native_decide) : CoordinateX := ⟨f, h⟩
def cY (f : Millimeters) (h : (-limit ≤ f && f ≤ limit) = true := by native_decide) : CoordinateY := ⟨f, h⟩
def cZ (f : Millimeters) (h : (0.0 ≤ f && f ≤ limit) = true := by native_decide) : CoordinateZ := ⟨f, h⟩
def gA (f : Degrees) (h : (0.0 ≤ f && f ≤ maxGrip) = true := by native_decide) : GripAngle := ⟨f, h⟩

def cX_safe (f : Float) : Option CoordinateX :=
  if h : (-limit ≤ f && f ≤ limit) then some ⟨f, h⟩ else none

def cY_safe (f : Float) : Option CoordinateY :=
  if h : (-limit ≤ f && f ≤ limit) then some ⟨f, h⟩ else none

def cZ_safe (f : Float) : Option CoordinateZ :=
  if h : (0.0 ≤ f && f ≤ limit) then some ⟨f, h⟩ else none

def gA_safe (f : Float) : Option GripAngle :=
  if h : (0.0 ≤ f && f ≤ maxGrip) then some ⟨f, h⟩ else none

-- Allowable pose coordinates
def xPlane : CoordinateX := cX 200
def Pose := CoordinateX × CoordinateY × CoordinateZ × GripAngle

def writeTrajectory (path : FilePath) (trajectory : List Pose) (delay : Seconds) : IO Unit := do
  let trajectoryJson : List Json := trajectory.map λ (x, y, z, t) =>
    Json.arr #[Json.str (toString x), Json.str (toString y), Json.str (toString z), Json.str (toString t)]
  let jsonObj := Json.mkObj [("trajectory", Json.arr (trajectoryJson.toArray)), ("delay", toJson delay)]
  IO.FS.writeFile path (jsonObj.pretty 4)

def createPlaneTrajectory (planeFigure : List Point2D) (slice : CoordinateX) : List Pose :=
  planeFigure.filterMap λ p => do
    let yCoord ← cY_safe p.x
    let zCoord ← cZ_safe (p.y + floor)
    let angle  ← gA_safe 0.0
    return (slice, yCoord, zCoord, angle)
    
def createCurveTrajectory (pointSamples : List Point3D) : List Pose :=
  pointSamples.filterMap λ p => do
    let xCoord ← cX_safe p.x
    let yCoord ← cY_safe p.y
    let zCoord ← cZ_safe (p.z + floor)
    let angle  ← gA_safe 0.0
    return (xCoord, yCoord, zCoord, angle)

def lemniscateTrajectory : List Pose := createPlaneTrajectory Lemniscate.planeSamples xPlane
def waveTrajectory : List Pose := createPlaneTrajectory Wave.planeSamples xPlane
def springTrajectory : List Pose := createPlaneTrajectory Spring.planeSamples xPlane
def springShiftTrajectory1 : List Pose := createPlaneTrajectory Spring.planeSamplesShift1 xPlane
def springShiftTrajectory2 : List Pose := createPlaneTrajectory Spring.planeSamplesShift2 xPlane
def springShiftTrajectory3 : List Pose := createPlaneTrajectory Spring.planeSamplesShift3 xPlane
def lorenzTrajectory : List Pose := createCurveTrajectory Lorenz.pointSamples
def rosslerTrajectory : List Pose := createCurveTrajectory Rossler.pointSamples
def helixTrajectory : List Pose := createCurveTrajectory Helix.pointSamples

def main : IO Unit := do
  writeTrajectory (trajectoryDir ++ "lemniscate.json") lemniscateTrajectory 0.005
  writeTrajectory (trajectoryDir ++ "wave.json") waveTrajectory 0.02
  writeTrajectory (trajectoryDir ++ "spring.json") springTrajectory 0.005
  writeTrajectory (trajectoryDir ++ "springShift1.json") springShiftTrajectory1 0.005
  writeTrajectory (trajectoryDir ++ "springShift2.json") springShiftTrajectory2 0.005
  writeTrajectory (trajectoryDir ++ "springShift3.json") springShiftTrajectory3 0.005
  writeTrajectory (trajectoryDir ++ "lorenz.json") lorenzTrajectory 0.02
  writeTrajectory (trajectoryDir ++ "rossler.json") rosslerTrajectory 0.02
  writeTrajectory (trajectoryDir ++ "helix.json") helixTrajectory 0.02
