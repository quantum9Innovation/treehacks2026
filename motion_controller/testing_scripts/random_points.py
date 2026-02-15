"""Move to random reachable points around the arm's workspace sphere."""

import math
import random
import sys
import time

sys.path.insert(0, ".")
from motion import Motion

m = Motion()

print("=== Probing Ground ===")
m.probe_ground()
print(f"Ground Z: {m.ground_z:.0f}mm")
print()

# Generate random reachable points in cylindrical coords
# Radius: 100-280mm, Angle: -180 to 180, Z: 10-400mm
random.seed(42)
N = 20

print(f"=== Moving to {N} random reachable points ===")
print(
    f"{'#':>2} {'Tgt X':>6} {'Tgt Y':>6} {'Tgt Z':>6} | {'Act X':>6} {'Act Y':>6} {'Act Z':>6}"
)
print("-" * 58)

for i in range(N):
    r = random.uniform(100, 280)
    angle = random.uniform(-math.pi, math.pi)
    z = random.uniform(10, 400)
    x = r * math.cos(angle)
    y = r * math.sin(angle)

    try:
        m.move_to(x, y, z, speed=3000, acc=200)
        time.sleep(0.5)
        pose = m.get_pose()
        print(
            f"{i + 1:2d} {x:6.0f} {y:6.0f} {z:6.0f} | {pose[0]:6.0f} {pose[1]:6.0f} {pose[2]:6.0f}"
        )
    except Exception as e:
        print(f"{i + 1:2d} {x:6.0f} {y:6.0f} {z:6.0f} | FAILED: {e}")

print()
m.home()
print("Done.")
