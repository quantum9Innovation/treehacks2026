"""Smooth diagonal sweep: X=150, Y from -50 to 50, Z from 0 to 400."""

import sys
import time

sys.path.insert(0, ".")
from motion import Motion

m = Motion()

print("=== Probing Ground ===")
m.probe_ground()
print(f"Ground Z: {m.ground_z:.0f}mm")
print()

X = 150
Y_START, Y_END = -50, 50
Z_START, Z_END = 0, 400
STEPS = 20

print(
    f"=== Linear Sweep: X={X}, Y=[{Y_START},{Y_END}], Z=[{Z_START},{Z_END}], {STEPS} steps ==="
)
print(f"{'Step':>4} {'Tgt Y':>6} {'Tgt Z':>6} | {'Act X':>6} {'Act Y':>6} {'Act Z':>6}")
print("-" * 55)

for i in range(STEPS + 1):
    t = i / STEPS
    y = Y_START + t * (Y_END - Y_START)
    z = Z_START + t * (Z_END - Z_START)
    try:
        m.move_to(X, y, z, speed=2000, acc=200)
        time.sleep(0.3)
        pose = m.get_pose()
        print(
            f"{i:4d} {y:6.0f} {z:6.0f} | {pose[0]:6.0f} {pose[1]:6.0f} {pose[2]:6.0f}"
        )
    except Exception as e:
        print(f"{i:4d} {y:6.0f} {z:6.0f} | FAILED: {e}")

print()
m.home()
print("Done.")
