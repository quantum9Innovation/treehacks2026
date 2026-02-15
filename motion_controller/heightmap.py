import math
import time

from motion import Motion


def main():
    m = Motion()

    # Calibrate ground at 90 deg left (away from scan area)
    print("=== Ground Calibration ===")
    m.probe_ground(base_angle=math.pi / 2)
    m.home()
    time.sleep(2)

    # Grid parameters
    n_angles = 9
    n_radii = 9
    angle_min, angle_max = -30, 30  # degrees
    radius_min, radius_max = 150, 280  # mm (arm max reach ~316mm, ~299mm at ground)

    angles_deg = [
        angle_min + i * (angle_max - angle_min) / (n_angles - 1)
        for i in range(n_angles)
    ]
    radii = [
        radius_min + i * (radius_max - radius_min) / (n_radii - 1)
        for i in range(n_radii)
    ]

    print(
        f"\n=== Heightmap Scan ({n_angles}x{n_radii} = {n_angles * n_radii} points) ==="
    )
    print(f"Base angles: {[f'{a:.1f}' for a in angles_deg]} deg")
    print(f"Radii: {[int(r) for r in radii]} mm\n")

    heights = [[0.0] * n_radii for _ in range(n_angles)]

    for i, angle_deg in enumerate(angles_deg):
        angle_rad = math.radians(angle_deg)
        for j, radius in enumerate(radii):
            x = radius * math.cos(angle_rad)
            y = radius * math.sin(angle_rad)

            h = m.probe_height_at(x, y)
            heights[i][j] = h

            label = f"[{i * n_radii + j + 1}/{n_angles * n_radii}]"
            if h is None:
                print(f"  {label} angle={angle_deg:+.1f}° r={radius:.0f}mm -> SKIP")
            else:
                print(
                    f"  {label} angle={angle_deg:+.1f}° r={radius:.0f}mm -> z={h:.0f}mm"
                )

    # Print results grid
    print("\n=== Heightmap (mm above ground) ===\n")
    header = "         " + "  ".join(f"{int(r):>5}" for r in radii)
    print(header)
    print("         " + "  ".join("-----" for _ in radii))
    for i, angle_deg in enumerate(angles_deg):
        row = f"{angle_deg:>+6.1f}° |"
        for j in range(n_radii):
            h = heights[i][j]
            row += "    N/A" if h is None else f"  {h:>5.0f}"
        print(row)

    print("\nReturning home...")
    m.home()


if __name__ == "__main__":
    main()
