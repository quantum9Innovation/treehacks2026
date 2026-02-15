import { CAMERA_WIDTH, CAMERA_HEIGHT, GRID_SPACING } from '@/lib/constants'

export function GridOverlay() {
  const lines: React.ReactNode[] = []

  for (let x = 0; x <= CAMERA_WIDTH; x += GRID_SPACING) {
    lines.push(
      <line key={`v${x}`} x1={x} y1={0} x2={x} y2={CAMERA_HEIGHT} stroke="#b4b4b4" strokeWidth={0.5} opacity={0.4} />,
    )
    lines.push(
      <text key={`vt${x}`} x={x + 2} y={12} fill="#b4b4b4" fontSize={9}>
        {x}
      </text>,
    )
  }

  for (let y = 0; y <= CAMERA_HEIGHT; y += GRID_SPACING) {
    lines.push(
      <line key={`h${y}`} x1={0} y1={y} x2={CAMERA_WIDTH} y2={y} stroke="#b4b4b4" strokeWidth={0.5} opacity={0.4} />,
    )
    lines.push(
      <text key={`ht${y}`} x={2} y={y + 12} fill="#b4b4b4" fontSize={9}>
        {y}
      </text>,
    )
  }

  return <g>{lines}</g>
}
