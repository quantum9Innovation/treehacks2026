import { useEffect, useRef } from 'react'

interface VideoPanelProps {
  track: MediaStreamTrack | null
  label: string
}

export function VideoPanel({ track, label }: VideoPanelProps) {
  const videoRef = useRef<HTMLVideoElement>(null)

  useEffect(() => {
    if (!videoRef.current) return
    if (track) {
      videoRef.current.srcObject = new MediaStream([track])
    } else {
      videoRef.current.srcObject = null
    }
  }, [track])

  return (
    <div className="flex flex-col gap-1.5">
      <span className="text-xs font-medium text-muted-foreground">{label}</span>
      <div className="relative aspect-[4/3] overflow-hidden rounded-xl border bg-black">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="absolute inset-0 h-full w-full object-contain"
        />
        {!track && (
          <div className="absolute inset-0 flex items-center justify-center">
            <p className="text-sm text-muted-foreground">No signal</p>
          </div>
        )}
      </div>
    </div>
  )
}
