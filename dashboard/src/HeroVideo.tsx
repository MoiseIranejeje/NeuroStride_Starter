import { useEffect, useRef, useState } from 'react'

type VideoSource = {
  url: string
  mime: string
}

// Only use published MuJoCo footage. Missing media leaves the graphite hero in place.
export function HeroVideo() {
  const [source, setSource] = useState<VideoSource | null>(null)
  const [userPaused, setUserPaused] = useState(false)
  const videoRef = useRef<HTMLVideoElement | null>(null)

  useEffect(() => {
    const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)')
    const reducedData = window.matchMedia('(prefers-reduced-data: reduce)')
    const controller = new AbortController()

    if (reducedMotion.matches || reducedData.matches) {
      return () => controller.abort()
    }

    const probeRealMedia = async () => {
      const decoder = document.createElement('video')
      const candidates: VideoSource[] = [
        { url: '/media/go2-hero.webm', mime: 'video/webm' },
        { url: '/media/go2-hero.mp4', mime: 'video/mp4' },
      ]

      for (const candidate of candidates) {
        if (decoder.canPlayType(candidate.mime) === '') continue

        try {
          const response = await fetch(candidate.url, {
            method: 'HEAD',
            signal: controller.signal,
            cache: 'force-cache',
          })
          const contentType = response.headers.get('content-type') ?? ''
          if (response.ok && /^(video\/|application\/octet-stream)/.test(contentType)) {
            if (!controller.signal.aborted) setSource(candidate)
            return
          }
        } catch {
          if (controller.signal.aborted) return
        }
      }
    }

    void probeRealMedia()
    return () => controller.abort()
  }, [])

  useEffect(() => {
    const video = videoRef.current
    if (!video || !source) return

    let inView = false
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)')
    const data = window.matchMedia('(prefers-reduced-data: reduce)')

    const syncPlayback = () => {
      if (userPaused || motion.matches || data.matches || !inView || document.hidden) {
        video.pause()
      } else {
        void video.play().catch(() => {
          // Browser autoplay permissions vary; the static background remains.
        })
      }
    }

    const observer = new IntersectionObserver(([entry]) => {
      inView = entry.isIntersecting
      syncPlayback()
    }, { threshold: 0.15 })

    observer.observe(video)
    document.addEventListener('visibilitychange', syncPlayback)
    motion.addEventListener('change', syncPlayback)
    data.addEventListener('change', syncPlayback)

    return () => {
      observer.disconnect()
      video.pause()
      document.removeEventListener('visibilitychange', syncPlayback)
      motion.removeEventListener('change', syncPlayback)
      data.removeEventListener('change', syncPlayback)
    }
  }, [source, userPaused])

  if (!source) return null

  return (
    <div className="hero-media">
      <video
        ref={videoRef}
        aria-hidden="true"
        className="hero-media-video"
        src={source.url}
        muted
        loop
        playsInline
        preload="none"
        tabIndex={-1}
        disablePictureInPicture
      />
      <button type="button" className="hero-media-toggle" onClick={() => setUserPaused(!userPaused)}>
        {userPaused ? 'Play replay' : 'Pause replay'}
      </button>
    </div>
  )
}
