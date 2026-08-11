import { useCallback, useRef, useState } from 'react'
import { NOW_PLAYING } from '@/consts'

/* Ported from componentry.dev/docs/components/music-player. A port, not a
   vendor drop — `shadcn add` output would not survive contact with this
   codebase, so the deviations are listed here rather than pretended away:

     - No framer-motion. Upstream pulls the whole library in to rotate the
       tonearm by 30 degrees, which is a `transition: transform`. This repo has
       no animation dependency and does not need one for a two-state swing.
     - Every colour is a project token. Upstream is wall-to-wall `zinc-*`, and
       `--color-*: initial` in global.css deleted those utilities, so a literal
       port renders as an invisible blob rather than failing loudly.
     - `<audio>` only. Upstream's other branch plays YouTube through a hidden
       iframe; see the NOW_PLAYING comment in src/consts.ts for why that branch
       is not here.
     - Fixed at 112px. Upstream is 256/320px, sized for a hero. At footer scale
       its tonearm geometry collapses to sub-pixel lint, so the arm below is
       drawn for this diameter rather than scaled down proportionally.

   Sizing note: the tonearm base is anchored at the record's top-right corner
   and translated out by half its own width, so it overhangs ~5px past the disc
   on both axes. That fits inside the layout's px-6 gutter; widening the arm
   without checking that will push a horizontal scrollbar onto the page. */

// Grooves and glare are gradients over the cover art, and both need partial
// transparency. `color-mix` against the palette keeps them derived from the
// tokens instead of hardcoding an rgba() that would drift on a repaint (and
// that tests/tokens.test.ts would reject on sight).
const GROOVES =
  'repeating-radial-gradient(circle, transparent 0 2px, color-mix(in srgb, var(--color-bg) 60%, transparent) 2px 3px)'
const GLARE =
  'linear-gradient(135deg, color-mix(in srgb, var(--color-fg) 28%, transparent) 0%, transparent 42%, transparent 60%, color-mix(in srgb, var(--color-fg) 14%, transparent) 100%)'

export default function MusicPlayer() {
  const [playing, setPlaying] = useState(false)
  const audioRef = useRef<HTMLAudioElement | null>(null)

  const toggle = useCallback(() => {
    const audio = audioRef.current
    if (!audio) return
    if (audio.paused) {
      // play() rejects when the browser blocks autoplay or the file is
      // missing. Only claim to be playing once it actually resolves, or the
      // record spins over silence.
      audio.play().then(
        () => setPlaying(true),
        () => setPlaying(false),
      )
    } else {
      audio.pause()
      setPlaying(false)
    }
  }, [])

  return (
    <>
      {/* preload="none": the home page should not spend a request, or the
          visitor's bandwidth, on audio nobody has asked to hear yet. */}
      <audio
        ref={audioRef}
        src={NOW_PLAYING.src}
        loop
        preload="none"
        onEnded={() => setPlaying(false)}
      />
      <button
        type="button"
        onClick={toggle}
        aria-pressed={playing}
        aria-label={`${playing ? 'Pause' : 'Play'} ${NOW_PLAYING.title} by ${NOW_PLAYING.artist}`}
        className="relative h-28 w-28 cursor-pointer"
      >
        {/* Tonearm. Swings in over the record on play; pure CSS transition. */}
        <div
          className="pointer-events-none absolute right-0 top-0 z-20 origin-top-right transition-transform duration-500 ease-in-out"
          style={{ transform: `rotate(${playing ? -20 : 10}deg)` }}
        >
          <div className="absolute right-0 top-0 h-2.5 w-2.5 -translate-y-1/2 translate-x-1/2 rounded-full border border-border bg-muted" />
          <div className="absolute right-0 top-0 h-0.5 w-9 origin-right -rotate-12 rounded-full bg-muted">
            <div className="absolute left-0 top-1/2 h-1 w-1 -translate-x-1/2 -translate-y-1/2 rounded-full bg-fg" />
          </div>
        </div>

        {/* The record. `animation-play-state: paused` rather than dropping the
            animation: a paused record holds its angle the way a real one does,
            where removing the class would snap it back to zero. */}
        <div
          className="animate-record-spin relative h-28 w-28 overflow-hidden rounded-full border border-border motion-reduce:animate-none"
          style={{
            backgroundImage: `${GLARE}, ${GROOVES}, url(${NOW_PLAYING.coverArt})`,
            backgroundSize: 'cover',
            animationPlayState: playing ? 'running' : 'paused',
          }}
        >
          <div className="absolute left-1/2 top-1/2 flex h-1/3 w-1/3 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-border bg-bg">
            <div className="h-1 w-1 rounded-full bg-muted" />
          </div>
        </div>
      </button>
    </>
  )
}
