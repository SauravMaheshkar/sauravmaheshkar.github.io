import { useEffect, useState } from 'react'
import FlipClock from '@/components/8starlabs-ui/flip-clock'
import { HERO } from '@/consts'

const PARTS = new Intl.DateTimeFormat('en-US', {
  timeZone: HERO.timeZone,
  hour12: false,
  year: 'numeric',
  month: '2-digit',
  day: '2-digit',
  hour: '2-digit',
  minute: '2-digit',
  second: '2-digit',
})

/**
 * FlipClock's real-time mode has no `value`/timezone prop: it reads
 * `now.getHours()`/`getMinutes()`/`getSeconds()`, which are always the
 * *local* getters of whatever Date it's handed. To show HERO.timeZone's
 * wall clock to every visitor regardless of their own timezone, build a
 * Date whose local fields already equal that zone's civil time — those
 * getters then echo the right numbers back, no matter what timezone the
 * browser itself runs in.
 */
function zonedNow(): Date {
  const parts = PARTS.formatToParts(new Date())
  const get = (type: string) => Number(parts.find((p) => p.type === type)?.value)

  return new Date(get('year'), get('month') - 1, get('day'), get('hour') % 24, get('minute'), get('second'))
}

export default function Clock() {
  // This is a static build (no adapter in astro.config.mjs): `useState(zonedNow)`
  // below still runs at build time on the server, so its very first value is
  // whenever the site was last deployed, not "now". Left unguarded, that
  // stale snapshot would sit in dist/index.html and read as the live time to
  // every visitor until the client:visible island hydrates — on a personal
  // site, potentially days later than it claims. `mounted` keeps the
  // server-rendered and first client-rendered output identical — a placeholder, not a fake
  // reading — and only swaps in the live, client-computed value once an
  // effect (which never runs during SSR) actually confirms we're in a browser.
  const [mounted, setMounted] = useState(false)
  const [now, setNow] = useState(zonedNow)
  const [paused, setPaused] = useState(false)

  useEffect(() => {
    setMounted(true)
    setNow(zonedNow())

    let timer: ReturnType<typeof setInterval> | undefined

    const start = () => {
      if (timer) return
      timer = setInterval(() => setNow(zonedNow()), 1000)
    }
    const stop = () => {
      clearInterval(timer)
      timer = undefined
    }
    // A backgrounded tab has no business ticking a clock, or keeping the
    // flip-clock widget's own animation timer alive, every second. Fully
    // unmounting FlipClock (via `paused`) also clears its internal poll,
    // not just ours.
    const onVisibility = () => {
      if (document.hidden) {
        stop()
        setPaused(true)
      } else {
        setNow(zonedNow())
        setPaused(false)
        start()
      }
    }

    start()
    document.addEventListener('visibilitychange', onVisibility)
    return () => {
      stop()
      document.removeEventListener('visibilitychange', onVisibility)
    }
  }, [])

  if (!mounted) {
    return (
      <div className="flex h-14 items-center justify-center gap-1 font-mono text-3xl text-muted" aria-hidden="true">
        --:--:--
      </div>
    )
  }

  if (paused) return null

  return <FlipClock now={now} size="sm" variant="outline" />
}
