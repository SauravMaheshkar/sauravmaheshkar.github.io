import { useEffect, useRef, type CSSProperties } from 'react'
import createGlobe from 'cobe'
import { GLOBE_COLORS, JOURNEY } from '@/consts'

/** JOURNEY is `as const`; cobe's types ask for mutable tuples. */
const tuple = (pair: readonly [number, number]): [number, number] => [...pair]

/**
 * Markers and arcs are exported because tests/globe.test.ts pins them. They are
 * derived data, not internals: everything here is a pure function of JOURNEY.
 */
export const markers = JOURNEY.map((stop) => ({
  id: stop.id,
  location: tuple(stop.location),
  size: 0.025,
}))

/**
 * Central angle between two [lat, lng] points, in degrees. Haversine rather
 * than the shorter acos form because the case that matters here is a *small*
 * angle, which is exactly where acos loses its precision.
 */
export function degreesApart(a: readonly [number, number], b: readonly [number, number]): number {
  const rad = (d: number) => (d * Math.PI) / 180
  const h =
    Math.sin(rad(b[0] - a[0]) / 2) ** 2 +
    Math.cos(rad(a[0])) * Math.cos(rad(b[0])) * Math.sin(rad(b[1] - a[1]) / 2) ** 2
  return (2 * Math.asin(Math.min(1, Math.sqrt(h))) * 180) / Math.PI
}

/**
 * Legs shorter than this are marked but not drawn.
 *
 * cobe's arcHeight is global: `updateArcs` writes the one value into every
 * instance, and `Arc` has no per-arc override. The shader then puts the Bezier
 * control point at a fixed radius, `GLOBE_R + arcHeight`, so the apex rises
 * about the same 26px above its endpoints whatever the span. Over the 78°
 * London leg that is spread across 210px and reads as a lazy arch. Over
 * Manchester to London — 2.4°, under 7px — the identical 26px becomes a spike
 * nearly four times taller than it is wide.
 *
 * Lowering the global height does not fix it, it just breaks the other end:
 * the smallest value that keeps the London leg from sinking into the globe is
 * 0.158, and the hop still spikes 17px over 7px there while the long leg lies
 * flat on the surface. Below roughly 9° an arc is taller than it is wide, so
 * that is the line. 10° keeps a little margin, and a spike between two dots
 * already touching carries nothing their labels don't.
 */
export const MIN_ARC_DEGREES = 10

/** One arc per leg of the route, so the pairs can't drift from the stops. */
export const arcs = JOURNEY.slice(1)
  .map((stop, i) => [JOURNEY[i]!, stop] as const)
  .filter(([from, to]) => degreesApart(from.location, to.location) >= MIN_ARC_DEGREES)
  .map(([from, to]) => ({ from: tuple(from.location), to: tuple(to.location) }))

/**
 * cobe defaults that are wrong at this size, and what this globe uses instead.
 * Grouped because they are one decision — how the route should sit on the
 * planet — and because leaving any of them off is what made the previous arcs
 * look clumsy. The values match the "COBE v2" slide on cobe.vercel.app, which
 * is where the look came from.
 */
export const GEOMETRY = {
  /** Default 1, which the shader widens to 0.005. At 420px that is a rope. */
  arcWidth: 0.5,
  /** Default 0.2, and the previous 0.18 was flatter still, so the legs
      crowded the surface instead of arching clear of it. */
  arcHeight: 0.25,
  /** Default 0.05, and it is not marker-only: arcs are built at
      `arcHeight + markerElevation` and terminate at `GLOBE_R +
      markerElevation`, so the default left both ends of every leg hovering
      above the planet with a visible gap under them. 0.01 sets them down
      without flattening the dots back into the map. */
  markerElevation: 0.01,
} as const

export default function Globe() {
  const wrapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const wrap = wrapRef.current
    if (!canvas || !wrap) return

    // Measured on the wrapper, not the canvas: cobe reparents the canvas under
    // its own container, so the canvas is not a reliable thing to measure.
    let size = wrap.clientWidth || 420
    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches

    const globe = createGlobe(canvas, {
      devicePixelRatio: Math.min(window.devicePixelRatio || 1, 2),
      width: size * 2,
      height: size * 2,
      phi: 0,
      theta: 0.25,
      dark: 1,
      diffuse: 1.2,
      mapSamples: 16000,
      mapBrightness: 6,
      baseColor: [...GLOBE_COLORS.base],
      markerColor: [...GLOBE_COLORS.marker],
      glowColor: [...GLOBE_COLORS.glow],
      markers,
      arcs,
      arcColor: [...GLOBE_COLORS.arc],
      ...GEOMETRY,
    })

    // cobe 2 dropped its `phenomenon` dependency, and that package was what
    // owned the render loop. v2 contains no requestAnimationFrame at all and
    // redraws only when update() is called, so nothing moves unless we drive it.
    //
    // Two things drive rotation, kept as separate accumulators so neither
    // clobbers the other: `spin` advances on its own, `drag` is whatever the
    // pointer has added. `rendered` eases toward their sum, which is what makes
    // releasing a drag hand back to the idle spin without a jolt.
    let spin = 0
    let drag = 0
    let rendered = 0
    let dragStartX: number | null = null
    let dragStartOffset = 0

    // Reduced-motion suppresses the *idle* spin only. Dragging is user-initiated,
    // so taking it away would remove function rather than remove motion.
    const settled = () => Math.abs(spin + drag - rendered) < 1e-4
    const idle = () => dragStartX === null && reduced && settled()

    let raf = 0
    const tick = () => {
      if (dragStartX === null && !reduced) spin += 0.004
      rendered += (spin + drag - rendered) * 0.12
      globe.update({ phi: rendered })
      raf = idle() ? 0 : requestAnimationFrame(tick)
    }
    const ensureRunning = () => {
      if (!raf) raf = requestAnimationFrame(tick)
    }

    // Bound to the wrapper rather than the canvas. The labels are interactive
    // now, so they sit above the canvas and would otherwise swallow the
    // pointerdown that starts a drag — grabbing the globe by a city name would
    // do nothing. Everything inside the wrapper bubbles here instead.
    const onPointerDown = (event: PointerEvent) => {
      dragStartX = event.clientX
      dragStartOffset = drag
      wrap.setPointerCapture(event.pointerId)
      wrap.style.cursor = 'grabbing'
      ensureRunning()
    }
    const onPointerMove = (event: PointerEvent) => {
      if (dragStartX === null) return
      // One canvas width of travel is half a turn, so the surface tracks the
      // pointer at roughly 1:1 rather than at some arbitrary sensitivity.
      drag = dragStartOffset + ((event.clientX - dragStartX) / size) * Math.PI
      ensureRunning()
    }
    const endDrag = (event: PointerEvent) => {
      if (dragStartX === null) return
      dragStartX = null
      if (wrap.hasPointerCapture(event.pointerId)) wrap.releasePointerCapture(event.pointerId)
      wrap.style.cursor = 'grab'
      ensureRunning()
    }

    wrap.addEventListener('pointerdown', onPointerDown)
    wrap.addEventListener('pointermove', onPointerMove)
    wrap.addEventListener('pointerup', endDrag)
    wrap.addEventListener('pointercancel', endDrag)

    raf = requestAnimationFrame(tick)

    // The Hugo implementation kept the backing resolution in sync with the
    // container. Without this the globe renders once at its mount-time size and
    // then goes blurry, because the canvas is CSS-stretched to fill its box.
    const observer = new ResizeObserver(() => {
      const next = wrap.clientWidth
      if (!next || next === size) return
      size = next
      globe.update({ width: size * 2, height: size * 2 })
    })
    observer.observe(wrap)

    canvas.style.opacity = '1'

    return () => {
      wrap.removeEventListener('pointerdown', onPointerDown)
      wrap.removeEventListener('pointermove', onPointerMove)
      wrap.removeEventListener('pointerup', endDrag)
      wrap.removeEventListener('pointercancel', endDrag)
      observer.disconnect()
      cancelAnimationFrame(raf)
      globe.destroy()
    }
  }, [])

  // The canvas is wrapped deliberately. createGlobe inserts a container div and
  // moves the canvas inside it, and destroy() never undoes that, so React's
  // recorded parent for the canvas goes stale. React must own a node cobe has
  // not reparented, or removeChild can throw NotFoundError on unmount.
  //
  // `relative` is for the labels, which are absolutely positioned: it makes the
  // globe's own box their containing block instead of letting them resolve
  // against the initial containing block several ancestors up.
  return (
    <div
      ref={wrapRef}
      className="relative h-full w-full cursor-grab"
      // `pan-y` keeps vertical page scrolling working on touch: only the
      // horizontal axis is claimed for rotation, which is the only axis
      // dragging actually affects. It lives on the wrapper rather than the
      // canvas because a pointer's allowed behaviours are the intersection of
      // touch-action down the ancestor chain, so one declaration here covers
      // the canvas and every label without repeating itself.
      style={{ touchAction: 'pan-y' }}
    >
      <canvas
        ref={canvasRef}
        className="h-full w-full opacity-0 transition-opacity duration-1000"
        style={{ contain: 'layout paint size' }}
      />
      {/* Labels MUST stay after the canvas here. cobe drops its own wrapper in
          where the canvas was, moves the canvas inside it, then appends the
          1px anchor divs alongside — so everything cobe owns ends up in a
          subtree that precedes these spans. CSS anchor positioning only accepts
          an anchor "laid out strictly before" the element querying it, so
          hoisting these above the canvas breaks every label at once, with no
          error: an unresolvable anchor is not a failure, it just positions
          nothing. tests/globe.test.ts pins the order.

          The var() fallbacks below look inverted and are not. cobe sets
          --cobe-visible-{id} to a deliberately invalid token while a marker
          faces the camera, and deletes it while the marker is round the back:

            facing us  -> `opacity: <invalid>` is invalid at computed-value
                          time, and opacity does not inherit, so it computes to
                          its initial value, 1. Likewise `filter: blur(...)`
                          collapses to the initial `none`.
            round back -> the property is missing, so the fallback applies: 0
                          and blur(6px).

          Hence "visible" is spelt as the fallback being *ignored*. Rewriting
          these into something that reads the variable as a number gets you a
          label that never fades. */}
      {JOURNEY.map((stop) => (
        <span
          key={stop.id}
          className={stop.labelBelow ? 'globe-label globe-label--below' : 'globe-label'}
          // Only a label with something to reveal is worth a tab stop.
          tabIndex={stop.photo ? 0 : undefined}
          style={
            {
              // positionAnchor is newer than the csstype build React types
              // against, so this object needs the cast to typecheck.
              positionAnchor: `--cobe-${stop.id}`,
              opacity: `var(--cobe-visible-${stop.id}, 0)`,
              filter: `blur(var(--cobe-visible-${stop.id}, 6px))`,
              // Whether this label can be hovered at all. Without it a label
              // round the back of the globe stays a live hit target: an
              // invisible chip you can hover over empty space, and one that
              // swallows a drag.
              //
              // It goes through a custom property rather than straight onto
              // `pointer-events` because csstype types that one as a keyword
              // union, so unlike positionAnchor (which it has never heard of)
              // the cast below cannot widen it. A custom property holds any
              // token sequence, so the invalid marker survives the hop intact
              // and global.css consumes it.
              //
              // Same trick as opacity and filter, different mechanism, and the
              // difference is worth knowing: those two do not inherit, so the
              // invalid token drops them to their INITIAL value.
              // pointer-events does inherit, so it drops to the INHERITED one
              // — `auto` here only because nothing above sets otherwise. Put
              // `pointer-events: none` on any ancestor and hovering dies.
              '--label-hit': `var(--cobe-visible-${stop.id}, none)`,
            } as CSSProperties
          }
        >
          {stop.label}
          {stop.photo && (
            <span className="globe-polaroid">
              {/* alt="" deliberately: the chip beside it already names the
                  place, and we cannot describe a photo we have not seen. */}
              {/* No width/height attributes on purpose: they are presentational
                  hints that would pin an aspect ratio, and these photos keep
                  whichever one they were shot in. The element is absolutely
                  positioned and hidden until hover, so there is no layout to
                  shift while the intrinsic size arrives. */}
              <img src={stop.photo} alt="" loading="lazy" />
            </span>
          )}
        </span>
      ))}
    </div>
  )
}
