import { useEffect, useRef } from 'react'
import createGlobe from 'cobe'
import { GLOBE_COLORS } from '@/consts'

/** [lat, lng] of the consecutive points the journey arcs between. */
const DELHI: [number, number] = [28.6139, 77.209]
const LONDON: [number, number] = [51.5074, -0.1278]
const SF: [number, number] = [37.7749, -122.4194]

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
      markers: [
        { location: DELHI, size: 0.05 },
        { location: LONDON, size: 0.05 },
        { location: SF, size: 0.05 },
      ],
      arcs: [
        { from: DELHI, to: LONDON },
        { from: LONDON, to: SF },
      ],
      arcColor: [...GLOBE_COLORS.arc],
      arcHeight: 0.18,
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

    const onPointerDown = (event: PointerEvent) => {
      dragStartX = event.clientX
      dragStartOffset = drag
      canvas.setPointerCapture(event.pointerId)
      canvas.style.cursor = 'grabbing'
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
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId)
      canvas.style.cursor = 'grab'
      ensureRunning()
    }

    canvas.addEventListener('pointerdown', onPointerDown)
    canvas.addEventListener('pointermove', onPointerMove)
    canvas.addEventListener('pointerup', endDrag)
    canvas.addEventListener('pointercancel', endDrag)

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
      canvas.removeEventListener('pointerdown', onPointerDown)
      canvas.removeEventListener('pointermove', onPointerMove)
      canvas.removeEventListener('pointerup', endDrag)
      canvas.removeEventListener('pointercancel', endDrag)
      observer.disconnect()
      cancelAnimationFrame(raf)
      globe.destroy()
    }
  }, [])

  // The canvas is wrapped deliberately. createGlobe inserts a container div and
  // moves the canvas inside it, and destroy() never undoes that, so React's
  // recorded parent for the canvas goes stale. React must own a node cobe has
  // not reparented, or removeChild can throw NotFoundError on unmount.
  return (
    <div ref={wrapRef} className="h-full w-full">
      <canvas
        ref={canvasRef}
        className="h-full w-full cursor-grab opacity-0 transition-opacity duration-1000"
        // `pan-y` keeps vertical page scrolling working on touch: only the
        // horizontal axis is claimed for rotation, which is the only axis
        // dragging actually affects.
        style={{ contain: 'layout paint size', touchAction: 'pan-y' }}
      />
    </div>
  )
}
