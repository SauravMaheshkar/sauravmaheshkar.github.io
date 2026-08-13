import { test, expect } from 'bun:test'
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { JOURNEY } from '../src/consts'
import { markers, arcs, GEOMETRY, degreesApart, MIN_ARC_DEGREES } from '../src/components/Globe'

/**
 * The globe's floating labels.
 *
 * Everything here guards the same class of failure: CSS anchor positioning has
 * no error state. An anchor that is never minted, or is minted under a name
 * nothing binds to, or is minted after the element querying it, all produce
 * exactly one symptom — `bottom: anchor(top)` resolves to nothing, the label
 * falls back to `auto` insets, and three city names quietly pile up over the
 * corner of the globe. No console warning, no unstyled markup, and the page
 * builds and hydrates perfectly either way.
 */

const DIST = path.resolve(import.meta.dir, '..', 'dist')

const home = () => readFileSync(path.join(DIST, 'index.html'), 'utf8')

/** Astro inlines small stylesheets and emits larger ones; look in both. */
function allCss(): string {
  const dir = path.join(DIST, '_astro')
  const sheets = existsSync(dir)
    ? readdirSync(dir)
        .filter((f) => f.endsWith('.css'))
        .map((f) => readFileSync(path.join(dir, f), 'utf8'))
    : []
  return [...sheets, home()].join('\n')
}

test('every marker carries the id its label anchors to', () => {
  // The load-bearing half, and the one that cannot be seen in the built HTML:
  // cobe mints `--cobe-{id}` only for markers that carry an id, and a marker
  // without one is drawn exactly the same. Drop the id here and the labels
  // keep declaring `position-anchor: --cobe-delhi` at an anchor that no longer
  // exists, so the markup still looks completely correct.
  expect(
    markers.map((m) => m.id),
    'the markers handed to cobe no longer match the stops',
  ).toEqual(JOURNEY.map((s) => s.id))
})

test('every arc joins two stops that are consecutive, in route order', () => {
  // A separately maintained pair list is how a fourth city gets added and
  // silently never joins up, or joins up to the wrong neighbour. Guards the
  // direction too: from/to reversed draws the identical curve, so nothing
  // about the globe would look wrong, but the arc anchors would be named for
  // the wrong leg.
  const legs = JOURNEY.slice(1).map((stop, i) => `${JOURNEY[i]!.id}->${stop.id}`)
  const idAt = (loc: readonly [number, number]) =>
    JOURNEY.find((s) => s.location[0] === loc[0] && s.location[1] === loc[1])?.id ?? '?'

  expect(arcs.length, 'no arcs are drawn at all').toBeGreaterThan(0)
  for (const arc of arcs) {
    expect(legs, 'an arc joins two stops that are not neighbours').toContain(
      `${idAt(arc.from)}->${idAt(arc.to)}`,
    )
  }
})

test('no arc is drawn across a span too short to look like an arc', () => {
  // cobe gives every arc the same apex height regardless of span, so a short
  // enough leg renders as a spike rather than an arch — see MIN_ARC_DEGREES.
  // Nothing errors when that happens; it just looks daft, and only on the one
  // pair of stops that happen to sit close together.
  for (const arc of arcs) {
    const span = degreesApart(arc.from, arc.to)
    expect(span, `an arc spans only ${span.toFixed(1)}°, which renders as a spike`).toBeGreaterThanOrEqual(
      MIN_ARC_DEGREES,
    )
  }
})

test('every stop renders a label bound to its own anchor', () => {
  const html = home()

  for (const stop of JOURNEY) {
    expect(html, `no label is anchored to --cobe-${stop.id}`).toContain(
      `position-anchor:--cobe-${stop.id}`,
    )
    expect(html, `${stop.id} has an anchor but renders no name`).toContain(stop.label)
  }

  // Fixes the count too: a stop added to JOURNEY without reaching the globe
  // would otherwise pass every per-stop assertion above.
  expect(html.match(/position-anchor:--cobe-/g) ?? []).toHaveLength(JOURNEY.length)
})

test('labels still fade out when their marker rotates behind the globe', () => {
  // THE subtle one. cobe signals "facing the camera" by setting
  // --cobe-visible-{id} to a deliberately invalid token, so the visible state
  // is spelled as `opacity` being invalid at computed-value time and falling
  // back to its initial 1. The hidden state is the var() *fallback*.
  //
  // That reads backwards, which is the problem: correcting the fallback to 1,
  // or removing it, yields labels that are pinned on and hang over the back of
  // the globe forever. Nothing about that looks like a bug in a screenshot.
  const html = home()

  for (const stop of JOURNEY) {
    expect(html, `${stop.id}'s label has lost its hidden state`).toContain(
      `opacity:var(--cobe-visible-${stop.id}, 0)`,
    )
  }
})

test('stops that ask to hang below their marker actually do', () => {
  // Two stops close enough together put their chips on top of each other, and
  // the result still renders two perfectly valid labels — just illegibly, in
  // the same place. `labelBelow` is the hand placement that separates them, so
  // it fails silently in both directions: unwired class, or a class with no
  // rules behind it.
  const html = home()
  const css = allCss()

  for (const stop of JOURNEY.filter((s) => s.labelBelow)) {
    const label = html.match(new RegExp(`<span[^>]*--cobe-${stop.id}[^>]*>`))?.[0] ?? ''
    expect(label, `${stop.id} asks to hang below and is not marked as doing so`).toContain(
      'globe-label--below',
    )
  }

  if (JOURNEY.some((s) => s.labelBelow)) {
    expect(css, 'the below-marker placement has no rules behind it').toContain(
      'top:anchor(bottom)',
    )
  }
})

test('a label is hoverable only while its marker faces the camera', () => {
  // `pointer-events` defaults to none in the stylesheet, so a missing
  // --label-hit is not a degraded hover, it is no hover at all: the chip goes
  // inert, nothing errors, and the polaroid simply never appears. The gating
  // matters in the other direction too — an ungated label round the back of
  // the globe stays a live hit target floating over empty space.
  const html = home()

  expect(allCss(), '.globe-label no longer reads the hit variable').toContain(
    'pointer-events:var(--label-hit,none)',
  )
  for (const stop of JOURNEY) {
    expect(html, `${stop.id}'s label never becomes hoverable`).toContain(
      `--label-hit:var(--cobe-visible-${stop.id}, none)`,
    )
  }
})

test('a polaroid takes no pointer events until it is actually shown', () => {
  // An opacity-0 polaroid that still accepts pointer events is an invisible
  // ~70px dead zone hanging over every label, eating drags aimed at the globe
  // behind it. Nothing looks broken; the globe just stops responding in
  // patches, in a place with no visible element to blame.
  const css = allCss()

  expect(css, 'the polaroid is a hit target before it is visible').toMatch(
    /\.globe-polaroid\{[^}]*pointer-events:none/,
  )
  expect(css, 'nothing reveals the polaroid on hover').toMatch(
    /\.globe-label:hover \.globe-polaroid[^{]*\{[^}]*opacity:1/,
  )
  // Hover-only would leave the photos unreachable without a mouse.
  expect(css, 'the polaroid is reachable by pointer but not by keyboard').toContain(
    ':focus-visible .globe-polaroid',
  )
})

test('the photo takes the chip\'s place rather than piling on top of it', () => {
  // The polaroid is 72px wide and sits flush over the chip. "SAN FRANCISCO" is
  // wider than that, so a chip that keeps painting pokes out of both sides of
  // its own photo — and the box has to stay put regardless, because it is what
  // holds the hover and the photo. Only the paint may go.
  const css = allCss()

  expect(css, 'the polaroid no longer sits flush with the chip').toMatch(
    /\.globe-polaroid\{[^}]*bottom:0[;}]/,
  )

  const swap = css.match(/\.globe-label:hover,[^{]*:focus-visible\{([^}]*)\}/)?.[1]
  expect(swap, 'nothing stops the chip painting while its photo is up').toBeTruthy()
  expect(swap!, 'the chip still paints its own background under the photo').not.toContain(
    '--color-fg',
  )
})

test('polaroids exist for exactly the stops that have a photo', () => {
  // Holds at zero photos as well as four: it catches a polaroid rendered
  // unconditionally, which would ship an <img> with an undefined src on every
  // label.
  const html = home()
  const withPhotos = JOURNEY.filter((s) => s.photo)

  expect(
    html.match(/class="globe-polaroid"/g) ?? [],
    'a polaroid rendered for a stop that has no photo',
  ).toHaveLength(withPhotos.length)

  for (const stop of withPhotos) {
    expect(html, `${stop.id}'s photo never reaches the page`).toContain(`src="${stop.photo}"`)
    expect(
      existsSync(path.join(DIST, stop.photo!)),
      `${stop.id} points at ${stop.photo}, which is not in the build`,
    ).toBe(true)
  }
})

test('the globe keeps enough headroom above it for a polaroid to pop into', () => {
  // The requirement is a function of how big the photo is, and the two live in
  // different files, so this derives it from the built CSS rather than
  // restating a number. Restating one is how the photo gets enlarged and the
  // margin does not, which lands a picture on top of the nav — and only while
  // somebody happens to be hovering a northern city, so it ships.

  /** Manchester's marker climbs to within this many px of the box's top edge,
   *  swept over a full rotation through cobe's own projection at theta 0.25. */
  const NORTHERNMOST_PEAK = 53
  /** The header's py-8 sits empty below the nav, so a photo may eat into it. */
  const HEADER_PADDING = 32

  const northernmost = JOURNEY.reduce((a, b) => (b.location[0] > a.location[0] ? b : a))
  expect(
    northernmost.id,
    'a stop further north than Manchester needs NORTHERNMOST_PEAK recomputed',
  ).toBe('manchester')

  const css = allCss()
  const px = (pattern: RegExp, what: string) => {
    const hit = css.match(pattern)
    expect(hit, `cannot read ${what} out of the built CSS`).not.toBeNull()
    return Number(hit![1])
  }
  const gap = px(/\.globe-label\{[^}]*margin-bottom:(\d+)px/, "the label's gap above its marker")
  const border = px(/\.globe-polaroid img\{[^}]*border:(\d+)px/, "the photo's border")
  const photo = px(/\.globe-polaroid img\{[^}]*max-height:(\d+)px/, "the photo's height cap")

  // The cap, not the rendered height: a landscape photo comes out shorter, but
  // the margin has to hold the tallest thing that can appear, and swapping in
  // a portrait later must not silently need more room than the page has.
  const reach = gap + border * 2 + photo
  const needed = reach - NORTHERNMOST_PEAK - HEADER_PADDING

  const section = home().match(/<section class="[^"]*\bmt-(\d+)\b[^"]*aspect-square/)
  expect(section, 'the globe is no longer in the section this test looks for').not.toBeNull()

  // Tailwind's spacing scale is 0.25rem a step. A floor: more is always fine.
  expect(
    Number(section![1]) * 4,
    `a polaroid reaches ${reach}px above its marker, so the globe needs ${needed}px of margin`,
  ).toBeGreaterThanOrEqual(needed)
})

test('the anchors are laid out before the labels that query them', () => {
  // cobe replaces the canvas with a wrapper of its own, moves the canvas
  // inside, and appends the anchor divs there — so the whole cobe subtree sits
  // wherever the canvas sits. Anchor positioning only accepts an anchor "laid
  // out strictly before" its querying element, so labels emitted above the
  // canvas would resolve against nothing.
  const html = home()
  const canvas = html.indexOf('<canvas')
  const firstLabel = html.indexOf('position-anchor:--cobe-')

  expect(canvas, 'the globe canvas is missing entirely').toBeGreaterThan(-1)
  expect(
    firstLabel,
    'a label is emitted before the canvas, so its anchor never resolves',
  ).toBeGreaterThan(canvas)
})

test('the positioning the labels depend on survives the build', () => {
  // anchor() is newer than most of the toolchain. If Lightning CSS ever drops
  // what it cannot parse, `.globe-label` stays `position: absolute` with no
  // insets and every label heaps up in the same corner.
  const css = allCss()
  expect(css, 'the labels have no anchored inset').toContain('bottom:anchor(top)')
  expect(css, 'the labels are not centred on their marker').toContain('left:anchor(center)')
})

test('browsers without anchor positioning get no labels rather than a heap', () => {
  // Chrome 125+, Safari 26+, Firefox 147+. Everywhere else the anchor() insets
  // are simply invalid declarations — but cobe keeps setting the visibility
  // custom properties regardless, so the labels would be fully opaque while
  // stacked on top of each other. Easy to miss when you only test in Chrome.
  expect(
    /@supports\s+not\s*\(anchor-name:\s*--\w+\)/.test(allCss()),
    'nothing hides the labels where anchor positioning is unsupported',
  ).toBe(true)
})

test('every cobe default this globe disagrees with is still overridden', () => {
  // markerElevation feeds the arcs as well as the dots, so at cobe's default
  // both ends of every leg hover clear of the surface. Dropping any of these
  // three in a refactor restores a thick, flat, visibly detached arc — the
  // exact look they were added to fix, and nothing errors on the way there.
  //
  // The defaults are restated rather than imported because cobe applies them
  // internally via `??` and exports nothing. If a cobe upgrade moves one, this
  // fails, which is the correct outcome: the override may have become moot.
  const COBE_DEFAULTS = { arcWidth: 1, arcHeight: 0.2, markerElevation: 0.05 } as const

  expect(Object.keys(GEOMETRY).sort(), 'an override was dropped or added').toEqual(
    Object.keys(COBE_DEFAULTS).sort(),
  )

  // Values stay free to move; restating a default is what is not allowed,
  // since that is an override in name only.
  for (const [option, fallback] of Object.entries(COBE_DEFAULTS)) {
    expect(
      GEOMETRY[option as keyof typeof GEOMETRY],
      `${option} now just restates cobe's default`,
    ).not.toBe(fallback)
  }
})
