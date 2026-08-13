export const SITE = {
  title: 'Saurav :)',
  description: "Saurav Maheshkar's personal site",
  // Must stay in sync with `site` in astro.config.mjs. Endpoints that run
  // inside a request (e.g. src/pages/index.xml.ts, src/pages/llms.txt.ts)
  // should prefer `context.site` over this field, since that reads
  // astro.config.mjs directly and can't drift from it.
  url: 'https://sauravmaheshkar.github.io',
} as const

export const NAV = [
  // The label is free to change; the href is not. /archives/ has been live
  // since the Hugo site and is pinned by the URL parity suite.
  { name: 'Writing', href: '/archives/' },
  { name: 'Talks', href: '/talks/' },
] as const

export const SOCIALS = [
  { name: 'github', url: 'https://github.com/SauravMaheshkar' },
  { name: 'twitter', url: 'https://twitter.com/MaheshkarSaurav' },
  { name: 'instagram', url: 'https://instagram.com/sauravvmaheshkar' },
] as const

export const POSTS_DIR = './src/content/posts'

/**
 * The globe's palette, as the normalised 0..1 triples cobe expects.
 *
 * This is the one colour on the site no CSS token can reach: it goes to WebGL,
 * not to a stylesheet, so `--color-*` is invisible to it. Keeping it here means
 * every colour decision still lives in one file, and a palette repaint has an
 * obvious place to follow through to instead of silently leaving the globe
 * behind. Approximate CSS equivalents are noted for orientation.
 */
export const GLOBE_COLORS = {
  base: [0.28, 0.28, 0.32], // ≈ rgb 71 71 82
  marker: [1, 1, 1], // white
  glow: [0.12, 0.14, 0.18], // ≈ rgb 31 36 46
  arc: [1, 1, 1], // white
} as const satisfies Record<string, readonly [number, number, number]>

/**
 * The places the globe marks, in the order they were lived in. Arcs are drawn
 * between consecutive stops rather than listed as their own pairs, so adding a
 * city extends the route instead of quietly relying on whoever adds it to
 * remember a second edit.
 *
 * `id` is not decoration and is not free to churn. cobe mints a CSS anchor
 * named `--cobe-{id}` and a visibility custom property `--cobe-visible-{id}`
 * for every marker carrying one, and `.globe-label` in global.css binds to
 * those names. Renaming an id renames a live CSS custom property; the label
 * then anchors to nothing and, since an unresolved anchor is not an error,
 * fails silently. tests/globe.test.ts pins the pairing.
 */
export type Stop = {
  readonly id: string
  readonly label: string
  readonly location: readonly [number, number]
  /**
   * Hangs this stop's label under its marker instead of over it. A hand
   * placement, and it exists for one reason: two stops close enough together
   * anchor two ~80px chips within a few pixels of each other, and neither is
   * then readable. Nothing detects that automatically, so it is declared.
   */
  readonly labelBelow?: boolean
  /**
   * A photo of this stop, popped as a polaroid while its label is hovered or
   * focused. Optional, and a stop without one renders no polaroid and takes no
   * tab stop, so the feature simply does not exist for it.
   *
   * Square crop, 120px minimum: they display at 60px and would otherwise be
   * soft on any retina screen. Anything here ships in the build, so it wants
   * to be owned or clearly licensed, same bar as the record cover.
   */
  readonly photo?: string
}

/*
 * The photos below started as Saurav's phone shots (DEL/MCR/LDN/SFO, 1.5k-4k
 * on the long edge, up to 4MB each). They are resampled so the LONG edge is
 * 260px, at JPEG quality 80 — 19-25KB apiece:
 *
 *     sips -Z 260 -s format jpeg -s formatOptions 80 IN.jpg --out OUT.jpg
 *
 * Not cropped. Three are portrait and one is landscape, and they keep it;
 * global.css caps both axes at 130px and lets each photo pick its own shape,
 * so a 2x screen still gets them pixel for pixel. There is no hover on touch,
 * so 3x phone screens never see these at all.
 *
 * Resize the same way rather than dropping a full-size photo in: at 4MB each
 * these would outweigh every other asset on the site put together, eighty
 * times over, to be shown two inches across.
 */
export const JOURNEY: readonly Stop[] = [
  { id: 'delhi', label: 'Delhi', location: [28.6139, 77.209], photo: '/images/delhi.jpg' },
  {
    id: 'manchester',
    label: 'Manchester',
    location: [53.4808, -2.2426],
    photo: '/images/manchester.jpg',
  },
  // 262km from Manchester, which is 2.36° of arc — under 7px on a 420px globe,
  // against chips an order of magnitude wider than that. London goes below and
  // Manchester stays above, in that order rather than the reverse: Manchester
  // is the northern of the two and so sits higher on screen at every rotation,
  // so this is the arrangement where neither chip lands on the other's dot.
  {
    id: 'london',
    label: 'London',
    location: [51.5074, -0.1278],
    labelBelow: true,
    photo: '/images/london.jpg',
  },
  {
    id: 'sf',
    label: 'San Francisco',
    location: [37.7749, -122.4194],
    photo: '/images/sf.jpg',
  },
]

/**
 * The record on the home page footer.
 *
 * The track is Saurav's own, written and exported from GarageBand, so the site
 * owns it outright with no licence and no attribution owed to anyone. That is
 * worth more than it sounds: the obvious ways to fill this slot are both
 * traps. A film's promo site is studio-copyrighted and gets torn down within a
 * year of release, and the upstream componentry.dev component's other branch
 * plays a YouTube URL through a `display: none` iframe, which can serve an
 * unskippable pre-roll the visitor can neither see nor skip and is squarely
 * against YouTube's developer policies (no background players, no separating
 * audio from video, 200x200px minimum).
 *
 * The .m4a is byte-for-byte the GarageBand export, and should stay that way.
 * Measured at 147.66bpm it is exactly 43.00 beats long, so it already ends on
 * the grid and `<audio loop>` restarts it in time. Two edits that look like
 * housekeeping would break that:
 *
 *   - Trimming the ~66ms of quiet at the head. That quiet is inside the 43
 *     beats; removing it shifts every subsequent loop off the beat.
 *   - Re-encoding to save bandwidth. AAC-to-AAC rewrites the encoder priming
 *     and padding, which can introduce the very gap the beat alignment exists
 *     to avoid, and costs a generation of quality to do it.
 *
 * Caveat for a future re-export: 43 beats is 10.75 bars in 4/4, so the
 * downbeat walks a beat every time the loop wraps. Fixing that needs a whole
 * bar count out of GarageBand, not a cut here — cutting to 40 beats would
 * loop correctly but would throw away 1.2s of the ending.
 */
export const NOW_PLAYING = {
  src: '/audio/spiderman.m4a',
  /**
   * Square-cropped from Miles_Silhouette.png, which is 5500x4000 with the
   * figure filling 12.7% of it. Dropped in whole it renders about 15px tall on
   * a 112px record, so the crop is not optional. Recrop with
   * scratchpad/crop-cover.ts if the source ever changes; naive min/max bounds
   * do not work on it (stray near-black pixels near two corners drag the box
   * out to half the canvas), which is why that script votes by ink density.
   */
  coverArt: '/images/record-cover.png',
  /** Provisional — GarageBand called it "My Song 2". Rename freely. */
  title: 'Theme',
  artist: SITE.title,
} as const

/** IANA zone the home page clock displays, to every visitor. Change on moving. */
const TIME_ZONE = 'Europe/London'

export const HERO = {
  timeZone: TIME_ZONE,
  /** Derived, not typed twice: the label can't drift from the zone it labels. */
  city: TIME_ZONE.split('/').pop()!.replace(/_/g, ' '),
} as const
