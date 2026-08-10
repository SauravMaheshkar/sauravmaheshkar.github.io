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

/** IANA zone the home page clock displays, to every visitor. Change on moving. */
const TIME_ZONE = 'Europe/London'

export const HERO = {
  timeZone: TIME_ZONE,
  /** Derived, not typed twice: the label can't drift from the zone it labels. */
  city: TIME_ZONE.split('/').pop()!.replace(/_/g, ' '),
} as const
