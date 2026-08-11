import { test, expect } from 'bun:test'
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { NOW_PLAYING } from '../src/consts'

/**
 * The footer record player. Every assertion here exists because the failure it
 * catches is *silent* — the page still builds, still renders, still looks
 * roughly right, and the feature is simply dead. None of it is catchable by
 * glancing at the home page, which is exactly how a static record survived
 * review once already.
 */

const DIST = path.resolve(import.meta.dir, '..', 'dist')

const html = (...segments: string[]) =>
  readFileSync(path.join(DIST, ...segments, 'index.html'), 'utf8')

/** Astro inlines small stylesheets into the document and emits larger ones to
 *  dist/_astro/, so a CSS assertion has to look in both places. */
function allCss(): string[] {
  const dir = path.join(DIST, '_astro')
  const sheets = existsSync(dir)
    ? readdirSync(dir)
        .filter((f) => f.endsWith('.css'))
        .map((f) => readFileSync(path.join(dir, f), 'utf8'))
    : []
  return [...sheets, html()]
}

/** The <footer> element of a built page, tag and contents. */
function footerOf(page: string): string {
  const match = page.match(/<footer[\s\S]*?<\/footer>/)
  if (!match) throw new Error('no <footer> in this page at all')
  return match[0]
}

test('the record actually spins: the keyframes it names are emitted', () => {
  // THE bug this suite exists for. Tailwind v4 emits `@keyframes spin` only as
  // a side effect of the `animate-spin` utility appearing in scanned source.
  // Nothing in this project uses it, so an inline `animation: spin 4s ...`
  // names a keyframe that does not exist: no error, no warning, no unstyled
  // markup, just a record that never turns. global.css declares
  // `record-spin` explicitly to avoid depending on that side effect — this
  // asserts the declaration survives.
  const css = allCss()
  expect(
    css.some((sheet) => /@keyframes\s+record-spin/.test(sheet)),
    'nothing declares @keyframes record-spin, so the record is frozen',
  ).toBe(true)

  // The reference half. A keyframe nothing uses is as dead as a use with no
  // keyframe, and each alone would pass the other's assertion.
  expect(html(), 'the record is not asking for the spin animation').toContain(
    'animate-record-spin',
  )
  expect(
    css.some((sheet) => sheet.includes('animation:var(--animate-record-spin)')),
    'animate-record-spin is used but Tailwind emitted no such utility',
  ).toBe(true)
})

test('the player names media that exists in the build', () => {
  // A typo'd path yields a player that renders perfectly and plays nothing:
  // audio.play() rejects, the catch resets state, and the record just refuses
  // to start with no indication why.
  for (const asset of [NOW_PLAYING.src, NOW_PLAYING.coverArt]) {
    expect(
      existsSync(path.join(DIST, asset)),
      `NOW_PLAYING points at ${asset}, which is not in the build`,
    ).toBe(true)
  }
  expect(html(), 'the audio element is missing its source').toContain(NOW_PLAYING.src)
  expect(html(), 'the record is missing its cover art').toContain(NOW_PLAYING.coverArt)
})

test('the separator rule appears on the home page and nowhere else', () => {
  // Owner decision: the line separates the copyright from the player, so a page
  // without a player has nothing to separate. Base.astro ties it to the footer
  // slot being filled rather than to a prop, and this pins that wiring — a
  // refactor to an always-on border would otherwise pass every other test.
  expect(footerOf(html()), 'the home page footer lost its separator').toContain('border-t')

  // The rule needs air on BOTH sides. Its py-6 sits inside the border, so it
  // only separates the rule from the copyright below; the gap above has to
  // come from a margin outside the border or the rule butts into the last
  // card. Asserted as "some top margin" rather than a pinned value — the exact
  // step is taste, its presence is not.
  expect(
    /\bmt-\d+\b/.test(footerOf(html())),
    'the separator has no margin above it, so it collides with the last card',
  ).toBe(true)

  for (const route of ['archives', 'talks', 'posts']) {
    expect(
      footerOf(html(route)),
      `/${route}/ has a separator but no player to separate`,
    ).not.toContain('border-t')
  }
})

test('the player is on the home page only', () => {
  expect(html(), 'the home page is missing the player').toContain('animate-record-spin')

  for (const route of ['archives', 'talks', 'posts']) {
    expect(html(route), `/${route}/ is carrying the player`).not.toContain('animate-record-spin')
    // The audio file is the heavier half of the mistake: shipping the element
    // sitewide costs every page a media request it has no use for.
    expect(html(route), `/${route}/ is carrying the audio element`).not.toContain(NOW_PLAYING.src)
  }
})

test('every page still carries the copyright, player or not', () => {
  // The separator work moved the copyright out of a bare <footer> and next to a
  // slot. Getting that wrong drops it from every page but the home page, which
  // is the sort of thing nobody notices for a year.
  for (const page of [html(), html('archives'), html('talks'), html('posts')]) {
    expect(footerOf(page)).toContain(`${new Date().getFullYear()}`)
    expect(footerOf(page)).toContain('Saurav')
  }
})
