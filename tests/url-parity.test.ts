import { test, expect } from 'bun:test'
import { existsSync, readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { readPostFiles } from './frontmatter'
import { SITE } from '../src/consts'

const DIST = path.resolve(import.meta.dir, '..', 'dist')

test('every source post emits its exact URL', async () => {
  const posts = await readPostFiles()
  const missing = posts
    .map((p) => p.slug)
    .filter((slug) => !existsSync(path.join(DIST, 'posts', slug, 'index.html')))
  expect(missing).toEqual([])
})

test('every emitted redirect points at its source externalURL', async () => {
  const posts = await readPostFiles()
  for (const { slug, fm } of posts) {
    const html = readFileSync(path.join(DIST, 'posts', slug, 'index.html'), 'utf8')
    const url = fm.externalURL
    // Asserted independently, not just "the URL appears somewhere in the
    // page" — dropping any one of these three still leaves the URL string
    // present elsewhere in the document, which would let a weaker check
    // pass on a broken redirect.
    expect(html, `${slug} is missing the meta refresh redirect`).toContain(
      `<meta http-equiv="refresh" content="0; url=${url}">`,
    )
    expect(html, `${slug} is missing the canonical link`).toContain(
      `<link rel="canonical" href="${url}">`,
    )
    expect(html, `${slug} is missing the visible fallback anchor`).toContain(
      `<a href="${url}">${url}</a>`,
    )
  }
})

test('no unexpected extra post URLs are emitted', async () => {
  const expected = new Set((await readPostFiles()).map((p) => p.slug))
  const emitted = readdirSync(path.join(DIST, 'posts'), { withFileTypes: true })
    .filter((d) => d.isDirectory())
    .map((d) => d.name)
  expect(emitted.filter((s) => !expected.has(s))).toEqual([])
})

test('every URL live in Hugo production still resolves (tests/legacy-urls.txt)', () => {
  // Unlike every other test in this file, the expected set here is NOT
  // derived from src/content/posts/ — it is a fixture snapshotted from
  // production history. Every other assertion proves "dist matches the
  // filenames currently on disk"; this one proves "dist matches what is
  // actually live", so a renamed or deleted post file fails this test
  // instead of silently 404ing a real inbound link.
  const raw = readFileSync(path.join(import.meta.dir, 'legacy-urls.txt'), 'utf8')
  const slugs = raw
    .split('\n')
    .map((l) => l.trim())
    .filter((l) => l && !l.startsWith('#'))
  expect(slugs).toHaveLength(32)
  const missing = slugs.filter((slug) => !existsSync(path.join(DIST, 'posts', slug, 'index.html')))
  expect(missing, `production URLs 404 in this build: ${missing.join(', ')}`).toEqual([])
})

test('FNet.md is served lowercase, matching its live Hugo URL', () => {
  // GitHub Pages is case-sensitive; macOS is not, so this cannot be caught by eye.
  expect(existsSync(path.join(DIST, 'posts', 'fnet', 'index.html'))).toBe(true)
  expect(readdirSync(path.join(DIST, 'posts')).includes('FNet')).toBe(false)
})

test('non-post routes and favicons all exist', () => {
  const required = [
    'index.html',
    'index.xml',
    'llms.txt',
    'archives/index.html',
    'talks/index.html',
    'posts/index.html',
    'sitemap-index.xml',
    'robots.txt',
    '404.html',
    'favicon.ico',
    'favicon-16x16.png',
    'favicon-32x32.png',
    'apple-touch-icon.png',
  ]
  const missing = required.filter((f) => !existsSync(path.join(DIST, f)))
  expect(missing).toEqual([])
})

test('index.json is not emitted', () => {
  expect(existsSync(path.join(DIST, 'index.json'))).toBe(false)
})

test('RSS items link to the local redirect stub, not the external article', async () => {
  const posts = await readPostFiles()
  const xml = readFileSync(path.join(DIST, 'index.xml'), 'utf8')
  // Owner decision, reversing an earlier choice: @astrojs/rss hardcodes
  // item.guid = link, and Hugo's GUIDs were the local /posts/<slug>/
  // permalinks. Linking to externalURL would make every existing subscriber
  // see all 32 posts as unread on the next migration, and two posts
  // currently share an externalURL, which would give them identical GUIDs
  // and make conforming readers silently drop one. Local permalinks restore
  // Hugo's exact GUID values, so subscriptions carry over cleanly.
  for (const { slug } of posts) {
    expect(xml, `${slug} is missing its local permalink as an RSS item link`).toContain(
      `<link>${SITE.url}/posts/${slug}/</link>`,
    )
  }
  // Inverse of the above: a feed can contain both the local link and an
  // external link at once, so "the local permalink is present somewhere"
  // alone would not catch a regression back to linking externalURL.
  for (const { slug, fm } of posts) {
    expect(
      xml,
      `${slug} links to its externalURL instead of (or as well as) the local redirect stub`,
    ).not.toContain(`<link>${fm.externalURL}</link>`)
  }
})

test('llms.txt lists every post once, correctly formatted, in date-descending order', async () => {
  const posts = await readPostFiles()
  const txt = readFileSync(path.join(DIST, 'llms.txt'), 'utf8')
  const lines = txt.split('\n').filter((l) => l.startsWith('- ['))

  // One Talks line plus one line per post. This is the Step 7 completeness
  // check from the task brief turned into a committed assertion instead of
  // a one-off shell command: a post silently dropped by getCollection()
  // shrinks this count without any assertion below noticing on its own.
  expect(lines).toHaveLength(posts.length + 1)

  const fnet = posts.find((p) => p.slug === 'fnet')
  expect(fnet, 'fnet.md fixture is missing from content/posts').toBeDefined()
  expect(lines, 'known post is missing its exact expected llms.txt line').toContain(
    `- [${fnet!.fm.title}](${SITE.url}/posts/fnet/)`,
  )

  const talksLine = `- [Talks](${SITE.url}/talks/)`
  const postLines = lines.filter((l) => l !== talksLine)
  expect(postLines, 'Talks line is missing or duplicated').toHaveLength(posts.length)

  const dateBySlug = new Map(posts.map((p) => [p.slug, new Date(p.fm.date).getTime()]))
  const dates = postLines.map((line) => {
    const slug = line.match(/\/posts\/([a-z0-9-]+)\/\)$/)?.[1]
    const date = slug ? dateBySlug.get(slug) : undefined
    if (date === undefined) throw new Error(`could not resolve a post date for line: ${line}`)
    return date
  })
  for (let i = 1; i < dates.length; i++) {
    expect(dates[i], `llms.txt order breaks at post index ${i} (not date-descending)`).toBeLessThanOrEqual(
      dates[i - 1],
    )
  }
})

test('the home page links every highlighted post straight to its externalURL', async () => {
  const posts = await readPostFiles()
  const highlighted = posts.filter((p) => p.fm.highlight === 'true')
  // Without this, every assertion below passes vacuously the day `highlight`
  // is renamed or dropped from the frontmatter schema.
  expect(highlighted.length, 'no posts are marked `highlight: true`').toBeGreaterThan(0)

  const html = readFileSync(path.join(DIST, 'index.html'), 'utf8')

  for (const { slug, fm } of highlighted) {
    expect(html, `${slug} is missing from the home page highlights`).toContain(
      `href="${fm.externalURL}"`,
    )
    // src/pages/index.astro documents this choice in a comment; a comment is
    // not a test. Someone "fixing" the card to use the local permalink would
    // break nothing else in this suite.
    //
    // Both forms, and the relative one is the load-bearing half: Astro emits
    // site-root hrefs as-authored, so the absolute form is what a regression
    // would *not* look like. Asserting only against SITE.url gives an assertion
    // that cannot fail.
    for (const stub of [`href="/posts/${slug}/"`, `href="${SITE.url}/posts/${slug}/"`]) {
      expect(html, `${slug} bounces through the local redirect stub`).not.toContain(stub)
    }
  }

  // Two posts share an externalURL (see the RSS GUID test above), so a
  // non-highlighted post can legitimately carry a URL a highlighted one also
  // uses. Those have to be dropped before asserting absence, or this false-fails
  // on a page that is perfectly correct.
  const shown = new Set(highlighted.map((p) => p.fm.externalURL))
  for (const { slug, fm } of posts) {
    if (fm.highlight === 'true' || shown.has(fm.externalURL)) continue
    expect(html, `${slug} is not highlighted but appears on the home page`).not.toContain(
      `href="${fm.externalURL}"`,
    )
  }
})

test('the highlight card hatch references a colour token that actually exists', () => {
  // @cardcn/card-10 upstream reads var(--border). This project declares its
  // tokens inside @theme, so Tailwind emits --color-border and there is no
  // --border alias. An unresolvable colour stop invalidates the entire
  // repeating-linear-gradient, and the card silently degrades to a plain
  // bordered box — which is exactly what it looked like before the port, so no
  // other assertion here would notice.
  const html = readFileSync(path.join(DIST, 'index.html'), 'utf8')
  expect(html, 'the highlight card hatch is missing from the home page').toContain(
    'repeating-linear-gradient(45deg',
  )
  expect(html, 'the hatch names a token this project does not declare').toContain(
    'var(--color-border)',
  )

  // Asserting the reference alone would still pass if the token were renamed in
  // global.css, so check that something actually declares it. Astro emits the
  // stylesheet to dist/_astro/, but inlines small ones into the document, so
  // both are candidates.
  const sheets = readdirSync(path.join(DIST, '_astro'))
    .filter((f) => f.endsWith('.css'))
    .map((f) => readFileSync(path.join(DIST, '_astro', f), 'utf8'))
  expect(
    [...sheets, html].some((css) => css.includes('--color-border:')),
    'nothing declares --color-border, so the hatch gradient resolves to nothing',
  ).toBe(true)
})
