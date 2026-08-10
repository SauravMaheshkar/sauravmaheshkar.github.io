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
