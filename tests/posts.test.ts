import { test, expect } from 'bun:test'
import { readPostFiles } from './frontmatter'

test('every post has a title and a valid externalURL', async () => {
  const posts = await readPostFiles()
  expect(posts.length).toBe(32)
  for (const { slug, fm } of posts) {
    expect(fm.title, `${slug} is missing a title`).toBeTruthy()
    expect(() => new URL(fm.externalURL), `${slug} has a bad externalURL`).not.toThrow()
  }
})

test('no two posts share a title', async () => {
  const titles = (await readPostFiles()).map((p) => p.fm.title)
  expect(new Set(titles).size).toBe(titles.length)
})

// Two posts pointing at the same article is the failure this catches, and it has
// happened twice in this repo: sam2.md carried chatbot-docs.md's title and date, and
// vicreg.md redirected to the Multi-Task GRL report for years. Both were copy-paste
// slips invisible to every other check, since each file is individually well-formed.
test('no two posts share an externalURL', async () => {
  const posts = await readPostFiles()
  const seen = new Map<string, string>()
  for (const { slug, fm } of posts) {
    const first = seen.get(fm.externalURL)
    expect(first, `${slug} redirects to the same article as ${first}`).toBeUndefined()
    seen.set(fm.externalURL, slug)
  }
})
