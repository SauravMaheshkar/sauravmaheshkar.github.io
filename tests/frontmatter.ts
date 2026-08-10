import { readdir, readFile } from 'node:fs/promises'
import path from 'node:path'
import { POSTS_DIR } from '../src/consts'

export function parseFrontmatter(src: string): Record<string, string> {
  const block = src.match(/^---\r?\n([\s\S]*?)\r?\n---/)
  if (!block) return {}
  const out: Record<string, string> = {}
  for (const line of block[1].split(/\r?\n/)) {
    const m = line.match(/^([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$/)
    if (!m) continue
    out[m[1]] = m[2].trim().replace(/^["'](.*)["']$/, '$1')
  }
  return out
}

export async function readPostFiles() {
  const dir = path.resolve(import.meta.dir, '..', POSTS_DIR)
  const files = (await readdir(dir)).filter((f) => f.endsWith('.md'))
  return Promise.all(
    files.map(async (f) => ({
      // Lowercased: Hugo urlizes paths, so FNet.md lives at /posts/fnet/.
      // Astro's default glob() id does the same. GitHub Pages is
      // case-sensitive even though macOS is not.
      // .toLowerCase() is a stand-in for github-slugger (what Astro's
      // default glob() id generator actually uses), not a full
      // reimplementation. It matches all 32 current filenames because none
      // contain spaces, accents, or other non-ASCII characters that
      // github-slugger would additionally strip or transliterate. If a
      // future post filename does, this diverges from Astro's real id and
      // fails loud (a "missing" slug), not silent.
      slug: f.replace(/\.md$/, '').toLowerCase(),
      fm: parseFrontmatter(await readFile(path.join(dir, f), 'utf8')),
    })),
  )
}
