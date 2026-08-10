import { test, expect } from 'bun:test'
import { readdirSync, readFileSync, statSync } from 'node:fs'
import path from 'node:path'

/**
 * The palette is locked in `src/styles/global.css` with `--color-*: initial`,
 * which deletes Tailwind's defaults so `bg-zinc-900` stops existing. That stops
 * off-palette colour *working* — but in Tailwind v4 an unknown utility silently
 * generates nothing rather than erroring, so on its own it fails quietly and
 * ships unstyled markup.
 *
 * This is the loud half. It scans authored source for the three ways colour and
 * spacing escape the token system, so drift fails in CI rather than in a review
 * somebody might not run.
 */

const SRC = path.resolve(import.meta.dir, '..', 'src')

/** Vendored third-party code plays by its own rules; content is prose, not UI. */
const EXCLUDED_DIRS = ['8starlabs-ui', 'content', 'data']

/**
 * Escapes that are deliberate. Each needs a reason, and adding one should be a
 * conscious act rather than something a regex quietly stopped noticing.
 */
const ALLOWED = {
  // <meta> cannot read a CSS custom property, so theme-color has to be literal.
  // Must stay in step with --color-bg.
  'layouts/Base.astro': ['#000000'],
} as const satisfies Record<string, readonly string[]>

/** Arbitrary Tailwind values we have accepted, with the same bar as above. */
const ALLOWED_ARBITRARY = new Set([
  'max-w-[420px]', // the globe's ceiling; matches the old Hugo .globe-wrap
  'text-[10px]', // below Tailwind's text-xs, for the small caps labels
])

const STOCK_PALETTE =
  'slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose'

function sourceFiles(dir: string, opts: { skipVendored: boolean }, acc: string[] = []): string[] {
  for (const entry of readdirSync(dir)) {
    const full = path.join(dir, entry)
    if (statSync(full).isDirectory()) {
      if (!opts.skipVendored || !EXCLUDED_DIRS.includes(entry)) sourceFiles(full, opts, acc)
    } else if (/\.(astro|tsx|ts|css)$/.test(entry)) {
      acc.push(full)
    }
  }
  return acc
}

/** Comments explain colours as often as they declare them; don't scan them. */
function stripComments(code: string): string {
  return code.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/(^|[^:])\/\/[^\n]*/g, '$1 ')
}

const read = (full: string) => ({
  rel: path.relative(SRC, full),
  code: stripComments(readFileSync(full, 'utf8')),
})

/** Files we hold to the rules: ours. */
const files = sourceFiles(SRC, { skipVendored: true }).map(read)

/**
 * Files that count as *usage*: everything, vendored included. The shadcn alias
 * tokens (background/foreground/input) and `inherit` exist solely because the
 * vendored flip clock asks for them, so judging them unused by looking only at
 * our own code would be wrong.
 */
const allFiles = sourceFiles(SRC, { skipVendored: false }).map(read)

test('the token scan actually covers the source', () => {
  expect(files.length).toBeGreaterThan(8)
  expect(files.some((f) => f.rel === 'styles/global.css')).toBe(true)
  expect(files.some((f) => f.rel.includes('8starlabs-ui'))).toBe(false)
})

test('no raw colour literals outside the palette definition', () => {
  const offenders: string[] = []
  for (const { rel, code } of files) {
    // global.css is where the palette is declared; literals are the point there.
    if (rel === 'styles/global.css') continue
    const allowed: readonly string[] = ALLOWED[rel as keyof typeof ALLOWED] ?? []
    for (const match of code.match(/#[0-9a-fA-F]{3,8}\b|rgba?\([^)]*\)/g) ?? []) {
      if (!allowed.includes(match)) offenders.push(`${rel}: ${match}`)
    }
  }
  expect(offenders).toEqual([])
})

test('no stock Tailwind palette utilities', () => {
  const pattern = new RegExp(
    `\\b(bg|text|border|fill|stroke|ring|divide|placeholder|from|via|to)-(${STOCK_PALETTE})(-\\d{2,3})?\\b`,
    'g',
  )
  const offenders: string[] = []
  for (const { rel, code } of files) {
    for (const match of code.match(pattern) ?? []) offenders.push(`${rel}: ${match}`)
  }
  expect(offenders).toEqual([])
})

test('no unapproved arbitrary values', () => {
  const offenders: string[] = []
  for (const { rel, code } of files) {
    if (rel.endsWith('.css')) continue // raw CSS is not Tailwind's arbitrary syntax
    for (const match of code.match(/\b[a-z][a-z-]*-\[[^\]\s]+\]/g) ?? []) {
      if (!ALLOWED_ARBITRARY.has(match)) offenders.push(`${rel}: ${match}`)
    }
  }
  expect(offenders).toEqual([])
})

test('every colour token is actually referenced somewhere', () => {
  const css = files.find((f) => f.rel === 'styles/global.css')!.code
  const declared = [...css.matchAll(/--color-([a-z-]+):/g)].map((m) => m[1])
  const everything = allFiles.map((f) => f.code).join('\n')

  const unused = declared.filter((name) => {
    if (name === '*') return false
    // Either as a Tailwind utility (bg-fg, text-muted) or a var() reference.
    return !new RegExp(`(-${name}\\b|--color-${name}\\b)`).test(
      everything.replace(new RegExp(`--color-${name}:`, 'g'), ''),
    )
  })
  expect(unused).toEqual([])
})
