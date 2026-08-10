import { SITE } from '@/consts'
import { getSortedPosts } from '@/lib/posts'
import type { APIContext } from 'astro'

export async function GET(context: APIContext) {
  const posts = await getSortedPosts()

  // Same site origin as index.xml.ts (context.site!), not SITE.url, so the
  // two endpoints can never disagree on the site's own base URL.
  const site = context.site!.origin

  const lines = [
    `# ${SITE.title}`,
    `- [Talks](${site}/talks/)`,
    '',
    '## Posts',
    ...posts.map((p) => `- [${p.data.title}](${site}/posts/${p.id}/)`),
    '',
  ]

  return new Response(lines.join('\n'), {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  })
}
