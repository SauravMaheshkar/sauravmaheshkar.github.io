import rss from '@astrojs/rss'
import { SITE } from '@/consts'
import { getSortedPosts } from '@/lib/posts'
import type { APIContext } from 'astro'

export async function GET(context: APIContext) {
  const posts = await getSortedPosts()

  return rss({
    title: SITE.title,
    description: SITE.description,
    site: context.site!,
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.date,
      // Local permalink, not post.data.externalURL: @astrojs/rss hardcodes
      // item.guid = link, and Hugo's GUIDs were these local permalinks. RSS
      // readers key "read" state off the GUID, so linking to externalURL
      // would make every existing subscriber see all 32 posts as unread —
      // and worse, two posts currently share an externalURL, which would
      // give them identical GUIDs and make conforming readers silently drop
      // one. Resolved against `site` (passed above) by @astrojs/rss itself.
      link: `/posts/${post.id}/`,
    })),
  })
}
