import { getCollection } from 'astro:content'

/**
 * The single source of ordering for every surface that lists posts:
 * index.astro, archives.astro, posts/index.astro, index.xml.ts and
 * llms.txt.ts all sort off this instead of each re-implementing
 * date-descending order. A regression here fails the llms.txt ordering
 * test, which transitively covers all five call sites.
 */
export async function getSortedPosts() {
  return (await getCollection('posts')).sort(
    (a, b) => b.data.date.getTime() - a.data.date.getTime(),
  )
}
