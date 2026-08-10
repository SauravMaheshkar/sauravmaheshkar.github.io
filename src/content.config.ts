import { defineCollection } from 'astro:content'
import { glob } from 'astro/loaders'
import { z } from 'astro/zod'
import { POSTS_DIR } from '@/consts'

const posts = defineCollection({
  loader: glob({ pattern: '*.md', base: POSTS_DIR }),
  schema: z.object({
    title: z.string(),
    date: z.coerce.date(),
    externalURL: z.string().url(),
    tags: z.array(z.string()).optional(),
    highlight: z.boolean().optional(),
  }),
})

export const collections = { posts }
