[![Deploy Astro site to Pages](https://github.com/SauravMaheshkar/sauravmaheshkar.github.io/actions/workflows/deploy.yml/badge.svg?branch=main)](https://github.com/SauravMaheshkar/sauravmaheshkar.github.io/actions/workflows/deploy.yml)

Source code for my personal website, built with [Astro](https://astro.build) and deployed to GitHub Pages.

* Most of my writing lives on external sites (Weights & Biases reports). `/posts/<slug>/` pages are frontmatter-only stubs that redirect there.
* React islands (globe, SF clock, flip clock, stretchy footer) hydrate via `@astrojs/react`; everything else ships as static HTML.
* Styling is Tailwind CSS, dark theme only.
