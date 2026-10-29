# Data licences

| What | Licence |
|---|---|
| Scripts and code | MIT ([LICENSE](LICENSE)) |
| Data made by us: `decision/data/` (synthetic and model-written cases), all manifests (`*/manifest.json`, `holdouts/sets/`, `holdouts/upstream.json`), `decision/writer-prompts/` | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) — attribution: "llm-eval-corpora, hyukudan" |
| Project Gutenberg book texts (rebuilt by `prose/`) | Public domain in the USA. The Project Gutenberg licence and trademark text is removed; check the copyright law of your country. |
| Wikipedia text (rebuilt by `teacher/`, `holdouts/`) | CC BY-SA 4.0 (older revisions also CC BY-SA 3.0 / GFDL). Attribute "Wikipedia contributors" with the page URL; anything you publish from it must be CC BY-SA. |
| Aya dataset (rebuilt by `teacher/`) | Apache-2.0 |
| oasst2 (rebuilt by `holdouts/`) | Apache-2.0 |
| databricks-dolly-15k, curated multilingual (rebuilt by `holdouts/`) | CC BY-SA 3.0 (share-alike) |
| MASSIVE 1.1, When2Call (converted by `decision/`) | CC BY 4.0, with the attributions in [decision/README.md](decision/README.md) |
| Source code used as text (rebuilt by `holdouts/`) | SQLite: public domain; CPython: PSF-2.0; Lua, lodash: MIT; d3: ISC |

The manifests contain identifiers, hashes and short titles of third-party items, not their text.
Third-party data is never relicensed: each rebuilt record carries its own `licence` and `attribution`.
