# llm-eval-corpora

Real text for evaluating and training language models, rebuilt byte for byte from public sources.

These are the corpora we use to qualify inference changes: long-context prose, a multilingual quality set,
chat/prose/code holdouts and closed-question decision cases. They are not tied to any model, runtime or
hardware. Every dataset comes with:

- a **manifest** that pins each item to its upstream source (id, revision, row, offset) and its sha256;
- a **standard-library rebuild script** that downloads the upstream data and reproduces the exact bytes;
- `make verify` / `python verify.py`, which rebuilds everything and checks every hash.

Data we made ourselves (synthetic and model-written decision cases, manifests) is in the repository. Third-party text is not
stored here; the scripts fetch it from the original publishers. A ready-made file set for a Hugging Face
dataset repository is described in [hf/](hf/).

## Datasets

| Dataset | Content | Languages | Size | Source and licence | Rebuild | Frozen sha256 |
|---|---|---|---|---|---|---|
| [prose](prose/) | 50 public-domain books, header/licence/footer stripped, concatenated in a fixed order; optional prompts of 262,144 / 393,216 / 524,032 / 1,048,320 tokens with an equal share from every book (`--lab-exact`: the original list of our measurements) | en (40 books), fr (3), es (2, one bilingual), de, it, pt | 39.4 MB text, 10.0 M tokens* | Project Gutenberg, public domain (USA) | `python prose/build_prose.py --out work/prose` | corpus `3cdea9c7…881bd0` (lab-exact `4f2f4118…6851f`) |
| [teacher](teacher/) | 40 parents: 24 Wikipedia articles (leading paragraphs) + 16 human-written chat pairs | en es fr de it pt zh ja | 0.1 MB (articles cut at whole paragraphs) | Wikipedia (CC BY-SA), Aya dataset (Apache-2.0) | `python teacher/build_teacher.py --out work/teacher.jsonl` | `33f10b72…dd329b` |
| [holdouts](holdouts/) | 4 frozen sets (short-v3, long-v3, ho2, ho3b) of 64 parents (long-v3: 12): Wikipedia, oasst2 and Dolly conversations, C/Python/Lua/JS source | en, es, code | 14,654 items, 52 MB | Wikipedia (CC BY-SA 4.0), oasst2 (Apache-2.0), Dolly curated multilingual (CC BY-SA 3.0), SQLite (public domain), CPython (PSF-2.0), Lua/lodash (MIT), d3 (ISC) | `python holdouts/build_holdouts.py --out work/holdouts` | per item; set lists below |
| [decision](decision/) synthetic | 5,499 closed-question cases from 8 code generators, answers computed in code | en es fr | 6.0 MB | ours, CC BY 4.0 | `python decision/verify.py` | `5eaffeed…f24a` |
| [decision](decision/) model-written | 2,377 labelled requests written by a language model from our prompt templates (+16 unlabelled), labels from teacher models with provenance (338 agreed by two teachers) | es en pt de fr it ca | 3.5 MB | ours, CC BY 4.0 | stored; `python decision/verify.py` re-validates | `60ce946a…97da48` |
| [decision](decision/) MASSIVE | 101,031 intent-choice cases | en es fr de it pt ca | converted locally | Amazon MASSIVE 1.1, CC BY 4.0 | `python decision/fetch_public.py …` then `convert_public.py massive` | `3d7a5e35…2dedfd` |
| [decision](decision/) When2Call | 12,575 next-step cases (call a tool / ask / decline / answer) | en | converted locally | NVIDIA When2Call, CC BY 4.0 | `convert_public.py when2call` | `0dc7a0e0…2def68` |

\* Token counts are for the one pinned tokenizer used in the manifests (Qwen3.8-Flash-Next, revision
`de4b8e4d`); everything else is tokenizer-agnostic. Full hashes are in each directory's manifest.

Holdout set manifests (`holdouts/sets/<set>.items.jsonl`, sha256):
short-v3 `18e7162e…69cb` (64 parents, 2,076 items), long-v3 `b640917d…54fd` (12, 2,726),
ho2 `8c4b7da3…6b67` (64, 4,317), ho3b `582a7428…0cb0c` (64, 5,535). A fifth set, ho4d, stays
unpublished while it is still used as an unseen holdout; it will be added with a note when that ends.

## Quick start

```sh
make verify-quick        # seconds, offline: regenerate the synthetic decision set
make verify              # download and rebuild prose, quality corpus and holdouts (~0.2 GB, ~30 min cold)
make verify-full         # + token-ID prompts (pip install tokenizers) and the MASSIVE / When2Call conversions
```

Python 3.9+ and the standard library are enough, except for token-ID prompts (`tokenizers`). Outputs and the
download cache go to `./work/` (ignored by git); a second run reuses the cache. Nothing needs a GPU.

## Determinism

- Selection was fixed before any model output existed: items are ordered by salted sha256 (salts recorded
  in the manifests), never by model behaviour.
- Upstream data is pinned: dataset revisions (Hugging Face commit ids), release tarballs with sha256, and
  Gutenberg files with the sha256 of the download used. If Gutenberg revises a book, the script says so and
  stops; the frozen bodies are in the published file set.
- Synthetic decision cases: `random.Random("decision-synth-v1:20261029")`, 6,000 scenarios; same bytes on any
  OS and Python 3.10+. Splits: `sha256(canonical_json([rule_id, salt, group]))`, 70/10/10/10, all variants of
  a scenario (and all translations of a MASSIVE utterance) share a group, so they never cross splits.
- Holdout de-duplication (origins, identical bytes, Dolly translation pairs bucketed by their original English
  example, near-duplicate groups) is documented and implemented in [holdouts/dedup.py](holdouts/dedup.py).

## Licences

- **Code** (all scripts): MIT, see [LICENSE](LICENSE).
- **Our data** (synthetic and model-written decision cases, manifests, writer prompt templates): CC BY 4.0, see
  [LICENSE-DATA.md](LICENSE-DATA.md).
- **Third-party data keeps its upstream licence.** Rebuilt Wikipedia and Dolly text is CC BY-SA
  (share-alike: anything you publish from it must carry the same licence and the attribution); oasst2 and Aya
  are Apache-2.0; MASSIVE and When2Call are CC BY 4.0; code files keep their project licences. Every rebuilt
  record carries its `licence` and `attribution`.

## Attribution and citation

If you use the rebuilt data, cite the upstream sources:

- Project Gutenberg, <https://www.gutenberg.org> (public-domain texts).
- Wikipedia contributors, Wikimedia Foundation; `wikimedia/wikipedia` dump 20231101. Each item links its page
  (and, for the quality corpus, the page revision).
- Singh et al. 2024, *Aya Dataset: An Open-Access Collection for Multilingual Instruction Tuning* (Apache-2.0).
- Köpf et al. 2023, *OpenAssistant Conversations – Democratizing Large Language Model Alignment* (oasst2, Apache-2.0).
- Conover et al. 2023, *Free Dolly* (databricks-dolly-15k, CC BY-SA 3.0); curated multilingual version by Argilla.
- FitzGerald et al. 2022, *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset* (CC BY 4.0),
  and Bastianelli et al. 2020, *SLURP* (CC BY 4.0).
- Ross et al. 2025, *When2Call: When (not) to Call Tools* (NVIDIA, CC BY 4.0).

BibTeX entries are in [decision/README.md](decision/README.md#licences-and-attribution). To cite this
repository:

```bibtex
@misc{llm_eval_corpora_2026,
  title  = {llm-eval-corpora: reproducible real-text corpora for language-model evaluation},
  author = {hyukudan},
  year   = {2026},
  url    = {https://github.com/hyukudan/llm-eval-corpora}
}
```

## Caveats

- The prose set is mostly English (40 of 50 books). The default list replaces one explicit Victorian novel
  (Gutenberg #30254) of our original measurements by *Little Women* (#514); `--lab-exact` rebuilds the original
  list and prompts (downloaded by id, nothing bundled).
- Published holdouts are no longer unseen: once public they can enter training data. Use the scripts with a
  new salt and fresh offsets to draw new holdouts.
- Model-written cases: requests and first-pass labels come from language models. Only the 338
  `teacher_agreed` cases (two independent teacher models agree) are meant for evaluation; `teacher_single`
  labels are training-grade. See [decision/README.md](decision/README.md#model-written-cases).
- Two tiny oasst2 conversations (about 120 bytes) are byte-identical across two early sets; they are listed in
  `holdouts/dedup.py` and kept so the published sets equal the ones that were used.

---

## En español

**llm-eval-corpora** publica los textos reales que usamos para evaluar modelos de lenguaje: prosa larga
(50 libros de dominio público de Project Gutenberg, hasta 1 M de tokens), un corpus multilingüe de calidad
(Wikipedia y Aya, 8 idiomas), conjuntos holdout de chat/prosa/código (Wikipedia, oasst2, Dolly, código abierto)
y casos de decisión (sintéticos propios en inglés, español y francés, y conversores de MASSIVE y When2Call).

No depende de ningún modelo, motor ni hardware. Cada conjunto tiene un manifiesto con el origen exacto y el
sha256 de cada elemento, y un script que lo reconstruye byte a byte desde la fuente original:
`make verify` (o `python verify.py`) lo reconstruye todo y comprueba los hashes.

Licencias: el código es MIT; nuestros datos sintéticos y manifiestos, CC BY 4.0; los datos de terceros
mantienen su licencia original (Wikipedia y Dolly, CC BY-SA con atribución y compartir-igual; oasst2 y Aya,
Apache-2.0; MASSIVE y When2Call, CC BY 4.0). Cita las fuentes originales indicadas arriba.
