# Decision cases

Closed-question decision cases. Each case is a question, a short list of
options and a context (evidence, open items, small facts), with exactly one
correct option. There are two kinds of data:

- **Synthetic cases**, written by the generators in this directory. Every
  label is computed in code from the scenario; no model and no outside text
  is involved. The frozen set is in `data/synthetic-v1.jsonl`.
- **Converters** for two public datasets, MASSIVE 1.1 and When2Call. Their
  converted records are **not** stored here; rebuild them from the original
  downloads (see below). `manifest.json` pins their digests and counts.

The record format is described in [SCHEMA.md](SCHEMA.md). All scripts use only
the Python standard library (3.10 or newer).

## Files

| File | Purpose |
|---|---|
| `gen_synthetic.py` | Synthetic generators (`--seed S --count N --out FILE.jsonl`). |
| `splits.py` | Split rule, record validation, integrity checks, canonical JSON. |
| `convert_public.py` | `massive` / `when2call` converters (`--input FILES --out FILE.jsonl`). |
| `fetch_public.py` | Downloads the upstream files to a cache directory and checks their sha256. |
| `freeze.py` | Writes `data/synthetic-v1.jsonl` and `manifest.json`. |
| `verify.py` | Regenerates and checks everything against `manifest.json`. |
| `writer-prompts/` | Prompt templates for collecting extra cases written by a language model (see below). |

## Synthetic generators

Each generator draws a scenario as a small structure, computes the answer from
it, and only then renders the text from short English, Spanish and French
templates.

| Generator | Family | Kind | Decides |
|---|---|---|---|
| `budget-gate` | `effort` | yes/no | Whether a step's cost fits the remaining request budget. The cost sits at the edge or one either side of it. |
| `deadline` | `effort` | yes/no | Whether a deadline is still ahead at a given time: one minute either side, equal times, 90 minutes and a day apart, including across midnight. |
| `ready-receipt` | `completion` | choice | Ready to validate; a stale test receipt for an older candidate (insufficient evidence); or work still pending. |
| `claim-review` | `review_need` | choice | Continue (a receipt confirms the claim), ask for independent review (claim only), or clarify (a receipt contradicts it). |
| `topic-route` | `classify` | choice | The topic of a support message among four. In about a quarter of cases the right topic is withheld, so "none of the listed topics" is correct. |
| `severity-score` | `grade` | score | Severity on a 0-3 scale as the number of failed checks out of five; the rubric is in the question. |
| `entailment` | `entail` | yes/no | Whether stated facts support, contradict or do not decide a statement about an object's status or owner. |
| `tool-select` | `next_tool` | choice | Which of three offered tools to call; ask for a missing required parameter; or say that no offered tool fits. |

Generation details:

- Scenario `i` uses generator `sorted(names)[i % 8]` and language
  `("en","es","fr")[(i // 8) % 3]`. The random stream is
  `random.Random("decision-synth-v1:<seed>")`.
- Every `choice` scenario is emitted twice: once in canonical option order
  (variant `a`) and once with its options shuffled (variant `b`, with
  `parent` = the `a` id). Both share the group and the label. Yes/no and
  score cases keep their fixed option order.
- A scenario whose content (question, facts, option set, evidence, pending
  items, language) repeats an earlier one is redrawn up to 32 times, then
  skipped. Small scenario spaces therefore saturate: `budget-gate`,
  `severity-score` and `claim-review` contribute only their distinct cases.

Frozen set (`freeze.py`: seed `20261029`, 6000 scenarios):

| | Records |
|---|---|
| Total | 5,499 (1,833 each in en/es/fr) |
| train / dev / calibration / test | 3,816 / 596 / 549 / 538 |
| By generator | ready-receipt 1,500; topic-route 1,500; deadline 750; entailment 750; tool-select 672; claim-review 144; budget-gate 105; severity-score 78 |

`manifest.json` lists the sha256 of the data file and of each split, and the counts per family, language and family/language.

## Split rule

`splits.py` assigns a split from the record's `group` alone:

```
bucket = int.from_bytes(sha256(canonical_json(["decision-split-v1", "llm-eval-corpora/decision/v1", group]))[:8], "big") % 100
0-69 train | 70-79 dev | 80-89 calibration | 90-99 test
```

`canonical_json` is `json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(",", ":"))` encoded as UTF-8.
The salt is `llm-eval-corpora/decision/v1`. Changing it reshuffles every split,
so it is part of the dataset version.

Records derived from one source always share a group: a permuted copy and its
parent, the translations of one MASSIVE utterance across locales, and the
replies to one When2Call conversation. Records from an upstream evaluation
partition are not hashed. They are labelled `external_dev` or `external_test`
and must never be used for training.

`Checker` in `splits.py` refuses a duplicate id, a split that does not follow
the rule, a `parent` that is missing or sits in another group or split, and
any content (see SCHEMA.md) that appears in two splits. The converters also
drop a record whose content repeats an earlier one, keep the first copy and
count the drops.

## Determinism

Output is a function of the seed, the count and the language list only:

- canonical JSON lines (keys sorted, UTF-8 without escaping, `\n` line ends);
- written in binary mode, so the bytes are the same on any OS;
- the RNG is Python's Mersenne Twister with string seeding.

The frozen file hashes identically under Python 3.10 and 3.13. If a set ever
grows past 15 MB, `freeze.py` stores it gzip-compressed with a fixed header
(mtime 0, no file name). `verify.py` then also checks the decompressed
content, because compressor output may differ between zlib builds.
`.gitattributes` stops git from converting the data's line endings.

## Public datasets

```
python fetch_public.py --cache CACHE
python convert_public.py massive   --input CACHE/massive/amazon-massive-dataset-1.1.tar.gz --out massive.jsonl
python convert_public.py when2call --input CACHE/when2call/when2call_train_pref.jsonl CACHE/when2call/when2call_test_mcq.jsonl --out when2call.jsonl
```

**MASSIVE 1.1** comes from
`https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz`
(sha256 `4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577`).
The converter reads the tarball directly and takes the locales en-US, es-ES,
fr-FR, de-DE, it-IT, pt-PT and ca-ES.

- Each utterance becomes a `classify` choice among the intents of its
  scenario. The options are all intents of that scenario across the input
  locales, sorted.
- The group is `massive-<id>`, shared across locales.
- Upstream `train` records are hashed into the four splits; `dev` and `test` become `external_dev` and `external_test`.
- The `weather` and `news` scenarios have a single intent, so they cannot form a choice and are skipped.

Result: 101,031 records (49,366 / 7,375 / 6,832 / 6,897 hashed; 12,325 external_dev; 18,236 external_test). 10,948 single-option records were skipped and 3,668 repeated-content records dropped.

**When2Call** comes from Hugging Face `nvidia/When2Call`, pinned to revision
`0582f7749df63a96fdc3070932e83e72396ace53`. It uses two files:

- `train/when2call_train_pref.jsonl` (sha256 `d90637f1...65bc4`). Each
  preference record becomes a choice between the chosen and the rejected
  reply. The order is fixed by the record's digest. The group is the digest
  of the conversation, and records are hashed into the four splits.
- `test/when2call_test_mcq.jsonl` (sha256 `8c3694e5...b14c`). Each
  multiple-choice record becomes a choice among calling a tool, asking for
  information, declining and answering directly, in a fixed order. These are
  always `external_test`.

Tool definitions are `confirmed` evidence, and the conversation turns are `claim` evidence.

Result: 12,575 records (6,271 / 889 / 879 / 915 hashed; 3,621 external_test). 76 records with an option over 512 characters (or otherwise over the limits) were skipped and 1 repeat dropped.

Labels are the upstream labels. Converted records carry `licence` and `attribution` fields.

## Licences and attribution

- **Synthetic cases** (`data/`): CC BY 4.0. Attribution: "llm-eval-corpora
  decision synthetic cases". The generators use no outside text.
- **MASSIVE 1.1**: CC BY 4.0, (c) Amazon.com, Inc. or its affiliates. It
  includes and translates SLURP (CC BY 4.0). Licence checked in the dataset's
  `LICENSE` and `NOTICE.md` and at <https://github.com/alexa/massive>.
- **When2Call**: CC BY 4.0, NVIDIA Corporation. Licence checked on the dataset
  card at <https://huggingface.co/datasets/nvidia/When2Call> (front matter
  `license: cc-by-4.0` and the "License/Terms of Use" section).

Converted records state their changes in `attribution`: question, options and
context are added, while texts and labels are unchanged. Cite the originals
when you use them:

```bibtex
@misc{fitzgerald2022massive,
  title         = {MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages},
  author        = {Jack FitzGerald and Christopher Hench and Charith Peris and Scott Mackie and Kay Rottmann and Ana Sanchez and Aaron Nash and Liam Urbach and Vishesh Kakarala and Richa Singh and Swetha Ranganath and Laurie Crist and Misha Britan and Wouter Leeuwis and Gokhan Tur and Prem Natarajan},
  year          = {2022},
  eprint        = {2204.08582},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}

@inproceedings{bastianelli-etal-2020-slurp,
  title     = {{SLURP}: A Spoken Language Understanding Resource Package},
  author    = {Bastianelli, Emanuele and Vanzo, Andrea and Swietojanski, Pawel and Rieser, Verena},
  booktitle = {Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year      = {2020},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2020.emnlp-main.588},
  doi       = {10.18653/v1/2020.emnlp-main.588},
  pages     = {7252--7262}
}

@inproceedings{ross-etal-2025-when2call,
  title     = {{W}hen2{C}all: When (not) to Call Tools},
  author    = {Ross, Hayley and Mahabaleshwarkar, Ameya Sunil and Suhara, Yoshi},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  year      = {2025},
  address   = {Albuquerque, New Mexico},
  publisher = {Association for Computational Linguistics},
  url       = {https://aclanthology.org/2025.naacl-long.174/},
  pages     = {3391--3409}
}
```

## Writer prompt templates

`writer-prompts/*.txt` are templates for asking a language model to write
*unlabelled* decision requests for families and languages that the
generators do not cover:

- `mixed.txt`: a blend of tasks.
- `route.txt`, `entail.txt`, `grade.txt`, `tool.txt`, `yesno.txt`: one task each.

Fill `{count}` with the number of lines and `{languages}` with the language
mix, e.g. `Spanish (es) 40, English (en) 25, French (fr) 15`.

The writer's output uses its own intake format: `family`, `kind`,
`language`, `question`, `options` with `option_id`, `insufficient_option`,
`objective` and `evidence`. It carries no label. Any such case needs an
independent labelling pass before use. Before splitting:

- give the options a writer-independent order and ids;
- group duplicate content together.

## Model-written cases

`data/authored-v1.jsonl` holds 2,377 requests written by a language model (`gpt-6-luna`) from these
templates, in Spanish (1,360), English (436), Portuguese (350), German (80), French (73), Italian (53) and
Catalan (25); families classify 1,644, route 189, entail 198, next_tool 196, grade 150. They are our data,
CC BY 4.0, and use the same record schema with `generator: "writer-luna"`.

- Options were given a writer-independent order and ids (`option-1`.., sorted and shuffled by content), because
  writers tend to list the right option first. Duplicate content (in any option order) was dropped.
- Labels come from teacher models, never from the writer's own answer. `meta.label.reviews` lists each pass
  with its model (`gpt-6-luna` first pass; `gpt-6.1-sol` on a random 15 %, every abstention and every grade
  case) and the sha256 of the exact prompt it was given. `meta.label.provenance` is `teacher_agreed` (two
  independent teachers agree, 338 cases: suitable for evaluation) or `teacher_single` (2,039: training-grade).
  Measured agreement of the two teachers: classify 98.4 %, entail 96.6 %, next_tool and route 100 %, grade
  66.7 % (the writers stated no rubric, so the second teacher labelled every grade case).
- Splits follow the same hashed group rule; groups are derived from the content.
- `data/authored-v1-unlabelled.jsonl`: 16 further requests with `answer: null` (13 where the teachers
  disagreed, 3 left unanswered).

`verify.py` re-validates both files and checks their digests (there is no generator to rerun).

## Rebuild and verify

```
python gen_synthetic.py --seed 20261029 --count 6000 --out /tmp/synthetic.jsonl   # same bytes as data/synthetic-v1.jsonl
python freeze.py                                    # rewrite data/ and manifest.json (keeps recorded public digests)
python fetch_public.py --cache CACHE
python freeze.py --public CACHE --public-out OUT    # also convert and record the public digests
python verify.py [--public CACHE]                   # exit status 1 on any mismatch
```
