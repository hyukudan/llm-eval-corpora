---
license: cc-by-4.0
language: [en, es, fr, pt, de, it, ca]
pretty_name: llm-eval-corpora
size_categories: [1K<n<10K]
task_categories: [text-classification, multiple-choice]
configs:
  - config_name: decision_synthetic
    data_files:
      - split: train
        path: decision/synthetic-v1.jsonl
  - config_name: decision_model_written
    data_files:
      - split: train
        path: decision/authored-v1.jsonl
---

# llm-eval-corpora (our data)

The data we made ourselves for <https://github.com/hyukudan/llm-eval-corpora>, all CC BY 4.0. Third-party
text (Project Gutenberg, Wikipedia, Aya, oasst2, Dolly, MASSIVE, When2Call, source code) is **not** here;
the repository's scripts rebuild it byte for byte from the original publishers, using the manifests below.
`SHA256SUMS` lists every file.

| Path | Content |
|---|---|
| `decision/synthetic-v1.jsonl` | 5,499 closed-question decision cases in en/es/fr from 8 code generators; every answer computed in code |
| `decision/authored-v1.jsonl` | 2,377 decision requests written by a language model from our prompt templates (es, en, pt, de, fr, it, ca), labelled by teacher models; provenance per record |
| `decision/authored-v1-unlabelled.jsonl` | 16 more written requests without a label (teachers disagreed or did not answer) |
| `decision/writer-prompts/` | the prompt templates used to collect the written requests |
| `decision/manifest.json`, `decision/SCHEMA.md` | counts, per-split sha256, split rule, record schema |
| `manifests/` | the manifests of the rebuildable corpora: prose (default and lab-exact), quality corpus, holdout sets, upstream pins |

Each record has an `id`, a `group` and a `split` (`train`/`dev`/`calibration`/`test`, by a salted hash of the
group, so variants of one scenario never cross splits). The `split` column is a field, not a Hugging Face
split: filter on it.

**Labels.** Synthetic answers are exact (computed by the generator). Model-written labels come from teacher
models: use the 338 records with `meta.label.provenance == "teacher_agreed"` (two independent teacher models
agree) for evaluation; `teacher_single` labels are training-grade.

**Licence.** CC BY 4.0. Attribution: "llm-eval-corpora, hyukudan".
