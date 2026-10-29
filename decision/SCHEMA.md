# Decision record schema (v1)

One JSON object per line (UTF-8, `\n` line ends, keys sorted, no spaces between
tokens). Every record, synthetic or converted, has exactly these keys:

| Key | Type | Meaning |
|---|---|---|
| `id` | string | Unique record id. Synthetic: `syn-<generator>-<seed>-<index>-<variant>`; MASSIVE: `massive-<utterance id>-<locale>`; When2Call: `when2call-mcq-<uuid>` or `when2call-pref-<digest>`. |
| `group` | string | Split group. Every record derived from one scenario, utterance or conversation shares it. |
| `family` | string | `route`, `next_tool`, `review_need`, `effort`, `completion`, `classify`, `entail` or `grade`. |
| `kind` | string | `choice`, `yes_no` or `score`. |
| `lang` | string | ISO 639-1 code of the request text (`en`, `es`, `fr`, ...). |
| `question` | string | The closed question (at most 2048 characters). |
| `options` | list | 2 to 32 objects `{"id", "text"}` (text at most 512 characters). `score` options also carry an integer `level`, strictly increasing in list order. |
| `insufficient_option` | string or null | Id of the option meaning "cannot tell from the evidence", if any. |
| `context` | object | `{"objective", "evidence", "pending", "facts"}`. See below. |
| `answer` | string | Id of the correct option. Synthetic: computed by the generator's code. Public: the upstream label, unchanged. |
| `rationale` | string or null | Synthetic only: a short explanation produced by the generator, e.g. `cost 7 > remaining 6`. |
| `generator` | string | Synthetic generator name, or `massive` / `when2call`. |
| `parent` | string or null | For a derived record (a permuted copy), the id of the record it was derived from. |
| `split` | string | `train`, `dev`, `calibration`, `test`, `external_dev` or `external_test`. |
| `licence` | string | SPDX identifier of the record's licence (`CC-BY-4.0`). |
| `attribution` | string | Attribution text required by that licence. |
| `meta` | object | Provenance that the model does not see: generator version, seed, index and variant; or upstream id, locale, partition and file. |

`context`:

- `objective` (string): what the decision is for.
- `evidence` (list of at most 64 objects): `{"ref", "source", "text"}`. `ref`
  is unique within the record. `source` is `confirmed` (an established fact),
  `receipt` (the output of a check or test) or `claim` (someone's statement,
  or the message being classified). `text` is at most 4096 characters.
- `pending` (list of strings): open work items, if any.
- `facts` (object): small computed values that the request was built from
  (strings, integers or booleans), e.g. `{"model_requests_remaining": 6}`.

Rules:

- `kind: yes_no` options are `yes`, `no` and an optional insufficient option, in that order.
- `answer` is always one of the option ids.
- The canonical serialisation of a record is at most 32 KiB.
- A record's *content* is the question, family, kind, language, the multiset
  of option texts and levels, the insufficient option's text and the
  context. Ids, option ids, option order, answer, rationale and metadata are
  not part of it. The same content never appears in two splits.

What a model is shown: `question`, `options` (ids and texts), `insufficient_option` and `context`.
What it is scored against: `answer`. The other keys are bookkeeping.
