# Chat / prose / code holdout sets

Four published frozen sets of **parents** (a fifth, ho4d, is still in use as an unseen holdout and will
be added here when its evaluations are finished). A parent is an ordered list of items from one stratum, long enough to cut
nested evaluation prompts (8,192 and 32,768 tokens; long-v3: 131,072 and 261,888) with a continuation of
up to 256 tokens to score.

| Set | Parents (en / es / chat / code) | SEARCH / HOLDOUT | Items | Bytes | Chat sources |
|---|---|---|---:|---:|---|
| short-v3 | 16 / 16 / 16 / 16 | 16 / 48 | 2,076 | 12.9 M | oasst2 en, es |
| long-v3 | 3 / 3 / 3 / 3 | 4 / 8 | 2,726 | 16.1 M | oasst2 en, es |
| ho2 | 8 / 8 / 40 / 8 | 0 / 64 | 4,317 | 11.6 M | oasst2 en, es |
| ho3b | 8 / 8 / 40 / 8 | 0 / 64 | 5,535 | 11.7 M | oasst2 en (rest), Dolly es |

Strata:
- **en / es**: Wikipedia 20231101 articles of >= 8,000 characters, no list pages, `"<title>\n\n<text>"`.
- **chat**: oasst2 ready trees, following the best-ranked reply at each turn; or Dolly (curated multilingual)
  instruction [+ context] -> response. Items with chat-template markup in their text are skipped.
- **code**: CPython 3.13 standard-library packages, 3,000-line chunks of the SQLite 3.46.1 amalgamation,
  Lua 5.4.7 sources, lodash 4.17.21 and d3 7.9.0 (whole files).

`sets/<set>.items.jsonl` lists every item: `source_id`, `parent_id`, `kind`, `licence`, `origin`, upstream
location `loc` (rows-API page and row index for Wikipedia and Dolly), `sha256` and `bytes`. `sets/<set>.json`
has the parents (stratum, language, SEARCH/HOLDOUT split by `sha256(salt, parent)`), the case parameters and,
for reference, the prompt/continuation token counts and hashes of the cases cut with the pinned Qwen3.8
tokenizer. `upstream.json` pins dataset revisions and file sha256s.

```sh
python build_holdouts.py --out OUT [--sets ho3b] [--cache DIR] [--offline]   # OUT/<set>/items/*, sources.jsonl
python dedup.py check OUT                                                   # cross-set checks
```

The Wikipedia and Dolly rows come from the Hugging Face datasets-server rows API, 100 rows per request,
which is rate-limited: a cold rebuild of all sets makes about 540 requests and takes 20–30 minutes; the
script backs off and resumes from its cache.

## De-duplication

Sets were drawn one after another, each excluding everything earlier. The rules (implemented in `dedup.py`):

1. an upstream origin is used by one set only;
2. items byte-identical to an earlier item are skipped (from ho3b on; two ~120-byte oasst2 items in the
   earliest sets predate this rule and are listed in `dedup.py`);
3. **translation pairs**: Dolly es is a translation of Dolly en. Rows are keyed by their original-English
   example; each example is used once (smallest id over both languages), in one language of one set, by
   `bucket = int(key, 16) % 3`;
4. **semantic groups** (ho4d, not yet published): examples join a group when their normalized prompts are equal (also against
   every earlier set's user messages, at any length), or when a passage of >= 40 words shares >= 80 % of its
   5-word shingles with another; one example per group, bucketed by group; groups touching an earlier set
   are dropped.

The published sets share about 35 short first user prompts (greetings and one-liners from oasst2); `dedup.py check`
reports them.

## Licences

Wikipedia CC BY-SA 4.0, Dolly CC BY-SA 3.0 (share-alike, attribution in every record); oasst2 Apache-2.0;
SQLite public domain; CPython PSF-2.0; Lua and lodash MIT; d3 ISC.
