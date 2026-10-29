# Multilingual quality corpus

40 short parents in 8 languages (en es fr de it pt zh ja), used as a fixed reference set for comparing a
model's next-token distribution against a reference implementation (teacher-forced, every position).

- **24 Wikipedia documents**, 3 per language, from `wikimedia/wikipedia` 20231101.<lang> at revision
  `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`: `"<title>\n\n"` + the first N whole paragraphs (N frozen per
  document; originally chosen so that each document is 450–650 tokens, with a prose filter: at least 70 % of
  the characters in paragraphs of >= 120 characters, >= 40 for zh/ja). Licence CC BY-SA; the manifest has
  the page URL and the page revision that was live on 2023-11-01 for attribution.
- **16 chat pairs**, 2 per language, human-written prompt and answer from `CohereLabs/aya_dataset` train at
  revision `f9ea04583f02a8f86404ff6c58bf75fe637df8a2` (Apache-2.0), without URLs, 260–700 tokens once
  rendered. Stored as plain messages; apply your model's chat template.

Selection order inside each pool was `sha256(salt | key)` with the salt in `manifest.json`, fixed before any
model output existed; Wikipedia rows were drawn at salted offsets that avoid an unrelated calibration set.
The rebuild does not rerun the selection: it fetches the frozen rows and checks them.

```sh
python build_teacher.py --out corpus.jsonl [--cache DIR] [--offline]
```

Each output line: `id, lang, kind, source, dataset_revision, row_idx, licence, attribution, sha256` and
`text` (documents) or `messages` (chat). Item sha256 = sha256 of `text`, or of the messages as canonical JSON
(sorted keys, no spaces, UTF-8). Output file sha256 `33f10b723da9195cafc895b925a3cdf3ec0bab2636b0e03cc1095c4344dd329b`.
