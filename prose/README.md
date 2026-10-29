# Long-context prose (Project Gutenberg)

50 public-domain books in a fixed order (`manifest.json`, or `manifest-lab-exact.json` with `--lab-exact`: Gutenberg id, title, author, language, sha256 of
the downloaded file and of the stripped body). The body is the text between the `*** START OF` line and the
`*** END OF` marker, stripped, with LF line ends and two trailing newlines.

```sh
python build_prose.py --out OUT                                # bodies/<id>.txt + corpus.txt (text only)
python build_prose.py --out OUT --profile qwen3.8-flash-next   # + PROSE-<N>.ids, checked against the manifest
python build_prose.py --out OUT --tokenizer-json tokenizer.json --lengths 131072,262144 \
       --head "..." --foot "..."                               # any Hugging Face tokenizer
python build_prose.py --out OUT --char-lengths 1000000         # text prompts cut by characters
```

**Equal share.** For a target length N (tokens or characters) minus the header and footer, every book that
still has text gets `floor(left / open_books)` units per round in manifest order until N is reached; short
books give their remainder away. So every book and language appears at every length. Each book is
tokenized on its own.

**Pinned profile `qwen3.8-flash-next`.** Tokenizer `Qwen/Qwen3.8-Flash-Next` `tokenizer.json` at revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540` (downloaded by the script, not redistributed). Header
`<|im_start|>user\nRead the following collection of books carefully.\n\n`, footer
`\n\nWrite a short essay comparing the main characters of these books.<|im_end|>\n<|im_start|>assistant\n`.
Output: space-separated ids and one newline.

| Length | Default: distinct ids, sha256 of `PROSE-<N>.ids` | `--lab-exact`: distinct ids, sha256 |
|---:|---|---|
| 262,144 | 23,857 `6f614817a19642f0386bc28d9e86425c30fd5e1f4565eea22cf4c02788e6b7f1` | 23,838 `df35b33d29b1ade3d9ba378e69fa7d6a508f55f1d36be58a6dde3bcf0d5b5bbc` |
| 393,216 | 27,948 `dcba8b5d4fafac759ac5f87e3af10687bfb993c2889783f0030e4f772d4f31b1` | 27,932 `de8d16434b2fdaa6bd4bf3b3c815f4c0e731d6a3e701c06714ecf9fe861c606f` |
| 524,032 | 30,844 `ef5389582b7ac7695879128e226c646e199e1eac71f8ea6f6f597fe3df3f3b11` | 30,800 `1373dd99c15170092e33db1cf08f30d9fc90fe50b5413347d9f58071033ed3de` |
| 1,048,320 | 38,121 `57ea5269f2d07e0bcb73c9cddfa8a6723d2900599b95d0e430fe9eab15a03232` | 38,049 `55e144d53886e447db2d8b605074d3f613081e5c39f447f228203180d91368b9` |

Corpus (default): 39,363,235 bytes, sha256 `3cdea9c7ef44b50646a1b956ed22e7f798249fa540a13ddef2a6644178881bd0`,
10,040,638 tokens with the pinned tokenizer. Lab-exact: 39,370,785 bytes,
`4f2f41183319f184775a041f06feeb3a2dbe9cde85aa8c767944e6bc99d6851f`, 10,035,309 tokens.

**Default vs lab-exact.** `manifest.json` is the default list. `manifest-lab-exact.json` is the list used
in our original long-context measurements; it differs in one book: Gutenberg #30254 (an explicit Victorian
novel) instead of #514 (*Little Women*, same position, similar size). `--lab-exact` downloads it by id like
every other book; nothing is bundled.

Languages by book: 40 en, 3 fr, 1 es, 1 es/en, 1 de, 1 it, 1 pt,
plus English translations of French, Russian, German, Greek and Italian classics.

Gutenberg occasionally revises a file. The script then reports which bodies changed and stops
(`--allow-drift` continues); the frozen bodies are in the published file set (see `../hf/`).
