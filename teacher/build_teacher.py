#!/usr/bin/env python3
"""Multilingual quality corpus: 40 parents in 8 languages (en es fr de it pt zh ja).

  * 24 Wikipedia documents (3 per language): wikimedia/wikipedia 20231101.<lang>, "<title>\n\n" followed by
    the first N whole paragraphs of the article (N is frozen per document in manifest.json).
    Licence CC BY-SA (see README); attribution: Wikipedia contributors, page URL and revision in the manifest.
  * 16 chat pairs (2 per language): CohereLabs/aya_dataset train, human-written (prompt, answer),
    Apache-2.0. Stored as plain {"role","content"} messages; render them with any chat template.

The rows are fetched one by one from the Hugging Face datasets-server rows API by their frozen row index and
checked against the frozen sha256 of the source row, then the rebuilt item is checked against its own sha256.

Usage: python build_teacher.py --out OUT.jsonl [--cache DIR] [--offline]
Standard library only.
"""
import argparse, hashlib, json, os, sys, time, urllib.parse, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "llm-eval-corpora/1.0 (corpus rebuild; python-urllib)"}
ROWS = "https://datasets-server.huggingface.co/rows?"


def h(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def canon(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def get_row(dataset, config, split, idx, cache, offline):
    path = os.path.join(cache, f"{dataset.replace('/', '__')}-{config}-{split}-{idx}.json")
    if not os.path.exists(path):
        if offline:
            raise SystemExit("missing in cache: " + path)
        url = ROWS + urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split, offset=idx, length=1))
        for k in range(6):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                    data = r.read()
                json.loads(data)["rows"][0]
                with open(path, "wb") as f:
                    f.write(data)
                time.sleep(1)
                break
            except Exception as e:
                print("retry", k, url, e, file=sys.stderr)
                time.sleep(5 * (k + 1))
        else:
            raise SystemExit("fetch failed: " + url)
    rows = json.load(open(path, encoding="utf-8"))["rows"]
    assert rows[0]["row_idx"] == idx, (path, rows[0]["row_idx"])
    return rows[0]["row"]


def build(item, cache, offline):
    if item["kind"] == "doc":
        row = get_row(item["source"], item["config"], "train", item["row_idx"], cache, offline)
        if str(row["id"]) != item["page_id"] or h(row["text"]) != item["source_text_sha256"]:
            raise ValueError("source row changed upstream")
        paras = [p.strip() for p in row["text"].split("\n") if p.strip()]
        text = row["title"] + "\n\n" + "\n\n".join(paras[:item["paragraphs"]])
        out = {"text": text}
        digest = h(text)
    else:
        row = get_row(item["source"], "default", "train", item["row_idx"], cache, offline)
        if h(row["inputs"] + "\0" + row["targets"]) != item["source_text_sha256"]:
            raise ValueError("source row changed upstream")
        msgs = [{"role": "user", "content": row["inputs"].strip()},
                {"role": "assistant", "content": row["targets"].strip()}]
        out = {"messages": msgs}
        digest = h(canon(msgs))
    if digest != item["sha256"]:
        raise ValueError("rebuilt item differs from the manifest")
    keep = ("id", "lang", "kind", "source", "dataset_revision", "row_idx", "licence", "attribution")
    return dict({k: item[k] for k in keep if k in item}, **out, sha256=digest)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", help="row cache (default: next to OUT)")
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    man = json.load(open(os.path.join(HERE, "manifest.json"), encoding="utf-8"))
    cache = a.cache or os.path.join(os.path.dirname(os.path.abspath(a.out)), "teacher-rows")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    recs, bad = [], []
    for item in man["items"]:
        try:
            recs.append(build(item, cache, a.offline))
        except ValueError as e:
            bad.append((item["id"], str(e)))
    with open(a.out, "w", encoding="utf-8", newline="\n") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
    print("items", len(recs), "of", len(man["items"]))
    for i, e in bad:
        print("MISMATCH", i, e, file=sys.stderr)
    if bad:
        raise SystemExit(1)
    corpus = hashlib.sha256(open(a.out, "rb").read()).hexdigest()
    print("output sha256", corpus, "OK" if corpus == man["output_sha256"] else "MISMATCH")
    if corpus != man["output_sha256"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
