#!/usr/bin/env python3
"""Chat / prose / code holdout sets (EN, ES, code): rebuild every item from its upstream source.

Each set (sets/<name>.json + sets/<name>.items.jsonl) is a list of parents; each parent is a list of items,
and each item names its origin, its upstream location and the sha256 of its bytes. Item forms:

  wikipedia  "<title>\n\n<article text>"   wikimedia/wikipedia 20231101.en|es       CC BY-SA 4.0
  oasst2     {"messages": [...]}            top-ranked path of a ready tree          Apache-2.0
  dolly      {"messages": [...]}            argilla/databricks-dolly-15k-curated-multilingual en|es, CC BY-SA 3.0
             user = instruction [+ "\n\n" + context], assistant = response
  sqlite     3,000-line chunk of sqlite3.c (amalgamation 3.46.1)                 public domain
  cpython    "# file: Lib/<path>\n" + file      CPython 3.13.7 standard library   PSF-2.0
  lua        "/* file: <member> */\n" + file    Lua 5.4.7                         MIT
  lodash/d3  whole file                         lodash 4.17.21 (MIT), d3 7.9.0 (ISC)

Usage: python build_holdouts.py --out OUT [--cache DIR] [--sets short-v3,long-v3,ho2,ho3b] [--offline]
Writes OUT/<set>/items/<source_id>.{txt,json} and OUT/<set>/sources.jsonl. Standard library only.
"""
import argparse, gzip, hashlib, io, json, os, sys, tarfile, time, urllib.parse, urllib.request, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "llm-eval-corpora/1.0 (corpus rebuild; python-urllib)"}
ROWS = "https://datasets-server.huggingface.co/rows?"
WIKI = ("wikimedia/wikipedia", "b04c8d1ceb2f5cd4588862100d08de323dccfbaa")
DOLLY = ("argilla/databricks-dolly-15k-curated-multilingual", "5f466e5af11fb82ea610acc5cf983dcb3a025840")
FILES = {  # name: (url, sha256)
    "oasst2": ("https://huggingface.co/datasets/OpenAssistant/oasst2/resolve/179dd21fc55192153d94adb0e0ce8f69e222bf75/"
               "2023-11-05_oasst2_ready.trees.jsonl.gz", None),
    "sqlite": ("https://www.sqlite.org/2024/sqlite-amalgamation-3460100.zip", None),
    "lua": ("https://www.lua.org/ftp/lua-5.4.7.tar.gz", None),
    "lodash": ("https://cdn.jsdelivr.net/npm/lodash@4.17.21/lodash.js", None),
    "d3": ("https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.js", None),
    "cpython": ("https://www.python.org/ftp/python/3.13.7/Python-3.13.7.tgz", None),
}
ALL_SETS = ["short-v3", "long-v3", "ho2", "ho3b"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def download(url, path, offline):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    if offline:
        raise SystemExit("missing in cache: " + path)
    for k in range(8):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                data = r.read()
            if url.startswith(ROWS):
                json.loads(data)["rows"]
            with open(path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(path + ".tmp", path)
            time.sleep(2 if url.startswith(ROWS) else 0)  # rate-limit the rows API
            return path
        except Exception as e:
            print("retry", k, url, e, file=sys.stderr)
            time.sleep(min(120, 10 * (k + 1)))
    raise SystemExit("download failed: " + url)


class Sources:
    def __init__(self, cache, offline):
        self.cache, self.offline, self.pages, self._mem = cache, offline, {}, {}
        os.makedirs(os.path.join(cache, "rows"), exist_ok=True)
        pins = json.load(open(os.path.join(HERE, "upstream.json"), encoding="utf-8"))
        self.pins = pins

    def file(self, name):
        url, _ = FILES[name]
        p = download(url, os.path.join(self.cache, os.path.basename(urllib.parse.urlparse(url).path)), self.offline)
        want = self.pins["files"][name]["sha256"]
        if want and sha(open(p, "rb").read()) != want:
            raise SystemExit(f"{name}: upstream file sha256 differs from upstream.json")
        return p

    def row(self, dataset, config, split, page, row_idx):
        key = (dataset, config, split, page)
        if key not in self.pages:
            q = urllib.parse.urlencode(dict(dataset=dataset, config=config, split=split, offset=page, length=100))
            p = os.path.join(self.cache, "rows", f"{dataset.replace('/', '__')}-{config}-{split}-{page}.json")
            rows = json.load(open(download(ROWS + q, p, self.offline), encoding="utf-8"))["rows"]
            self.pages[key] = {r["row_idx"]: r["row"] for r in rows}
        return self.pages[key][row_idx]

    def oasst_trees(self):
        if "oasst" not in self._mem:
            trees = {}
            with gzip.open(self.file("oasst2"), "rt", encoding="utf-8") as f:
                for line in f:
                    t = json.loads(line)
                    trees[t["message_tree_id"]] = t
            self._mem["oasst"] = trees
        return self._mem["oasst"]

    def sqlite_lines(self):
        if "sqlite" not in self._mem:
            z = zipfile.ZipFile(self.file("sqlite"))
            name = [n for n in z.namelist() if n.endswith("/sqlite3.c")][0]
            self._mem["sqlite"] = z.read(name).decode("utf-8", "replace").splitlines(keepends=True)
        return self._mem["sqlite"]

    def tar_member(self, name, member):
        k = (name, "tar")
        if k not in self._mem:
            self._mem[k] = tarfile.open(self.file(name))
        return self._mem[k].extractfile(member).read()


def oasst_top_path(node):
    """Follow the best-ranked non-deleted reply (ties by message id) and end on an assistant turn."""
    msgs = []
    while node:
        msgs.append({"role": "user" if node["role"] == "prompter" else "assistant", "content": node["text"]})
        reps = [r for r in node.get("replies", []) if not r.get("deleted")]
        if not reps:
            break
        reps.sort(key=lambda r: (r.get("rank") if r.get("rank") is not None else 99, r["message_id"]))
        node = reps[0]
    while msgs and msgs[-1]["role"] != "assistant":
        msgs.pop()
    return msgs


def attribution(o):
    if o.startswith("https://"):
        return "Wikipedia contributors, " + o
    for prefix, text in (("oasst2:", "OpenAssistant Conversations (oasst2), OpenAssistant contributors"),
                         ("dolly-ml:", "databricks-dolly-15k (Databricks, Inc.), curated multilingual version by Argilla"),
                         ("sqlite-", "SQLite 3.46.1 amalgamation, sqlite.org"),
                         ("cpython-", "CPython 3.13.7 standard library, Python Software Foundation"),
                         ("lua-", "Lua 5.4.7, Lua.org, PUC-Rio"),
                         ("lodash-", "lodash 4.17.21, OpenJS Foundation and contributors"),
                         ("d3-", "d3 7.9.0, Mike Bostock")):
        if o.startswith(prefix):
            return text
    raise ValueError(o)


def text_mode(b):
    """Decode as Python's text mode would: UTF-8 with universal newlines."""
    return io.TextIOWrapper(io.BytesIO(b), encoding="utf-8", errors="replace").read()


def build_item(it, src):
    o = it["origin"]
    if o.startswith("https://"):
        L = it["loc"]
        row = src.row(WIKI[0], L["config"], "train", L["page"], L["row_idx"])
        assert row["url"] == o
        return (row["title"] + "\n\n" + row["text"]).encode("utf-8")
    if o.startswith("oasst2:"):
        msgs = oasst_top_path(src.oasst_trees()[o.split(":", 1)[1]]["prompt"])
        return json.dumps({"messages": msgs}, ensure_ascii=False).encode("utf-8")
    if o.startswith("dolly-ml:"):
        L = it["loc"]
        row = src.row(DOLLY[0], "default", L["split"], L["page"], L["row_idx"])
        assert f"dolly-ml:{L['split']}:{row['id']}" == o
        ask, ctx, ans = ((row.get(k) or "").strip() for k in ("instruction", "context", "response"))
        msgs = [{"role": "user", "content": ask + ("\n\n" + ctx if ctx else "")}, {"role": "assistant", "content": ans}]
        return json.dumps({"messages": msgs}, ensure_ascii=False).encode("utf-8")
    if o.startswith("sqlite-3.46.1:sqlite3.c:"):
        i = int(o.rsplit(":", 1)[1])
        return "".join(src.sqlite_lines()[i:i + 3000]).encode("utf-8")
    if o.startswith("cpython-3.13:Lib/"):
        rel = o.split(":", 1)[1]
        return (f"# file: {rel}\n" + text_mode(src.tar_member("cpython", "Python-3.13.7/" + rel))).encode("utf-8")
    if o.startswith("lua-5.4.7:"):
        m = o.split(":", 1)[1]
        return (f"/* file: {m} */\n" + src.tar_member("lua", m).decode("utf-8", "replace")).encode("utf-8")
    if o in ("lodash-4.17.21", "d3-7.9.0"):
        return text_mode(open(src.file(o.split("-")[0]), "rb").read()).encode("utf-8")
    raise ValueError("unknown origin " + o)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", help="download cache (default OUT/cache)")
    ap.add_argument("--sets", default=",".join(ALL_SETS))
    ap.add_argument("--offline", action="store_true")
    a = ap.parse_args()
    src = Sources(a.cache or os.path.join(a.out, "cache"), a.offline)
    bad = 0
    for name in a.sets.split(","):
        d = os.path.join(a.out, name, "items")
        os.makedirs(d, exist_ok=True)
        items = [json.loads(l) for l in open(os.path.join(HERE, "sets", f"{name}.items.jsonl"), encoding="utf-8")]
        with open(os.path.join(a.out, name, "sources.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for it in items:
                data = build_item(it, src)
                ok = sha(data) == it["sha256"]
                if not ok:
                    bad += 1
                    print("MISMATCH", name, it["source_id"], it["origin"], file=sys.stderr)
                ext = "json" if it["kind"] == "chat" else "txt"
                with open(os.path.join(d, f"{it['source_id']}.{ext}"), "wb") as g:
                    g.write(data)
                rec = dict(it, path=f"items/{it['source_id']}.{ext}", attribution=attribution(it["origin"]))
                f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
        print(name, "items", len(items), "mismatches so far", bad)
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
