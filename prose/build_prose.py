#!/usr/bin/env python3
"""Long-context real prose: 50 public-domain Project Gutenberg books in six languages.

Default (tokenizer-agnostic): download every book by its Gutenberg id, strip the Project Gutenberg
header/licence and footer, and write

    OUT/bodies/<id>.txt   one body per book
    OUT/corpus.txt        the 50 bodies concatenated in manifest order

checking each sha256 against manifest.json.

Optional token-ID prompts (equal share from every book at every length, chat-wrapped):

    --profile qwen3.8-flash-next          pinned tokenizer + wrapper; ids are checked against the manifest
    --tokenizer-json PATH [--head T --foot T]   any Hugging Face tokenizer.json (no expected hashes)
    --char-lengths 1000000,4000000        text prompts cut by characters instead of tokens

Equal share: for a target of N units (tokens or characters) between the header and footer, every book
that still has text gets floor(left / open_books) units per round, in manifest order, until N is reached;
short books give their remainder away. Each book is encoded on its own, never across book boundaries.

Usage: python build_prose.py --out OUT [--cache DIR] [--profile qwen3.8-flash-next] [--lab-exact] [--offline]

--lab-exact uses manifest-lab-exact.json: the original book list of our long-context measurements, which
includes Gutenberg #30254 (an explicit Victorian novel); the default set replaces it by #514 (Little Women).
Needs Python 3.9+; token modes also need `pip install tokenizers`.
"""
import argparse, hashlib, json, os, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "llm-eval-corpora/1.0 (prose rebuild; python-urllib)"}
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"
PROFILES = {
    "qwen3.8-flash-next": {
        "tokenizer_url": "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/"
                         "de4b8e4d43b917e7706784d8bb445c9af86a3540/tokenizer.json",
        "head": "<|im_start|>user\nRead the following collection of books carefully.\n\n",
        "foot": "\n\nWrite a short essay comparing the main characters of these books.<|im_end|>\n"
                "<|im_start|>assistant\n",
        "lengths": [262144, 393216, 524032, 1048320],
    },
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def fetch(url, path, tries=6):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return open(path, "rb").read()
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                data = r.read()
            with open(path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(path + ".tmp", path)
            time.sleep(1)  # be polite to the mirror
            return data
        except Exception as e:
            print("retry", k, url, e, file=sys.stderr)
            time.sleep(5 * (k + 1))
    raise SystemExit("download failed: " + url)


def strip_gutenberg(raw):
    """Body between the '*** START OF' line and the '*** END OF' marker, LF line ends, two trailing LFs."""
    t = raw.decode("utf-8", errors="replace")
    a = t.find("*** START OF")
    a = t.find("\n", a) + 1 if a >= 0 else 0
    b = t.find("*** END OF")
    b = b if b > 0 else len(t)
    return t[a:b].strip().replace("\r\n", "\n") + "\n\n"


def equal_share(lengths, n):
    take, left = [0] * len(lengths), n
    if n > sum(lengths):
        raise SystemExit(f"target {n} exceeds the corpus ({sum(lengths)})")
    while left:
        open_ = [k for k, L in enumerate(lengths) if take[k] < L]
        share = max(1, left // len(open_))
        for k in open_:
            d = min(share, lengths[k] - take[k], left)
            take[k] += d
            left -= d
            if not left:
                break
    return take


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", help="raw download cache (default OUT/raw)")
    ap.add_argument("--profile", choices=sorted(PROFILES))
    ap.add_argument("--tokenizer-json")
    ap.add_argument("--head", default="")
    ap.add_argument("--foot", default="")
    ap.add_argument("--lengths", help="token lengths, comma separated (default: the profile's)")
    ap.add_argument("--char-lengths", help="character lengths for text prompts, comma separated")
    ap.add_argument("--offline", action="store_true", help="use the cache only")
    ap.add_argument("--lab-exact", action="store_true", help="original lab book list (manifest-lab-exact.json)")
    ap.add_argument("--allow-drift", action="store_true",
                    help="continue when Gutenberg has revised a book (hashes then differ from the manifest)")
    a = ap.parse_args()
    man = json.load(open(os.path.join(HERE, "manifest-lab-exact.json" if a.lab_exact else "manifest.json"),
                         encoding="utf-8"))
    cache = a.cache or os.path.join(a.out, "raw")
    os.makedirs(os.path.join(a.out, "bodies"), exist_ok=True)
    os.makedirs(cache, exist_ok=True)

    bodies, drift = [], []
    for b in man["books"]:
        p = os.path.join(cache, f"pg{b['id']}.txt")
        if a.offline and not os.path.exists(p):
            raise SystemExit("missing in cache: " + p)
        raw = fetch(URL.format(id=b["id"]), p)
        body = strip_gutenberg(raw).encode("utf-8")
        if sha(raw) != b["raw_sha256"]:
            print(f"note: pg{b['id']} raw file differs from the frozen download (upstream revision)", file=sys.stderr)
        if sha(body) != b["body_sha256"]:
            drift.append(b["id"])
        with open(os.path.join(a.out, "bodies", f"{b['id']}.txt"), "wb") as f:
            f.write(body)
        bodies.append(body)
    corpus = b"".join(bodies)
    with open(os.path.join(a.out, "corpus.txt"), "wb") as f:
        f.write(corpus)
    print("books", len(bodies), "corpus bytes", len(corpus), "sha256", sha(corpus))
    if drift:
        msg = f"{len(drift)} book bodies differ from the manifest: {drift}. Use the frozen bodies from the " \
              f"published file set, or pass --allow-drift."
        if not a.allow_drift:
            raise SystemExit(msg)
        print("warning:", msg, file=sys.stderr)
    elif sha(corpus) != man["corpus_sha256"]:
        raise SystemExit("corpus sha256 mismatch")
    texts = [x.decode("utf-8") for x in bodies]
    ok = True

    if a.char_lengths:
        for n in map(int, a.char_lengths.split(",")):
            take = equal_share([len(t) for t in texts], n - len(a.head) - len(a.foot))
            s = (a.head + "".join(t[:k] for t, k in zip(texts, take)) + a.foot).encode("utf-8")
            with open(os.path.join(a.out, f"PROSE-{n}.chars.txt"), "wb") as f:
                f.write(s)
            print(n, "chars sha256", sha(s))

    if a.profile or a.tokenizer_json:
        from tokenizers import Tokenizer
        prof = PROFILES.get(a.profile, {})
        if a.profile:
            tj = os.path.join(cache, f"tokenizer-{a.profile}.json")
            if not (a.offline and os.path.exists(tj)):
                fetch(prof["tokenizer_url"], tj)
            head, foot = prof["head"], prof["foot"]
        else:
            tj, head, foot = a.tokenizer_json, a.head, a.foot
        tok = Tokenizer.from_file(tj)
        enc = lambda s: tok.encode(s, add_special_tokens=False).ids
        h_ids, f_ids = enc(head), enc(foot)
        books = [enc(t) for t in texts]
        expect = man.get("token_profiles", {}).get(a.profile or "", {})
        if expect:
            if expect["book_tokens"] != [len(x) for x in books] or expect["head"] != h_ids or expect["foot"] != f_ids:
                print("MISMATCH: per-book token counts or wrapper ids differ from the manifest", file=sys.stderr)
                ok = False
        lengths = [int(x) for x in a.lengths.split(",")] if a.lengths else prof.get("lengths", [])
        if not lengths:
            raise SystemExit("--lengths is required with --tokenizer-json")
        for p in lengths:
            take = equal_share([len(x) for x in books], p - len(h_ids) - len(f_ids))
            t = h_ids + [i for k, b in enumerate(books) for i in b[:take[k]]] + f_ids
            assert len(t) == p
            data = (" ".join(map(str, t)) + "\n").encode()
            path = os.path.join(a.out, f"PROSE-{p}.ids")
            with open(path, "wb") as f:
                f.write(data)
            want = expect.get("prompts", {}).get(str(p), {}).get("sha256")
            status = "" if not want else ("OK" if want == sha(data) else "MISMATCH")
            ok &= status != "MISMATCH"
            print(p, "distinct", len(set(t)), "sha256", sha(data), status)
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
