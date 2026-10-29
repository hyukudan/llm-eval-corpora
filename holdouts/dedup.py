#!/usr/bin/env python3
"""De-duplication rules used to keep the holdout sets disjoint, and a checker for rebuilt sets.

Rules (applied in this order when a set was selected; the frozen sets in sets/ are their output):

1. Origin: an upstream origin (Wikipedia page URL, oasst2 tree id, Dolly split:id, code file or chunk) is used by
   at most one set.
2. Bytes: an item whose bytes equal an item of an earlier set is skipped.
3. Translation pairs (Dolly): the Spanish split is a translation of the English one. Every row is keyed by its
   original-English example, sha256(instruction_original_en \\0 context_original_en \\0 response_original_en);
   the example is represented once (smallest id over both languages) and goes to ONE language of ONE set:
   bucket = int(key, 16) % 3 (ho3b es = bucket 0; ho4* en = bucket 1, es = bucket 2).
4. Semantic groups (Dolly; used by the newest set, ho4d, which is not published yet): examples are joined (union-find) when
   - their normalized original prompts (instruction + context words) are equal (formatting variants);
   - any normalized prompt, original or curated en/es, of any length is equal, also against the user messages
     of earlier sets;
   - a passage of >= 40 words (whole example, prompt, instruction, context or response; original or curated)
     has >= 80 % of its 5-word shingles inside another passage (paraphrases, shared passages).
   One example per group (smallest id), bucketed by the group key; groups touching an earlier set are not used.
5. Parents are split into SEARCH / HOLDOUT by sha256(salt, parent_id) before any result exists.

Usage:
  python dedup.py check BUILT_DIR      cross-set checks on the output of build_holdouts.py
"""
import hashlib, json, os, re, sys
from collections import Counter


def h(s):
    return hashlib.sha256(s.encode()).hexdigest()


def dolly_key(row):
    """The original-English example: translations and re-listed duplicates share it whatever their ids."""
    return h("\0".join((row.get("instruction_original_en") or "", row.get("context_original_en") or "",
                        row.get("response_original_en") or "")))


def dolly_bucket(key, n=3):
    return int(key, 16) % n


def words(s):
    return re.findall(r"\w+", (s or "").lower())


def shingles(w):
    return {h(" ".join(w[i:i + 5]))[:16] for i in range(max(1, len(w) - 4))}


class Groups:
    """Union-find over example keys (and reserved earlier-set nodes)."""

    def __init__(self):
        self.parent = {}

    def add(self, x):
        self.parent.setdefault(x, x)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def semantic_groups(rows, earlier_chats=(), every_prompt=True, threshold=0.8, min_words=40):
    """rows: every Dolly row of both languages (dicts with id and *_original_en / curated fields).
    earlier_chats: (origin, messages) of chat items already used. Returns (example key -> group, reserved groups)."""
    canon, every = {}, []
    for r in rows:
        k = dolly_key(r)
        canon[k] = min(canon.get(k, r["id"]), r["id"])
        every.append((k, r))
    reps = {k: r for k, r in every if r["id"] == canon[k]}
    g = Groups()
    for k in reps:
        g.add(k)
    passages, prompt, same = [], {}, {}

    def join_prompt(node, w):
        if every_prompt and w:
            t = " ".join(w)
            if t in same:
                g.union(node, same[t])
            same.setdefault(t, node)

    def add_passages(node, parts):
        for w in parts:
            if len(w) >= min_words:
                passages.append((node, shingles(w)))

    for k, r in reps.items():
        ins, ctx, rsp = (words(r.get(f)) for f in ("instruction_original_en", "context_original_en", "response_original_en"))
        key = " ".join(ins + ["\0"] + ctx)
        if len(ins) + len(ctx) >= 5:
            if key in prompt:
                g.union(k, prompt[key])
            prompt.setdefault(key, k)
        add_passages(k, (ins + ctx + rsp, ins + ctx, ins, ctx, rsp))
    for k, r in every:
        ins, ctx, rsp = (words(r.get(f)) for f in ("instruction", "context", "response"))
        add_passages(k, (ins + ctx + rsp, ins + ctx, ins, ctx, rsp))
    for k, r in every:
        join_prompt(k, words(r.get("instruction_original_en")) + words(r.get("context_original_en")))
        join_prompt(k, words(r.get("instruction")) + words(r.get("context")))
    for origin, msgs in earlier_chats:
        node = "R:" + origin
        g.add(node)
        for m in msgs:
            if m["role"] == "user":
                join_prompt(node, words(m["content"]))
        add_passages(node, [sum((words(m["content"]) for m in msgs), [])] + [words(m["content"]) for m in msgs])
    index = {}
    for i, (_, s) in enumerate(passages):
        for x in s:
            index.setdefault(x, []).append(i)
    # exact prefix filter: if >= 80 % of s's shingles are in another set, that set holds one of s's
    # int(0.2*|s|)+1 rarest shingles, so probing only those finds every such pair
    for i, (node, s) in enumerate(passages):
        probe = sorted(s, key=lambda x: (len(index[x]), x))[:int((1 - threshold) * len(s)) + 1]
        for j in {j for x in probe for j in index[x]}:
            other = passages[j][1]
            if j != i and len(other) >= len(s) and len(s & other) >= threshold * len(s):
                g.union(node, passages[j][0])
    reserved = {g.find(n) for n in list(g.parent) if n.startswith("R:")}
    return {k: g.find(k) for k in reps}, reserved


# Byte-identical items in the earliest sets, frozen before rule 2 existed: two ~120-byte oasst2 conversations
# that appear in two different trees each. Kept so that the published sets equal the ones that were used.
KNOWN_SHARED = {"8b54ae40de2825359316e32a1be53eba14ad90fcb048d896cc36c891e276d490": ("short-v3", "long-v3"),
                "c3fa4cae3a661b944196b18bf5d9128f6202c0648a9a6c132433a752d86f6c07": ("short-v3", "ho2")}


def check(built):
    """Cross-set checks on rebuilt sets: shared origins or identical bytes fail; repeated user prompts are reported."""
    sets = sorted(d for d in os.listdir(built) if os.path.exists(os.path.join(built, d, "sources.jsonl")))
    origin, digest, prompts, fail = {}, {}, {}, 0
    for s in sets:
        for line in open(os.path.join(built, s, "sources.jsonl"), encoding="utf-8"):
            it = json.loads(line)
            for table, key in ((origin, it["origin"]), (digest, it["sha256"])):
                if key in table and table[key] != s and key not in KNOWN_SHARED:
                    print(f"FAIL {s} shares {key[:60]} with {table[key]}")
                    fail += 1
                table.setdefault(key, s)
            if it["kind"] == "chat":
                msgs = json.load(open(os.path.join(built, s, it["path"]), encoding="utf-8"))["messages"]
                p = " ".join(words(msgs[0]["content"]))
                prompts.setdefault(p, set()).add(s)
    cross = Counter(tuple(sorted(v)) for v in prompts.values() if len(v) > 1)
    print("sets", sets, "items", len(origin))
    print("first user prompts repeated across sets (normalized, informational):", dict(cross) or "none")
    return fail


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "check":
        raise SystemExit(__doc__)
    raise SystemExit(1 if check(sys.argv[2]) else 0)
