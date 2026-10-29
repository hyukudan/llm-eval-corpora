#!/usr/bin/env python3
"""Assemble the Hugging Face file set of OUR data (CC BY 4.0) from this checkout.

Usage: python hf/assemble.py OUT
Writes OUT/ with the layout described in hf/README.md and OUT/SHA256SUMS. Does not upload anything.
Third-party text is never copied: only our decision cases, prompt templates and the manifests.
"""
import hashlib, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FILES = {
    "README.md": "hf/README.md",
    "decision/synthetic-v1.jsonl": "decision/data/synthetic-v1.jsonl",
    "decision/authored-v1.jsonl": "decision/data/authored-v1.jsonl",
    "decision/authored-v1-unlabelled.jsonl": "decision/data/authored-v1-unlabelled.jsonl",
    "decision/manifest.json": "decision/manifest.json",
    "decision/SCHEMA.md": "decision/SCHEMA.md",
    "manifests/prose/manifest.json": "prose/manifest.json",
    "manifests/prose/manifest-lab-exact.json": "prose/manifest-lab-exact.json",
    "manifests/quality/manifest.json": "teacher/manifest.json",
    "manifests/holdouts/upstream.json": "holdouts/upstream.json",
}


def main():
    out = sys.argv[1]
    files = dict(FILES)
    for d, key in (("decision/writer-prompts", "decision/writer-prompts"), ("holdouts/sets", "manifests/holdouts/sets")):
        for f in sorted(os.listdir(os.path.join(ROOT, d))):
            files[f"{key}/{f}"] = f"{d}/{f}"
    sums = []
    for dst, src in sorted(files.items()):
        p = os.path.join(out, dst)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        shutil.copyfile(os.path.join(ROOT, src), p)
        sums.append(f"{hashlib.sha256(open(p, 'rb').read()).hexdigest()}  {dst}")
    with open(os.path.join(out, "SHA256SUMS"), "w", newline="\n") as f:
        f.write("\n".join(sums) + "\n")
    print("files", len(sums))


if __name__ == "__main__":
    main()
