#!/usr/bin/env python3
"""Rebuild every dataset and check it against the frozen sha256s.

  python verify.py --quick              offline: regenerate the synthetic decision set only
  python verify.py                      + download and rebuild prose, quality corpus and holdout sets
  python verify.py --tokens             + prose token-ID prompts for the pinned tokenizer (needs `tokenizers`)
  python verify.py --public             + MASSIVE / When2Call conversions
  python verify.py --work DIR           where downloads and outputs go (default ./work, reused as a cache)

Exits non-zero on the first failing step. Downloads total about 0.2 GB (rows API pages are rate-limited,
so a cold holdout rebuild takes ~25 minutes); everything runs on one CPU core.
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))


def run(label, *args):
    t = time.time()
    print(f"== {label}", flush=True)
    r = subprocess.run([sys.executable, *args], cwd=HERE)
    print(f"== {label}: {'OK' if r.returncode == 0 else 'FAILED'} ({time.time() - t:.0f} s)", flush=True)
    if r.returncode:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--tokens", action="store_true")
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--work", default=os.path.join(HERE, "work"))
    a = ap.parse_args()
    w = os.path.abspath(a.work)
    if a.public:
        run("decision: fetch MASSIVE / When2Call", "decision/fetch_public.py", "--cache", os.path.join(w, "decision"))
    run("decision: synthetic set" + (" + public conversions" if a.public else ""), "decision/verify.py", *(["--public", os.path.join(w, "decision")] if a.public else []))
    if a.quick:
        return
    prose = ["prose/build_prose.py", "--out", os.path.join(w, "prose"), "--cache", os.path.join(w, "cache", "prose")]
    run("prose: 50 Gutenberg books", *prose, *(["--profile", "qwen3.8-flash-next"] if a.tokens else []))
    if a.tokens:
        run("prose: lab-exact list", *prose[:2], os.path.join(w, "prose-lab-exact"), *prose[3:],
            "--profile", "qwen3.8-flash-next", "--lab-exact")
    run("quality corpus: 40 parents", "teacher/build_teacher.py", "--out", os.path.join(w, "teacher", "corpus.jsonl"),
        "--cache", os.path.join(w, "cache", "teacher"))
    run("holdouts: 4 sets", "holdouts/build_holdouts.py", "--out", os.path.join(w, "holdouts"),
        "--cache", os.path.join(w, "cache", "holdouts"))
    run("holdouts: cross-set checks", "holdouts/dedup.py", "check", os.path.join(w, "holdouts"))
    print("all checks passed")


if __name__ == "__main__":
    main()
