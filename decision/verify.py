#!/usr/bin/env python3
"""Check the frozen decision data against manifest.json.

Usage:
    python verify.py [--public CACHE_DIR]

Regenerates the synthetic set from the manifest's seed and count, and checks
the stored file, the whole-content digest and every split's digest and
count. With ``--public`` it also re-runs the converters on the fetched files
(see fetch_public.py) and checks the upstream file digests and the converted
digests and counts. Exits 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import freeze
from splits import file_sha256, sha256_hex

HERE = Path(__file__).resolve().parent


def compare(label: str, expected, actual, problems: list) -> None:
    if expected != actual:
        problems.append(f"{label}: expected {expected!r}, got {actual!r}")


def check_summary(label: str, expected: dict, actual: dict, problems: list) -> None:
    for key in ("count", "content_sha256", "splits", "families", "langs", "family_lang", "generators"):
        compare(f"{label}.{key}", expected.get(key), actual.get(key), problems)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify frozen decision data.")
    parser.add_argument("--public", type=Path, help="cache dir filled by fetch_public.py")
    args = parser.parse_args(argv)
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    problems: list[str] = []

    # Synthetic: regenerate and compare with the manifest and with the stored file.
    syn = manifest["synthetic"]
    data, summary = freeze.build_synthetic(syn["seed"], syn["scenarios"], tuple(syn["languages"]))
    check_summary("synthetic", syn, summary, problems)
    name, payload = freeze.synthetic_payload(data)
    compare("synthetic.file", syn["file"], name, problems)
    compare("synthetic.file_sha256 (regenerated)", syn["file_sha256"], sha256_hex(payload), problems)
    stored = HERE / syn["file"]
    if not stored.exists():
        problems.append(f"missing {syn['file']}")
    else:
        compare("synthetic.file_sha256 (stored)", syn["file_sha256"], file_sha256(stored), problems)
        content = stored.read_bytes()
        if stored.suffix == ".gz":
            content = gzip.decompress(content)  # compressor output may differ; content must not
        compare("synthetic.content_sha256 (stored)", syn["content_sha256"], sha256_hex(content), problems)

    # Model-written requests: no generator to rerun; re-validate the stored file and check its digests.
    auth = manifest.get("authored")
    if auth:
        checker = freeze.Checker()
        for line in (HERE / auth["file"]).read_bytes().splitlines():
            checker.add(json.loads(line))
        check_summary("authored", auth, checker.summary(), problems)
        compare("authored.file_sha256", auth["file_sha256"], file_sha256(HERE / auth["file"]), problems)
        compare("authored.unlabelled_sha256", auth["unlabelled_sha256"], file_sha256(HERE / auth["unlabelled_file"]), problems)

    # Public: re-run converters on the cached upstream files.
    if args.public:
        for source, entry in manifest["public"].items():
            if entry["output"] is None:
                problems.append(f"{source}: manifest has no recorded output to compare")
                continue
            inputs = freeze.public_inputs(args.public, source)
            if inputs is None:
                problems.append(f"{source}: upstream files missing under {args.public}")
                continue
            for item, path in zip(entry["upstream"], inputs):
                compare(f"{source}.upstream.{item['file']}", item["sha256"], file_sha256(path), problems)
            actual = freeze.convert_source(source, inputs, None)
            check_summary(f"public.{source}", entry["output"], actual, problems)
            for key in ("skipped_over_limits", "dropped_duplicates"):
                compare(f"public.{source}.{key}", entry["output"][key], actual[key], problems)

    for problem in problems:
        print("MISMATCH", problem)
    print("FAIL" if problems else "OK")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
