#!/usr/bin/env python3
"""Check the frozen decision data against manifest.json.

Usage:
    python verify.py [--public CACHE_DIR]
                     [--robustness-public FILE.jsonl ... --robustness-public-out FILE]

Regenerates the synthetic set from the manifest's seed and count, and checks
the stored file, the whole-content digest and every split's digest and
count. It rebuilds the robustness set from the stored synthetic and authored
files and checks it against the manifest and data/robustness-v1.jsonl (both
are required). With ``--public`` it also re-runs the converters on the
fetched files (see fetch_public.py) and checks the upstream file digests and
the converted digests and counts. With ``--robustness-public-out`` it checks
the stored public perturbation file (required, bytes and counts) against
``robustness.public``; adding ``--robustness-public`` also rebuilds it from
the converted public files and compares the bytes. Only the keys a manifest
entry stores are compared. Exits 1 on any mismatch.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import freeze
import robustness
from splits import canonical, file_sha256, sha256_hex

HERE = Path(__file__).resolve().parent


def compare(label: str, expected, actual, problems: list) -> None:
    if expected != actual:
        problems.append(f"{label}: expected {expected!r}, got {actual!r}")


def check_summary(label: str, expected: dict, actual: dict, problems: list) -> None:
    # A fixed key list: summary keys the manifest does not store (the family_splits audit) are ignored.
    for key in ("count", "content_sha256", "splits", "families", "langs", "family_lang", "generators"):
        compare(f"{label}.{key}", expected.get(key), actual.get(key), problems)


def perturbation_counts(data: bytes) -> dict:
    """Count, per-perturbation and per-family counts of robustness JSONL bytes."""
    records = [json.loads(line) for line in data.splitlines()]
    by_perturbation, by_family = Counter(r["meta"]["perturbation"] for r in records), Counter(r["family"] for r in records)
    return {
        "count": len(records),
        "by_perturbation": {name: by_perturbation[name] for name in robustness.PERTURBATIONS if by_perturbation[name]},
        "by_family": dict(sorted(by_family.items())),
    }


def check_robustness_public(rob: dict, manifest: dict, args, problems: list) -> None:
    """Stored (and, with inputs, rebuilt) public perturbations against robustness.public."""
    expected = rob.get("public")
    if not expected:
        problems.append("robustness: manifest has no recorded public entry to compare")
        return
    out = args.robustness_public_out
    if not out.is_file():
        problems.append(f"robustness.public: missing output file {out}")
        stored = None
    else:
        stored = out.read_bytes()
        compare("robustness.public.file_sha256 (stored)", expected["file_sha256"], sha256_hex(stored), problems)
        for key, value in perturbation_counts(stored).items():
            compare(f"robustness.public.{key} (stored)", expected[key], value, problems)
        if any(canonical(json.loads(line)) != line for line in stored.splitlines()):
            problems.append("robustness.public: stored file is not canonical JSONL")
    if args.robustness_public:
        data, summary = robustness.build_public(manifest, args.robustness_public)
        for key in ("inputs_sha256", "count", "by_perturbation", "by_family"):
            compare(f"robustness.public.{key}", expected[key], summary[key], problems)
        compare("robustness.public.file_sha256 (rebuilt)", expected["file_sha256"], sha256_hex(data), problems)
        if stored is not None and stored != data:
            problems.append("robustness.public: stored file differs from the rebuilt bytes")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Verify frozen decision data.")
    parser.add_argument("--public", type=Path, help="cache dir filled by fetch_public.py")
    parser.add_argument("--robustness-public", type=Path, nargs="+", help="converted public JSONL files")
    parser.add_argument("--robustness-public-out", type=Path, help="stored public perturbation file to check")
    args = parser.parse_args(argv)
    if args.robustness_public and args.robustness_public_out is None:
        parser.error("--robustness-public needs --robustness-public-out (the stored file to check)")
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

    # Robustness: rebuild from the stored data and compare with the manifest and the stored file.
    rob = manifest.get("robustness")
    if not isinstance(rob, dict) or not rob:
        problems.append("robustness: manifest entry missing or empty")
    else:
        data, entry = freeze.robustness_entry(manifest)
        for key, value in entry.items():
            compare(f"robustness.{key}", rob.get(key), value, problems)
        compare("robustness.sources.synthetic_test_sha256 (manifest)",
                syn["splits"]["test"]["content_sha256"], rob["sources"]["synthetic_test_sha256"], problems)
        if auth:
            compare("robustness.sources.authored_test_sha256 (manifest)",
                    auth["splits"]["test"]["content_sha256"], rob["sources"]["authored_test_sha256"], problems)
        stored = HERE / str(rob.get("file"))
        if not stored.is_file():
            problems.append(f"missing {rob.get('file')}")
        else:
            compare("robustness.file_sha256 (stored)", rob.get("file_sha256"), file_sha256(stored), problems)
            if stored.read_bytes() != data:
                problems.append("robustness: stored file differs from the rebuilt bytes")
        if args.robustness_public_out:
            check_robustness_public(rob, manifest, args, problems)

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
