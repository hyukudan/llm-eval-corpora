#!/usr/bin/env python3
"""Freeze the synthetic and robustness sets and write manifest.json.

Usage:
    python freeze.py [--public CACHE_DIR --public-out DIR]
                     [--robustness-public FILE.jsonl ... --robustness-public-out FILE]

Writes data/synthetic-v1.jsonl (or .jsonl.gz with a deterministic gzip
header when over 15 MB), data/robustness-v1.jsonl (see robustness.py) and
manifest.json next to this script. Manifest keys that this script does not
build (such as ``authored``) are kept as they are. With
``--public`` it also converts the fetched MASSIVE and When2Call files (see
fetch_public.py) into DIR and records their digests and counts; the
converted third-party records are not written into this directory. Without
it, the public section of an existing manifest is kept; when a source file
is missing, its hashes are left null. With ``--robustness-public`` it also
perturbs the test records of converted public files into a file outside
``data/`` and records its digest under ``robustness.public``; without it, a
recorded ``robustness.public`` entry is kept.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import convert_public
import fetch_public
import gen_synthetic
import robustness
from splits import BOUNDS, RULE_ID, SALT, Checker, canonical, file_sha256, gzip_bytes, sha256_hex

HERE = Path(__file__).resolve().parent
SEED = 20261029
COUNT = 6000  # scenarios; choice scenarios add one permuted child each
LANGUAGES = gen_synthetic.LANGUAGES
GZIP_OVER = 15 * 1024 * 1024
MANIFEST = HERE / "manifest.json"
LICENCE_PAGES = {
    "massive": "https://github.com/alexa/massive (LICENSE, NOTICE.md)",
    "when2call": "https://huggingface.co/datasets/nvidia/When2Call (dataset card, licence: cc-by-4.0)",
}


def build_synthetic(seed=SEED, count=COUNT, languages=LANGUAGES) -> tuple[bytes, dict]:
    """Canonical JSONL bytes of the synthetic set and its Checker summary."""
    buffer = io.BytesIO()
    checker = Checker(buffer)
    for record in gen_synthetic.generate(seed, count, languages):
        checker.add(record)
    return buffer.getvalue(), checker.summary()


def synthetic_payload(data: bytes) -> tuple[str, bytes]:
    """File name and bytes as stored: plain JSONL, or gzip when large."""
    if len(data) > GZIP_OVER:
        return "data/synthetic-v1.jsonl.gz", gzip_bytes(data)
    return "data/synthetic-v1.jsonl", data


def stored_summary(summary: dict) -> dict:
    """A Checker summary as recorded in the manifest (the split audit is not stored)."""
    return {key: value for key, value in summary.items() if key != "family_splits"}


def robustness_entry(manifest: dict) -> tuple[bytes, dict]:
    """Bytes of data/robustness-v1.jsonl and its manifest entry."""
    data, summary = robustness.build_default(manifest)
    entry = {
        "version": robustness.VERSION,
        "file": robustness.FILE,
        "file_sha256": sha256_hex(data),
        "count": summary["count"],
        "by_perturbation": summary["by_perturbation"],
        "by_family": summary["by_family"],
        "skipped_over_limits": summary["skipped_over_limits"],
        "sources": summary["sources"],
        "licence": robustness.LICENCE,
    }
    return data, entry


def robustness_public_entry(manifest: dict, paths: list[Path], out: Path) -> dict:
    """Perturb converted public test records into ``out`` and describe the result."""
    data, summary = robustness.build_public(manifest, paths)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return {
        "inputs_sha256": summary["inputs_sha256"],
        "file_sha256": sha256_hex(data),
        "count": summary["count"],
        "by_perturbation": summary["by_perturbation"],
        "by_family": summary["by_family"],
        "skipped_over_limits": summary["skipped_over_limits"],
    }


def public_inputs(cache: Path, source: str) -> list[Path] | None:
    paths = [cache / source / name for name, _, _ in fetch_public.FILES[source]]
    return paths if all(path.exists() for path in paths) else None


def convert_source(source: str, inputs: list[Path], out: Path | None) -> dict:
    if out is None:
        return convert_public.convert(source, inputs)
    with open(out, "wb") as sink:
        return convert_public.convert(source, inputs, sink)


def public_entry(source: str, cache: Path | None, out_dir: Path | None) -> dict:
    entry = {
        "converter_version": convert_public.CONVERTER_VERSION,
        "licence": convert_public.LICENCE,
        "licence_checked_at": LICENCE_PAGES[source],
        "upstream": [
            {"file": name, "url": url, "sha256": expected}
            for name, url, expected in fetch_public.FILES[source]
        ],
        "output": None,
    }
    if source == "massive":
        entry["locales"] = list(convert_public.MASSIVE_LOCALES)
    inputs = public_inputs(cache, source) if cache else None
    if inputs is None:
        return entry
    for item, path in zip(entry["upstream"], inputs):
        if file_sha256(path) != item["sha256"]:
            raise SystemExit(f"upstream file differs from the pinned sha256: {path}")
    out = out_dir / f"{source}.jsonl" if out_dir else None
    entry["output"] = stored_summary(convert_source(source, inputs, out))
    return entry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Freeze the synthetic and robustness sets and write the manifest.")
    parser.add_argument("--public", type=Path, help="cache dir filled by fetch_public.py")
    parser.add_argument("--public-out", type=Path, help="where converted public records go")
    parser.add_argument("--robustness-public", type=Path, nargs="+", help="converted public JSONL files to perturb")
    parser.add_argument("--robustness-public-out", type=Path, help="where their perturbations go (outside data/)")
    args = parser.parse_args(argv)
    if args.robustness_public and (
        args.robustness_public_out is None or robustness.inside(args.robustness_public_out, HERE / "data")
    ):
        parser.error("--robustness-public needs --robustness-public-out outside data/")
    previous = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    if args.public_out:
        args.public_out.mkdir(parents=True, exist_ok=True)

    data, summary = build_synthetic()
    name, payload = synthetic_payload(data)
    target = HERE / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)

    manifest = {
        **previous,  # keeps keys built elsewhere, such as "authored"
        "schema": "llm-eval-corpora/decision-manifest-v1",
        "split_rule": {
            "rule_id": RULE_ID,
            "salt": SALT,
            "bounds": list(BOUNDS),
            "bucket": "int.from_bytes(sha256(canonical_json([rule_id, salt, group]))[:8], 'big') % 100",
        },
        "synthetic": {
            "generator_version": gen_synthetic.GENERATOR_VERSION,
            "seed": SEED,
            "scenarios": COUNT,
            "languages": list(LANGUAGES),
            "file": name,
            "file_sha256": sha256_hex(payload),
            "licence": gen_synthetic.LICENCE,
            **stored_summary(summary),
        },
        "public": {source: public_entry(source, args.public, args.public_out) for source in ("massive", "when2call")},
    }
    if args.public is None and "public" in previous:
        # Without --public, keep the public digests recorded by an earlier run.
        manifest["public"] = previous["public"]

    robust_data, manifest["robustness"] = robustness_entry(manifest)
    (HERE / robustness.FILE).write_bytes(robust_data)
    if args.robustness_public:
        manifest["robustness"]["public"] = robustness_public_entry(
            manifest, args.robustness_public, args.robustness_public_out
        )
    elif "public" in previous.get("robustness", {}):
        manifest["robustness"]["public"] = previous["robustness"]["public"]
    text = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    MANIFEST.write_bytes(text.encode("utf-8"))
    print(canonical({"file": name, "records": summary["count"], "sha256": sha256_hex(payload)}).decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
