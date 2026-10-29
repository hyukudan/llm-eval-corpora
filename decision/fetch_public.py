#!/usr/bin/env python3
"""Download the public source files used by convert_public.py, with sha256 checks.

Usage:
    python fetch_public.py --cache CACHE_DIR [--only massive|when2call]

Files land under CACHE_DIR/<source>/ and are verified against the pinned
sha256 below; a file already present with the right digest is not fetched
again. A mismatch deletes the partial download and exits non-zero. Nothing is
extracted; the converter reads the MASSIVE tarball directly.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from splits import file_sha256

# When2Call files are pinned to one dataset revision so the bytes cannot drift.
WHEN2CALL_REVISION = "0582f7749df63a96fdc3070932e83e72396ace53"
WHEN2CALL_BASE = f"https://huggingface.co/datasets/nvidia/When2Call/resolve/{WHEN2CALL_REVISION}"

FILES = {
    "massive": [
        (
            "amazon-massive-dataset-1.1.tar.gz",
            "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz",
            "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577",
        ),
    ],
    "when2call": [
        (
            "when2call_train_pref.jsonl",
            f"{WHEN2CALL_BASE}/train/when2call_train_pref.jsonl",
            "d90637f108fabf1b097493c5818c3692e1d73140259d2f8e490536255e765bc4",
        ),
        (
            "when2call_test_mcq.jsonl",
            f"{WHEN2CALL_BASE}/test/when2call_test_mcq.jsonl",
            "8c3694e583eeeb8dbc297e6cd90da70efc68efa4b6adb7227523e828c6b7b14c",
        ),
    ],
}


def fetch(cache: Path, source: str) -> list[Path]:
    paths = []
    for name, url, expected in FILES[source]:
        target = cache / source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and file_sha256(target) == expected:
            print(f"ok (cached) {target}")
            paths.append(target)
            continue
        partial = target.with_name(target.name + ".partial")
        print(f"downloading {url}")
        request = urllib.request.Request(url, headers={"User-Agent": "decision-fetch/1"})
        with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
            for chunk in iter(lambda: response.read(1 << 20), b""):
                out.write(chunk)
        actual = file_sha256(partial)
        if actual != expected:
            partial.unlink()
            raise SystemExit(f"sha256 mismatch for {name}: expected {expected}, got {actual}")
        partial.replace(target)
        print(f"ok {target}")
        paths.append(target)
    return paths


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Download public decision sources.")
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--only", choices=sorted(FILES))
    args = parser.parse_args(argv)
    for source in [args.only] if args.only else sorted(FILES):
        fetch(args.cache, source)
    return 0


if __name__ == "__main__":
    sys.exit(main())
