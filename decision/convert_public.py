#!/usr/bin/env python3
"""Convert public datasets into decision records (see SCHEMA.md).

Usage:
    python convert_public.py massive --input amazon-massive-dataset-1.1.tar.gz --out massive.jsonl
    python convert_public.py when2call --input when2call_train_pref.jsonl when2call_test_mcq.jsonl --out when2call.jsonl

MASSIVE 1.1 (CC BY 4.0): one record per utterance and locale. The question
is which intent of the utterance's scenario it expresses; the options are all
intents of that scenario seen in the input, sorted. Translations of one
utterance id share the group ``massive-<id>``, so they never straddle splits.
Upstream ``train`` records get the hashed split; upstream ``dev`` and
``test`` records become ``external_dev`` / ``external_test``.

When2Call (CC BY 4.0): multiple-choice test records become a choice among
calling a tool, asking for information, declining and answering directly
(fixed option order), always ``external_test``. Preference training records
become a choice between the chosen and the rejected reply, in an order fixed
by the record's digest; replies to one conversation share a group.

Records over the size limits in splits.py are skipped and counted; a record
whose content repeats an earlier one is dropped and counted. Labels are the
upstream labels, unchanged. Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

from splits import Checker, RecordError, assign_split, canonical, content_key, sha256_hex, validate

CONVERTER_VERSION = "decision-public-v1"
LICENCE = "CC-BY-4.0"
MASSIVE_ATTRIBUTION = (
    "MASSIVE 1.1, (c) Amazon.com, Inc. or its affiliates, CC BY 4.0 (FitzGerald et al., 2022, "
    "https://github.com/alexa/massive); includes and translates SLURP (Bastianelli et al., 2020), "
    "CC BY 4.0. Converted to the decision schema: question, options and context added; "
    "utterances and intent labels unchanged."
)
WHEN2CALL_ATTRIBUTION = (
    "When2Call, NVIDIA Corporation, CC BY 4.0 (Ross et al., 2025, "
    "https://huggingface.co/datasets/nvidia/When2Call). Converted to the decision schema: "
    "question and context added; texts and labels unchanged."
)
# Languages of the original source entry: en, es, fr, de, it, pt, ca.
MASSIVE_LOCALES = ("en-US", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-PT", "ca-ES")
MASSIVE_SPLITS = {"train": None, "dev": "external_dev", "test": "external_test"}
# Fixed order: the option list never hints at the answer.
WHEN2CALL_ANSWERS = (
    ("tool_call", "tool-call"),
    ("request_for_info", "ask-for-information"),
    ("cannot_answer", "cannot-answer"),
    ("direct", "direct-answer"),
)


def _jsonl(lines):
    for line in lines:
        if line.strip():
            yield json.loads(line)


def _massive_lines(paths, locales):
    """Lines of each wanted locale file, from the tarball or from extracted .jsonl files."""
    for path in paths:
        path = Path(path)
        if path.name.endswith((".tar.gz", ".tgz")):
            with tarfile.open(path, "r:gz") as archive:
                members = {Path(m.name).name: m for m in archive.getmembers() if m.isfile()}
                for locale in locales:
                    member = members.get(f"{locale}.jsonl")
                    if member is None:
                        raise SystemExit(f"locale {locale} not in {path.name}")
                    yield from archive.extractfile(member).read().decode("utf-8").splitlines()
        elif path.stem in locales:
            yield from path.read_text(encoding="utf-8").splitlines()


def massive_records(paths, locales=MASSIVE_LOCALES):
    rows = []
    for record in _jsonl(_massive_lines(paths, locales)):
        if record["locale"] not in locales:
            continue
        rows.append(
            tuple(record[k] for k in ("id", "locale", "partition", "scenario", "intent", "utt"))
        )
    intents: dict[str, set[str]] = {}
    for row in rows:
        intents.setdefault(row[3], set()).add(row[4])
    for upstream_id, locale, partition, scenario, intent, utt in rows:
        group = f"massive-{upstream_id}"
        split = MASSIVE_SPLITS[partition] or assign_split(group)
        yield {
            "id": f"massive-{upstream_id}-{locale}",
            "group": group,
            "family": "classify",
            "kind": "choice",
            "lang": locale.split("-")[0],
            "question": "Which intent does the user's message express?",
            "options": [
                {"id": name.replace("_", "-"), "text": name.replace("_", " ")}
                for name in sorted(intents[scenario])
            ],
            "insufficient_option": None,
            "context": {
                "objective": "Route a voice-assistant request to the intent that handles it.",
                "evidence": [{"ref": "e1", "source": "claim", "text": utt}],
                "pending": [],
                "facts": {"scenario": scenario},
            },
            "answer": intent.replace("_", "-"),
            "rationale": None,
            "generator": "massive",
            "parent": None,
            "split": split,
            "licence": LICENCE,
            "attribution": MASSIVE_ATTRIBUTION,
            "meta": {
                "converter_version": CONVERTER_VERSION,
                "dataset": "MASSIVE 1.1",
                "locale": locale,
                "upstream_id": upstream_id,
                "upstream_partition": partition,
            },
        }


def when2call_records(paths):
    for path in paths:
        name = Path(path).name
        with open(path, encoding="utf-8") as handle:
            for record in _jsonl(handle):
                evidence = [
                    {"ref": f"t{n}", "source": "confirmed", "text": f"Available tool: {tool}"}
                    for n, tool in enumerate(record["tools"], 1)
                ]
                meta = {"converter_version": CONVERTER_VERSION, "dataset": "When2Call", "upstream_file": name}
                if "answers" in record:  # multiple-choice evaluation record
                    group = f"when2call-mcq-{record['uuid']}"
                    record_id, split = group, "external_test"
                    turns = [{"role": "user", "content": record["question"]}]
                    options = [
                        {"id": option_id, "text": record["answers"][key]}
                        for key, option_id in WHEN2CALL_ANSWERS
                        if key in record["answers"]
                    ]
                    answer = dict(WHEN2CALL_ANSWERS)[record["correct_answer"]]
                    meta.update(upstream_id=record["uuid"], upstream_source=record.get("source"))
                else:  # preference training record
                    turns = record["messages"]
                    # One conversation may have several rejected replies: one group.
                    group = "when2call-pref-" + sha256_hex(canonical(turns))[:32]
                    key = sha256_hex(canonical(record))[:32]
                    record_id, split = f"when2call-pref-{key}", assign_split(group)
                    replies = [
                        ("chosen", record["chosen_response"]["content"]),
                        ("rejected", record["rejected_response"]["content"]),
                    ]
                    if int(key[0], 16) % 2:
                        replies.reverse()
                    options = [{"id": f"reply-{'ab'[n]}", "text": text} for n, (_, text) in enumerate(replies)]
                    answer = options[[label for label, _ in replies].index("chosen")]["id"]
                evidence += [
                    {"ref": f"m{n}", "source": "claim", "text": f"{turn['role']}: {turn.get('content') or ''}"}
                    for n, turn in enumerate(turns, 1)
                ]
                yield {
                    "id": record_id,
                    "group": group,
                    "family": "next_tool",
                    "kind": "choice",
                    "lang": "en",
                    "question": "Which reply should the assistant give next?",
                    "options": options,
                    "insufficient_option": None,
                    "context": {
                        "objective": "Use a listed tool only when it fits and its inputs are known.",
                        "evidence": evidence,
                        "pending": [],
                        "facts": {},
                    },
                    "answer": answer,
                    "rationale": None,
                    "generator": "when2call",
                    "parent": None,
                    "split": split,
                    "licence": LICENCE,
                    "attribution": WHEN2CALL_ATTRIBUTION,
                    "meta": meta,
                }


def convert(layout: str, inputs, sink=None, locales=MASSIVE_LOCALES) -> dict:
    """Convert, check and (optionally) write; return counts and digests."""
    records = massive_records(inputs, locales) if layout == "massive" else when2call_records(inputs)
    checker, seen = Checker(sink), set()
    skipped = duplicates = 0
    for record in records:
        try:
            validate(record)
        except RecordError:
            skipped += 1  # over a size limit or not a decision (e.g. one option)
            continue
        key = content_key(record)
        if key in seen:
            duplicates += 1  # identical content already emitted: keep the first only
            continue
        seen.add(key)
        checker.add(record)
    return {**checker.summary(), "skipped_over_limits": skipped, "dropped_duplicates": duplicates}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert public datasets to decision records.")
    parser.add_argument("layout", choices=("massive", "when2call"))
    parser.add_argument("--input", nargs="+", required=True, help="downloaded upstream file(s)")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--locales", default=",".join(MASSIVE_LOCALES), help="MASSIVE only")
    args = parser.parse_args(argv)
    locales = tuple(part for part in args.locales.split(",") if part)
    with open(args.out, "wb") as sink:
        summary = convert(args.layout, args.input, sink, locales)
    print(canonical(summary).decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
