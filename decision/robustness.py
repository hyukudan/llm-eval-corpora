#!/usr/bin/env python3
"""Robustness set: label-preserving perturbations of test-split decision cases.

Usage:
    python robustness.py [--out FILE]                      synthetic + authored (default: print the summary)
    python robustness.py --public FILE.jsonl [...] --out PATH   converted MASSIVE / When2Call records

Only ``test`` records are perturbed: the stored synthetic test split and the
authored test records whose ``meta.label.provenance`` is ``teacher_agreed``
(or, with ``--public``, the ``test`` records of converted public files, whose
output must be written outside ``data/``). Each source yields, in this order:

1. ``option_reverse``: the option list reversed (``choice`` kind only).
2. ``neutral_ids``: option ids renamed ``opt-1``..``opt-n`` in list order,
   with ``answer`` and ``insufficient_option`` remapped (``choice`` only).
3. ``distractor_evidence``: one irrelevant ``confirmed`` evidence item with a
   fresh ref, from a fixed per-language pool, chosen by a salted sha256 of the
   record id (skipped when the evidence already has 64 items).
4. ``evidence_reverse``: the evidence list reversed (at least two items).

Perturbations 3 and 4 touch the evidence, so they apply to synthetic records
only: their generators compute the answer from specific facts that a neutral
fact cannot bear on, and their evidence items are unordered. Authored and
public requests are open-domain (a "neutral" fact could be what the question
asks about) and their evidence can be ordered conversation turns, so they get
only perturbations 1 and 2.

A variant whose canonical form would exceed the 32 KiB record limit (a source
near the limit whose new option ids are longer, for example) is skipped and
counted in ``skipped_over_limits``; the build never aborts on it.

The answer always names the option with the same text. Output records keep
the source's group, family, kind and language, have split ``test``, id
``<source id>~<perturbation>`` and ``parent`` = the source id. Output is
canonical JSONL in source order, then perturbation order. Standard library
only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from splits import (
    MAX_EVIDENCE, MAX_RECORD_BYTES, SALT, Checker, RecordError, canonical, content_key, sha256_hex, time_block_of,
)

HERE = Path(__file__).resolve().parent
VERSION = "decision-robustness-v1"
FILE = "data/robustness-v1.jsonl"
LICENCE = "CC-BY-4.0"
PERTURBATIONS = ("option_reverse", "neutral_ids", "distractor_evidence", "evidence_reverse")

# Neutral facts for synthetic records. They avoid every subject the generators
# ask about (budgets, deadlines and times, tests and checks, candidates and
# changelogs, support topics, parcels, servers, reports, invoices and their
# status or owning team, tools, weather, translation, bookings) and contain no
# numbers, dates, colours, people or money. test_decision.py checks the
# overlap with the generators' vocabulary.
DISTRACTORS = {
    "en": (
        "The coffee machine in the staff kitchen has been descaled.",
        "A new doormat was placed at the main entrance.",
        "The hallway light bulbs were replaced with energy-saving ones.",
        "The shelves in the staff kitchen were tidied and wiped down.",
        "The stairwell handrails were repainted.",
        "The windows of the neighbouring building were cleaned.",
    ),
    "es": (
        "La cafetera de la cocina del personal ha sido descalcificada.",
        "Se ha colocado un felpudo nuevo en la entrada principal.",
        "Las bombillas del pasillo se cambiaron por otras de bajo consumo.",
        "Las estanterías de la cocina del personal se ordenaron y limpiaron.",
        "Se repintaron los pasamanos de la escalera.",
        "Se limpiaron las ventanas del edificio vecino.",
    ),
    "fr": (
        "La machine à café de la cuisine du personnel a été détartrée.",
        "Un nouveau paillasson a été posé à l'entrée principale.",
        "Les ampoules du couloir ont été remplacées par des modèles basse consommation.",
        "Les étagères de la cuisine du personnel ont été rangées et nettoyées.",
        "Les rampes de l'escalier ont été repeintes.",
        "Les fenêtres de l'immeuble voisin ont été nettoyées.",
    ),
    "de": (
        "Die Kaffeemaschine in der Teeküche wurde entkalkt.",
        "Am Haupteingang wurde eine neue Fußmatte ausgelegt.",
        "Die Glühbirnen im Flur wurden durch Energiesparlampen ersetzt.",
        "Die Regale in der Teeküche wurden aufgeräumt und abgewischt.",
        "Die Handläufe im Treppenhaus wurden neu gestrichen.",
        "Die Fenster des Nachbargebäudes wurden geputzt.",
    ),
    "it": (
        "La macchina del caffè della cucina del personale è stata decalcificata.",
        "All'ingresso principale è stato messo un nuovo zerbino.",
        "Le lampadine del corridoio sono state sostituite con modelli a basso consumo.",
        "Gli scaffali della cucina del personale sono stati riordinati e puliti.",
        "I corrimano delle scale sono stati ridipinti.",
        "Le finestre dell'edificio vicino sono state pulite.",
    ),
    "pt": (
        "A máquina de café da copa dos funcionários foi descalcificada.",
        "Foi colocado um tapete novo na entrada principal.",
        "As lâmpadas do corredor foram trocadas por outras de baixo consumo.",
        "As prateleiras da copa dos funcionários foram arrumadas e limpas.",
        "Os corrimãos da escada foram repintados.",
        "As janelas do edifício vizinho foram limpas.",
    ),
    "ca": (
        "La cafetera de la cuina del personal s'ha descalcificat.",
        "S'ha posat una estora nova a l'entrada principal.",
        "Les bombetes del passadís s'han canviat per unes de baix consum.",
        "Els prestatges de la cuina del personal s'han endreçat i netejat.",
        "S'han repintat les baranes de l'escala.",
        "S'han netejat les finestres de l'edifici veí.",
    ),
}


def load_jsonl(path) -> list[dict]:
    """Records of a JSONL file (gzip-compressed when the name ends in .gz)."""
    data = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.splitlines() if line.strip()]


def presented(record: dict) -> bytes:
    """What a model is shown, in order: question, options, insufficient option, context."""
    return canonical([record["question"], record["options"], record["insufficient_option"], record["context"]])


def _derive(source: dict, name: str, **changes) -> dict:
    meta = {
        "perturbation": name,
        "robustness_version": VERSION,
        "source_id": source["id"],
        "source_meta": source["meta"],
    }
    time_block = time_block_of(source)
    if time_block is not None:
        meta["time_block"] = time_block  # keeps the split unit of the source
    record = {**source, "id": f"{source['id']}~{name}", "parent": source["id"], "split": "test", "meta": meta}
    record.update(changes)
    return record


def option_reverse(source: dict):
    if source["kind"] != "choice":
        return None
    return _derive(source, "option_reverse", options=list(reversed(source["options"])))


def neutral_ids(source: dict):
    if source["kind"] != "choice":
        return None
    mapping = {option["id"]: f"opt-{n}" for n, option in enumerate(source["options"], 1)}
    if all(old == new for old, new in mapping.items()):
        return None  # already neutral: nothing would change
    options = [{**option, "id": mapping[option["id"]]} for option in source["options"]]
    insufficient = source["insufficient_option"]
    return _derive(
        source, "neutral_ids", options=options, answer=mapping[source["answer"]],
        insufficient_option=None if insufficient is None else mapping[insufficient],
    )


def distractor_text(record_id: str, lang: str) -> str:
    """The neutral fact for a record, fixed by a salted sha256 of its id."""
    pool = DISTRACTORS[lang]
    digest = hashlib.sha256(canonical([VERSION, SALT, record_id])).digest()
    return pool[int.from_bytes(digest[:8], "big") % len(pool)]


def distractor_evidence(source: dict):
    evidence = source["context"]["evidence"]
    if len(evidence) >= MAX_EVIDENCE:
        return None
    if source["lang"] not in DISTRACTORS:
        raise RecordError(f"no distractor pool for language {source['lang']}: {source['id']}")
    refs = {item["ref"] for item in evidence}
    n = len(evidence) + 1
    while f"e{n}" in refs:
        n += 1
    item = {"ref": f"e{n}", "source": "confirmed", "text": distractor_text(source["id"], source["lang"])}
    context = {**source["context"], "evidence": [*evidence, item]}
    return _derive(source, "distractor_evidence", context=context)


def evidence_reverse(source: dict):
    evidence = source["context"]["evidence"]
    if len(evidence) < 2:
        return None
    context = {**source["context"], "evidence": list(reversed(evidence))}
    return _derive(source, "evidence_reverse", context=context)


def perturb(source: dict, synthetic: bool) -> list[dict]:
    """The perturbations of one test record, in PERTURBATIONS order.

    Evidence perturbations (distractor, reversal) only when ``synthetic``.
    """
    if source["split"] != "test":
        raise RecordError(f"only test records are perturbed: {source['id']}")
    makers = [option_reverse, neutral_ids]
    if synthetic:
        makers += [distractor_evidence, evidence_reverse]
    out = []
    for make in makers:
        record = make(source)
        if record is None:
            continue
        if presented(record) == presented(source):
            raise RecordError(f"perturbation changes nothing: {record['id']}")
        answer_text = {o["id"]: o["text"] for o in record["options"]}[record["answer"]]
        if answer_text != {o["id"]: o["text"] for o in source["options"]}[source["answer"]]:
            raise RecordError(f"perturbation changes the answer: {record['id']}")
        out.append(record)
    return out


def build(sources, others) -> tuple[bytes, dict]:
    """Canonical JSONL bytes and a summary.

    ``sources`` is a list of (records, synthetic); only their test records
    are perturbed. ``others`` are all base records: no output may share the
    content of one outside the test split. A variant over the record size
    limit is skipped and counted in ``skipped_over_limits``.
    """
    foreign = {content_key(r) for r in others if r["split"] != "test"}
    checker = Checker()  # ids, parent/group/split and the split rule
    tests = [(r, synthetic) for records, synthetic in sources for r in records if r["split"] == "test"]
    for record, _ in tests:
        checker.add(record)
    lines, by_perturbation, by_family, skipped = [], Counter(), Counter(), 0
    for source, synthetic in tests:
        for record in perturb(source, synthetic):
            if len(canonical(record)) > MAX_RECORD_BYTES:
                skipped += 1
                continue
            checker.add(record)
            if content_key(record) in foreign:
                raise RecordError(f"content of a non-test record: {record['id']}")
            lines.append(canonical(record) + b"\n")
            by_perturbation[record["meta"]["perturbation"]] += 1
            by_family[record["family"]] += 1
    data = b"".join(lines)
    summary = {
        "count": len(lines),
        "content_sha256": sha256_hex(data),
        "by_perturbation": {name: by_perturbation[name] for name in PERTURBATIONS if by_perturbation[name]},
        "by_family": dict(sorted(by_family.items())),
        "skipped_over_limits": skipped,
    }
    return data, summary


def agreed(records) -> list[dict]:
    """Authored records whose label two independent teachers agree on."""
    return [r for r in records if r["meta"].get("label", {}).get("provenance") == "teacher_agreed"]


def base_sets(manifest: dict) -> tuple[list[dict], list[dict]]:
    """Stored synthetic and authored records named by the manifest."""
    synthetic = load_jsonl(HERE / manifest["synthetic"]["file"])
    authored = load_jsonl(HERE / manifest["authored"]["file"]) if manifest.get("authored") else []
    return synthetic, authored


def build_default(manifest: dict) -> tuple[bytes, dict]:
    """The stored set: synthetic test records and agreed authored test records."""
    synthetic, authored = base_sets(manifest)
    data, summary = build([(synthetic, True), (agreed(authored), False)], synthetic + authored)
    summary["sources"] = {
        "synthetic_test_sha256": manifest["synthetic"]["splits"]["test"]["content_sha256"],
        "authored_test_sha256": manifest["authored"]["splits"]["test"]["content_sha256"] if authored else None,
    }
    return data, summary


def build_public(manifest: dict, paths) -> tuple[bytes, dict]:
    """Perturbations of the test records of converted public files."""
    synthetic, authored = base_sets(manifest)
    public = [record for path in paths for record in load_jsonl(path)]
    data, summary = build([(public, False)], synthetic + authored + public)
    summary["inputs_sha256"] = [sha256_hex(Path(path).read_bytes()) for path in paths]
    return data, summary


def inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build the decision robustness set.")
    parser.add_argument("--public", type=Path, nargs="+", help="converted public JSONL files")
    parser.add_argument("--out", type=Path, help="output JSONL file")
    args = parser.parse_args(argv)
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    if args.public:
        if args.out is None or inside(args.out, HERE / "data"):
            parser.error("--public needs an --out path outside data/ (third-party records are not stored here)")
        data, summary = build_public(manifest, args.public)
    else:
        data, summary = build_default(manifest)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(data)
    print(canonical(summary).decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
