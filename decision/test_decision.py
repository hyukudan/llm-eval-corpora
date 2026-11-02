#!/usr/bin/env python3
"""Self-checks for the split rule and the robustness set (standard library unittest).

Run from the repository root:
    python -m unittest discover -s decision -p "test_*.py"
or directly:
    python decision/test_decision.py
"""

from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import robustness  # noqa: E402
import splits  # noqa: E402
from splits import Checker, RecordError, assign_split  # noqa: E402

TEST_GROUP = next(f"g-{n}" for n in range(1000) if assign_split(f"g-{n}") == "test")
MANIFEST = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
SYNTHETIC, AUTHORED = robustness.base_sets(MANIFEST)


def record(group="g-1", family="classify", kind="choice", split=None, time_block=None, **changes) -> dict:
    """A small valid record; its split follows the rule unless given."""
    if kind == "yes_no":
        options = [{"id": "yes", "text": "Yes."}, {"id": "no", "text": "No."}]
        answer = "yes"
    elif kind == "score":
        options = [{"id": "s0", "text": "Low.", "level": 0}, {"id": "s1", "text": "High.", "level": 1}]
        answer = "s1"
    else:
        options = [{"id": "a", "text": "First."}, {"id": "b", "text": "Second."}, {"id": "c", "text": "Unknown."}]
        answer = "b"
    meta = {} if time_block is None else {"time_block": time_block}
    base = {
        "id": f"{group}-{family}-{kind}-{time_block}", "group": group, "family": family, "kind": kind,
        "lang": "en", "question": f"Question for {group} {family} {time_block}?", "options": options,
        "insufficient_option": "c" if kind == "choice" else None,
        "context": {"objective": "Decide.", "evidence": [
            {"ref": "e1", "source": "claim", "text": "A first statement."},
            {"ref": "e2", "source": "receipt", "text": "A second statement."},
        ], "pending": [], "facts": {}},
        "answer": answer, "rationale": None, "generator": "test", "parent": None,
        "split": split or assign_split(group, family, time_block), "licence": "CC-BY-4.0",
        "attribution": "test", "meta": meta,
    }
    base.update(changes)
    return base


class SplitRuleV1(unittest.TestCase):
    def test_known_groups(self):
        known = {
            "syn-budget-gate-20261029-0": "train",
            "syn-severity-score-20261029-5": "dev",
            "syn-entailment-20261029-3": "calibration",
            "syn-tool-select-20261029-6": "test",
            "authored-341c2719750c14d054324a462ad4c74b": "test",
            "authored-a466e0e11064e7edb715d3818d0748b8": "dev",
        }
        for group, split in known.items():
            self.assertEqual(assign_split(group), split, group)
            self.assertEqual(assign_split(group, "route"), split, group)  # family alone changes nothing

    def test_stored_files_follow_v1(self):
        for item in SYNTHETIC + AUTHORED:
            self.assertEqual(assign_split(item["group"]), item["split"], item["id"])

    def test_summary_keeps_manifest_keys(self):
        checker = Checker()
        for item in AUTHORED:
            checker.add(item)
        summary = checker.summary()
        for key in ("count", "content_sha256", "splits", "families", "langs", "family_lang", "generators"):
            self.assertEqual(summary[key], MANIFEST["authored"][key], key)
        audit = summary["family_splits"]
        self.assertEqual(sum(sum(v.values()) for v in audit.values()), MANIFEST["authored"]["count"])
        self.assertEqual({f: sum(v.values()) for f, v in audit.items()}, MANIFEST["authored"]["families"])


class TimeBlockRule(unittest.TestCase):
    def test_same_unit_same_split(self):
        split = assign_split("x", "route", "2026-03")
        for group in ("a", "b", "c", "another group"):
            self.assertEqual(assign_split(group, "route", "2026-03"), split)
        checker = Checker()
        for group in ("a", "b", "c"):
            checker.add(record(group=group, family="route", time_block="2026-03"))
        self.assertEqual(checker.summary()["family_splits"]["route"], {split: 3})

    def test_formula(self):
        key = splits.canonical(["decision-split-v1-timeblock", splits.SALT, "entail", "2026-01-31"])
        bucket = int.from_bytes(splits.hashlib.sha256(key).digest()[:8], "big") % 100
        expected = "train" if bucket < 70 else "dev" if bucket < 80 else "calibration" if bucket < 90 else "test"
        self.assertEqual(assign_split("ignored", "entail", "2026-01-31"), expected)

    def test_units_spread_over_splits(self):
        months = [f"20{y:02d}-{m:02d}" for y in range(20, 30) for m in range(1, 13)]
        self.assertEqual({assign_split("g", "classify", tb) for tb in months}, set(splits.HASHED_SPLITS))

    def test_group_straddling_units_refused(self):
        for second in (
            record(group="g", family="route", time_block="2026-04", id="r2"),  # another time block
            record(group="g", family="entail", time_block="2026-03", id="r2"),  # another family
            record(group="g", family="route", id="r2"),  # no time block
        ):
            checker = Checker()
            checker.add(record(group="g", family="route", time_block="2026-03", id="r1"))
            with self.assertRaises(RecordError):
                checker.add(second)

    def test_v1_group_may_mix_families(self):
        checker = Checker()
        checker.add(record(group="g", family="route", id="r1"))
        checker.add(record(group="g", family="entail", id="r2"))

    def test_bad_format_refused(self):
        for bad in ("2026", "2026-13", "2026-00", "2026-02-30", "2026-1-05", "26-01", "2026-01-01T00",
                    " 2026-01", "2026-01\n", "２０２６-01", 202601, ""):
            with self.assertRaises(RecordError, msg=repr(bad)):
                assign_split("g", "route", bad)
        with self.assertRaises(RecordError):
            Checker().add(record(time_block="2026-02-30", split="test"))
        for good in ("2026-01", "2024-02-29", "1999-12"):
            self.assertIn(assign_split("g", "route", good), splits.HASHED_SPLITS)

    def test_split_must_follow_time_block_rule(self):
        item = record(group="g", family="route", time_block="2026-03")
        wrong = next(s for s in splits.HASHED_SPLITS if s != item["split"])
        with self.assertRaises(RecordError):
            Checker().add({**item, "split": wrong})


class Robustness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data, cls.summary = robustness.build_default(MANIFEST)
        cls.records = [json.loads(line) for line in cls.data.splitlines()]
        cls.sources = {r["id"]: r for r in SYNTHETIC + AUTHORED}

    def answer_text(self, item):
        return {o["id"]: o["text"] for o in item["options"]}[item["answer"]]

    def test_matches_stored_file_and_manifest(self):
        self.assertEqual((HERE / robustness.FILE).read_bytes(), self.data)
        entry = MANIFEST["robustness"]
        self.assertEqual(entry["count"], len(self.records))
        self.assertEqual(entry["by_perturbation"], self.summary["by_perturbation"])

    def test_deterministic(self):
        self.assertEqual(robustness.build_default(MANIFEST)[0], self.data)

    def test_only_test_sources(self):
        ids = set()
        for item in self.records:
            source = self.sources[item["parent"]]
            self.assertEqual(source["split"], "test")
            self.assertEqual(item["split"], "test")
            if source["generator"] == "writer-luna":
                self.assertEqual(source["meta"]["label"]["provenance"], "teacher_agreed")
            self.assertEqual(item["id"], f"{source['id']}~{item['meta']['perturbation']}")
            self.assertEqual(item["meta"]["source_meta"], source["meta"])
            for key in ("group", "family", "kind", "lang"):
                self.assertEqual(item[key], source[key])
            splits.validate(item)
            ids.add(item["id"])
        self.assertEqual(len(ids), len(self.records))
        with self.assertRaises(RecordError):
            robustness.perturb(next(r for r in SYNTHETIC if r["split"] == "train"), True)

    def test_answer_text_preserved_and_content_changed(self):
        for item in self.records:
            source = self.sources[item["parent"]]
            self.assertEqual(self.answer_text(item), self.answer_text(source), item["id"])
            self.assertNotEqual(robustness.presented(item), robustness.presented(source), item["id"])

    def test_neutral_ids_remap(self):
        seen = 0
        for item in self.records:
            if item["meta"]["perturbation"] != "neutral_ids":
                continue
            source = self.sources[item["parent"]]
            self.assertEqual([o["id"] for o in item["options"]], [f"opt-{n}" for n in range(1, len(source["options"]) + 1)])
            self.assertEqual([o["text"] for o in item["options"]], [o["text"] for o in source["options"]])
            if source["insufficient_option"] is None:
                self.assertIsNone(item["insufficient_option"])
            else:
                old = {o["id"]: o["text"] for o in source["options"]}[source["insufficient_option"]]
                new = {o["id"]: o["text"] for o in item["options"]}[item["insufficient_option"]]
                self.assertEqual(old, new)
            seen += 1
        self.assertGreater(seen, 0)

    def test_order_and_id_perturbations_skip_yes_no_and_score(self):
        for item in self.records:
            if item["meta"]["perturbation"] in ("option_reverse", "neutral_ids"):
                self.assertEqual(item["kind"], "choice")
        for kind in ("yes_no", "score"):
            names = [r["meta"]["perturbation"] for r in robustness.perturb(record(kind=kind, split="test"), True)]
            self.assertEqual(names, ["distractor_evidence", "evidence_reverse"])

    def test_choice_perturbations(self):
        source = record(split="test")
        out = {r["meta"]["perturbation"]: r for r in robustness.perturb(source, True)}
        self.assertEqual(list(out), list(robustness.PERTURBATIONS))
        self.assertEqual([o["id"] for o in out["option_reverse"]["options"]], ["c", "b", "a"])
        self.assertEqual(out["option_reverse"]["answer"], "b")
        self.assertEqual(out["neutral_ids"]["answer"], "opt-2")
        self.assertEqual(out["neutral_ids"]["insufficient_option"], "opt-3")
        added = out["distractor_evidence"]["context"]["evidence"][-1]
        self.assertEqual((added["ref"], added["source"]), ("e3", "confirmed"))
        self.assertIn(added["text"], robustness.DISTRACTORS["en"])
        self.assertEqual([e["ref"] for e in out["evidence_reverse"]["context"]["evidence"]], ["e2", "e1"])
        self.assertNotIn("evidence_reverse", [r["meta"]["perturbation"] for r in robustness.perturb(source, False)])

    def test_evidence_reverse_synthetic_only(self):
        for item in self.records:
            if item["meta"]["perturbation"] == "evidence_reverse":
                self.assertNotEqual(item["generator"], "writer-luna")

    def test_full_evidence_skips_distractor(self):
        source = record(split="test")
        source["context"]["evidence"] = [
            {"ref": f"e{n}", "source": "claim", "text": "Statement."} for n in range(1, 65)
        ]
        names = [r["meta"]["perturbation"] for r in robustness.perturb(source, True)]
        self.assertNotIn("distractor_evidence", names)

    def test_distractor_pools(self):
        for lang in ("en", "es", "fr", "de", "it", "pt", "ca"):
            pool = robustness.DISTRACTORS[lang]
            self.assertEqual(len(pool), len(set(pool)))
            self.assertFalse(any(ch.isdigit() for text in pool for ch in text), lang)
        self.assertEqual(robustness.distractor_text("x", "es"), robustness.distractor_text("x", "es"))

    def test_collision_with_other_split_refused(self):
        source = record(group=TEST_GROUP)
        clash = copy.deepcopy(robustness.perturb(source, True)[2])  # distractor_evidence
        clash.update(id="elsewhere", group="elsewhere", split=assign_split("elsewhere"), parent=None, meta={})
        self.assertNotEqual(clash["split"], "test")
        with self.assertRaisesRegex(RecordError, "non-test"):
            robustness.build([([source], True)], [source, clash])
        data, summary = robustness.build([([source], True)], [source])
        self.assertEqual(summary["count"], 4)

    def test_public_out_inside_data_refused(self):
        with self.assertRaises(SystemExit):
            robustness.main(["--public", str(HERE / "data" / "x.jsonl"), "--out", str(HERE / "data" / "y.jsonl")])


if __name__ == "__main__":
    unittest.main()
