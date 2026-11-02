#!/usr/bin/env python3
"""Self-checks for the split rule and the robustness set (standard library unittest).

Run from the repository root:
    python -m unittest discover -s decision -p "test_*.py"
or directly:
    python decision/test_decision.py
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import freeze  # noqa: E402
import gen_synthetic  # noqa: E402
import robustness  # noqa: E402
import verify  # noqa: E402
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


def near_limit(group: str, record_id: str) -> dict:
    """A valid choice record whose option_reverse variant is 5 bytes under the size limit.

    Its neutral_ids variant (longer option ids, shorter perturbation name) is over it.
    """
    source = record(group=group, split="test", id=record_id, question=f"Near-limit question {record_id}?")
    source["context"]["evidence"] = [
        {"ref": f"e{n}", "source": "claim", "text": "x" * 4000} for n in range(1, 8)
    ] + [{"ref": "e8", "source": "claim", "text": "y" * 100}]

    def variant_size():
        return len(splits.canonical(robustness.option_reverse(source)))

    source["context"]["evidence"][-1]["text"] = "y" * (100 + splits.MAX_RECORD_BYTES - 5 - variant_size())
    assert variant_size() == splits.MAX_RECORD_BYTES - 5
    return splits.validate(source)


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
                    " 2026-01", "2026-01\n", "２０２６-01", 202601, "", "0000-01", "0000-01-01", "0000-12"):
            with self.assertRaises(RecordError, msg=repr(bad)):
                assign_split("g", "route", bad)
        with self.assertRaises(RecordError):
            Checker().add(record(time_block="2026-02-30", split="test"))
        for good in ("2026-01", "2024-02-29", "1999-12", "0001-01", "9999-12", "9999-12-31"):
            self.assertIn(assign_split("g", "route", good), splits.HASHED_SPLITS)

    def test_present_non_string_time_block_refused(self):
        for bad in (None, 202601, ["2026-01"], {"block": "2026-01"}, "0000-01"):
            item = record(split="test", meta={"time_block": bad})
            with self.assertRaises(RecordError, msg=repr(bad)):
                splits.time_block_of(item)
            with self.assertRaises(RecordError, msg=repr(bad)):
                Checker().add(item)
        self.assertIsNone(splits.time_block_of(record(meta={})))  # only an absent key means none
        self.assertIsNone(splits.time_block_of(record(meta="not an object")))

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
            self.assertEqual(robustness.perturb(record(kind=kind, split="test"), False), [])

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
        names = [r["meta"]["perturbation"] for r in robustness.perturb(source, False)]
        self.assertEqual(names, ["option_reverse", "neutral_ids"])  # open-domain: no evidence perturbations

    def test_evidence_perturbations_synthetic_only(self):
        for item in self.records:
            if item["meta"]["perturbation"] in ("distractor_evidence", "evidence_reverse"):
                self.assertIn(item["generator"], gen_synthetic.GENERATORS, item["id"])
            else:
                self.assertIn(item["meta"]["perturbation"], ("option_reverse", "neutral_ids"))
        authored = [r for r in self.records if r["generator"] == "writer-luna"]
        self.assertTrue(authored)
        self.assertEqual({r["meta"]["perturbation"] for r in authored}, {"option_reverse", "neutral_ids"})

    def test_distractors_avoid_generator_vocabulary(self):
        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for key, item in value.items():
                    yield from strings(key)
                    yield from strings(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    yield from strings(item)

        def words(text):
            return {w.lower() for w in re.findall(r"[^\W\d_]{4,}", text)}

        vocabulary = set()
        for name, value in vars(gen_synthetic).items():
            if name.isupper():  # every text table, topic, entity, tool and value list
                for text in strings(value):
                    vocabulary |= words(text)
        function_words = {"with"}
        for lang in gen_synthetic.LANGUAGES:
            for sentence in robustness.DISTRACTORS[lang]:
                self.assertLessEqual(words(sentence) & vocabulary, function_words, sentence)
        self.assertIn("weather", vocabulary)  # the check sees the tool-select requests

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

    def test_near_limit_variant_skipped_and_counted(self):
        source = near_limit(TEST_GROUP, "near")
        data, summary = robustness.build([([source], False)], [source])
        self.assertEqual(summary["count"], 1)
        self.assertEqual(summary["by_perturbation"], {"option_reverse": 1})
        self.assertEqual(summary["skipped_over_limits"], 1)
        (kept,) = [json.loads(line) for line in data.splitlines()]
        self.assertEqual(len(splits.canonical(kept)), splits.MAX_RECORD_BYTES - 5)
        self.assertEqual(robustness.build([([source], False)], [source])[0], data)

    def test_public_out_inside_data_refused(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            robustness.main(["--public", str(HERE / "data" / "x.jsonl"), "--out", str(HERE / "data" / "y.jsonl")])


def run_quiet(function, argv) -> tuple[int, str]:
    """Exit status and stdout of a script's main()."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        status = function(argv)
    return status, buffer.getvalue()


class VerifyAndFreeze(unittest.TestCase):
    """verify.main and freeze.main on a temporary copy of this directory's data."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.here = root / "decision"
        (cls.here / "data").mkdir(parents=True)
        for name in ("synthetic-v1.jsonl", "authored-v1.jsonl", "authored-v1-unlabelled.jsonl", "robustness-v1.jsonl"):
            shutil.copyfile(HERE / "data" / name, cls.here / "data" / name)
        shutil.copyfile(HERE / "manifest.json", cls.here / "manifest.json")
        cls.patches = [
            mock.patch.object(verify, "HERE", cls.here),
            mock.patch.object(robustness, "HERE", cls.here),
            mock.patch.object(freeze, "HERE", cls.here),
            mock.patch.object(freeze, "MANIFEST", cls.here / "manifest.json"),
        ]
        for patch in cls.patches:
            patch.start()
        # A stand-in for converted public records: synthetic test records with new ids and text.
        cls.public_in = root / "public" / "converted.jsonl"
        cls.public_in.parent.mkdir()
        lines = []
        for item in SYNTHETIC:
            if item["split"] == "test" and item["parent"] is None:
                item = {**item, "id": "pub-" + item["id"], "generator": "massive", "question": item["question"] + " (pub)"}
                lines.append(splits.canonical(item) + b"\n")
        lines.append(splits.canonical(near_limit(TEST_GROUP, "pub-near-limit")) + b"\n")
        cls.public_in.write_bytes(b"".join(lines))
        cls.public_out = root / "out" / "robustness-public.jsonl"
        status, _ = run_quiet(freeze.main, ["--robustness-public", str(cls.public_in),
                                            "--robustness-public-out", str(cls.public_out)])
        assert status == 0
        cls.manifest = (cls.here / "manifest.json").read_bytes()
        cls.robust = (cls.here / "data" / "robustness-v1.jsonl").read_bytes()
        cls.public = cls.public_out.read_bytes()

    @classmethod
    def tearDownClass(cls):
        for patch in cls.patches:
            patch.stop()
        cls.tmp.cleanup()

    def setUp(self):
        (self.here / "manifest.json").write_bytes(self.manifest)
        (self.here / "data" / "robustness-v1.jsonl").write_bytes(self.robust)
        self.public_out.write_bytes(self.public)

    def set_manifest(self, change):
        manifest = json.loads(self.manifest)
        change(manifest)
        (self.here / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def assert_mismatch(self, argv, text):
        status, out = run_quiet(verify.main, argv)
        self.assertEqual(status, 1, out)
        self.assertIn(text, out)

    def test_freeze_adds_only_the_public_entry(self):
        before = json.loads((HERE / "manifest.json").read_bytes())
        after = json.loads(self.manifest)
        public = after["robustness"].pop("public")
        self.assertEqual(after, before)
        self.assertEqual(public["file_sha256"], splits.sha256_hex(self.public))
        self.assertEqual(public["skipped_over_limits"], 1)  # the near-limit neutral_ids variant
        self.assertEqual(set(public["by_perturbation"]), {"option_reverse", "neutral_ids"})
        self.assertEqual(self.robust, (HERE / "data" / "robustness-v1.jsonl").read_bytes())

    def test_freeze_keeps_recorded_public_entry(self):
        status, _ = run_quiet(freeze.main, [])
        self.assertEqual(status, 0)
        self.assertEqual((self.here / "manifest.json").read_bytes(), self.manifest)

    def test_ok(self):
        public = ["--robustness-public", str(self.public_in), "--robustness-public-out", str(self.public_out)]
        for argv in ([], ["--robustness-public-out", str(self.public_out)], public):
            status, out = run_quiet(verify.main, argv)
            self.assertEqual((status, out.strip()), (0, "OK"), argv)

    def test_missing_or_empty_robustness_entry(self):
        self.set_manifest(lambda m: m.pop("robustness"))
        self.assert_mismatch([], "robustness: manifest entry missing or empty")
        self.set_manifest(lambda m: m.update(robustness={}))
        self.assert_mismatch([], "robustness: manifest entry missing or empty")

    def test_stored_robustness_file_required_and_checked(self):
        (self.here / "data" / "robustness-v1.jsonl").unlink()
        self.assert_mismatch([], "missing data/robustness-v1.jsonl")
        (self.here / "data" / "robustness-v1.jsonl").write_bytes(self.robust[:-1])
        self.assert_mismatch([], "robustness.file_sha256 (stored)")

    def test_public_output_required(self):
        self.public_out.unlink()
        self.assert_mismatch(["--robustness-public-out", str(self.public_out)], "missing output file")
        self.assert_mismatch(["--robustness-public", str(self.public_in), "--robustness-public-out",
                              str(self.public_out)], "missing output file")

    def test_public_output_edited(self):
        lines = self.public.splitlines(keepends=True)
        for edited in (b"".join(lines[:-1]), self.public.replace(b"(pub)", b"(PUB)", 1)):
            self.public_out.write_bytes(edited)
            self.assert_mismatch(["--robustness-public-out", str(self.public_out)], "robustness.public.file_sha256 (stored)")
            self.assert_mismatch(["--robustness-public", str(self.public_in), "--robustness-public-out",
                                  str(self.public_out)], "stored file differs from the rebuilt bytes")
        self.public_out.write_bytes(b"".join(lines[:-1]))
        self.assert_mismatch(["--robustness-public-out", str(self.public_out)], "robustness.public.count (stored)")

    def test_public_needs_output_argument_and_entry(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            verify.main(["--robustness-public", str(self.public_in)])
        self.set_manifest(lambda m: m["robustness"].pop("public"))
        self.assert_mismatch(["--robustness-public-out", str(self.public_out)], "no recorded public entry")


if __name__ == "__main__":
    unittest.main()
