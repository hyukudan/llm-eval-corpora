#!/usr/bin/env python3
"""Checks for v2; run with python -m unittest discover -s decision -p "test_*.py"."""

from __future__ import annotations

import contextlib
import io
import json
import random
import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import gen_synthetic_v2 as gen
from splits import Checker, assign_split, canonical, content_key, file_sha256, validate

HERE = Path(__file__).resolve().parent
FROZEN = {
    "gen_synthetic.py": "d1fe56f62145300a28621c2c1f196f0ac5d230300cfde4c4fc9df60809c91ebc",
    "manifest.json": "625c4f9523253a68cb4f82f164967705532aa85b557894e842483812ce8bd862",
    "data/authored-v1-unlabelled.jsonl": "5af5603c3a9c45e4317b606bc3ee0e3dbe832fa9b74f42bc2c394fc8b2053c72",
    "data/authored-v1.jsonl": "60ce946ab5155b2ac8754d6705e224e80e7118f14b655671bd0989e92397da48",
    "data/robustness-v1.jsonl": "bd3f71762f5a232abaa88e2646ed5146e58e5493831dd85b28695060d637756e",
    "data/synthetic-v1.jsonl": "5eaffeed930f8d5e7abd0bebe56ea0474942a23dea2366afaa17aa9a335af24a",
}


class GeneratorV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = list(gen.generate(17, 2400))

    def test_determinism_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.jsonl", Path(tmp) / "b.jsonl"
            argv = ["--seed", "21", "--count", "120", "--families", "completion,next_tool"]
            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(gen.main(argv + ["--out", str(a)]), 0)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["count"], 240)
            self.assertEqual(set(summary["families"]), {"completion", "next_tool"})
            gen.write(21, 120, b, ("completion", "next_tool"))
            self.assertEqual(a.read_bytes(), b.read_bytes())
            self.assertEqual(summary["content_sha256"], file_sha256(a))
            lines = a.read_bytes().splitlines(keepends=True)
            self.assertTrue(all(line == canonical(json.loads(line)) + b"\n" for line in lines))
            gen.write(22, 120, b, ("completion", "next_tool"))
            self.assertNotEqual(a.read_bytes(), b.read_bytes())

    def test_schema_splits_and_children(self):
        checker = Checker()
        parents = {}
        for record in self.records:
            validate(record)
            checker.add(record)
            self.assertEqual(record["meta"]["generator_version"], "decision-synth-v2")
            self.assertEqual(record["split"], assign_split(record["group"]))
            self.assertTrue(record["group"].startswith(f"syn2-{record['generator']}-17-"))
            self.assertTrue(record["rationale"])
            if record["parent"] is None:
                parents[record["id"]] = record
            else:
                parent = parents[record["parent"]]
                self.assertNotEqual(record["options"], parent["options"])
                for key in ("group", "answer", "split", "context"):
                    self.assertEqual(record[key], parent[key])
                self.assertEqual(content_key(record), content_key(parent))
        self.assertEqual(len(parents), 2400)
        self.assertEqual(len({content_key(r) for r in parents.values()}), 2400)
        for record in parents.values():
            self.assertEqual(
                sum(r["parent"] == record["id"] for r in self.records),
                0 if record["kind"] == "score" else 1,
            )

    def test_tool_catalogue_and_structured_labels(self):
        self.assertGreaterEqual(len(gen.TOOLS), 30)
        for required, optional, actions in gen.TOOLS.values():
            self.assertTrue(1 <= len(required) <= 4)
            self.assertLessEqual(len(optional), 2)
            self.assertEqual(len(actions), 3)
            for param in required + optional:
                self.assertGreaterEqual(len(set(gen.VALUES[param])), 8)
        seen = Counter()
        for record in self.records:
            if record["family"] != "next_tool" or record["parent"]:
                continue
            state = record["meta"]["scenario"]
            shape, tool = state["shape"], state["tool"]
            seen[(record["lang"], shape)] += 1
            required, optional, _ = gen.TOOLS[tool]
            texts = [e["text"] for e in record["context"]["evidence"]]
            # Recompute availability and conflicts from structured turns, never
            # from wording or the generator's cached missing/ambiguous fields.
            found = {p: set() for p in required + optional}
            for turn in (state["given"], state["earlier"], state["conflicting"]):
                for param, value in turn.items():
                    self.assertIn(value, gen.parameter_values(tool, param, record["lang"]))
                    found[param].add(value)
            unresolved = [p for p in required if len(found[p]) != 1]
            self.assertEqual(state["missing"], [p for p in required if not found[p]])
            offered = [o["id"][5:] for o in record["options"] if o["id"].startswith("call_")]
            self.assertIn(len(offered), (3, 4))
            expected = "none" if tool not in offered else "ask" if unresolved else f"call_{tool}"
            self.assertEqual(record["answer"], expected)
            if shape.startswith("missing_"):
                self.assertEqual(
                    sum(not found[p] for p in required), 2 if shape == "missing_two" else 1
                )
                self.assertFalse(any("…" in text or "..." in text for text in texts))
            if shape == "ambiguous":
                self.assertEqual(len(found[state["ambiguous"]]), 2)
            if shape == "optional_missing":
                self.assertTrue(any(not found[p] for p in optional))
                self.assertEqual(expected, f"call_{tool}")
            if shape == "earlier":
                self.assertEqual(len(texts), 2)
                for param, value in state["earlier"].items():
                    self.assertNotIn(param, state["given"])
                    self.assertEqual(found[param], {value})
            if shape == "shared_params":
                self.assertTrue(
                    any(set(required) & set(gen.TOOLS[n][0]) for n in offered if n != tool)
                )
        self.assertEqual(
            set(seen), {(lang, shape) for lang in gen.LANGUAGES for shape in gen.TOOL_SHAPES}
        )
        self.assertTrue(
            all(
                40 <= sum(n for (lang, s), n in seen.items() if s == shape) <= 110
                for shape in gen.TOOL_SHAPES
            )
        )

    def test_natural_tool_requests_and_safe_options(self):
        old = re.compile(r"\bwith \w+ [^;]+;|\b(?:zone|note|depot|article|collection)-[0-9]+|\. with\b")
        for record in self.records:
            if record["family"] != "next_tool":
                continue
            for evidence in record["context"]["evidence"]:
                self.assertIsNone(old.search(evidence["text"]), evidence["text"])
            for option in record["options"]:
                for forbidden in ("\n", "\r", " ]", "Answer code:"):
                    self.assertNotIn(forbidden, option["text"])
        for tool, (required, optional, _) in gen.TOOLS.items():
            for lang in gen.LANGUAGES:
                values = {p: gen.parameter_values(tool, p, lang)[0] for p in required + optional}
                requests = set()
                for frame in range(3):
                    rng = mock.Mock()
                    rng.randrange.return_value = frame
                    requests.add(gen.tool_request(rng, lang, tool, values))
                self.assertEqual(len(requests), 3)

    def test_grade_balance(self):
        counts = Counter()
        for record in gen.generate(7001, 4000, ("grade",)):
            counts[record["answer"]] += 1
            state = record["meta"]["scenario"]
            self.assertEqual(record["answer"], f"level{min(3, len(state['failing']))}")
            self.assertLessEqual(len(state["failing"]), 5)
        self.assertEqual(set(counts), {f"level{n}" for n in range(4)})
        self.assertTrue(all(0.22 <= n / 4000 <= 0.28 for n in counts.values()))

    def test_completion_rendered_labels(self):
        seen = set()
        positives = total = 0
        for record in self.records:
            if record["family"] != "completion" or record["parent"]:
                continue
            lang = record["lang"]
            seen.add((lang, record["meta"]["scenario"]["shape"]))
            evidence = record["context"]["evidence"]
            component = evidence[0]["text"].split(":")[0].split()[-1]
            changed = re.search(r"\d+", evidence[0]["text"]).group()
            # The header names the component and timestamp of this check run.
            header = evidence[1]["text"]
            receipt_time = int(re.search(r"(\d+)\s*:", header).group(1))
            valid = (
                evidence[1]["source"] == "receipt"
                and component in header
                and receipt_time >= int(changed)
            )
            failed_word = gen.localized(lang, "failed", "fallida", "échec")
            failure = any(failed_word in e["text"] for e in evidence[2:])
            expected = (
                "missing_work"
                if failure or record["context"]["pending"]
                else "ready_to_validate"
                if valid
                else "missing_evidence"
            )
            self.assertEqual(record["answer"], expected, evidence)
            positives += expected == "ready_to_validate"
            total += 1
        self.assertEqual(seen, {(lang, s) for lang in gen.LANGUAGES for s in gen.READY_SHAPES})
        self.assertTrue(0.35 <= positives / total <= 0.45)

    def test_grade_and_review_rendered_labels(self):
        review_shapes = set()
        for record in self.records:
            if record["parent"]:
                continue
            lang = record["lang"]
            evidence = record["context"]["evidence"]
            if record["family"] == "grade":
                failed_word = gen.localized(lang, "failed", "fallida", "échec")
                warning_word = gen.localized(lang, "non-blocking", "no bloqueante", "non bloquant")
                count = sum(
                    failed_word in e["text"] and warning_word not in e["text"] for e in evidence
                )
                self.assertEqual(record["answer"], f"level{min(3, count)}")
                self.assertEqual([o["level"] for o in record["options"]], list(range(4)))
                self.assertIn(
                    gen.localized(lang, "Ignore", "Ignorar", "Ignorer"),
                    record["context"]["objective"],
                )
            elif record["family"] == "review_need":
                review_shapes.add((lang, record["meta"]["scenario"]["shape"]))
                component = evidence[0]["text"].split(":")[0].strip()
                value = int(re.search(r"\d+", evidence[0]["text"]).group())
                if len(evidence) == 1:
                    expected = "semantic_review"
                else:
                    other = evidence[1]
                    other_component = other["text"].split(":")[0].strip()
                    other_value = int(re.search(r"\d+", other["text"]).group())
                    expected = (
                        "semantic_review"
                        if component != other_component
                        else "clarify"
                        if value != other_value
                        else "continue"
                    )
                self.assertEqual(record["answer"], expected)
        self.assertEqual(
            review_shapes, {(lang, s) for lang in gen.LANGUAGES for s in gen.REVIEW_SHAPES}
        )

    def test_repeatable_exclude_and_redraw(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b, out = [Path(tmp) / name for name in ("a.jsonl", "b.jsonl", "out.jsonl")]
            gen.write(1, 200, a)
            gen.write(2, 200, b)
            excluded = gen.exclusion_keys((a, b))
            args = [
                "--seed",
                "1",
                "--count",
                "200",
                "--exclude",
                str(a),
                "--exclude",
                str(b),
                "--out",
                str(out),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                gen.main(args)
            first = out.read_bytes()
            records = [json.loads(line) for line in first.splitlines()]
            self.assertEqual(sum(r["parent"] is None for r in records), 200)
            self.assertFalse({content_key(r) for r in records} & excluded)
            with contextlib.redirect_stdout(io.StringIO()):
                gen.main(args)
            self.assertEqual(first, out.read_bytes())
        # A saturated scenario space is skipped after the bounded redraw loop.
        state = gen._scenario_review(random.Random(1), "en")
        with mock.patch.dict(gen.GENERATORS, {"claim-review-v2": lambda rng, lang: state}):
            records = list(gen.generate(1, 12, ("review_need",)))
            self.assertEqual(len(records), 6)  # one distinct content per language, plus children

    def test_content_disjoint_from_v1(self):
        with (HERE / "data" / "synthetic-v1.jsonl").open(encoding="utf-8") as source:
            frozen = {content_key(json.loads(line)) for line in source}
        self.assertFalse(frozen & {content_key(r) for r in self.records})

    def test_frozen_v1_unchanged(self):
        for name, digest in FROZEN.items():
            self.assertEqual(file_sha256(HERE / name), digest, name)

    def test_novelty_4000_per_family(self):
        for family in gen.FAMILIES:
            first = {content_key(r) for r in gen.generate(1, 4000, (family,))}
            total = novel = 0
            for record in gen.generate(2, 4000, (family,)):
                total += 1
                novel += content_key(record) not in first
            self.assertEqual(total, 4000 if family == "grade" else 8000)
            self.assertGreaterEqual(novel / total, 0.9, family)


if __name__ == "__main__":
    unittest.main()
