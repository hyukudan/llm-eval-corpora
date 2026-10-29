"""Split rule, record checks and canonical serialisation for decision cases.

Shared by gen_synthetic.py, convert_public.py, freeze.py and verify.py.
Standard library only.

Split rule (``decision-split-v1``): a record's split is a function of its
``group`` alone. The bucket is

    int.from_bytes(sha256(canonical_json([RULE_ID, SALT, group]))[:8], "big") % 100

and buckets 0-69 are ``train``, 70-79 ``dev``, 80-89 ``calibration`` and
90-99 ``test``. Every derived record (a permuted copy, a translation of the
same utterance, another reply to the same conversation) shares its source's
group, so it always lands in the same split. Records taken from an upstream
evaluation partition are never hashed: they keep ``external_dev`` or
``external_test`` and must not be used for training.

Integrity (``Checker``): ids are unique; a record naming a ``parent`` must
follow that parent and share its group and split; and the same content
(ignoring id, option ids and option order) may never appear in two splits.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections import Counter

RULE_ID = "decision-split-v1"
SALT = "llm-eval-corpora/decision/v1"
BOUNDS = (70, 80, 90)  # cumulative percent: train | dev | calibration | test
HASHED_SPLITS = ("train", "dev", "calibration", "test")
EXTERNAL_SPLITS = ("external_dev", "external_test")
SPLITS = HASHED_SPLITS + EXTERNAL_SPLITS

FAMILIES = ("route", "next_tool", "review_need", "effort", "completion", "classify", "entail", "grade")
KINDS = ("choice", "yes_no", "score")
EVIDENCE_SOURCES = ("confirmed", "receipt", "claim")
RECORD_KEYS = {
    "id", "group", "family", "kind", "lang", "question", "options", "insufficient_option",
    "context", "answer", "rationale", "generator", "parent", "split", "licence",
    "attribution", "meta",
}
# Size limits; a record over any of them is refused (converters skip and count it).
MAX_OPTIONS = 32
MAX_QUESTION = 2048
MAX_OPTION_TEXT = 512
MAX_EVIDENCE = 64
MAX_EVIDENCE_TEXT = 4096
MAX_RECORD_BYTES = 32 * 1024


class RecordError(ValueError):
    """A record breaks the schema, a size limit or the split integrity rules."""


def canonical(value) -> bytes:
    """Canonical JSON: sorted keys, no spaces, UTF-8 (not ASCII-escaped)."""
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def assign_split(group: str) -> str:
    """Hashed split of a group key (see module docstring)."""
    bucket = int.from_bytes(hashlib.sha256(canonical([RULE_ID, SALT, group])).digest()[:8], "big") % 100
    low, middle, high = BOUNDS
    if bucket < low:
        return "train"
    if bucket < middle:
        return "dev"
    if bucket < high:
        return "calibration"
    return "test"


def content_key(record: dict) -> str:
    """Digest of what a model sees, blind to the id, option ids and option order."""
    texts = {option["id"]: option["text"] for option in record["options"]}
    body = {
        "family": record["family"],
        "kind": record["kind"],
        "lang": record["lang"],
        "question": record["question"],
        "options": sorted(canonical([o["text"], o.get("level")]).decode() for o in record["options"]),
        "insufficient": texts.get(record["insufficient_option"]),
        "context": record["context"],
    }
    return sha256_hex(canonical(body))


def validate(record: dict) -> dict:
    """Check one record against SCHEMA.md and the size limits; return it unchanged."""
    if type(record) is not dict or set(record) != RECORD_KEYS:
        raise RecordError("record_keys")
    for key in ("id", "group", "lang", "question", "answer", "generator", "licence", "attribution"):
        if type(record[key]) is not str or not record[key]:
            raise RecordError(f"record_{key}")
    if record["family"] not in FAMILIES or record["kind"] not in KINDS:
        raise RecordError("record_family_or_kind")
    if record["split"] not in SPLITS:
        raise RecordError("record_split")
    if len(record["question"]) > MAX_QUESTION:
        raise RecordError("question_too_long")
    options = record["options"]
    if type(options) is not list or not 2 <= len(options) <= MAX_OPTIONS:
        raise RecordError("options_count")
    ids = []
    for option in options:
        if type(option) is not dict or not {"id", "text"} <= set(option) <= {"id", "text", "level"}:
            raise RecordError("option_keys")
        if type(option["id"]) is not str or not option["id"]:
            raise RecordError("option_id")
        if type(option["text"]) is not str or not option["text"] or len(option["text"]) > MAX_OPTION_TEXT:
            raise RecordError("option_text")
        ids.append(option["id"])
    if len(set(ids)) != len(ids):
        raise RecordError("option_duplicate")
    insufficient = record["insufficient_option"]
    if insufficient is not None and insufficient not in ids:
        raise RecordError("insufficient_option")
    answers = [o for o in options if o["id"] != insufficient]
    if record["kind"] == "yes_no" and [o["id"] for o in answers] != ["yes", "no"]:
        raise RecordError("yes_no_options")
    if record["kind"] == "score":
        levels = [o.get("level") for o in answers]
        if any(type(level) is not int for level in levels) or levels != sorted(set(levels)):
            raise RecordError("score_levels")
    elif any("level" in o for o in options):
        raise RecordError("level_unexpected")
    if record["answer"] not in ids:
        raise RecordError("answer_not_an_option")
    context = record["context"]
    if type(context) is not dict or set(context) != {"objective", "evidence", "pending", "facts"}:
        raise RecordError("context_keys")
    evidence = context["evidence"]
    if type(evidence) is not list or len(evidence) > MAX_EVIDENCE:
        raise RecordError("evidence_count")
    for item in evidence:
        if type(item) is not dict or set(item) != {"ref", "source", "text"}:
            raise RecordError("evidence_keys")
        if item["source"] not in EVIDENCE_SOURCES:
            raise RecordError("evidence_source")
        if type(item["text"]) is not str or not item["text"] or len(item["text"]) > MAX_EVIDENCE_TEXT:
            raise RecordError("evidence_text")
    if len({item["ref"] for item in evidence}) != len(evidence):
        raise RecordError("evidence_ref_duplicate")
    if len(canonical(record)) > MAX_RECORD_BYTES:
        raise RecordError("record_too_large")
    return record


class Checker:
    """Streaming integrity check that also hashes the canonical output.

    Feed records in output order with ``add``; ``summary`` returns counts and
    sha256 digests (whole file and per split, each over the canonical JSONL
    lines of that split in file order).
    """

    def __init__(self, sink=None):
        self.sink = sink  # optional binary file object receiving the JSONL bytes
        self.whole = hashlib.sha256()
        self.per_split: dict[str, hashlib._Hash] = {}
        self.counts: Counter = Counter()
        self.families: Counter = Counter()
        self.langs: Counter = Counter()
        self.family_lang: Counter = Counter()
        self.generators: Counter = Counter()
        self.ids: dict[str, tuple[str, str]] = {}  # id -> (group, split)
        self.content: dict[str, str] = {}  # content key -> split

    def add(self, record: dict) -> None:
        validate(record)
        group, split = record["group"], record["split"]
        if record["id"] in self.ids:
            raise RecordError(f"duplicate id {record['id']}")
        if split in HASHED_SPLITS and assign_split(group) != split:
            raise RecordError(f"split does not follow the rule: {record['id']}")
        parent = record["parent"]
        if parent is not None and self.ids.get(parent) != (group, split):
            raise RecordError(f"parent missing or in another group/split: {record['id']}")
        key = content_key(record)
        if self.content.setdefault(key, split) != split:
            raise RecordError(f"same content in two splits: {record['id']}")
        self.ids[record["id"]] = (group, split)
        line = canonical(record) + b"\n"
        self.whole.update(line)
        self.per_split.setdefault(split, hashlib.sha256()).update(line)
        self.counts[split] += 1
        self.families[record["family"]] += 1
        self.langs[record["lang"]] += 1
        self.family_lang[f"{record['family']}/{record['lang']}"] += 1
        self.generators[record["generator"]] += 1
        if self.sink is not None:
            self.sink.write(line)

    def summary(self) -> dict:
        return {
            "count": sum(self.counts.values()),
            "content_sha256": self.whole.hexdigest(),
            "splits": {
                name: {"count": self.counts[name], "content_sha256": self.per_split[name].hexdigest()}
                for name in sorted(self.counts)
            },
            "families": dict(sorted(self.families.items())),
            "langs": dict(sorted(self.langs.items())),
            "family_lang": dict(sorted(self.family_lang.items())),
            "generators": dict(sorted(self.generators.items())),
        }


def gzip_bytes(data: bytes) -> bytes:
    """Deterministic gzip: mtime 0, no file name, fixed compression level."""
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, compresslevel=9, mtime=0) as handle:
        handle.write(data)
    return buffer.getvalue()


def file_sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
