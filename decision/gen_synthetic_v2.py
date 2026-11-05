#!/usr/bin/env python3
"""Second synthetic decision generator, with labels computed from scenario state."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

from gen_synthetic import ATTRIBUTION, LANGUAGES, LICENCE, TEXT, TOOL_TEXT
from splits import Checker, assign_split, canonical, content_key, validate

GENERATOR_VERSION = "decision-synth-v2"


def localized(lang, en, es, fr):
    return (en, es, fr)[LANGUAGES.index(lang)]


# Tool id, required parameters, optional parameters, and the operation in en/es/fr.
TOOLS = {
    "reserve_room": (
        ("location", "date", "quantity"),
        ("note",),
        ("reserve a room", "reservar una sala", "réserver une salle"),
    ),
    "cancel_room": (
        ("location", "date"),
        ("note",),
        (
            "cancel a room reservation",
            "cancelar una reserva de sala",
            "annuler une réservation de salle",
        ),
    ),
    "rent_bicycle": (
        ("location", "date"),
        ("quantity",),
        ("rent bicycles", "alquilar bicicletas", "louer des vélos"),
    ),
    "return_bicycle": (
        ("location", "reference"),
        (),
        ("return a bicycle", "devolver una bicicleta", "rendre un vélo"),
    ),
    "order_books": (
        ("item", "quantity", "location"),
        ("note",),
        ("order books", "pedir libros", "commander des livres"),
    ),
    "renew_loan": (
        ("reference", "date"),
        ("note",),
        (
            "renew a library loan",
            "renovar un préstamo de biblioteca",
            "renouveler un prêt de bibliothèque",
        ),
    ),
    "track_parcel": (
        ("reference",),
        ("note",),
        ("track a parcel", "rastrear un paquete", "suivre un colis"),
    ),
    "ship_parcel": (
        ("reference", "location", "date", "quantity"),
        ("note",),
        ("ship a parcel", "enviar un paquete", "expédier un colis"),
    ),
    "book_train": (
        ("location", "destination", "date"),
        ("quantity", "note"),
        ("book train travel", "reservar un viaje en tren", "réserver un voyage en train"),
    ),
    "book_ferry": (
        ("location", "destination", "date"),
        ("quantity",),
        ("book ferry travel", "reservar un viaje en ferry", "réserver un voyage en ferry"),
    ),
    "water_garden": (
        ("location", "quantity"),
        ("date",),
        ("water a garden", "regar un jardín", "arroser un jardin"),
    ),
    "inspect_garden": (
        ("location", "date"),
        ("note",),
        ("inspect a garden", "inspeccionar un jardín", "inspecter un jardin"),
    ),
    "schedule_repair": (
        ("item", "location", "date"),
        ("note",),
        ("schedule a repair", "programar una reparación", "planifier une réparation"),
    ),
    "quote_repair": (
        ("item", "location"),
        ("note",),
        (
            "estimate repair costs",
            "estimar el coste de reparación",
            "estimer le coût de réparation",
        ),
    ),
    "convert_units": (
        ("quantity", "unit"),
        (),
        ("convert a measurement", "convertir una medida", "convertir une mesure"),
    ),
    "calculate_area": (
        ("quantity", "unit"),
        ("note",),
        ("calculate an area", "calcular un área", "calculer une surface"),
    ),
    "archive_document": (
        ("reference", "collection"),
        ("note",),
        ("archive a document", "archivar un documento", "archiver un document"),
    ),
    "retrieve_document": (
        ("reference", "collection"),
        (),
        ("retrieve a document", "recuperar un documento", "récupérer un document"),
    ),
    "reserve_telescope": (
        ("location", "date"),
        ("note",),
        ("reserve a telescope", "reservar un telescopio", "réserver un télescope"),
    ),
    "reserve_microscope": (
        ("location", "date"),
        ("note",),
        ("reserve a microscope", "reservar un microscopio", "réserver un microscope"),
    ),
    "order_meal": (
        ("item", "quantity", "location"),
        ("note",),
        ("order a meal", "pedir una comida", "commander un repas"),
    ),
    "plan_menu": (
        ("item", "quantity"),
        ("note",),
        ("plan a menu", "planificar un menú", "planifier un menu"),
    ),
    "print_poster": (
        ("reference", "quantity"),
        ("note",),
        ("print posters", "imprimir carteles", "imprimer des affiches"),
    ),
    "scan_poster": (
        ("reference", "location"),
        ("note",),
        ("scan a poster", "escanear un cartel", "numériser une affiche"),
    ),
    "reserve_court": (
        ("location", "date"),
        ("note",),
        ("reserve a sports court", "reservar una pista deportiva", "réserver un terrain de sport"),
    ),
    "rent_rackets": (
        ("location", "quantity"),
        ("date",),
        ("rent rackets", "alquilar raquetas", "louer des raquettes"),
    ),
    "create_backup": (
        ("collection", "destination"),
        ("note",),
        ("create a backup", "crear una copia de seguridad", "créer une sauvegarde"),
    ),
    "restore_backup": (
        ("collection", "destination", "reference"),
        ("note",),
        ("restore a backup", "restaurar una copia de seguridad", "restaurer une sauvegarde"),
    ),
    "check_inventory": (
        ("item", "location"),
        ("note",),
        ("check inventory", "consultar el inventario", "consulter le stock"),
    ),
    "move_inventory": (
        ("item", "location", "destination", "quantity"),
        ("note",),
        ("move inventory", "trasladar existencias", "déplacer du stock"),
    ),
    "reserve_exhibit": (
        ("reference", "date"),
        ("quantity",),
        (
            "reserve exhibition admission",
            "reservar entrada a una exposición",
            "réserver une entrée à une exposition",
        ),
    ),
    "cancel_exhibit": (
        ("reference", "date"),
        ("note",),
        (
            "cancel exhibition admission",
            "cancelar entrada a una exposición",
            "annuler une entrée à une exposition",
        ),
    ),
}
VALUES = {
    "location": tuple(f"zone-{n}" for n in range(1, 17)),
    "destination": tuple(f"depot-{n}" for n in range(1, 17)),
    "date": tuple(f"2031-06-{n:02d}" for n in range(1, 25)),
    "quantity": tuple(str(n) for n in range(2, 26)),
    "reference": tuple(f"ref-{n}" for n in range(101, 125)),
    "item": tuple(f"article-{n}" for n in range(1, 17)),
    "collection": tuple(f"collection-{n}" for n in range(1, 17)),
    "unit": ("mm", "cm", "m", "km", "mg", "g", "kg", "litre"),
    "note": tuple(f"note-{n}" for n in range(1, 17)),
}
PARAM_LABELS = {
    "en": (
        "location",
        "destination",
        "date",
        "quantity",
        "reference",
        "item",
        "collection",
        "unit",
        "note",
    ),
    "es": (
        "ubicación",
        "destino",
        "fecha",
        "cantidad",
        "referencia",
        "artículo",
        "colección",
        "unidad",
        "nota",
    ),
    "fr": (
        "lieu",
        "destination",
        "date",
        "quantité",
        "référence",
        "article",
        "collection",
        "unité",
        "note",
    ),
}
TOOL_SHAPES = (
    "complete",
    "missing_one",
    "missing_two",
    "optional_missing",
    "earlier",
    "ambiguous",
    "unsupported",
    "shared_params",
)
COMPONENTS = (
    "renderer",
    "importer",
    "serializer",
    "validator",
    "router",
    "catalogue",
    "search",
    "queue",
    "ledger",
    "notifier",
    "compiler",
    "resolver",
    "uploader",
    "downloader",
    "aggregator",
    "reporter",
)
CHECKS = (
    "syntax",
    "roundtrip",
    "encoding",
    "pagination",
    "ordering",
    "escaping",
    "precision",
    "recovery",
    "isolation",
    "compatibility",
    "bounds",
    "concurrency",
    "migration",
    "checksum",
    "permissions",
    "deduplication",
)
REVIEW_SHAPES = ("confirmed", "claim_only", "contradiction", "other_component", "disagree")
READY_SHAPES = (
    "clean",
    "clean",
    "clean",
    "clean",
    "stale",
    "other_component",
    "failure",
    "pending",
    "claim_only",
    "stale",
)


def scenario(
    family, lang, options, evidence, answer, why, facts=None, pending=None, objective=None
):
    keys = {
        "next_tool": ("q", "obj"),
        "completion": ("ready_q", "ready_obj"),
        "review_need": ("review_q", "review_obj"),
        "grade": ("severity_q", "severity_obj"),
    }
    q, obj = keys[family]
    words = TOOL_TEXT[lang] if family == "next_tool" else TEXT[lang]
    return {
        "family": family,
        "kind": "score" if family == "grade" else "choice",
        "question": words[q],
        "objective": objective or words[obj],
        "options": options,
        "insufficient": "missing_evidence" if family == "completion" else None,
        "evidence": evidence,
        "pending": pending or [],
        "facts": facts or {},
        "answer": answer,
        "why": why,
    }


def _scenario_tool(rng, lang):
    shape = rng.choice(TOOL_SHAPES)
    eligible = [
        name
        for name, (req, opt, _) in TOOLS.items()
        if (shape != "missing_two" or len(req) >= 2) and (shape != "optional_missing" or opt)
    ]
    tool = rng.choice(eligible)
    required, optional, actions = TOOLS[tool]
    peers = [name for name in TOOLS if name != tool and set(TOOLS[name][0]) & set(required)]
    peer = rng.choice(peers)
    offered = [tool, peer] + rng.sample(
        [n for n in TOOLS if n not in (tool, peer)], rng.choice((1, 2))
    )
    if shape == "unsupported":
        offered.remove(tool)
        offered.append(rng.choice([n for n in TOOLS if n not in offered and n != tool]))
    rng.shuffle(offered)
    given = {p: rng.choice(VALUES[p]) for p in required + optional}
    removed = (
        rng.sample(required, 2 if shape == "missing_two" else 1)
        if shape in ("missing_one", "missing_two")
        else []
    )
    for p in removed:
        del given[p]
    if shape == "optional_missing":
        del given[rng.choice(optional)]
    earlier = {}
    if shape == "earlier":
        p = rng.choice(required)
        earlier[p] = given.pop(p)
    labels = dict(zip(VALUES, PARAM_LABELS[lang]))

    def details(values):
        return "; ".join(
            localized(lang, "with {p} {v}", "con {p} {v}", "avec {p} {v}").format(p=labels[p], v=v)
            for p, v in values.items()
        )

    action = actions[LANGUAGES.index(lang)]
    phrasing = rng.randrange(3)
    request = localized(
        lang,
        ("Please {a}.", "I need to {a}.", "Can you {a}?")[phrasing],
        ("Por favor, {a}.", "Necesito {a}.", "¿Puedes {a}?")[phrasing],
        ("Merci de {a}.", "Je souhaite {a}.", "Pouvez-vous {a} ?")[phrasing],
    ).format(a=action)
    evidence = [("claim", request + " " + details(given))]
    if earlier:
        evidence.append(
            (
                "claim",
                localized(
                    lang,
                    "Earlier user turn: ",
                    "Turno anterior del usuario: ",
                    "Message précédent de l'utilisateur : ",
                )
                + details(earlier),
            )
        )
    ambiguous = None
    if shape == "ambiguous":
        ambiguous = rng.choice(required)
        other = rng.choice([v for v in VALUES[ambiguous] if v != given[ambiguous]])
        evidence.append(
            (
                "claim",
                localized(
                    lang,
                    "Another unresolved candidate: ",
                    "Otro candidato sin resolver: ",
                    "Autre candidat non résolu : ",
                )
                + details({ambiguous: other}),
            )
        )
    unresolved = [p for p in required if p not in given and p not in earlier]
    answer = "none" if tool not in offered else "ask" if unresolved or ambiguous else f"call_{tool}"
    words = TOOL_TEXT[lang]
    options = [
        {
            "id": f"call_{name}",
            "text": words["tool"].format(name=name, params=", ".join(TOOLS[name][0]))
            + " "
            + localized(lang, "Operation: ", "Operación: ", "Opération : ")
            + TOOLS[name][2][LANGUAGES.index(lang)]
            + ". "
            + localized(lang, "Optional: ", "Opcionales: ", "Facultatifs : ")
            + ", ".join(TOOLS[name][1]),
        }
        for name in offered
    ]
    options += [{"id": key, "text": words[key]} for key in ("ask", "none")]
    result = scenario(
        "next_tool",
        lang,
        options,
        evidence,
        answer,
        "no offered operation fits"
        if answer == "none"
        else f"unresolved required parameters: {', '.join(unresolved) or ambiguous}"
        if answer == "ask"
        else "all required values are available and unambiguous; optional values are not required",
    )
    result["audit"] = {
        "shape": shape,
        "tool": tool,
        "given": given,
        "earlier": earlier,
        "missing": unresolved,
        "ambiguous": ambiguous,
    }
    return result


def check_text(rng, lang, component, name, failed):
    forms = localized(
        lang,
        ("{c}: check {n} {s}.", "{c}: result for {n}: {s}.", "{c}: {n} finished with status {s}."),
        (
            "{c}: comprobación {n}: {s}.",
            "{c}: resultado de {n}: {s}.",
            "{c}: {n} terminó con estado {s}.",
        ),
        (
            "{c} : vérification {n} : {s}.",
            "{c} : résultat de {n} : {s}.",
            "{c} : {n} se termine avec le statut {s}.",
        ),
    )
    status = localized(
        lang,
        "failed" if failed else "passed",
        "fallida" if failed else "superada",
        "échec" if failed else "réussite",
    )
    return rng.choice(forms).format(c=component, n=name, s=status)


def _scenario_ready(rng, lang):
    shape = rng.choice(READY_SHAPES)
    component, other = rng.sample(COMPONENTS, 2)
    changed = rng.randint(10, 900)
    receipt_time = changed - rng.randint(1, 9) if shape == "stale" else changed + rng.randint(1, 9)
    receipt_component = other if shape == "other_component" else component
    evidence = [
        (
            "confirmed",
            localized(
                lang,
                "Last change to {c}: minute {t}.",
                "Último cambio de {c}: minuto {t}.",
                "Dernier changement de {c} : minute {t}.",
            ).format(c=component, t=changed),
        )
    ]
    evidence.append(
        (
            "claim" if shape == "claim_only" else "receipt",
            localized(
                lang,
                "Check run for {c} at minute {t}:",
                "Ejecución de pruebas de {c} en el minuto {t}:",
                "Exécution des tests de {c} à la minute {t} :",
            ).format(c=receipt_component, t=receipt_time),
        )
    )
    checks = rng.sample(CHECKS, rng.randint(4, 8))
    failed = rng.choice(checks) if shape == "failure" else None
    for name in checks:
        evidence.append(
            (
                "claim" if shape == "claim_only" else "receipt",
                check_text(rng, lang, receipt_component, name, name == failed),
            )
        )
    pending = (
        [
            localized(
                lang,
                "Resolve the open migration item.",
                "Resolver la migración pendiente.",
                "Résoudre la migration en attente.",
            )
        ]
        if shape == "pending"
        else []
    )
    valid_receipt = (
        shape != "claim_only" and receipt_component == component and receipt_time >= changed
    )
    answer = (
        "missing_work"
        if pending or failed
        else "ready_to_validate"
        if valid_receipt
        else "missing_evidence"
    )
    options = [
        {"id": k, "text": TEXT[lang]["ready" if k == "ready_to_validate" else k]}
        for k in ("missing_evidence", "missing_work", "ready_to_validate")
    ]
    rubric = localized(
        lang,
        "Require a passing receipt for this component at or after its last change, with no failed checks or open items. Claims are not receipts.",
        "Se requiere un recibo aprobado de este componente posterior o igual al último cambio, sin fallos ni tareas abiertas. Las afirmaciones no son recibos.",
        "Exiger un reçu réussi pour ce composant à partir du dernier changement, sans échec ni tâche ouverte. Les affirmations ne sont pas des reçus.",
    )
    result = scenario(
        "completion",
        lang,
        options,
        evidence,
        answer,
        f"{shape}: receipt valid={valid_receipt}, failed={failed}, open items={len(pending)}",
        pending=pending,
        objective=rubric,
    )
    result["audit"] = {"shape": shape}
    return result


def _scenario_severity(rng, lang):
    component = rng.choice(COMPONENTS)
    checks = rng.sample(CHECKS, rng.randint(5, 12))
    failing = set(rng.sample(checks, rng.randint(0, len(checks))))
    evidence = [
        ("receipt", check_text(rng, lang, component, name, name in failing)) for name in checks
    ]
    warnings = rng.sample([n for n in CHECKS if n not in checks], rng.randint(0, 3))
    for name in warnings:
        evidence.append(
            (
                "receipt",
                localized(
                    lang,
                    "{c}: {n}: non-blocking warning, not a failure.",
                    "{c}: {n}: aviso no bloqueante, no es un fallo.",
                    "{c} : {n} : avertissement non bloquant, pas un échec.",
                ).format(c=component, n=name),
            )
        )
    options = [
        {"id": f"level{n}", "level": n, "text": TEXT[lang]["level"].format(level=n)}
        for n in range(4)
    ]
    rubric = localized(
        lang,
        "Count only failed checks: 0 failures = 0; 1 = 1; 2 = 2; 3 or more = 3. Ignore non-blocking warnings.",
        "Contar solo comprobaciones fallidas: 0 fallos = 0; 1 = 1; 2 = 2; 3 o más = 3. Ignorar avisos no bloqueantes.",
        "Compter seulement les vérifications échouées : 0 échec = 0 ; 1 = 1 ; 2 = 2 ; 3 ou plus = 3. Ignorer les avertissements non bloquants.",
    )
    result = scenario(
        "grade",
        lang,
        options,
        evidence,
        f"level{min(3, len(failing))}",
        f"{len(failing)} failed checks, capped at 3; {len(warnings)} warnings ignored",
        objective=rubric,
    )
    result["audit"] = {"shape": "checks", "failing": sorted(failing), "warnings": warnings}
    return result


def _scenario_review(rng, lang):
    shape = rng.choice(REVIEW_SHAPES)
    component, other = rng.sample(COMPONENTS, 2)
    check = rng.choice(CHECKS)
    # Each fact also varies in its meaningful result count, rather than an unrelated nonce.
    count = rng.randint(2, 40)
    opposite = count + rng.randint(1, 9)
    forms = localized(
        lang,
        (
            "{c}: {n} processed {v} entries",
            "{c}: {v} entries completed {n}",
            "{c}: entry count after {n} is {v}",
        ),
        (
            "{c}: {n} procesó {v} entradas",
            "{c}: {v} entradas completaron {n}",
            "{c}: el total de entradas tras {n} es {v}",
        ),
        (
            "{c} : {n} a traité {v} entrées",
            "{c} : {v} entrées ont terminé {n}",
            "{c} : le nombre d'entrées après {n} est {v}",
        ),
    )

    def fact(c, value):
        return rng.choice(forms).format(c=c, n=check, v=value) + "."

    evidence = [("claim", fact(component, count))]
    if shape != "claim_only":
        evidence.append(
            (
                "claim" if shape == "disagree" else "receipt",
                fact(
                    other if shape == "other_component" else component,
                    opposite if shape in ("contradiction", "disagree") else count,
                ),
            )
        )
    agrees = shape == "confirmed"
    conflict = shape in ("contradiction", "disagree")
    answer = "continue" if agrees else "clarify" if conflict else "semantic_review"
    options = [{"id": k, "text": TEXT[lang][k]} for k in ("continue", "semantic_review", "clarify")]
    result = scenario(
        "review_need",
        lang,
        options,
        evidence,
        answer,
        "matching receipt confirms the count"
        if agrees
        else "counts for the same component disagree"
        if conflict
        else "no receipt supports this component's claim",
    )
    result["audit"] = {"shape": shape}
    return result


GENERATORS = {
    "tool-select-v2": _scenario_tool,
    "ready-receipt-v2": _scenario_ready,
    "severity-score-v2": _scenario_severity,
    "claim-review-v2": _scenario_review,
}
FAMILIES = {
    "next_tool": "tool-select-v2",
    "completion": "ready-receipt-v2",
    "grade": "severity-score-v2",
    "review_need": "claim-review-v2",
}


def _record(generator, seed, index, variant, lang, state, options, parent):
    group = f"syn2-{generator}-{seed}-{index}"
    return {
        "id": f"{group}-{variant}",
        "group": group,
        "family": state["family"],
        "kind": state["kind"],
        "lang": lang,
        "question": state["question"],
        "options": [dict(o) for o in options],
        "insufficient_option": state["insufficient"],
        "context": {
            "objective": state["objective"],
            "evidence": [
                {"ref": f"e{n}", "source": source, "text": text}
                for n, (source, text) in enumerate(state["evidence"], 1)
            ],
            "pending": list(state["pending"]),
            "facts": dict(state["facts"]),
        },
        "answer": state["answer"],
        "rationale": state["why"],
        "generator": generator,
        "parent": parent,
        "split": assign_split(group),
        "licence": LICENCE,
        "attribution": ATTRIBUTION,
        "meta": {
            "generator_version": GENERATOR_VERSION,
            "seed": seed,
            "index": index,
            "variant": variant,
            "scenario": state["audit"],
        },
    }


def exclusion_keys(paths):
    keys = set()
    for path in paths:
        with open(path, encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    keys.add(content_key(json.loads(line)))
    return keys


def generate(seed: int, count: int, families=tuple(FAMILIES), exclude=()):
    """Draw count scenarios, cycling sorted generators and en/es/fr; redraw collisions 32 times."""
    families = tuple(families)
    if not families or set(families) - set(FAMILIES) or len(set(families)) != len(families):
        raise ValueError("unsupported or repeated family")
    if count < 0:
        raise ValueError("count must be nonnegative")
    rng = random.Random(f"{GENERATOR_VERSION}:{seed}")
    names = sorted(FAMILIES[f] for f in families)
    seen = exclusion_keys(exclude)
    for index in range(count):
        generator = names[index % len(names)]
        lang = LANGUAGES[(index // len(names)) % len(LANGUAGES)]
        for _ in range(32):
            state = GENERATORS[generator](rng, lang)
            base = _record(generator, seed, index, "a", lang, state, state["options"], None)
            key = content_key(base)
            if key not in seen:
                seen.add(key)
                break
        else:
            continue
        yield validate(base)
        if state["kind"] == "choice":
            permuted = list(state["options"])
            while permuted == state["options"]:
                rng.shuffle(permuted)
            yield validate(_record(generator, seed, index, "b", lang, state, permuted, base["id"]))


def write(seed: int, count: int, out, families=tuple(FAMILIES), exclude=()):
    # Read exclusions before opening the output (also permits excluding a prior output at this path).
    records = generate(seed, count, families, exclude)
    first = next(records, None)
    with open(out, "wb") as sink:
        checker = Checker(sink)
        if first is not None:
            checker.add(first)
        for record in records:
            checker.add(record)
    return checker.summary()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    families = tuple(args.families.split(","))
    if (
        args.count < 0
        or not families
        or set(families) - set(FAMILIES)
        or len(set(families)) != len(families)
    ):
        parser.error("count must be nonnegative and families must be distinct supported names")
    print(canonical(write(args.seed, args.count, args.out, families, args.exclude)).decode())
    return 0


if __name__ == "__main__":
    sys.exit(main())
