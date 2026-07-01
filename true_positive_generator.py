#!/usr/bin/env python3
"""
true_positive_generator.py
==========================

Generate a labeled evaluation set of *verified true-positive* name-screening
cases, scoped to the UK Sanctions List (UKSL), with case shapes matching
Identomat's screening API contract.

A true positive is a customer who genuinely IS a sanctioned entity. We build one
by taking a real listed person and producing a customer who refers to the same
person but as a real onboarding form would capture them: a name rendered
differently, or a missing identifier, while the remaining identifiers (full DOB,
nationality, place of birth, document number) corroborate. Identity is preserved
by construction, so the label is verified.

Same record schema, shared candidate builder, and screening gate as
sanctions_fp_generator (imported from it), so the two sets merge cleanly. That
module must sit in the same directory.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from sanctions_fp_generator import (
    OS_DATASET,
    Target,
    _first_year,
    candidate_block,
    dob_from_target,
    download,
    iter_ftm_lines,
    make_customer,
    normalize_entity,
    screening_match,
    sex_of,
    transliterate_variant,
)

GENERATOR_VERSION = "2.0.0"
STRATEGIES = ["exact_match", "name_variant", "partial_identity"]


def _corroborators(customer: dict, target: Target) -> list[str]:
    """Which customer fields actually agree with the listed entity."""
    agree = []
    if customer["full_name"] and customer["full_name"] == target.primary_name:
        agree.append("name")
    cy = _first_year(customer.get("birthday_time") or "")
    if cy and cy in target.birth_years:
        agree.append("birth_date")
    if customer["nationality"] and customer["nationality"] in target.nationalities:
        agree.append("nationality")
    if customer["birth_place"] and target.birth_places and \
            customer["birth_place"].lower() in {b.lower() for b in target.birth_places}:
        agree.append("birth_place")
    if customer["document_number"] and customer["document_number"] in target.id_numbers:
        agree.append("document_number")
    return agree


def _tp_customer(name: str, target: Target, rng: random.Random, drop: Optional[str] = None) -> dict:
    """Customer who IS the target: identifiers copied from the list entry.
    `drop` omits one identifier to model an incomplete onboarding form."""
    bd, bt, _ = dob_from_target(target, rng)
    fields = {
        "birthday": bd, "birthday_time": bt,
        "birth_place": target.birth_places[0] if target.birth_places else None,
        "nationality": (target.nationalities or [None])[0],
        "citizenship": (target.citizenships or target.nationalities or [None])[0],
        "document_number": target.id_numbers[0] if target.id_numbers else None,
        "sex": sex_of(target.gender),
    }
    if drop == "birth_year":
        fields["birthday"] = fields["birthday_time"] = None
    elif drop == "nationality":
        fields["nationality"] = fields["citizenship"] = None
    return make_customer(name, rng, **fields)


# --------------------------------------------------------------------------- #
# The three ways a real customer can be a true match without looking identical.
# Each returns (customer, perturbation) or None if the target lacks needed data.
# --------------------------------------------------------------------------- #
def exact_match(target: Target, rng: random.Random):
    return (_tp_customer(target.primary_name, target, rng), "none")


def name_variant(target: Target, rng: random.Random):
    pool = target.aliases + target.weak_aliases
    name = rng.choice(pool) if pool and rng.random() < 0.5 \
        else transliterate_variant(target.primary_name, rng)
    if name == target.primary_name:
        return None
    return (_tp_customer(name, target, rng), "name_rendered_differently")


def partial_identity(target: Target, rng: random.Random):
    if not (target.birth_years and target.nationalities):
        return None
    drop = rng.choice(["birth_year", "nationality"])
    # Half the time the name is also a variant -> the hard true positive: the
    # agent must confirm identity from a variant name plus a single corroborator.
    name = transliterate_variant(target.primary_name, rng) if rng.random() < 0.5 \
        else target.primary_name
    return (_tp_customer(name, target, rng, drop=drop), f"missing_{drop}")


STRATEGY_FNS = {"exact_match": exact_match, "name_variant": name_variant,
                "partial_identity": partial_identity}


def difficulty(tp_type: str, name_is_variant: bool, missing_id: bool, has_doc: bool) -> str:
    """Keyed on how obscured the confirming evidence is."""
    if has_doc:
        return "easy"                       # a matching document number is conclusive
    if name_is_variant and missing_id:
        return "hard"                       # variant name + one identifier gone
    if tp_type == "exact_match":
        return "easy"
    return "medium"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def generate(entities: Iterable[dict], *, n, per_entity, threshold, strategies, rng):
    cases, stats = [], {"entities_seen": 0, "constructed": 0,
                        "dropped_below_threshold": 0, "by_type": {}, "by_difficulty": {}}
    for ent in entities:
        if len(cases) >= n:
            break
        target = normalize_entity(ent)
        stats["entities_seen"] += 1
        if not target or target.schema != "Person" or not target.primary_name:
            continue
        produced = 0
        for tp_type in strategies:
            if len(cases) >= n or produced >= per_entity:
                break
            result = STRATEGY_FNS[tp_type](target, rng)
            if result is None:
                continue
            customer, perturbation = result

            cy = _first_year(customer.get("birthday_time") or "")
            sc = screening_match(customer["full_name"], cy, target)
            if sc["combined_score"] < threshold:
                stats["dropped_below_threshold"] += 1
                continue

            corr = _corroborators(customer, target)
            name_is_variant = customer["full_name"] != target.primary_name
            missing_id = customer["birthday_time"] is None or customer["nationality"] is None
            band = difficulty(tp_type, name_is_variant, missing_id, "document_number" in corr)

            cases.append({
                "case_id": f"tp-{len(cases):06d}", "label": "true_positive",
                "tp_type": tp_type, "perturbation": perturbation,
                "corroborators": corr, "difficulty": band,
                "customer": customer, "candidate": candidate_block(target, sc),
                "match": sc, "generator_version": GENERATOR_VERSION,
            })
            produced += 1
            stats["constructed"] += 1
            stats["by_type"][tp_type] = stats["by_type"].get(tp_type, 0) + 1
            stats["by_difficulty"][band] = stats["by_difficulty"].get(band, 0) + 1
    return cases, stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path)
    src.add_argument("--download", action="store_true")
    p.add_argument("--snapshot", default="latest")
    p.add_argument("--cache", type=Path, default=Path("gb_fcdo_sanctions.ftm.json"))
    p.add_argument("--out", type=Path, default=Path("tp_eval_set.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("tp_eval_manifest.json"))
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--per-entity", type=int, default=2)
    p.add_argument("--score-threshold", type=float, default=85.0,
                   help="min blended name+DOB score (0-100); mirrors minScreeningScore")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategies", nargs="*", default=STRATEGIES)
    args = p.parse_args(argv)

    if args.download:
        download(args.snapshot, args.cache)
        input_path = args.cache
    else:
        input_path = args.input
    if not input_path.exists():
        print(f"Input not found: {input_path}", file=sys.stderr)
        return 1

    rng = random.Random(args.seed)
    cases, stats = generate(iter_ftm_lines(input_path), n=args.n,
                            per_entity=args.per_entity, threshold=args.score_threshold,
                            strategies=args.strategies, rng=rng)
    with args.out.open("w", encoding="utf-8") as fh:
        for c in cases:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": {"publisher": "OpenSanctions", "dataset": OS_DATASET,
                   "represents": "UK Sanctions List (UKSL), FCDO, sole UK source since 2026-01-28",
                   "snapshot": args.snapshot,
                   "license_note": "Free for non-commercial use; commercial use needs a license."},
        "schema_note": "customer shaped like Identomat session person object; candidate "
                       "shaped like get-screening-person-details (shared with FP generator).",
        "label_definition": "true positive = a customer who genuinely is a UK-listed person, "
                            "captured with realistic divergence that preserves identity",
        "parameters": {"n": args.n, "per_entity": args.per_entity,
                       "score_threshold": args.score_threshold, "seed": args.seed,
                       "strategies": args.strategies},
        "stats": stats,
        "caveat": "True positives only. Merge with the false-positive set at a realistic "
                  "base rate (FPs heavily outnumber TPs) before scoring.",
    }
    with args.manifest.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(cases)} cases to {args.out}")
    print(f"By type: {stats['by_type']}")
    print(f"By difficulty: {stats['by_difficulty']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
