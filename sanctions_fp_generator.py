#!/usr/bin/env python3
"""
sanctions_fp_generator.py
=========================

Generate a labeled evaluation set of *verified false-positive* name-screening
cases, scoped to the current UK Sanctions List (UKSL), with case shapes that
mirrorign Identomat API screening.

Watchlist side : real UK designations from OpenSanctions `gb_fcdo_sanctions`
                 (the FCDO UK Sanctions List, sole UK source since 2026-01-28),
                 mapped onto Identomat's `get-screening-person-details` shape.
Customer side  : synthetic identities (NO real PII) shaped like Identomat's
                 session `person` object, constructed to match a listed entity
                 on name + DOB (what the screening engine keys on) while
                 differing on a hard identifier, so the label is verified by
                 construction.

The realism gate mirrors screening: a blended name+DOB score, cut at 85 to match
the `minScreeningScore` config default, decides whether a pair would surface as
an alert at all.

Output : JSONL (one case per line) + a manifest with provenance.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Optional

import jellyfish
from rapidfuzz import fuzz
from rapidfuzz.distance import DamerauLevenshtein
from unidecode import unidecode

GENERATOR_VERSION = "2.0.0"

OS_DATASET = "gb_fcdo_sanctions"
OS_URL_TEMPLATE = "https://data.opensanctions.org/datasets/{snapshot}/{dataset}/entities.ftm.json"

# Pools for fabricating plausible-but-different customer attributes (ISO-3166 a2).
COUNTRY_POOL = ["gb", "us", "ca", "au", "nz", "ie", "fr", "de", "es", "it", "pt",
                "nl", "se", "no", "pl", "in", "pk", "ng", "ke", "za", "br", "mx"]
GIVEN_NAME_POOL = ["James", "Mary", "John", "Patricia", "Robert", "Jennifer",
                   "Michael", "Linda", "David", "Sarah", "Omar", "Aisha", "Wei",
                   "Mei", "Carlos", "Sofia", "Ahmed", "Fatima", "Ivan", "Olga"]
BIRTHPLACE_POOL = ["London", "Manchester", "Paris", "Berlin", "Madrid", "Lagos",
                   "Mumbai", "Toronto", "Sydney", "Warsaw", "Nairobi", "Lisbon"]
ADDRESS_POOL = ["12 High Street, London, UK", "44 Oak Avenue, Manchester, UK",
                "8 Rue de Paris, Paris, FR", "210 King St, Toronto, CA",
                "5 George St, Sydney, AU", "77 Market Rd, Lagos, NG"]

# OpenSanctions topic codes -> Identomat screening glossary terms.
TOPIC_MAP = {
    "sanction": "Sanctioned entity",
    "sanction.linked": "Sanction-linked entity",
    "sanction.counter": "Counter-sanctioned entity",
    "debarment": "Debarred entity",
    "role.pep": "Politician",
    "role.rca": "Close associate",
    "poi": "Person of Interest",
    "reg.action": "Regulator action",
    "reg.warn": "Regulator warning",
    "wanted": "Wanted person",
    "crime": "Person of Interest",
}

TRANSLIT_SWAPS = [("ph", "f"), ("f", "ph"), ("ck", "k"), ("kh", "k"), ("k", "kh"),
                  ("ee", "i"), ("ou", "u"), ("y", "i"), ("ie", "y"), ("v", "w"),
                  ("z", "s"), ("dj", "j"), ("ll", "l"), ("ss", "s"), ("nn", "n")]


# --------------------------------------------------------------------------- #
# Normalized view of one sanctioned entity
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    id: str
    schema: str
    names: list[str]
    aliases: list[str]
    weak_aliases: list[str]
    birth_dates: list[str]     # raw, e.g. "1975-04-12" or "1968"
    nationalities: list[str]   # iso2 lowercase
    citizenships: list[str]    # iso2 lowercase
    birth_places: list[str]
    id_numbers: list[str]      # passport / national id / tax numbers
    gender: Optional[str]      # "male" / "female"
    topics: list[str]          # raw OS topic codes

    @property
    def primary_name(self) -> Optional[str]:
        return self.names[0] if self.names else (self.aliases[0] if self.aliases else None)

    @property
    def all_names(self) -> list[str]:
        seen, out = set(), []
        for n in self.names + self.aliases + self.weak_aliases:
            k = n.lower().strip()
            if k and k not in seen:
                seen.add(k)
                out.append(n)
        return out

    @property
    def birth_years(self) -> list[int]:
        return sorted({y for d in self.birth_dates if (y := _first_year(d)) is not None})

    @property
    def entity_type(self) -> str:
        return entity_type_of(self.schema)


# --------------------------------------------------------------------------- #
# Loading / normalizing
# --------------------------------------------------------------------------- #
def iter_ftm_lines(path: Path) -> Iterator[dict]:
    """Yield entities from a line-delimited FollowTheMoney JSON file."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue  # UK source data quality is poor; skip bad lines


_YEAR_RE = re.compile(r"\b(1[5-9]\d{2}|20\d{2})\b")


def _first_year(date_str) -> Optional[int]:
    m = _YEAR_RE.search(str(date_str))
    return int(m.group(1)) if m else None


def normalize_entity(ent: dict) -> Optional[Target]:
    props = ent.get("properties", {}) or {}

    def prop(name: str) -> list[str]:
        return [str(v).strip() for v in (props.get(name, []) or []) if str(v).strip()]

    names, aliases, weak = prop("name"), prop("alias"), prop("weakAlias")
    if not (names or aliases or weak):
        return None

    gender = prop("gender")
    return Target(
        id=ent.get("id", ""),
        schema=ent.get("schema", "Thing"),
        names=names,
        aliases=aliases,
        weak_aliases=weak,
        birth_dates=prop("birthDate"),
        nationalities=sorted({c.lower() for c in prop("nationality")}),
        citizenships=sorted({c.lower() for c in prop("citizenship")}),
        birth_places=prop("birthPlace"),
        id_numbers=prop("idNumber") + prop("passportNumber") + prop("taxNumber"),
        gender=gender[0].lower() if gender else None,
        topics=prop("topics"),
    )


# --------------------------------------------------------------------------- #
# Identity helpers (shared with the true-positive generator)
# --------------------------------------------------------------------------- #
def split_name(full: str) -> tuple[str, str, str]:
    """(first, middle, last). Heuristic for list names; do not over-trust on
    non-Western names -- the full string remains the value the matcher uses."""
    toks = (full or "").split()
    if not toks:
        return ("", "", "")
    if len(toks) == 1:
        return (toks[0], "", "")
    return (toks[0], " ".join(toks[1:-1]), toks[-1])


def make_dob(year: int, rng: random.Random) -> tuple[str, str]:
    """Fabricate a full DOB in a given year -> (display 'M/D/YYYY', ISO)."""
    month, day = rng.randint(1, 12), rng.randint(1, 28)
    return f"{month}/{day}/{year}", f"{year:04d}-{month:02d}-{day:02d}T00:00:00.000Z"


def dob_from_target(target: "Target", rng: random.Random):
    """Reproduce the listed person's DOB as a document would show it.
    Full list date -> use it; year-only -> fabricate plausible month/day.
    Returns (display, iso, year) or (None, None, None)."""
    if not target.birth_dates:
        return None, None, None
    raw = target.birth_dates[0]
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        y, mo, da = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return f"{mo}/{da}/{y}", f"{y:04d}-{mo:02d}-{da:02d}T00:00:00.000Z", y
    y = _first_year(raw)
    if y is None:
        return None, None, None
    disp, iso = make_dob(y, rng)
    return disp, iso, y


def sex_of(gender: Optional[str]) -> Optional[str]:
    return {"male": "M", "female": "F"}.get((gender or "").lower())


def entity_type_of(schema: str) -> str:
    if schema == "Person":
        return "person"
    if schema in {"Company", "Organization", "LegalEntity"}:
        return "company"
    if schema == "Vessel":
        return "vessel"
    return "entity"


def map_topics(os_topics: list[str]) -> list[str]:
    out = [TOPIC_MAP[t] for t in os_topics if t in TOPIC_MAP]
    return out or ["Sanctioned entity"]  # the UK list is all sanctions designations


def fake_doc_number(rng: random.Random) -> str:
    letters = "".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(2))
    return letters + "".join(str(rng.randint(0, 9)) for _ in range(7))


def fake_personal_number(rng: random.Random) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(11))


def transliterate_variant(name: str, rng: random.Random) -> str:
    base = unidecode(name)
    lowered = base.lower()
    cands = [re.sub(a, b, base, count=1, flags=re.IGNORECASE)
             for a, b in rng.sample(TRANSLIT_SWAPS, k=min(3, len(TRANSLIT_SWAPS)))
             if a in lowered]
    if not cands or all(c == name for c in cands):
        for va, vb in (("a", "e"), ("e", "a"), ("o", "u"), ("i", "y")):
            if va in lowered:
                cands.append(re.sub(va, vb, base, count=1, flags=re.IGNORECASE))
                break
    return next((c for c in cands if c and c != name), base)


# --------------------------------------------------------------------------- #
# Matching ensemble + the screening gate
# --------------------------------------------------------------------------- #
def name_scores(customer: str, target_names: list[str]) -> dict:
    best = {"jaro_winkler": 0.0, "dlev_norm": 0.0, "token_set": 0.0,
            "phonetic_match": False, "best_score": 0.0, "matched_name": None}
    c = (customer or "").lower().strip()
    c_meta = jellyfish.metaphone(c)
    for tname in target_names:
        t = tname.lower().strip()
        if not t:
            continue
        jw = jellyfish.jaro_winkler_similarity(c, t)
        dl = DamerauLevenshtein.normalized_similarity(c, t)
        ts = fuzz.token_set_ratio(c, t) / 100.0
        score = max(jw, dl, ts)
        if score > best["best_score"]:
            best = {"jaro_winkler": round(jw, 4), "dlev_norm": round(dl, 4),
                    "token_set": round(ts, 4),
                    "phonetic_match": bool(c_meta) and c_meta == jellyfish.metaphone(t),
                    "best_score": round(score, 4), "matched_name": tname}
    return best


def screening_match(customer_fullname: str, customer_year: Optional[int],
                    target: "Target") -> dict:
    """Blend name similarity with DOB agreement into a 0-100 score, approximating
    the screening engine. Name dominates so exact-name false positives still
    surface; DOB agreement lifts a weaker name match over the line."""
    sc = name_scores(customer_fullname, target.all_names)
    combined = 100.0 * sc["best_score"]
    tys = target.birth_years
    if customer_year and tys:
        gap = min(abs(customer_year - ty) for ty in tys)
        if gap == 0:
            combined += 5
        elif gap <= 2:
            combined += 2
        elif gap > 10:
            combined -= 10
    sc["combined_score"] = round(max(0.0, min(100.0, combined)), 2)
    return sc


# --------------------------------------------------------------------------- #
# Shared block builders (identical shape across both generators)
# --------------------------------------------------------------------------- #
def make_customer(full_name: str, rng: random.Random, *, birthday=None,
                  birthday_time=None, birth_place=None, nationality=None,
                  citizenship=None, document_number=None, personal_number=None,
                  sex=None, address=None) -> dict:
    """Customer identity shaped like Identomat's session `person` object."""
    first, _, last = split_name(full_name)
    return {
        "first_name": first, "last_name": last, "full_name": full_name,
        "birthday": birthday, "birthday_time": birthday_time,
        "birth_place": birth_place,
        "nationality": nationality, "citizenship": citizenship,
        "document_number": document_number or fake_doc_number(rng),
        "personal_number": personal_number or fake_personal_number(rng),
        "sex": sex, "address": address or rng.choice(ADDRESS_POOL),
    }


def candidate_block(target: "Target", sc: dict) -> dict:
    """Listed entity shaped like Identomat's `get-screening-person-details`."""
    first, middle, last = split_name(target.primary_name or "")
    matched = sc.get("matched_name")
    return {
        "personId": target.id,
        "entityType": target.entity_type,
        "name": target.primary_name,
        "firstName": first, "middleName": middle, "lastName": last,
        "gender": target.gender,
        "birthdayTime": target.birth_dates[0] if target.birth_dates else None,
        "nationality": (target.nationalities or [None])[0],
        "birthPlace": target.birth_places[0] if target.birth_places else None,
        "idNumbers": target.id_numbers,             # via extraDetails; availability varies
        "aliases": target.aliases + target.weak_aliases,
        "topics": map_topics(target.topics),
        "matched_name": matched,
        "matched_alias_is_weak": matched in set(target.weak_aliases),
        "list_source": "UK Sanctions List (FCDO) via OpenSanctions",
        "birth_years": target.birth_years,          # kept for gate/debug
        "nationalities": target.nationalities,
    }


# --------------------------------------------------------------------------- #
# Discriminator helpers (guarantee "different person" by construction)
# --------------------------------------------------------------------------- #
def differing_birth_year(target: Target, rng: random.Random, tol: int) -> Optional[int]:
    if not target.birth_years:
        return None
    for _ in range(50):
        y = rng.randint(1940, 2004)
        if all(abs(y - ty) > tol for ty in target.birth_years):
            return y
    return None


def close_birth_year(target: Target, rng: random.Random, max_gap: int) -> Optional[int]:
    if not target.birth_years:
        return None
    base = rng.choice(target.birth_years)
    return base + rng.choice([g for g in range(-max_gap, max_gap + 1) if g != 0])


def differing_nationality(target: Target, rng: random.Random) -> Optional[str]:
    opts = [c for c in COUNTRY_POOL if c not in set(target.nationalities)]
    return rng.choice(opts) if opts else None


def differing_birthplace(target: Target, rng: random.Random) -> Optional[str]:
    if not target.birth_places:
        return None
    opts = [p for p in BIRTHPLACE_POOL if p.lower() not in {b.lower() for b in target.birth_places}]
    return rng.choice(opts) if opts else None


# --------------------------------------------------------------------------- #
# False-positive strategies. Each yields (customer, fp_type, discriminator, trigger).
# A customer's own (fabricated) document number differs from any the list holds,
# so it becomes a discriminator whenever the list entry actually has one.
# --------------------------------------------------------------------------- #
def _doc_discriminator(target: Target, disc: list[str]) -> list[str]:
    return disc + (["document_number"] if target.id_numbers else [])


def build_cases_for_target(target: Target, strategies: set[str],
                           rng: random.Random, tol: int):
    pname = target.primary_name
    is_person = target.schema == "Person"

    def customer(name, year, nat, *, birth_place=None):
        bd, bt = (make_dob(year, rng) if year else (None, None))
        return make_customer(name, rng, birthday=bd, birthday_time=bt,
                             birth_place=birth_place, nationality=nat,
                             citizenship=nat, sex=sex_of(target.gender))

    if "common_name_collision" in strategies and pname and is_person:
        by, nat = differing_birth_year(target, rng, tol), differing_nationality(target, rng)
        bp = differing_birthplace(target, rng)
        disc = [d for d, v in (("birth_date", by), ("nationality", nat), ("birth_place", bp)) if v]
        if disc:
            yield (customer(pname, by, nat, birth_place=bp),
                   "common_name_collision", _doc_discriminator(target, disc), pname)

    if "transliteration_variant" in strategies and pname and is_person:
        variant = transliterate_variant(pname, rng)
        by, nat = differing_birth_year(target, rng, tol), differing_nationality(target, rng)
        disc = [d for d, v in (("birth_date", by), ("nationality", nat)) if v]
        if variant != pname and disc:
            yield (customer(variant, by, nat),
                   "transliteration_variant", _doc_discriminator(target, disc), pname)

    if "weak_alias_hit" in strategies and target.weak_aliases and is_person:
        wa = rng.choice(target.weak_aliases)
        by, nat = differing_birth_year(target, rng, tol), differing_nationality(target, rng)
        disc = ["weak_alias_only"] + [d for d, v in (("birth_date", by), ("nationality", nat)) if v]
        yield (customer(wa, by, nat), "weak_alias_hit", _doc_discriminator(target, disc), wa)

    if "partial_token_match" in strategies and pname and is_person:
        toks = pname.split()
        if len(toks) >= 2:
            swapped = f"{rng.choice(GIVEN_NAME_POOL)} {toks[-1]}"
            by = differing_birth_year(target, rng, tol)
            disc = ["given_name"] + (["birth_date"] if by else [])
            cust = make_customer(swapped, rng, sex=None,
                                 nationality=differing_nationality(target, rng),
                                 **dict(zip(("birthday", "birthday_time"),
                                            make_dob(by, rng) if by else (None, None))))
            yield (cust, "partial_token_match", disc, pname)

    if "type_mismatch" in strategies and pname and not is_person:
        yield (customer(pname, rng.randint(1950, 2000), differing_nationality(target, rng)),
               "type_mismatch", ["entity_type"], pname)

    if "hard_case" in strategies and pname and is_person and target.nationalities:
        nat, cby = differing_nationality(target, rng), close_birth_year(target, rng, 2)
        if nat and cby:
            yield (customer(pname, cby, nat), "hard_case", ["nationality"], pname)


def fp_difficulty(score: float, disc: list[str], fp_type: str) -> str:
    if fp_type == "type_mismatch":
        return "easy"
    if "document_number" in disc:
        return "easy"            # a conclusive identifier mismatch is trivial to clear
    if score >= 95 and disc == ["nationality"]:
        return "hard"            # exact name, same approx age, only nationality differs
    if score >= 92:
        return "medium"
    return "easy"


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
DEFAULT_STRATEGIES = {"common_name_collision", "transliteration_variant",
                      "weak_alias_hit", "partial_token_match", "type_mismatch", "hard_case"}


def generate(entities: Iterable[dict], *, n, per_entity, threshold, strategies, rng, tol):
    cases, stats = [], {"entities_seen": 0, "constructed": 0,
                        "dropped_below_threshold": 0, "by_type": {}, "by_difficulty": {}}
    for ent in entities:
        if len(cases) >= n:
            break
        target = normalize_entity(ent)
        stats["entities_seen"] += 1
        if not target:
            continue
        produced = 0
        for customer, fp_type, disc, trigger in build_cases_for_target(target, strategies, rng, tol):
            if len(cases) >= n or produced >= per_entity:
                break
            cyear = _first_year(customer.get("birthday_time") or "")
            sc = screening_match(customer["full_name"], cyear, target)
            if sc["combined_score"] < threshold:
                stats["dropped_below_threshold"] += 1
                continue
            band = fp_difficulty(sc["combined_score"], disc, fp_type)
            cases.append({
                "case_id": f"fp-{len(cases):06d}", "label": "false_positive",
                "fp_type": fp_type, "discriminator": disc, "difficulty": band,
                "customer": customer, "candidate": candidate_block(target, sc),
                "match": sc, "generator_version": GENERATOR_VERSION,
            })
            produced += 1
            stats["constructed"] += 1
            stats["by_type"][fp_type] = stats["by_type"].get(fp_type, 0) + 1
            stats["by_difficulty"][band] = stats["by_difficulty"].get(band, 0) + 1
    return cases, stats


def download(snapshot: str, dest: Path) -> None:
    url = OS_URL_TEMPLATE.format(snapshot=snapshot, dataset=OS_DATASET)
    print(f"Downloading {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, dest)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=Path)
    src.add_argument("--download", action="store_true")
    p.add_argument("--snapshot", default="latest")
    p.add_argument("--cache", type=Path, default=Path("gb_fcdo_sanctions.ftm.json"))
    p.add_argument("--out", type=Path, default=Path("fp_eval_set.jsonl"))
    p.add_argument("--manifest", type=Path, default=Path("fp_eval_manifest.json"))
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--per-entity", type=int, default=3)
    p.add_argument("--score-threshold", type=float, default=85.0,
                   help="min blended name+DOB score (0-100) to keep a case; "
                        "mirrors minScreeningScore")
    p.add_argument("--dob-tolerance", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strategies", nargs="*", default=sorted(DEFAULT_STRATEGIES))
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
                            strategies=set(args.strategies), rng=rng, tol=args.dob_tolerance)
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
                       "shaped like get-screening-person-details. Gate mirrors name+DOB "
                       "screening at minScreeningScore.",
        "label_definition": "false positive = matches a UK-listed entity on name+DOB but "
                            "differs on a hard identifier (verified by construction)",
        "parameters": {"n": args.n, "per_entity": args.per_entity,
                       "score_threshold": args.score_threshold,
                       "dob_tolerance_years": args.dob_tolerance, "seed": args.seed,
                       "strategies": sorted(set(args.strategies))},
        "stats": stats,
        "caveat": "False positives only. Pair with a true-positive set and a realistic "
                  "base rate before reporting accuracy.",
    }
    with args.manifest.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(cases)} cases to {args.out}")
    print(f"By type: {stats['by_type']}")
    print(f"By difficulty: {stats['by_difficulty']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
