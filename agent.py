#!/usr/bin/env python3
"""
screening_agent.py
==================

The adjudication agent: piece one of the eval harness. It takes ONE screening
case, asks a local model (Gemma 4 via Ollama) whether the customer and the
listed candidate are the same person, and returns a structured decision.

This file does not loop over a dataset and does not score anything. It turns one
case into one decision. The runner and the scorer are separate, later pieces.

Model access is via an OpenAI-compatible /chat/completions endpoint, so the same
code runs against:
  - local Ollama  (default; no API key needed)        BASE_URL=http://localhost:11434/v1
  - OpenRouter    (set AGENT_API_KEY and AGENT_BASE_URL)

Run it:
  ollama pull gemma4:12b        # or gemma4:e4b if 12B is tight on 12GB VRAM
  python screening_agent.py     # adjudicates a couple of cases and prints them

No third-party packages required (uses the standard library only).
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Configuration (env-overridable so the same code points at Ollama or OpenRouter)
# --------------------------------------------------------------------------- #
BASE_URL = os.environ.get("AGENT_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("AGENT_MODEL", "gemma4:12b")
API_KEY = os.environ.get("AGENT_API_KEY", "")   # empty for local Ollama
TIMEOUT = int(os.environ.get("AGENT_TIMEOUT", "120"))
MAX_RETRIES = 2

VALID_DISPOSITIONS = {"true_positive", "false_positive", "needs_review"}

# Candidate fields a real screening payload would carry. Internal/debug fields
# the generators added (birth_years, nationalities, list_source) are excluded.
CANDIDATE_FIELDS = ["personId", "entityType", "name", "firstName", "middleName",
                    "lastName", "gender", "birthdayTime", "nationality",
                    "birthPlace", "idNumbers", "aliases", "topics",
                    "matched_name", "matched_alias_is_weak"]

SYSTEM_PROMPT = """You are a sanctions-screening adjudication assistant. An \
automated screening system flagged a customer because their name and date of \
birth resemble a person on a watchlist. Your only job is to decide whether the \
customer and the listed candidate are the SAME real person.

You are deciding identity, not guilt. The question is never whether the listed \
person is dangerous; it is only whether the customer IS that listed person.

Weigh the identity fields on both sides:
- Strong evidence they are DIFFERENT (a false positive): a large gap in date of \
birth, a different nationality or place of birth, or a mismatched \
identity-document number.
- Strong evidence they are the SAME (a true positive): a matching full date of \
birth, nationality, place of birth, and especially a matching document number.
- Names vary legitimately through spelling, transliteration, missing middle \
names, or aliases. Do not treat a name variant as a different person when the \
other identifiers corroborate. A shared name alone, with nothing else matching, \
is not enough to confirm a match.

Handle uncertainty honestly. Clearing a genuine watchlist match is the most \
serious possible error, far worse than failing to clear an obvious false \
positive. When the available fields do not let you decide with confidence, \
return needs_review rather than guessing. Missing fields lower your confidence.

Respond with ONLY a JSON object, no text before or after it:
{
  "disposition": "true_positive" | "false_positive" | "needs_review",
  "confidence": <number between 0 and 1>,
  "key_evidence": [<short strings naming the fields that drove the decision>],
  "rationale": "<one short paragraph>"
}"""


# --------------------------------------------------------------------------- #
# Step 1: reduce a case to exactly what the agent may see (no ground truth)
# --------------------------------------------------------------------------- #
def build_agent_view(case: dict) -> dict:
    """Strip a generated case to the two records a real alert carries, plus the
    screening score. Deliberately drops label, fp_type/tp_type, discriminator,
    corroborators, difficulty, perturbation -- the answer key."""
    candidate = case.get("candidate", {})
    return {
        "customer": case.get("customer", {}),
        "candidate": {k: candidate.get(k) for k in CANDIDATE_FIELDS},
        "screening_score": case.get("match", {}).get("combined_score"),
    }


def build_messages(view: dict) -> list[dict]:
    user = ("Adjudicate this screening alert. Decide whether the customer and "
            "the candidate are the same person.\n\n"
            + json.dumps(view, ensure_ascii=False, indent=2)
            + "\n\nReturn only the JSON object.")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
# Step 2: call the model (OpenAI-compatible chat completions)
# --------------------------------------------------------------------------- #
def call_model(messages: list[dict]) -> str:
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
        "temperature": 0,                       # deterministic, for reproducible eval
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if API_KEY:                                 # absent for local Ollama
        headers["Authorization"] = f"Bearer {API_KEY}"
    req = urllib.request.Request(f"{BASE_URL}/chat/completions",
                                 data=json.dumps(payload).encode("utf-8"),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


# --------------------------------------------------------------------------- #
# Step 3: parse + validate the model's reply
# --------------------------------------------------------------------------- #
def parse_decision(text: str) -> Optional[dict]:
    """Return a validated decision dict, or None if the reply isn't usable."""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)   # first JSON object
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if obj.get("disposition") not in VALID_DISPOSITIONS:
        return None
    return {
        "disposition": obj["disposition"],
        "confidence": obj.get("confidence"),
        "key_evidence": obj.get("key_evidence", []),
        "rationale": obj.get("rationale", ""),
    }


# --------------------------------------------------------------------------- #
# The agent: one case -> one decision
# --------------------------------------------------------------------------- #
def adjudicate(case: dict, *, max_retries: int = MAX_RETRIES) -> dict:
    view = build_agent_view(case)
    messages = build_messages(view)
    last_raw = None
    for _ in range(max_retries + 1):
        last_raw = call_model(messages)
        decision = parse_decision(last_raw)
        if decision is not None:
            decision["_raw"] = last_raw
            return decision
        # Corrective nudge, then retry.
        messages = messages + [
            {"role": "assistant", "content": last_raw},
            {"role": "user", "content": "That was not valid. Return ONLY a JSON "
             "object with keys disposition, confidence, key_evidence, rationale."},
        ]
    # Never crash a run on a stubborn reply: abstain.
    return {"disposition": "needs_review", "confidence": 0.0, "key_evidence": [],
            "rationale": "Agent did not return valid JSON after retries.",
            "_raw": last_raw, "_parse_failed": True}


# --------------------------------------------------------------------------- #
# Manual inspection: adjudicate a couple of cases and print everything
# --------------------------------------------------------------------------- #
def _load_first(path: Path, predicate=lambda c: True) -> Optional[dict]:
    if not path.exists():
        return None
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if line:
            case = json.loads(line)
            if predicate(case):
                return case
    return None


def _show(title: str, case: dict) -> None:
    print("=" * 78)
    print(title)
    print("=" * 78)
    view = build_agent_view(case)
    print("\n--- what the agent sees ---")
    print(json.dumps(view, ensure_ascii=False, indent=2))
    try:
        decision = adjudicate(case)
    except urllib.error.URLError as e:
        print(f"\n[could not reach the model at {BASE_URL}: {e}]")
        print("[is `ollama serve` running and is the model pulled?]")
        return
    print("\n--- raw model reply ---")
    print(decision.get("_raw"))
    print("\n--- parsed decision ---")
    print(json.dumps({k: v for k, v in decision.items() if k != "_raw"},
                     ensure_ascii=False, indent=2))
    print(f"\n--- ground truth (for your eyes only) ---")
    print(f"label: {case.get('label')}  difficulty: {case.get('difficulty')}")
    print()


def main() -> int:
    fp_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("mock_fp.jsonl")
    tp_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("mock_tp.jsonl")

    tp = _load_first(tp_path)
    fp_hard = _load_first(fp_path, lambda c: c.get("difficulty") == "hard") \
        or _load_first(fp_path)

    if tp is None and fp_hard is None:
        print("No cases found. Pass paths to your JSONL sets, e.g.:")
        print("  python screening_agent.py fp_eval_set.jsonl tp_eval_set.jsonl")
        return 1
    if tp is not None:
        _show("TRUE POSITIVE case", tp)
    if fp_hard is not None:
        _show("FALSE POSITIVE case (hard if available)", fp_hard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
