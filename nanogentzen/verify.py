"""
verify.py - the verification core shared by the ClearCoT and Jev pipelines.

Both front ends (nanogentzen/cot, nanogentzen/jev) build their own formulas from their own inputs, then hand
them to the same engine through these functions. Nothing here knows about KG paths, natural language,
scenario JSON or criteria: it proves, decides, diagnoses and checks certificates.

    prove_sequent(engine, seq)                 decide one sequent (ClearCoT)
    decide(engine, request)                    decide a /v1/decide request: state + rules + questions (Jev)
    missing_facts(engine, goal, facts, rules)  minimal base literals that would make `goal` provable
    check_certificate(certificate)             re-check a typed lambda-term certificate offline
"""
from typing import Any, Dict, List, Optional, Tuple
import logging

from .engine_v4 import is_v4_engine

logger = logging.getLogger(__name__)

__all__ = ["is_v4_engine", "prove_sequent", "decide", "missing_facts", "check_certificate"]


def prove_sequent(engine: Any, seq: Any, **kw) -> Dict[str, Any]:
    """Kernel verdict for one sequent (v4 engines only; see NanoGentzenV4Engine.prove_sequent)."""
    return engine.prove_sequent(seq, **kw)


def decide(engine: Any, request: Dict[str, Any]) -> Dict[str, Any]:
    """Kernel verdicts for a decision request (v4 engines only; see NanoGentzenV4Engine.decide)."""
    return engine.decide(request)


def missing_facts(engine: Any, goal: Any, facts: List[Any], rules: List[Any]) -> List[str]:
    """
    Irredundant set of base literals (never the goal, never a derivable rule head) whose addition makes
    `goal` provable from facts + rules without contradiction. [] when none exists or the engine is not v4.
    Formulas must be v4 kernel formulas.
    """
    core = getattr(engine, "core", None)
    if core is None or not hasattr(core, "_missing_facts"):
        return []
    try:
        return list(core._missing_facts(goal, list(facts), list(rules)))
    except Exception as e:  # diagnosis only, never fail a sample for it
        logger.debug(f"missing-fact search failed for {goal}: {e}")
        return []


def check_certificate(certificate: Optional[Dict[str, Any]]) -> Tuple[bool, Optional[str]]:
    """Re-check a certificate with the independent v4 type checker (shares no code with the kernel)."""
    if not certificate:
        return False, "no certificate"
    from nanogentzen_v4.checker import check_certificate as _check
    return _check(certificate)
