"""
nanoGentzen Intuitionistic Logic (LI) Sequent Calculus engine.
Includes deterministic kernel, parser, SafeTensors inference engine, search, and logical tree architect.
Now upgraded with nanoGentzen-v3 Kripke semantics, Dyckhoff G4ip calculus, and SafeTensors model support.
v4 checkpoints are served by engine_v4.NanoGentzenV4Engine (use load_engine(), which picks v3 or v4).
"""

from .kernel import (
    Formula,
    Atom,
    Var,
    And,
    Or,
    Imp,
    Not,
    Sequent,
    ProofNode,
    RULES,
    G4IP_RULES,
    RULE_NAMES,
    apply_rule,
    verify_proof_tree,
    verify_proof_tree_strict,
    canonicalize_sequent,
    restore_proof_tree,
    extract_kripke_countermodel,
    KripkeModel,
    KripkeWorld,
    extract_relevance_witnesses,
    proof_tree_to_curry_howard,
)
from .configuration_nanogentzen import GentzenConfig, PolicyValueConfig
from .tokenization_nanogentzen import LogicTokenizerHF
try:
    from .tokenization_nanogentzen import LogicTokenizer
except ImportError:
    LogicTokenizer = LogicTokenizerHF
try:
    from .modeling_nanogentzen import GentzenPolicyValueModel
except ImportError:  # torch not installed: v3 neural weights unavailable, v4 NumPy backend still works
    GentzenPolicyValueModel = None
from .domain_registry import DomainCompilerRegistry, REGISTRY, register_domain
from .logic_syntax import parse_formula, parse_sequent
from .cot.parser_cot import text_to_sequent
from .engine import NanoGentzenEngine, NanoGentzenONNXEngine, load_engine, detect_model_version
from .search import NeuralProofSearch
from .cot.architect import CotLogicalArchitect
from .jev.architect import JevLogicalArchitect
from .logical_architect import LogicalArchitect

__all__ = [
    "Formula",
    "Atom",
    "Var",
    "And",
    "Or",
    "Imp",
    "Not",
    "Sequent",
    "ProofNode",
    "RULES",
    "G4IP_RULES",
    "RULE_NAMES",
    "apply_rule",
    "verify_proof_tree",
    "verify_proof_tree_strict",
    "canonicalize_sequent",
    "restore_proof_tree",
    "extract_kripke_countermodel",
    "KripkeModel",
    "KripkeWorld",
    "extract_relevance_witnesses",
    "proof_tree_to_curry_howard",
    "GentzenConfig",
    "PolicyValueConfig",
    "LogicTokenizer",
    "LogicTokenizerHF",
    "GentzenPolicyValueModel",
    "DomainCompilerRegistry",
    "REGISTRY",
    "register_domain",
    "parse_formula",
    "parse_sequent",
    "text_to_sequent",
    "NanoGentzenONNXEngine",
    "load_engine",
    "detect_model_version",
    "NeuralProofSearch",
    "CotLogicalArchitect",
    "JevLogicalArchitect",
    "LogicalArchitect",
]
