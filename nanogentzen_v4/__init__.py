"""
nanogentzen_v4 - the nanoGentzen v4 engine, vendored into the Studio as its own package.

Generated files (do not edit; regenerate with tools/vendor_v4_engine.py after retraining):
    kernel, parser, search, certificates, checker, jev_server, domain_registry,
    modeling_nanogentzen, configuration_nanogentzen, tokenization_nanogentzen
Studio files: backend.py (torch/NumPy glue), numpy_model.py (torch-free forward pass).

Everything here is kept separate from the v3 `nanogentzen` package: formulas, sequents and proof trees
of the two kernels are different classes. nanogentzen/engine_v4.py converts between them.
"""
from .kernel import (  # noqa: F401
    And, Bot, BOT, Formula, Imp, Not, Or, Sequent, Var,
    g4ip_prove, normalize_sequent, verify_proof_tree_strict, find_kripke_countermodel, model_view_sequent,
)
from .parser import parse_symbolic_sequent, compile_to_formula  # noqa: F401

__version__ = "4.1"
