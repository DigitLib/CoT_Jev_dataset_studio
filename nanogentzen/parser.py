"""
parser.py - compatibility facade over the split parsers.

The parser used to serve both dataset pipelines from one module. It is now three modules:
    nanogentzen/logic_syntax.py     symbolic formula / sequent syntax of the engine (shared)
    nanogentzen/cot/parser_cot.py   ClearCoT: natural language + KG paths -> sequents
    nanogentzen/jev/parser_jev.py   Jev: state JSON + policy rules -> facts / rules
New code should import from those modules directly; this facade re-exports them for old imports.
"""
from .logic_syntax import *  # noqa: F401,F403
from .logic_syntax import FormulaParser, Parser, parse_formula, parse_sequent, parse_symbolic_sequent  # noqa: F401
from .cot.parser_cot import *  # noqa: F401,F403
from .cot.parser_cot import (  # noqa: F401
    clean_atom_name, clean_term, parse_nl_formula, compile_to_formula, parse_natural_language,
    text_to_sequent, parse_path_edges, path_logic_report, path_to_sequent,
)
from .jev.parser_jev import *  # noqa: F401,F403
from .jev.parser_jev import (  # noqa: F401
    extract_vars, select_relevant_gamma, GENERIC_OPS, resolve_dot_path, ground_state_generic,
    _get_domain_registry, detect_workflow, get_calibrated_distribution, compile_structured_decision,
)
