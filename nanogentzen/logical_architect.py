"""
logical_architect.py (nanoGentzen-v3 Logic Tree Generator)
Backward-compatibility adapter module.
Imports and exposes CotLogicalArchitect (aliased as LogicalArchitect) from .architect_cot.
"""

from .cot.architect import (
    CotLogicalArchitect,
    LogicalArchitect,
    RULE_DESCRIPTIONS,
)

__all__ = [
    "CotLogicalArchitect",
    "LogicalArchitect",
    "RULE_DESCRIPTIONS",
]
