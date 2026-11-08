"""
nanogentzen.jev - Jev typed-decision pipeline front end.

    parser_jev  state JSON grounding, policy-rule compilation and validation, question specs
    architect   JevLogicalArchitect: decides scenarios, gold labels, factors, counterfactual labelling rules

Must not import nanogentzen.cot (checked by tests/test_pipeline_separation.py). Verification goes through
the shared engine (nanogentzen.engine / engine_v4 / kernel / search) and nanogentzen.verify.
"""
