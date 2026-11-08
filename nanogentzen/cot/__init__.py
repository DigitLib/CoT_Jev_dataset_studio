"""
nanogentzen.cot - ClearCoT reasoning pipeline front end.

    parser_cot      natural-language claims and KG paths -> sequents
    relation_logic  which KG relations license an inference
    architect       CotLogicalArchitect: logic trees, proof records for SFT/DPO samples

Must not import nanogentzen.jev (checked by tests/test_pipeline_separation.py). Verification goes through
the shared engine (nanogentzen.engine / engine_v4 / kernel / search) and nanogentzen.verify.
"""
