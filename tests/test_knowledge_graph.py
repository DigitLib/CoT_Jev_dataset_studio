"""
Knowledge graph: concept quality, relation logic, the evidence gate, taxonomy, concept repair,
document scoping, cross-document topic gating, path discovery and saving / loading.
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tests  # noqa: F401  (redirects all runtime data to a temp dir; see tests/__init__.py)

from core.chunker import DocumentChunker
from core.concepts import canonical_concept, check_inference, concept_issues, concept_key, concept_quality, fix_legacy_singular, inference_cue, read_relation, repair_from_evidence, specialization_of
from core.document_parser import DocumentPage
from core.evidence import ground_evidence
from core.kg_builder import KnowledgeGraphBuilder, Proposition
from nanogentzen.architect_cot import CotLogicalArchitect
from nanogentzen.cot.architect import CotLogicalArchitect
from nanogentzen.cot.parser_cot import path_logic_report, path_to_sequent
from nanogentzen.engine import NanoGentzenEngine, load_engine
from nanogentzen.parser import path_logic_report, path_to_sequent


def _add(kg, s, p, o, ev, chunk="c1"):
    kg.add_proposition(Proposition(subject=s, predicate=p, obj=o, statement=ev, evidence=ev, chunk_id=chunk,
                                   doc_name="d.pdf", page_number=1))


def _kg(*triples):
    b = KnowledgeGraphBuilder()
    for s_, p_, o_ in triples:
        b.add_proposition(Proposition(subject=s_, predicate=p_, obj=o_, statement=f"{s_} {p_} {o_}.",
                                      evidence=f"{s_} {p_} {o_}.", doc_name="d.pdf", page_number=1))
    return b


TEXT_P1 = ("Intentionality requires mental represen- tation in every cognitive system that we know of today.\n\n"
           "Brentano argued that mental phenomena exhibit “intentionality” as their defining mark here.")

TEXT_P2 = ("Mental representation implies a representational vehicle, according to most current accounts of it.\n\n"
           "Mental representation implies a representational vehicle, according to most current accounts of it.")


def _pages():
    return [DocumentPage(page_number=1, text=TEXT_P1, blocks=[]),
            DocumentPage(page_number=2, text=TEXT_P2, blocks=[])]


def _has_networkx():
    try:
        import networkx
        return hasattr(networkx, "all_simple_paths")
    except ImportError:
        return False


class FakeLLM:
    """Returns a fixed extraction for every chunk."""
    def __init__(self, triples):
        self.triples = triples

    def health_check(self, *a, **k):
        return True

    def generate(self, **kw):
        return json.dumps(self.triples)


def _add_crossdoc_cleanup(kg, s, p, o, doc):
    kg.add_proposition(Proposition(subject=s, predicate=p, obj=o, statement=f"{s} {p} {o}.", evidence=f"{s} {p} {o}",
                                   chunk_id=f"{doc}:{s}", doc_name=doc, page_number=1, evidence_status="exact",
                                   extractor="manual"))


MODEL_V4 = ROOT / "model"   # nanoGentzen v4 weights (config.MODEL_DIR)


def _add_logic_first_paths(b, s, p, o, doc="d.pdf"):
    b.add_proposition(Proposition(subject=s, predicate=p, obj=o, statement=f"{s} {p} {o}.",
                                  evidence=f"{s} {p} {o}.", doc_name=doc, page_number=1))


class TestConceptNames(unittest.TestCase):
    def test_issues(self):
        cases = {
            "Respect For Practical And": "truncated",
            "Principle Of Informed Cognitive": "truncated",          # cut after an adjective
            "Values And Interests Different": "truncated",
            "People Recognizing A Body'": "truncated",
            "People Are Blameworthy": "clause",
            "Getting A Smoke Alarm": "clause",
            "Autonomy Demands Guidance Control": "clause",
            "While Retaining Legitimacy": "clause",
            "Agent To Have Control": "clause",
            "Persons Subject To Them": "pronoun",
            "Individual": "generic",
        }
        for name, issue in cases.items():
            self.assertIn(issue, concept_issues(name), name)

    def test_clean_names_have_no_issues(self):
        for name in ["Substantive Independence", "Right To Explanation", "Capacity To Be Responsible", "COMPAS",
                     "Democratic Will", "Will Theory", "Theorem A", "Daily Me", "Financial Self-sufficiency",
                     "Moral Responsibility", "Autonomy As Ideal"]:
            self.assertEqual(concept_issues(name), [], name)

    def test_recurrence_does_not_rescue_fragments(self):
        self.assertLess(concept_quality("Respect For Practical And", mentions=20), 0.5)
        self.assertEqual(concept_quality("Substantive Independence", mentions=20), 1.0)

    def test_canonical_concept_never_cuts_mid_phrase(self):
        self.assertEqual(canonical_concept("the principle of informed cognitive agency"),
                         "principle of informed cognitive agency")
        self.assertEqual(canonical_concept("information that her file contained"), "information")
        self.assertEqual(canonical_concept("their autonomy"), "autonomy")

    def test_builder_keeps_long_names_whole(self):
        kg = KnowledgeGraphBuilder()
        self.assertEqual(kg._canonicalize("principles of informed practical agency"),
                         "Principle Of Informed Practical Agency")
        self.assertEqual(kg._canonicalize("large-scale social processes"), "Large-scale Social Process")
        self.assertFalse(kg._is_valid_entity("Respect For Practical And"))

    def test_concept_key_variants(self):
        self.assertEqual(concept_key("Conception Of Autonomy"), concept_key("Autonomy Conception"))
        self.assertEqual(concept_key("Rights To Explanation"), concept_key("Right To Explanation"))
        self.assertEqual(concept_key("Uber's Algorithm"), concept_key("Uber Algorithm"))
        self.assertNotEqual(concept_key("Practical Or Cognitive Agency"), concept_key("Practical And Cognitive Agency"))

    def test_repair_from_evidence(self):
        ev = ["They defend principles of practical and cognitive agency, which ground claims to information."]
        self.assertEqual(repair_from_evidence("Principles Of Practical And", ev),
                         "principles of practical and cognitive agency")
        self.assertIsNone(repair_from_evidence("Treating Someone As An", ["Treating someone as an end requires it."]))

    def test_legacy_singular(self):
        self.assertEqual(fix_legacy_singular("Legitimating Processe"), "Legitimating Process")
        self.assertEqual(fix_legacy_singular("Niche Market"), "Niche Market")


class TestRelations(unittest.TestCase):
    def test_reversed_phrasings(self):
        self.assertEqual(read_relation("is necessary for"), ("requires", True))
        self.assertEqual(read_relation("is based on"), ("depends on", False))

    def test_inference_cue_is_conservative(self):
        self.assertEqual(inference_cue("is the other key requirement for"), ("requires", True))
        self.assertEqual(inference_cue("presupposes"), ("requires", False))
        for raw in ["adds to rather than contradicts extant", "may be used to simultaneously violate",
                    "involves a kind of", "ensures causal", "is defined as the intentional use of", "does not require"]:
            self.assertIsNone(inference_cue(raw), raw)

    def test_kind_of_pattern_is_anchored(self):
        kg = KnowledgeGraphBuilder()
        self.assertEqual(kg._canonicalize_predicate("involves a kind of"), "involves")
        self.assertEqual(kg._canonicalize_predicate("is a kind of"), "is a kind of")
        self.assertEqual(kg._canonicalize_predicate("advance practical"), "advance")
        self.assertEqual(kg._canonicalize_predicate("argues against"), "argues against")

    def test_canonical_relation_swaps(self):
        kg = KnowledgeGraphBuilder()
        self.assertEqual(kg._canonical_relation("is a prerequisite for"), ("requires", True))


class TestEvidenceGate(unittest.TestCase):
    def test_no_cue_hedged_off_topic(self):
        self.assertFalse(check_inference("causes", "This is a common practice.", "ChoicePoint", "Instant Result")["ok"])
        r = check_inference("causes", "If she knows a system might produce a false report, she can act.",
                            "System", "False Report")
        self.assertFalse(r["ok"])
        self.assertIn("hedged", r["reason"])
        r = check_inference("requires", "Procedural independence requires several things. One is epistemic competence.",
                            "Personal Autonomy", "Epistemic Competence")
        self.assertFalse(r["ok"])
        r = check_inference("requires", "Cognitive agency, in contrast, requires only enough information.",
                            "Background Check System", "Cognitive Agency")
        self.assertFalse(r["ok"])

    def test_direction(self):
        ev = "That is because fully realized autonomy demands both procedural and substantive independence."
        self.assertEqual(check_inference("requires", ev, "Substantive Independence", "Fully Realized Autonomy")["direction"],
                         "reverse")
        self.assertEqual(check_inference("requires", ev, "Fully Realized Autonomy", "Substantive Independence")["direction"],
                         "forward")
        inv = "Epistemic competence is necessary for personal autonomy."
        self.assertEqual(check_inference("requires", inv, "Personal Autonomy", "Epistemic Competence")["direction"], "forward")
        self.assertEqual(check_inference("requires", inv, "Epistemic Competence", "Personal Autonomy")["direction"], "reverse")

    def test_extraction_moves_leading_preposition_into_predicate(self):
        kg = KnowledgeGraphBuilder()

        class FakeLLM:
            def generate(self, **kw):
                return ('[{"subject": "Sunstein", "predicate": "argues", "object": "against echo chambers", '
                        '"statement": "Sunstein argues against echo chambers.", '
                        '"evidence": "Sunstein argues against echo chambers.", "strength": 8}]')
        chunk = types.SimpleNamespace(text="Sunstein argues against echo chambers.", chunk_id="c1", doc_name="d.pdf",
                                      page_number=1, section_title="s")
        props = kg._extract_propositions_for_chunk(chunk, model="m", client=FakeLLM())
        self.assertEqual([(p.subject, p.predicate, p.obj) for p in props], [("Sunstein", "argues against", "Echo Chamber")])

    def test_extraction_applies_the_gate(self):
        kg = KnowledgeGraphBuilder()
        chunk = types.SimpleNamespace(text="This is a common practice. Autonomy demands substantive independence.",
                                      chunk_id="c1", doc_name="d.pdf", page_number=1, section_title="s")
        props = kg._grounded([
            Proposition("Choicepoint Report", "causes", "Instant Result", "s", "This is a common practice."),
            Proposition("Substantive Independence", "requires", "Autonomy", "s",
                        "Autonomy demands substantive independence."),
        ], chunk, "llm")
        demoted = next(p for p in props if p.claimed_predicate)
        self.assertEqual((demoted.predicate, demoted.claimed_predicate), ("relates to", "causes"))
        flipped = next(p for p in props if p.predicate == "requires")
        self.assertEqual((flipped.subject, flipped.obj), ("Autonomy", "Substantive Independence"))
        self.assertEqual(kg.extraction_stats.get("inference_reversed"), 1)


class TestTaxonomy(unittest.TestCase):
    def test_specialization(self):
        self.assertTrue(specialization_of("Positive Freedom", "Freedom"))
        self.assertTrue(specialization_of("Moral Responsibility", "Responsibility"))
        for s, g in [("Policies Using Technology", "Technology"), ("Forming Justified Belief", "Belief"),
                     ("Non-domination", "Domination"), ("Algorithmic Advertising Process", "Process"),
                     ("Other Relevant Concept", "Relevant Concept"), ("Lack Of Autonomy", "Autonomy")]:
            self.assertFalse(specialization_of(s, g), (s, g))


class TestRepairConcepts(unittest.TestCase):
    def _graph(self):
        kg = KnowledgeGraphBuilder()
        _add(kg, "Autonomy", "requires", "Substantive Independence", "Autonomy demands substantive independence.", "c1")
        _add(kg, "Personal Autonomy", "relates to", "Moral Concern", "Personal autonomy grounds moral concern.", "c2")
        _add(kg, "Substantive Independence", "requires", "Fully Realized Autonomy",
             "That is because fully realized autonomy demands substantive independence.", "c3")
        _add(kg, "ChoicePoint", "causes", "Instant Result", "This is a common practice.", "c4")
        _add(kg, "Legitimacy", "based on", "Democratic Will", "Legitimacy is based on democratic will.", "c5")
        _add(kg, "Conception Of Autonomy", "shapes", "Policy Design", "A conception of autonomy shapes policy design.", "c6")
        _add(kg, "Autonomy Conception", "informs", "Law", "An autonomy conception informs law.", "c7")
        # nodes as older builds left them (the current builder rejects such names at insertion)
        kg.graph.add_edge("Sunstein", "Against Echo Chamber", predicate="argues", chunk_id="c8",
                          statement="Sunstein argues against the echo chamber.", evidence="Sunstein argues against the echo chamber.")
        kg.graph.add_node("Respect For Practical And", label="x", doc_name="d.pdf", page_number=1)
        kg.graph.add_edge("Respect For Practical And", "Autonomy", predicate="grounds",
                          statement="Respect for practical and cognitive agency grounds autonomy.",
                          evidence="Respect for practical and cognitive agency grounds autonomy.", chunk_id="c9")
        return kg

    def test_repair(self):
        kg = self._graph()
        rep = kg.repair_concepts()
        G = kg.graph
        # truncated name completed from the evidence
        self.assertIn("Respect For Practical And Cognitive Agency", G)
        self.assertNotIn("Respect For Practical And", G)
        # variants merged
        self.assertEqual(sum(1 for n in G if "Conception" in n), 1)
        # unsupported inference label demoted, kept as claim
        d = next(iter(G.get_edge_data("ChoicePoint", "Instant Result").values()))
        self.assertEqual((d["predicate"], d["claimed_predicate"]), ("relates to", "causes"))
        # swapped edge flipped to match the evidence
        self.assertTrue(G.has_edge("Fully Realized Autonomy", "Substantive Independence"))
        self.assertFalse(G.has_edge("Substantive Independence", "Fully Realized Autonomy"))
        # "based on" read as an inference
        d = next(iter(G.get_edge_data("Legitimacy", "Democratic Will").values()))
        self.assertEqual(d["predicate"], "depends on")
        # leading preposition moved into the predicate
        d = next(iter(G.get_edge_data("Sunstein", "Echo Chamber").values()))
        self.assertEqual(d["predicate"], "argues against")
        # taxonomy from names
        self.assertTrue(any(dd.get("extractor") == "taxonomy" and dd["predicate"] == "is a kind of"
                            for dd in (G.get_edge_data("Personal Autonomy", "Autonomy") or {}).values()))
        self.assertTrue(G.has_edge("Fully Realized Autonomy", "Autonomy"))
        self.assertGreater(rep["inference_edges_after"], rep["inference_edges_before"])

    def test_idempotent(self):
        kg = self._graph()
        kg.repair_concepts()
        kg.repair_concepts()
        rep = kg.repair_concepts()
        self.assertEqual((rep["predicates_upgraded"], len(rep["inference_demoted"]), len(rep["taxonomy_added"]),
                          len(rep["merged"]), len(rep["renamed"])), (0, 0, 0, 0, 0))

    def test_sense_check_rejects_and_remembers(self):
        kg = KnowledgeGraphBuilder()
        _add(kg, "Agency", "is defined as", "Capacity To Act", "Agency is defined as the capacity to act.", "c1")
        _add(kg, "Talent Agency", "signs", "Actor", "The talent agency signs actors.", "c2")
        _add(kg, "Moral Agency", "requires", "Capacity To Act", "Moral agency requires the capacity to act.", "c3")
        vec = {"Agency": [1.0, 0.0], "Moral Agency": [0.9, 0.1], "Talent Agency": [0.0, 1.0]}
        fake = lambda texts: [vec[t.split(":")[0]] for t in texts]   # noqa: E731
        rep = kg.repair_concepts(embed_fn=fake)
        self.assertIn(("Moral Agency", "Agency"), rep["taxonomy_added"])
        self.assertIn("Talent Agency", [x[0] for x in rep["taxonomy_rejected_sense"]])
        rep2 = kg.repair_concepts(embed_fn=None)                      # without embeddings: still rejected
        self.assertNotIn(("Talent Agency", "Agency"), rep2["taxonomy_added"])
        self.assertEqual(concept_issues("Capacity To Act"), [])

    def test_paths_skip_meaningless_concepts(self):
        kg = KnowledgeGraphBuilder()
        _add(kg, "Autonomy", "requires", "Substantive Independence", "Autonomy requires substantive independence.")
        ev = "Substantive independence requires values and interests different from others."
        kg.graph.add_edge("Substantive Independence", "Values And Interests Different", predicate="requires",
                          statement=ev, evidence=ev, chunk_id="c2")   # an older build's truncated node
        self.assertEqual(kg.find_multihop_paths(require_entailment=True, verify_with_gentzen=False, limit=None), [])
        self.assertTrue(kg.find_multihop_paths(require_entailment=True, verify_with_gentzen=False, limit=None,
                                               min_concept_quality=0))


class TestGateCueSubject(unittest.TestCase):
    def rejected(self, pred, ev, s, o):
        return not check_inference(pred, ev, s, o)["ok"]

    def test_reported_speech_is_rejected(self):
        self.assertTrue(self.rejected("implies", "One argument for the claim that agency is necessary for "
                                      "consciousness is that this is implied by many scientific theories.",
                                      "Agency", "Scientific Theory"))
        self.assertTrue(self.rejected("is defined as", "It also means that scientific theories should be central "
                                      "to consciousness-related AI policy and regulation.",
                                      "Consciousness", "Scientific Theory"))
        self.assertTrue(self.rejected("requires", "PRM is particularly clear that agency is a necessary condition "
                                      "for consciousness.", "Consciousness", "PRM"))
        self.assertTrue(self.rejected("depends on", "Unsurprisingly, the answer depends on the essential structural "
                                      "and functional features.", "Consciousness", "Essential Structural Feature"))

    def test_real_subjects_still_pass(self):
        ok = [
            ("excludes", "The standard construal of IIT is incompatible with our working assumption of "
             "computational functionalism.", "Integrated Information Theory", "Computational Functionalism"),
            ("requires", "Important functions like feature grouping and binding and figure-ground segregation "
             "require recurrence.", "Feature Grouping", "Recurrent Processing"),
            ("requires", "These theories suggest that metacognitive monitoring of perceptual systems with relevant "
             "properties is necessary for consciousness in AI.", "AI System", "Metacognitive Monitoring"),
            ("depends on", "The ability to track the accuracy of responses (known as metacognitive sensitivity) "
             "depends on being conscious of the stimuli.", "Metacognitive Sensitivity", "Consciousness Of Stimuli"),
            ("implies", "The concept of rectangularity entails that the interior angles sum to 360 degrees.",
             "Rectangularity", "Interior Angles Sum To 360 Degrees"),
            ("is a kind of", "PredPol is a tool that helps structure how governments exercise power.",
             "PredPol", "Tool"),
            ("excludes", "Transform- ers lack the overall structure of a system with a global workspace.",
             "Transformer", "Global Workspace"),
            ("requires", "Agency is the key requirement for consciousness.", "Consciousness", "Agency"),
        ]
        for pred, ev, s, o in ok:
            with self.subTest(ev=ev[:40]):
                self.assertTrue(check_inference(pred, ev, s, o)["ok"], check_inference(pred, ev, s, o))


class TestGateMentionedAndReversed(unittest.TestCase):
    def test_required_for_is_reversed(self):
        r = check_inference("requires", "Doerig et al. interpret RPT as claiming that implementational recurrence "
                            "is required for consciousness.", "Implementational Recurrence", "Consciousness")
        self.assertEqual((r["ok"], r["direction"]), (True, "reverse"))

    def test_mentioned_claims_are_rejected_attributed_ones_kept(self):
        self.assertFalse(check_inference("depends on", "The proposal that consciousness depends on material "
                                         "composition is clearly incompatible with functionalism.",
                                         "Consciousness", "Material Composition")["ok"])
        self.assertTrue(check_inference("requires", "Hurley (1998) claims that consciousness requires intentional "
                                        "agency.", "Consciousness", "Intentional Agency")["ok"])
        self.assertTrue(check_inference("requires", "The crux of the account is her argument that autonomy demands "
                                        "substantive independence.", "Autonomy", "Substantive Independence")["ok"])

    def test_unresolved_pronoun_and_two_letter_acronym(self):
        # "AI System" must not match the word "as"; "it" has no endpoint before it
        self.assertFalse(check_inference("implies", "RPT may be interpreted as claiming that it is sufficient for "
                                         "consciousness that recurrence is used.", "AI System", "Consciousness")["ok"])
        self.assertTrue(check_inference("requires", "Consciousness is subtle, and it requires recurrent processing.",
                                        "Consciousness", "Recurrent Processing")["ok"])


class TestTaxonomyNames(unittest.TestCase):
    def test_verb_phrases_and_fragments(self):
        self.assertIn("verb_phrase", concept_issues("Assess Consciousness"))
        self.assertIn("truncated", concept_issues("Own Consciousness"))
        self.assertEqual(concept_issues("Lack Of Oversight"), [])
        self.assertEqual(concept_issues("Test Data"), [])

    def test_specialization_rejects_acts_and_vague_heads(self):
        self.assertFalse(specialization_of("Assess Consciousness", "Consciousness"))
        self.assertFalse(specialization_of("Own Consciousness", "Consciousness"))
        self.assertFalse(specialization_of("Scientific Theory", "Theory"))
        self.assertFalse(specialization_of("Recurrent Processing", "Processing"))
        self.assertTrue(specialization_of("Access Consciousness", "Consciousness"))


class TestTaxonomyPolicy(unittest.TestCase):
    def _paths(self, b, **kw):
        return b.find_multihop_paths(require_entailment=True, verify_with_gentzen=False, limit=None, **kw)

    def setUp(self):
        self.b = _kg(("Artificial Consciousness", "is a kind of", "Consciousness"),
                     ("Consciousness", "requires", "Recurrent Processing"),
                     ("Recurrent Processing", "requires", "Feedback Loop"),
                     ("Feedback Loop", "is a kind of", "Neural Loop"))

    @staticmethod
    def _tax_positions(p):
        return [i for i, e in enumerate(p["path_repr"].split(" -> ")) if "[is a kind of]" in e]

    def test_any_keeps_a_leading_taxonomy_hop(self):
        self.assertTrue(any(self._tax_positions(p)[:1] == [0] for p in self._paths(self.b)))

    def test_last_allows_one_taxonomy_hop_only_at_the_end(self):
        paths = self._paths(self.b, taxonomy_policy="last")
        self.assertTrue(paths)
        for p in paths:
            pos = self._tax_positions(p)
            self.assertLessEqual(len(pos), 1, p["path_repr"])
            if pos:
                self.assertEqual(pos[0], p["hops"] - 1, p["path_repr"])
        self.assertTrue(any(p["path_nodes"][-1] == "Neural Loop" for p in paths))

    def test_none_has_no_taxonomy_hop(self):
        paths = self._paths(self.b, taxonomy_policy="none")
        self.assertTrue(paths)
        self.assertTrue(all(p["taxonomy_hops"] == 0 for p in paths))

    def test_unknown_policy_is_an_error(self):
        with self.assertRaises(ValueError):
            self._paths(self.b, taxonomy_policy="first")

    def test_hub_cap(self):
        triples = [(f"Source {c}", "requires", "Hub Concept") for c in "ABCDE"]
        triples += [("Hub Concept", "requires", f"Target {c}") for c in "VWXYZ"]
        b = _kg(*triples)
        self.assertEqual(len(self._paths(b)), 25)
        capped = self._paths(b, max_paths_per_hub=3)
        self.assertEqual(len(capped), 3)
        self.assertTrue(all("Hub Concept" in p["path_nodes"][1:-1] for p in capped))


class TestRepairTaxonomy(unittest.TestCase):
    def test_lexical_edge_that_no_longer_qualifies_is_removed(self):
        b = _kg(("Assess Consciousness", "relates to", "Moral Status"), ("Consciousness", "relates to", "Mind"),
                ("Scientific Theory", "relates to", "Evidence"), ("Theory", "relates to", "Explanation"))
        for u, v in (("Assess Consciousness", "Consciousness"), ("Scientific Theory", "Theory")):
            b.graph.add_edge(u, v, predicate="is a kind of", statement=f"{u} is a kind of {v}.", evidence="",
                             extractor="taxonomy", evidence_status="lexical", strength=6)
        rep = b.repair_concepts(add_taxonomy=True, reread_predicates=False)
        self.assertEqual(sorted(rep["taxonomy_removed"]),
                         [("Assess Consciousness", "Consciousness"), ("Scientific Theory", "Theory")])
        self.assertFalse(any(d.get("extractor") == "taxonomy" for _, _, d in b.graph.edges(data=True)))


class TestEvidenceGrounding(unittest.TestCase):
    def test_classification(self):
        ch = TEXT_P1
        self.assertEqual(ground_evidence("Intentionality requires mental representation", ch,
                                         "Intentionality", "Mental Representation")[0], "exact")   # hyphenation
        self.assertEqual(ground_evidence('mental phenomena exhibit "intentionality"', ch,
                                         "Brentano", "Intentionality")[0], "exact")                # curly quotes
        st, ev = ground_evidence("", ch, "Brentano", "Intentionality")
        self.assertEqual(st, "located")
        self.assertIn("Brentano argued", ev)
        self.assertEqual(ground_evidence("Consciousness is fundamental to physics.", ch,
                                         "Consciousness", "Physics")[0], "unsupported")

    def _chunk(self):
        return DocumentChunker(min_chunk_chars=20).chunk_pages("doc.pdf", _pages()[:1])[0]

    def test_hallucinated_edges_are_dropped(self):
        kg = KnowledgeGraphBuilder()
        kg.llm_client = FakeLLM([
            {"subject": "Intentionality", "predicate": "requires", "object": "Mental Representation",
             "statement": "Intentionality requires mental representation.",
             "evidence": "Intentionality requires mental representation", "strength": "high"},
            {"subject": "Consciousness", "predicate": "implies", "object": "Quantum Collapse",
             "statement": "Consciousness implies quantum collapse.",
             "evidence": "Consciousness causes the collapse of the wave function.", "strength": 9},
        ])
        props = kg._extract_propositions_for_chunk(self._chunk(), model="m")
        self.assertEqual([(p.subject, p.obj) for p in props], [("Intentionality", "Mental Representation")])
        self.assertEqual(props[0].evidence_status, "exact")
        self.assertEqual(props[0].extractor, "llm")
        self.assertEqual(props[0].strength, 8)               # a non-numeric strength ("high") falls back to the default; the chunk is kept
        self.assertEqual(kg.extraction_stats.get("dropped_unsupported"), 1)

    def test_keep_policy_flags_instead_of_dropping(self):
        kg = KnowledgeGraphBuilder()
        kg.evidence_policy = "keep"
        kg.llm_client = FakeLLM([{"subject": "Consciousness", "predicate": "implies", "object": "Quantum Collapse",
                                  "evidence": "made up", "strength": 9}])
        props = kg._extract_propositions_for_chunk(self._chunk(), model="m")
        self.assertEqual(props[0].evidence_status, "unsupported")

    def test_missing_evidence_is_located_not_faked(self):
        kg = KnowledgeGraphBuilder()
        kg.llm_client = FakeLLM([{"subject": "Brentano", "predicate": "argued", "object": "Intentionality"}])
        chunk = next(c for c in DocumentChunker(min_chunk_chars=20).chunk_pages("doc.pdf", _pages()[:1])
                     if "Brentano" in c.text)
        props = kg._extract_propositions_for_chunk(chunk, model="m")
        self.assertEqual(props[0].evidence_status, "located")
        self.assertTrue(props[0].evidence.startswith("Brentano argued"))   # a real sentence, not the chunk prefix


class TestPersistence(unittest.TestCase):
    def setUp(self):
        if not _has_networkx():
            self.skipTest("networkx not installed")

    def _kg(self):
        kg = KnowledgeGraphBuilder()
        for s, p, o in [("Intentionality", "requires", "Mental Representation"),
                        ("Mental Representation", "implies", "Representational Vehicle"),
                        ("Brentano", "argued", "Intentionality")]:
            kg.add_proposition(Proposition(subject=s, predicate=p, obj=o, statement=f"{s} {p} {o}.", evidence="e",
                                           chunk_id="c1", doc_name="doc.pdf", page_number=1,
                                           evidence_status="exact", extractor="llm"))
        return kg

    def test_round_trip(self):
        kg = self._kg()
        chunks = DocumentChunker(min_chunk_chars=20).chunk_pages("doc.pdf", _pages())
        comms = [{"community_id": 0, "title": "Mind"}]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "kg" / "g.json"
            kg.save(str(path), chunks=chunks, communities=comms)
            info = KnowledgeGraphBuilder.saved_info(str(path))
            self.assertEqual((info["num_nodes"], info["num_edges"], info["num_chunks"]), (4, 3, len(chunks)))
            kg2 = KnowledgeGraphBuilder()
            res = kg2.load(str(path))
        self.assertEqual(sorted(kg2.graph.nodes()), sorted(kg.graph.nodes()))
        self.assertEqual(sorted((u, v, a["predicate"], a["evidence_status"]) for u, v, a in kg2.graph.edges(data=True)),
                         sorted((u, v, a["predicate"], a["evidence_status"]) for u, v, a in kg.graph.edges(data=True)))
        self.assertEqual([c.chunk_id for c in res["chunks"]], [c.chunk_id for c in chunks])
        self.assertEqual(res["communities"], comms)
        # the restored graph is fully usable: same paths, node lookup rebuilt
        p1 = [p["path_nodes"] for p in kg.find_multihop_paths(verify_with_gentzen=False)]
        p2 = [p["path_nodes"] for p in kg2.find_multihop_paths(verify_with_gentzen=False)]
        self.assertEqual(sorted(p1), sorted(p2))
        self.assertEqual(kg2._resolve_to_existing_node("mental representation"), "Mental Representation")

    def test_rejects_foreign_files_and_missing(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d) / "x.json"
            bad.write_text('{"nodes": []}')
            with self.assertRaises(ValueError):
                KnowledgeGraphBuilder().load(str(bad))
            self.assertIsNone(KnowledgeGraphBuilder.saved_info(str(Path(d) / "none.json")))

    def test_paths_report_evidence(self):
        paths = self._kg().find_multihop_paths(verify_with_gentzen=False)
        self.assertTrue(paths)
        for p in paths:
            self.assertEqual(p["evidence_statuses"], ["exact"] * (len(p["path_nodes"]) - 1))
            self.assertTrue(p["evidence_supported"])

    def test_append_from_chunks_skips_existing(self):
        kg = self._kg()
        chunks = DocumentChunker(min_chunk_chars=20).chunk_pages("doc.pdf", _pages())
        # c1 is already in kg edges
        res = kg.append_from_chunks(chunks)
        # Should report novel chunks (chunks with ids not yet in graph)
        self.assertIn("novel_chunks", res)
        self.assertGreaterEqual(res["total_nodes"], 4)


class TestDocumentScoping(unittest.TestCase):
    def setUp(self):
        self.builder = KnowledgeGraphBuilder()

        # Doc A (Cognitive Science)
        # Bayesian Model -> Causal Inference -> Decision Theory
        self.builder.add_proposition(Proposition(
            subject="Bayesian Model", predicate="formulates", obj="Causal Inference",
            statement="Bayesian Model formulates Causal Inference.",
            evidence="In cognitive modeling, the Bayesian Model formulates Causal Inference.",
            doc_name="cogsci.pdf", page_number=10
        ))
        self.builder.add_proposition(Proposition(
            subject="Causal Inference", predicate="enables", obj="Decision Theory",
            statement="Causal Inference enables Decision Theory.",
            evidence="Causal Inference enables Decision Theory under uncertainty.",
            doc_name="cogsci.pdf", page_number=12
        ))

        # Doc B (Information Retrieval)
        # Reasoning-8B -> QwenRerank -> Cross Encoder
        self.builder.add_proposition(Proposition(
            subject="Reasoning-8B", predicate="compares", obj="QwenRerank",
            statement="Reasoning-8B compares QwenRerank.",
            evidence="Reasoning-8B compares QwenRerank on BRIGHT benchmark.",
            doc_name="retrieval.pdf", page_number=5
        ))
        self.builder.add_proposition(Proposition(
            subject="QwenRerank", predicate="implements", obj="Cross Encoder",
            statement="QwenRerank implements Cross Encoder.",
            evidence="QwenRerank implements Cross Encoder ranking architectures.",
            doc_name="retrieval.pdf", page_number=6
        ))

    def test_intra_document_path_isolation(self):
        # Connect Doc A and Doc B through a bridge edge
        self.builder.add_proposition(Proposition(
            subject="Bayesian Model", predicate="utilizes", obj="Reasoning-8B",
            statement="Bayesian Model utilizes Reasoning-8B.",
            evidence="A hypothetical cross-edge connecting cogsci and retrieval.",
            doc_name="cogsci.pdf", page_number=90
        ))

        # Default mode: intra_document_only = True
        paths_intra = self.builder.find_multihop_paths(max_hops=3, min_hops=2, intra_document_only=True)
        self.assertTrue(len(paths_intra) > 0)
        for p in paths_intra:
            self.assertFalse(p.get("is_cross_document", False))
            self.assertEqual(len(p.get("provenance_docs", [])), 1)

    def test_scope_doc_name(self):
        # Scoping specifically to retrieval.pdf
        paths_retrieval = self.builder.find_multihop_paths(
            max_hops=3, min_hops=2, intra_document_only=True, scope_doc_name="retrieval.pdf"
        )
        self.assertTrue(len(paths_retrieval) > 0)
        for p in paths_retrieval:
            self.assertEqual(p["doc_name"], "retrieval.pdf")
            self.assertEqual(p["provenance_docs"], ["retrieval.pdf"])

    def test_generic_bridge_suppression_in_cross_document_mode(self):
        # Connect doc1 and doc2 through a generic academic bridge concept: "Model"
        builder2 = KnowledgeGraphBuilder()
        builder2.add_proposition(Proposition(
            subject="Theorem A", predicate="implies", obj="Model",
            statement="Theorem A implies Model.", evidence="Evidence A",
            doc_name="paper1.pdf", page_number=1
        ))
        builder2.add_proposition(Proposition(
            subject="Model", predicate="evaluates", obj="Benchmark B",
            statement="Model evaluates Benchmark B.", evidence="Evidence B",
            doc_name="paper2.pdf", page_number=2
        ))

        # With intra_document_only=False, generic bridge "Model" MUST be suppressed
        paths = builder2.find_multihop_paths(max_hops=3, min_hops=2, intra_document_only=False)
        self.assertEqual(len(paths), 0, "Generic bridge 'Model' should not allow cross-document path discovery")

    def test_valid_cross_document_synthesis_preserves_provenance(self):
        # Connect doc1 and doc2 through a domain-specific concept: "Vector Search Engine"
        builder3 = KnowledgeGraphBuilder()
        builder3.add_proposition(Proposition(
            subject="Hierarchical Index", predicate="enables", obj="Vector Search Engine",
            statement="Hierarchical Index enables Vector Search Engine.", evidence="Index evidence",
            doc_name="indexing.pdf", page_number=3
        ))
        builder3.add_proposition(Proposition(
            subject="Vector Search Engine", predicate="accelerates", obj="Dense Reranker",
            statement="Vector Search Engine accelerates Dense Reranker.", evidence="Search evidence",
            doc_name="reranking.pdf", page_number=8
        ))

        paths = builder3.find_multihop_paths(max_hops=3, min_hops=2, intra_document_only=False)
        self.assertTrue(len(paths) > 0)
        cross_path = paths[0]
        self.assertTrue(cross_path["is_cross_document"])
        self.assertIn("indexing.pdf", cross_path["provenance_docs"])
        self.assertIn("reranking.pdf", cross_path["provenance_docs"])

    def test_cross_document_only_mode(self):
        builder = KnowledgeGraphBuilder()
        # Intra doc 1: A -> B -> C
        builder.add_proposition(Proposition(
            subject="Concept A", predicate="relates to", obj="Concept B",
            statement="Concept A relates to Concept B.", evidence="ev",
            doc_name="doc1.pdf", page_number=1
        ))
        builder.add_proposition(Proposition(
            subject="Concept B", predicate="relates to", obj="Concept C",
            statement="Concept B relates to Concept C.", evidence="ev",
            doc_name="doc1.pdf", page_number=2
        ))
        # Cross doc: Concept B -> Concept D (in doc2.pdf)
        builder.add_proposition(Proposition(
            subject="Concept B", predicate="influences", obj="Concept D",
            statement="Concept B influences Concept D.", evidence="ev",
            doc_name="doc2.pdf", page_number=10
        ))

        # With cross_document_only=True, only cross-document paths must be returned
        cross_paths = builder.find_multihop_paths(max_hops=3, min_hops=2, cross_document_only=True, limit=50)
        self.assertTrue(len(cross_paths) > 0)
        for p in cross_paths:
            self.assertTrue(p["is_cross_document"])
            self.assertGreater(len(p["provenance_docs"]), 1)

        # With intra_document_only=True, no cross-document paths
        intra_paths = builder.find_multihop_paths(max_hops=3, min_hops=2, intra_document_only=True, limit=50)
        self.assertTrue(len(intra_paths) > 0)
        for p in intra_paths:
            self.assertFalse(p["is_cross_document"])
            self.assertEqual(len(p["provenance_docs"]), 1)

    def test_balanced_interleaving_and_limit_compliance(self):
        builder = KnowledgeGraphBuilder()
        # Intra doc 1
        builder.add_proposition(Proposition(
            subject="Brentano Thesis", predicate="requires", obj="Intentional Inexistence",
            statement="Brentano Thesis requires Intentional Inexistence.", evidence="ev",
            doc_name="brentano.pdf", page_number=28
        ))
        builder.add_proposition(Proposition(
            subject="Intentional Inexistence", predicate="defines", obj="Mental Phenomena",
            statement="Intentional Inexistence defines Mental Phenomena.", evidence="ev",
            doc_name="brentano.pdf", page_number=29
        ))
        # Cross doc to ai_consciousness.pdf
        builder.add_proposition(Proposition(
            subject="Mental Phenomena", predicate="challenges", obj="Artificial General Intelligence",
            statement="Mental Phenomena challenges Artificial General Intelligence.", evidence="ev",
            doc_name="ai_consciousness.pdf", page_number=15
        ))

        # Test limit=1
        paths_lim1 = builder.find_multihop_paths(max_hops=3, min_hops=2, intra_document_only=False, limit=1)
        self.assertEqual(len(paths_lim1), 1)

        # Test limit=10
        paths_lim10 = builder.find_multihop_paths(max_hops=3, min_hops=2, intra_document_only=False, limit=10)
        self.assertGreaterEqual(len(paths_lim10), 2)
        has_cross = any(p["is_cross_document"] for p in paths_lim10)
        has_intra = any(not p["is_cross_document"] for p in paths_lim10)
        self.assertTrue(has_cross, "Cross-document path should be discovered and included")
        self.assertTrue(has_intra, "Intra-document path should be discovered and included")


class TestTopicGate(unittest.TestCase):
    def setUp(self):
        if not _has_networkx():
            self.skipTest("networkx not installed")
        kg = KnowledgeGraphBuilder()
        # two papers on the same topic: they share several specific concepts
        for s, p, o in [("Intentionality", "requires", "Mental Representation"),
                        ("Mental Representation", "requires", "Representational Vehicle"),
                        ("Global Workspace", "enables", "Access Consciousness"),
                        ("Access Consciousness", "requires", "Attention")]:
            _add_crossdoc_cleanup(kg, s, p, o, "mind_a.pdf")
        for s, p, o in [("Representational Vehicle", "implies", "Neural Encoding"),
                        ("Global Workspace", "requires", "Attention"),
                        ("Mental Representation", "depends on", "Neural Encoding")]:
            _add_crossdoc_cleanup(kg, s, p, o, "mind_b.pdf")
        # an unrelated paper that shares exactly one concept with mind_b
        for s, p, o in [("Neural Encoding", "implies", "Compression Ratio"),
                        ("Compression Ratio", "requires", "Entropy Coder"), ("Entropy Coder", "implies", "Bitrate"),
                        ("Bitrate", "requires", "Bandwidth"), ("Bandwidth", "implies", "Latency"),
                        ("Latency", "requires", "Buffer Size"), ("Video Codec", "implies", "Entropy Coder"),
                        ("Video Codec", "requires", "Motion Estimation"), ("Motion Estimation", "implies", "Bitrate")]:
            _add_crossdoc_cleanup(kg, s, p, o, "codecs.pdf")
        self.kg = kg

    def test_overlap_scores(self):
        ov = self.kg.doc_topic_overlap()
        self.assertGreaterEqual(ov[("mind_a.pdf", "mind_b.pdf")], 0.5)
        self.assertLess(ov[("mind_b.pdf", "codecs.pdf")], 0.25)
        self.assertEqual(ov[("mind_a.pdf", "codecs.pdf")], 0.0)

    def test_same_topic_paths_kept_unrelated_blocked(self):
        paths = self.kg.find_multihop_paths(max_hops=3, min_hops=2, limit=200, intra_document_only=False,
                                            min_topic_overlap=0.3, verify_with_gentzen=False)
        cross = [p for p in paths if p["is_cross_document"]]
        self.assertTrue(cross)
        for p in cross:
            self.assertNotIn("codecs.pdf", p["provenance_docs"])
            self.assertGreaterEqual(p["doc_topic_overlap"], 0.3)
        self.assertTrue(any(set(p["provenance_docs"]) == {"mind_a.pdf", "mind_b.pdf"} for p in cross))
        everything = self.kg.find_multihop_paths(max_hops=3, min_hops=2, limit=500, intra_document_only=False,
                                                 min_topic_overlap=0.0, verify_with_gentzen=False)
        self.assertTrue(any("codecs.pdf" in p["provenance_docs"] and p["is_cross_document"] for p in everything))

    def test_ranking_prefers_entailment_chains_and_limits(self):
        paths = self.kg.find_multihop_paths(max_hops=3, min_hops=2, limit=5, intra_document_only=True,
                                            verify_with_gentzen=False)
        self.assertLessEqual(len(paths), 5)
        scores = [p["rank_score"] for p in paths]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(paths[0]["logic_basis"], "kg_entailment_chain")

    def test_composite_claim_keeps_every_hop(self):
        p = next(p for p in self.kg.find_multihop_paths(max_hops=3, min_hops=3, limit=50, verify_with_gentzen=False))
        for prop in p["propositions"]:
            self.assertIn(prop["statement"].rstrip(".")[1:].lower(), p["composite_claim"].lower())


class TestReversedExclusion(unittest.TestCase):
    REV = (["Human Agency", "Freedom", "Insular Life"],
           "(Human Agency) --[requires]--> (Freedom) -> (Freedom) --[excludes] <-- (Insular Life)")

    def test_reversed_exclusion_as_last_hop_is_a_chain(self):
        rep = path_logic_report(*self.REV)
        self.assertEqual(rep["logic_basis"], "kg_entailment_chain")
        self.assertEqual(path_to_sequent(*self.REV).to_str(),
                         "Human_Agency, (Human_Agency => Freedom), (Insular_Life => ~Freedom) |- ~Insular_Life")

    def test_reversed_exclusion_before_another_hop_breaks(self):
        rep = path_logic_report(["A", "B", "C"], "(A) --[excludes] <-- (B) -> (B) --[implies]--> (C)")
        self.assertEqual(rep["logic_basis"], "kg_relational")

    def test_reversed_entailment_still_breaks(self):
        rep = path_logic_report(["A", "B"], "(A) --[requires] <-- (B)")
        self.assertEqual(rep["logic_basis"], "kg_relational")


@unittest.skipUnless((MODEL_V4 / "config.json").exists(), "model/ (nanoGentzen v4 weights) not present")
class TestLogicFirstDiscovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arch = CotLogicalArchitect(engine=load_engine(str(MODEL_V4)))

    def _builder(self):
        b = KnowledgeGraphBuilder()
        # a busy hub with many non-inferential relations: it owns the top-degree source slots
        for i in range(12):
            _add_logic_first_paths(b, "Hub", "relates to", f"Topic {i}")
            _add_logic_first_paths(b, f"Topic {i}", "contrasts with", f"Other Topic {i}")
        # a quiet entailment chain far from the hub (low-degree nodes)
        _add_logic_first_paths(b, "Consent", "requires", "Disclosure")
        _add_logic_first_paths(b, "Disclosure", "requires", "Transparency")
        _add_logic_first_paths(b, "Opacity", "excludes", "Transparency")
        return b

    def test_logic_first_finds_low_degree_chains(self):
        b = self._builder()
        general = b.find_multihop_paths(require_entailment=True, logic_first=False, max_sources=3,
                                    verify_with_gentzen=False, limit=None)
        new = b.find_multihop_paths(require_entailment=True, max_sources=3, verify_with_gentzen=False, limit=None)
        reprs = {p["path_repr"] for p in new}
        # Consent has degree 1: never a source for the degree-ranked search, always for the logic-first one
        self.assertFalse(any(p["source"] == "Consent" for p in general))
        self.assertGreater(len(new), len(general))
        self.assertIn("(Consent) --[requires]--> (Disclosure) -> (Disclosure) --[requires]--> (Transparency)", reprs)
        self.assertIn("(Disclosure) --[requires]--> (Transparency) -> (Transparency) --[excludes] <-- (Opacity)", reprs)

    def test_verified_only_returns_proven_paths_and_limits_after_filtering(self):
        b = self._builder()
        paths = b.find_multihop_paths(verified_only=True, architect=self.arch, limit=None)
        self.assertTrue(paths)
        self.assertTrue(all(p["gentzen_verified"] and p["epistemic_status"] == "PROVEN_TRUE" for p in paths))
        rev = next(p for p in paths if "<--" in p["path_repr"])
        self.assertIn("¬Opacity", rev["gentzen_sequent"])
        one = b.find_multihop_paths(verified_only=True, architect=self.arch, limit=1)
        self.assertEqual(len(one), 1)
        self.assertTrue(one[0]["gentzen_verified"])

    def test_mixed_mode_keeps_relational_paths_and_adds_chains(self):
        b = self._builder()
        paths = b.find_multihop_paths(verify_with_gentzen=False, limit=None)
        bases = {p["logic_basis"] for p in paths}
        self.assertEqual(bases, {"kg_relational", "kg_entailment_chain"})


class TestPathSearch(unittest.TestCase):
    def setUp(self):
        try:
            import networkx  # noqa: F401
            if not hasattr(networkx, "all_simple_paths"):
                raise ImportError
        except ImportError:
            self.skipTest("networkx not installed")

    def test_relational_paths_are_marked_and_filterable(self):
        from core.kg_builder import Proposition
        kg = KnowledgeGraphBuilder()
        for s, p, o in [("Rain", "causes", "Wet Street"), ("Sprinkler", "causes", "Wet Street"),
                        ("Wet Street", "causes", "Slippery Road"), ("Slippery Road", "implies", "Accident Risk")]:
            kg.add_proposition(Proposition(subject=s, predicate=p, obj=o, statement=f"{s} {p} {o}.",
                                           evidence="", chunk_id="c", doc_name="d", page_number=1))
        arch = CotLogicalArchitect(engine=NanoGentzenEngine(model_path=str(ROOT / "model" / "model.safetensors")))
        paths = kg.find_multihop_paths(max_hops=3, min_hops=2, limit=50, architect=arch)
        by_nodes = {tuple(p["path_nodes"]): p for p in paths}
        chain = by_nodes.get(("Rain", "Wet Street", "Slippery Road"))
        self.assertIsNotNone(chain)
        self.assertEqual(chain["logic_basis"], "kg_entailment_chain")
        self.assertTrue(chain["gentzen_verified"])
        for nodes, p in by_nodes.items():
            if nodes[0] == "Rain" and nodes[-1] == "Sprinkler":
                self.assertFalse(p["gentzen_verified"])
        only = kg.find_multihop_paths(max_hops=3, min_hops=2, limit=50, architect=arch, require_entailment=True)
        self.assertTrue(only)
        self.assertTrue(all(p["logic_basis"] == "kg_entailment_chain" for p in only))


if __name__ == "__main__":
    unittest.main()
