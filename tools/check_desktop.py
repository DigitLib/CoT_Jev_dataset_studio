"""
check_desktop.py - one-command check that the Studio and the nanoGentzen v4 engine work on this machine.

    python tools/check_desktop.py                 # environment + engine + offline end-to-end pipeline
    python tools/check_desktop.py --llm http://localhost:11434/v1 --model qwen2.5:7b    # also test the LLM backend
    python tools/check_desktop.py --quick         # skip the 200-sequent soundness sweep

Nothing here needs an LLM or a network connection (except --llm). Exit code 0 = every required check passed.
Writes only to a temporary folder; your data/ folder is not touched.
"""
import argparse
import importlib
import json
import platform
import random
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS = []          # (section, name, status, detail)   status: PASS | FAIL | WARN | SKIP


def record(section, name, status, detail=""):
    RESULTS.append((section, name, status, detail))
    mark = {"PASS": "✔", "FAIL": "✘", "WARN": "!", "SKIP": "-"}[status]
    print(f"  [{mark}] {name}" + (f": {detail}" if detail else ""), flush=True)


def check(section, name, required=True):
    """Decorator: run fn, record PASS / FAIL (or WARN when not required); fn may return a detail string."""
    def deco(fn):
        def run(*a, **k):
            try:
                detail = fn(*a, **k)
                record(section, name, "PASS", detail or "")
                return True
            except SkipCheck as e:
                record(section, name, "SKIP", str(e))
            except Exception as e:
                record(section, name, "FAIL" if required else "WARN", f"{type(e).__name__}: {e}")
                if "--verbose" in sys.argv:
                    traceback.print_exc()
            return False
        return run
    return deco


class SkipCheck(Exception):
    pass


# ---------------------------------------------------------------------------------------------- environment
PACKAGES = [  # (module, pip name, required, purpose)
    ("streamlit", "streamlit", True, "UI"),
    ("pydantic", "pydantic", True, "v4 decide API"),
    ("numpy", "numpy", True, "v4 NumPy backend"),
    ("networkx", "networkx", True, "knowledge graph"),
    ("pandas", "pandas", True, "exports"),
    ("pyarrow", "pyarrow", True, "Parquet exports"),
    ("pymupdf", "pymupdf", True, "PDF parsing"),
    ("pymupdf4llm", "pymupdf4llm", False, "layout-aware PDF parsing (fallback: plain PyMuPDF)"),
    ("docx", "python-docx", False, "DOCX import"),
    ("httpx", "httpx", True, "LLM backend client"),
    ("pyvis", "pyvis", False, "graph visualisation"),
    ("torch", "torch", False, "v4 GPU/CPU torch backend (fallback: NumPy)"),
    ("transformers", "transformers", False, "torch backend model loading"),
    ("safetensors", "safetensors", False, "torch backend weights"),
]


def environment():
    print("\n== Environment")
    record("env", "python", "PASS" if (3, 10) <= sys.version_info[:2] <= (3, 13) else "WARN",
           f"{platform.python_version()} on {platform.system()} {platform.release()}")
    for mod, pip, required, why in PACKAGES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", getattr(m, "VersionBind", ""))
            extra = ""
            if mod == "torch":
                cuda = m.cuda.is_available()
                extra = f", CUDA {'available: ' + m.cuda.get_device_name(0) if cuda else 'not available'}"
            if mod == "pymupdf4llm" and not hasattr(m, "to_json"):
                record("env", mod, "WARN", f"{ver} has no to_json(); layout analysis falls back. pip install -U pymupdf4llm")
                continue
            record("env", mod, "PASS", f"{ver}{extra}")
        except Exception as e:
            record("env", mod, "FAIL" if required else "WARN", f"missing ({why}); pip install {pip}  [{type(e).__name__}]")
    for f in ("model.safetensors", "config.json", "vocab.json"):
        p = ROOT / "model" / f
        record("env", f"model/{f}", "PASS" if p.exists() else "FAIL", f"{p.stat().st_size:,} bytes" if p.exists() else "missing")


# ---------------------------------------------------------------------------------------------- engine
def engine(quick: bool):
    print("\n== nanoGentzen v4 engine")
    from nanogentzen.engine import load_engine
    from nanogentzen.kernel import Atom, And, Or, Imp, Not, Sequent
    from nanogentzen_v4.checker import check_certificate
    state = {}

    @check("engine", "load (NumPy backend)")
    def load_numpy():
        state["np"] = load_engine(str(ROOT / "model"), backend="numpy")
        s = state["np"].status()
        return f"gates {s['gates']}, s1_temperature {s['s1_temperature']}"

    @check("engine", "load (torch backend)", required=False)
    def load_torch():
        try:
            import torch  # noqa: F401
        except ImportError:
            raise SkipCheck("torch not installed: the Studio will use the NumPy backend")
        state["torch"] = load_engine(str(ROOT / "model"), backend="torch")
        return f"device {state['torch'].device}"

    @check("engine", "torch vs NumPy forward pass", required=False)
    def parity():
        if "torch" not in state:
            raise SkipCheck("no torch backend")
        import numpy as np
        import torch
        from nanogentzen_v4.backend import to_numpy
        from nanogentzen_v4.kernel import model_view_sequent
        eng_t, eng_n = state["torch"], state["np"]
        worst = 0.0
        rng = random.Random(1)
        for _ in range(20):
            seq = _rand_seq(rng, Atom, And, Or, Imp, Not, Sequent)
            v4, _ = eng_n.to_v4_sequent(seq)
            ids = eng_n.tokenizer.encode(model_view_sequent(v4)[0].model_str())
            if len(ids) > 256:
                continue
            row = ids + [0] * (256 - len(ids))
            t_out = [to_numpy(x) for x in eng_t.model.predict_epistemic(torch.tensor([row], device=eng_t.device))]
            n_out = eng_n.model.predict_epistemic(np.array([row]))
            worst = max(worst, max(float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(t_out, n_out)))
        if worst > 1e-3:
            raise AssertionError(f"max |diff| {worst:.2e} > 1e-3")
        return f"max |diff| {worst:.1e} over 20 sequents"

    @check("engine", "known verdicts")
    def verdicts():
        e = state["np"]
        a, b, c = Atom("a"), Atom("b"), Atom("c")
        cases = [
            (Sequent((a, Imp(a, b), Imp(b, c)), (c,)), "PROVEN_TRUE"),
            (Sequent((a, Imp(a, Not(b))), (b,)), "PROVEN_FALSE"),
            (Sequent((a, Not(a)), (b,)), "INCONSISTENT_PREMISES"),
            (Sequent((), (Or(a, Not(a)),)), "INTUITIONISTIC_UNDECIDED"),        # no excluded middle
            (Sequent((), (Not(Not(Or(a, Not(a)))),)), "PROVEN_TRUE"),
        ]
        for seq, want in cases:
            got = e.prove_sequent(seq)["status"]
            if got != want:
                raise AssertionError(f"{seq.to_str()}: {got} != {want}")
        return f"{len(cases)} sequents"

    @check("engine", "soundness sweep (network vs no network, certificates)")
    def sweep():
        if quick:
            raise SkipCheck("--quick")
        net = state.get("torch") or state["np"]
        sym = load_engine(str(ROOT / "model"), backend="symbolic")
        rng = random.Random(7)
        n, proofs, t0 = 200, 0, time.perf_counter()
        for _ in range(n):
            seq = _rand_seq(rng, Atom, And, Or, Imp, Not, Sequent)
            r1 = net.prove_sequent(seq, want_missing_facts=False)
            r2 = sym.prove_sequent(seq, want_missing_facts=False)
            if r1["status"] != r2["status"]:
                raise AssertionError(f"verdict changed by the network on {seq.to_str()}: {r1['status']} / {r2['status']}")
            if r1["status"] == "PROVEN_TRUE":
                proofs += 1
                ok, err = check_certificate(r1["certificate"])
                if not ok:
                    raise AssertionError(f"certificate rejected on {seq.to_str()}: {err}")
        dt = (time.perf_counter() - t0) / n * 1000
        return f"{n} sequents, {proofs} proofs certified, 0 disagreements, {dt:.1f} ms/sequent ({net.backend})"

    @check("engine", "System 1 latency", required=False)
    def latency():
        e = state.get("torch") or state["np"]
        seq = Sequent((Atom("x"), Imp(Atom("x"), Atom("y"))), (Atom("y"),))
        e.evaluate_sequent(seq)
        t0 = time.perf_counter()
        for _ in range(20):
            e.evaluate_sequent(seq)
        return f"{(time.perf_counter() - t0) / 20 * 1000:.1f} ms per call ({e.backend})"

    for fn in (load_numpy, load_torch, parity, verdicts, sweep, latency):
        fn()
    return state.get("torch") or state.get("np")


def _rand_seq(rng, Atom, And, Or, Imp, Not, Sequent):
    atoms = ["a", "b", "c", "d"]

    def f(depth):
        if depth == 0 or rng.random() < 0.3:
            x = Atom(rng.choice(atoms))
            return Not(x) if rng.random() < 0.2 else x
        op = rng.choice([And, Or, Imp, Imp, Not])
        return Not(f(depth - 1)) if op is Not else op(f(depth - 1), f(depth - 1))
    return Sequent(tuple(f(2) for _ in range(rng.randint(1, 4))), (f(2),))


# ---------------------------------------------------------------------------------------------- pipeline
DOC_A = ("Philosophy of Mind",
         ["Intentionality requires mental representation in every cognitive system.",
          "Mental representation requires a representational vehicle that carries content.",
          "Consciousness differs from attention in the global workspace literature.",
          "Phenomenal experience does not require linguistic report in most theories."])
DOC_B = ("Cognitive Neuroscience",
         ["A representational vehicle implies neural encoding in the cortex.",
          "Mental representation depends on neural encoding of stimulus features.",
          "Neural encoding requires synaptic plasticity during learning.",
          "Attention enables selective processing of sensory input."])


def pipeline(engine_obj, tmp: Path):
    print("\n== Offline end-to-end pipeline (no LLM)")
    st = {}

    @check("pipeline", "PDF parsing")
    def parse():
        from core.document_parser import DocumentParser, DocumentPage
        try:
            import pymupdf
        except ImportError:
            st["pages"] = {f"{n}.pdf": [DocumentPage(page_number=1, text="\n\n".join(s), blocks=[])] for n, s in (DOC_A, DOC_B)}
            raise SkipCheck("pymupdf not installed: using text pages for the rest of the pipeline")
        st["pages"] = {}
        parser = DocumentParser()
        for title, sents in (DOC_A, DOC_B):
            path = tmp / f"{title.replace(' ', '_')}.pdf"
            doc = pymupdf.open()
            page = doc.new_page()
            y = 72
            page.insert_text((72, y), title, fontsize=16)
            for s in sents:
                y += 40
                page.insert_textbox(pymupdf.Rect(72, y, 520, y + 36), s, fontsize=11)
            doc.save(str(path))
            doc.close()
            pages = parser.parse_file(str(path))
            text = " ".join(p.text for p in pages)
            if "representation" not in text.lower():
                raise AssertionError(f"text not extracted from {path.name}")
            st["pages"][path.name] = pages
            path.unlink()                                  # file must not stay locked (Windows)
        eng = type(parser.layout_engine).__name__ + ("" if getattr(parser.layout_engine, "pymupdf4llm", None) else " (fallback parser)")
        return f"2 PDFs, {eng}"

    @check("pipeline", "chunking (stable ids)")
    def chunk():
        from core.chunker import DocumentChunker
        st["chunks"] = []
        for name, pages in st["pages"].items():
            a = DocumentChunker(min_chunk_chars=30).chunk_pages(name, pages)
            b = DocumentChunker(min_chunk_chars=30).chunk_pages(name, pages)
            if [c.chunk_id for c in a] != [c.chunk_id for c in b]:
                raise AssertionError("chunk ids differ between identical runs")
            st["chunks"] += a
        if len(st["chunks"]) < 2:
            raise AssertionError(f"only {len(st['chunks'])} chunks")
        return f"{len(st['chunks'])} chunks"

    @check("pipeline", "knowledge graph (heuristic extractor, evidence check)")
    def kg():
        from core.kg_builder import KnowledgeGraphBuilder
        k = KnowledgeGraphBuilder(llm_client=None)
        for c in st["chunks"]:
            for p in k._extract_propositions_for_chunk(c, model="none"):
                k.add_proposition(p)
        s = k.get_stats()
        if s["num_edges"] < 3:
            raise AssertionError(f"only {s['num_edges']} relations extracted")
        neg = [d for _, _, d in k.graph.edges(data=True) if str(d.get("predicate", "")).startswith("not ")]
        if not neg:
            raise AssertionError("'does not require' was not kept as a negated relation")
        st["kg"] = k
        return f"{s['num_nodes']} entities, {s['num_edges']} relations, evidence {s['evidence']}"

    @check("pipeline", "cross-document KG paths (topic gate) + verification")
    def paths():
        from nanogentzen.architect_cot import CotLogicalArchitect
        k = st["kg"]
        arch = CotLogicalArchitect(engine=engine_obj)
        st["arch"] = arch
        ps = k.find_multihop_paths(max_hops=3, min_hops=2, limit=50, intra_document_only=False, architect=arch)
        cross = [p for p in ps if p["is_cross_document"]]
        verified = [p for p in ps if p["gentzen_verified"]]
        if not cross:
            raise AssertionError(f"no cross-document path between same-topic documents (overlap {k.doc_topic_overlap()})")
        for p in verified:
            if p["logic_basis"] != "kg_entailment_chain":
                raise AssertionError(f"verified path is not an entailment chain: {p['path_repr']}")
        blocked = k.find_multihop_paths(intra_document_only=False, min_topic_overlap=0.99, verify_with_gentzen=False)
        if any(p["is_cross_document"] for p in blocked):
            raise AssertionError("topic gate did not block cross-document paths at overlap 0.99")
        st["paths"] = ps
        return f"{len(ps)} paths, {len(cross)} cross-document, {len(verified)} verified entailment chains"

    @check("pipeline", "ClearCoT sample (logic tree + offline CoT + DB + exports)")
    def clearcot():
        from core.cot_generator import CoTGenerator
        from core.dataset_manager import DatasetManager, DatasetSample
        from core.batch_generator import decide_sample_status
        from nanogentzen.architect_cot import with_proof_record
        p = next((x for x in st["paths"] if x["gentzen_verified"]), st["paths"][0])
        tree = st["arch"].generate_best_logic_tree(premises=[p["composite_claim"]], conclusion=p["target"],
                                                   path_nodes=p["path_nodes"], path_repr=p["path_repr"])
        cot = CoTGenerator(llm_client=None).build_cot_from_logic_tree(
            question=f"How does {p['source']} relate to {p['target']}?", target_claim=p["composite_claim"],
            logic_tree_text=tree["logic_tree_text"], context_text=" ".join(c.text for c in st["chunks"]),
            relevance_info=tree.get("relevance"), path_nodes=p["path_nodes"], is_verified=tree["is_verified"],
            epistemic_status=tree["epistemic_status"])
        status, why = decide_sample_status("accepted", tree["is_verified"], cot.get("is_grounded", False), cot.get("is_fallback", False))
        if status != "draft" or "LLM output unavailable; template reasoning" not in why:
            raise AssertionError(f"offline template CoT must be stored as draft, got {status} {why}")
        db = DatasetManager(db_path=str(tmp / "t.db"))
        db.save_sample(DatasetSample(sample_id="", doc_name=p["doc_name"], question="q?", reasoning=cot["reasoning"],
                                     answer=cot["answer"], status="accepted", gentzen_verified=tree["is_verified"],
                                     provenance_json=json.dumps(with_proof_record({"documents": p["provenance_docs"]}, tree), default=str)))
        n = db.export_jsonl(str(tmp / "sft.jsonl"))
        row = json.loads((tmp / "sft.jsonl").read_text(encoding="utf-8").splitlines()[0])
        if "split" not in row:
            raise AssertionError("no split column")
        try:
            import pyarrow  # noqa: F401
            db.export_parquet(str(tmp / "sft.parquet"))
            pq = "parquet ok"
        except ImportError:
            pq = "parquet skipped (no pyarrow)"
        return f"{tree['logic_basis']} / {tree['epistemic_status']}, {n} SFT row, {pq}"

    @check("pipeline", "Jev scenario + counterfactuals + exports")
    def jev():
        from core.jev_generator import JevScenarioGenerator
        from core.jev_dataset_manager import JevDatasetManager, JevSample
        g = JevScenarioGenerator(llm_client=None, engine=engine_obj)
        sc = g._heuristic_fallback(context_text=" ".join(DOC_A[1]), doc_name="mind.pdf", pages=[1],
                                   workflow="scientific_theories", path_nodes=["Intentionality", "Mental Representation"])
        if not sc["gentzen_verified"]:
            raise AssertionError(f"offline scenario not verified: {sc['epistemic_status']}")
        variants = g.make_counterfactuals(sc)
        if not variants:
            raise AssertionError("no counterfactual variants")
        m = JevDatasetManager(db_path=str(tmp / "j.db"))

        def save(s, extra=None):
            prov = dict(s["provenance"], **(extra or {}))
            return m.save_sample(JevSample(sample_id="", doc_name=prov.get("doc_name", "d"), workflow=s["workflow"],
                                           state_json=json.dumps(s["state"]), questions_json=json.dumps(s["questions"]),
                                           gold_json=json.dumps(s["gold"], default=str), factors_json=json.dumps(s["factors"]),
                                           policy_rules_json=json.dumps(s["policy_rules"]), target_decision=s["target_decision"],
                                           gentzen_verified=s["gentzen_verified"], provenance_json=json.dumps(prov, default=str)))
        pid = save(sc)
        for v in variants:
            v["provenance"]["counterfactual"]["of"] = pid
            save(v)
        n = m.export_jev_jsonl(str(tmp / "jev.jsonl"))
        labels = sorted({next(iter(json.loads(l)["gold"].values()))["label"] for l in (tmp / "jev.jsonl").read_text(encoding="utf-8").splitlines()})
        try:
            import pyarrow  # noqa: F401
            m.export_open_jev_parquet(str(tmp / "s1mb.parquet"))
            m.export_typed_decisions_parquet(str(tmp / "td.parquet"), column_mode="positional")
            pq = "parquet ok"
        except ImportError:
            pq = "parquet skipped (no pyarrow)"
        return f"{n} records ({len(variants)} counterfactual), labels {labels}, {pq}"

    @check("pipeline", "knowledge graph save / load")
    def persist():
        from core.kg_builder import KnowledgeGraphBuilder
        k = st["kg"]
        k.save(str(tmp / "kg.json"), chunks=st["chunks"])
        k2 = KnowledgeGraphBuilder()
        res = k2.load(str(tmp / "kg.json"))
        if k2.graph.number_of_edges() != k.graph.number_of_edges() or len(res["chunks"]) != len(st["chunks"]):
            raise AssertionError("round trip changed the graph")
        return f"{k2.graph.number_of_edges()} relations, {len(res['chunks'])} chunks restored"

    for fn in (parse, chunk, kg, paths, clearcot, jev, persist):
        if not fn() and fn in (chunk, kg):
            print("  (later steps need this one; stopping the pipeline)")
            break


def llm(url: str, model: str):
    print("\n== LLM backend")
    from llm_backend.client import LocalLLMClient

    @check("llm", f"health {url}")
    def health():
        c = LocalLLMClient(base_url=url)
        if not c.health_check(force=True):
            raise AssertionError("backend not reachable")
        return f"models: {', '.join(c.list_models(force=True)[:5])}"

    @check("llm", f"generate with {model}")
    def gen():
        c = LocalLLMClient(base_url=url)
        out = c.generate(prompt="Reply with the single word: ready", model=model, max_tokens=8, temperature=0)
        if not out.strip():
            raise AssertionError("empty response")
        return repr(out.strip()[:40])

    health() and gen()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--llm", default=None, help="OpenAI-compatible base URL, e.g. http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen2.5:7b")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    print(f"nanoGentzen Studio desktop check  ({ROOT})")
    environment()
    eng = engine(a.quick)
    if eng is not None:
        with tempfile.TemporaryDirectory() as d:
            pipeline(eng, Path(d))
    if a.llm:
        llm(a.llm, a.model)
    fails = [r for r in RESULTS if r[2] == "FAIL"]
    warns = [r for r in RESULTS if r[2] == "WARN"]
    print(f"\n== Summary: {sum(r[2] == 'PASS' for r in RESULTS)} passed, {len(fails)} failed, {len(warns)} warnings, "
          f"{sum(r[2] == 'SKIP' for r in RESULTS)} skipped")
    for r in fails:
        print(f"  FAIL  {r[0]} / {r[1]}: {r[3]}")
    for r in warns:
        print(f"  WARN  {r[0]} / {r[1]}: {r[3]}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
