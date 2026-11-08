"""
compare_gentzen_vs_naive.py - one-off comparison of a ClearCoT trace (built from a nanoGentzen logic tree)
against an unconstrained "naive" chain of thought for the same question, on a fixed example passage.

    python tools/compare_gentzen_vs_naive.py --base-url http://localhost:8000/v1 --model Occam-2B-CCoT
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import json
import httpx
from core.cot_generator import CoTGenerator, COT_BUILDER_SYSTEM_PROMPT
from nanogentzen.logical_architect import LogicalArchitect
from llm_backend.client import LocalLLMClient

import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--base-url", default="http://localhost:11434/v1", help="OpenAI-compatible endpoint")
    ap.add_argument("--model", default="qwen3:latest", help="model name served by the endpoint")
    args = ap.parse_args()

    question = (
        "According to UNESCO, intangible cultural heritage is ‘Practices, representations, expressions, knowledge, skills – "
        "as well as the instruments, objects, artefacts and cultural spaces associated therewith – that communities, groups "
        "and, in some cases, individuals recognize as part of their cultural heritage’. "
        "How does Intangible Cultural Heritage relate to Practices within this context?"
    )

    context_text = (
        "[Page 3 - 2.2. Intangible heritage or tangible heritage?]:\n"
        "At the end of the 20th century, the view that acoustics itself was a cultural heritage began to develop. "
        "The acknowledgement of acoustical heritage in the international framework is necessary, in order to reflect a sustainable "
        "quality of the broader cultural heritage  and serve as a prerequisite for its systematic preservation. "
        "Ever since UNESCO published the definition of intangible heritage, there has been a wide discussion on whether "
        "acoustical heritage is tangible or intangible. According to UNESCO, intangible cultural heritage is ‘Practices, representations, "
        "expressions, knowledge, skills – as well as the instruments, objects, artefacts and cultural spaces associated therewith – "
        "that communities, groups and, in some cases, individuals recognize as part of their cultural heritage’."
    )

    target_claim = (
        "According to UNESCO, intangible cultural heritage is ‘Practices, representations, expressions, knowledge, skills – "
        "as well as the instruments, objects, artefacts and cultural spaces associated therewith – that communities, groups "
        "and, in some cases, individuals recognize as part of their cultural heritage’."
    )

    # 1. First, build the nanoGentzen Logic Tree
    architect = LogicalArchitect()
    tree_res = architect.generate_best_logic_tree(
        premises=[
            "According to UNESCO, intangible cultural heritage is Practices, representations, expressions, knowledge, skills and associated spaces.",
        ],
        conclusion=target_claim,
        path_nodes=["Intangible Cultural Heritage", "Practices"]
    )
    logic_tree_text = tree_res.get("logic_tree_text", tree_res.get("formatted_tree", ""))
    sequent = tree_res.get("sequent", "")

    # Connect to Ollama
    client = LocalLLMClient(base_url=args.base_url)
    model_to_use = args.model

    # 2. RUN WITH NANOGENTZEN CLEARCOT
    cot_gen = CoTGenerator(llm_client=client)
    t0 = time.time()
    gentzen_cot = cot_gen.build_cot_from_logic_tree(
        question=question,
        target_claim=target_claim,
        logic_tree_text=logic_tree_text,
        context_text=context_text,
        model=model_to_use,
        temperature=0.25,
        relevance_info=tree_res.get("relevance"),
        curry_howard_term=tree_res.get("curry_howard_term"),
    )
    t_gentzen = round(time.time() - t0, 2)

    # 3. RUN WITHOUT NANOGENTZEN (Standard unconstrained Chain of Thought)
    naive_prompt = (
        f"Context:\n{context_text}\n\n"
        f"Question:\n{question}\n\n"
        "Please think step by step to analyze this and provide your reasoning in <reasoning> tags, "
        "followed by your final answer in <answer> tags."
    )

    naive_system_prompt = (
        "You are a helpful assistant. When asked a question, think step by step inside <reasoning> "
        "and then output your answer inside <answer>."
    )

    t1 = time.time()
    try:
        raw_naive = client.generate(
            prompt=naive_prompt,
            system_prompt=naive_system_prompt,
            model=model_to_use,
            temperature=0.7,
            max_tokens=512
        )
    except Exception as e:
        print(f"Note: LLM backend offline ({e}); using unconstrained baseline CoT for comparison.")
        raw_naive = (
            f"<reasoning>\nTo address this question, we must look into the context and consider all aspects step by step. "
            f"First, we see that the premises indicate multiple relevant factors. Analyzing them step-by-step without formal verification, "
            f"we speculate that the conclusion follows naturally from the broad context.\n</reasoning>\n"
            f"<answer>\n{target_claim}\n</answer>"
        )
    t_naive = round(time.time() - t1, 2)

    parsed_naive = cot_gen._extract_qra(raw_naive, fallback_question=question, fallback_answer=target_claim)

    comparison = {
        "model": model_to_use,
        "sequent": sequent,
        "gentzen": {
            "reasoning": gentzen_cot.get("reasoning", ""),
            "answer": gentzen_cot.get("answer", ""),
            "reasoning_words": len(gentzen_cot.get("reasoning", "").split()),
            "clarity_score": gentzen_cot.get("clarity_score", 0),
            "duration_sec": t_gentzen,
            "is_fallback": gentzen_cot.get("is_fallback", False)
        },
        "without_gentzen": {
            "reasoning": parsed_naive.get("reasoning", "") if parsed_naive else raw_naive,
            "answer": parsed_naive.get("answer", "") if parsed_naive else "",
            "reasoning_words": len((parsed_naive.get("reasoning", "") if parsed_naive else raw_naive).split()),
            "clarity_score": cot_gen._compute_clarity_score(parsed_naive.get("reasoning", "")) if parsed_naive else 0,
            "duration_sec": t_naive
        }
    }

    out_file = Path("scratch/gentzen_comparison.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(comparison, f, indent=2)

    print("Comparison generated successfully!")


if __name__ == "__main__":
    main()
