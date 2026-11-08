# Roadmap

Planned work after 0.1.0. Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## Throughput

- [ ] **Async knowledge-graph extraction.** `KnowledgeGraphBuilder.build_from_chunks` uses a thread pool; move
  it to the client's async batch API (`LocalLLMClient.async_generate_batch`) with a concurrency limit, so large
  documents do not hit connection timeouts on local vLLM / Ollama servers.
- [x] **Incremental graphs.** `append_from_chunks` extracts only chunks that are not yet in the graph.

## Scale

- [ ] **Large-graph rendering.** The graph view shows the 800 highest-degree nodes; add cluster-level
  (community) views for graphs with thousands of entities.

## Code structure

- [ ] **Retire the v3 modules.** Move `nanogentzen/cot/` and `nanogentzen/jev/` onto the v4 formula types, then
  remove `nanogentzen/kernel.py`, `search.py`, `parser.py` and `domain_registry.py`, which duplicate code from
  `nanogentzen_v4/` (see [nanogentzen/README.md](nanogentzen/README.md)). The two pipelines stay separate.

## Dataset quality

- [ ] **Deeper ClearCoT proofs.** Prefer paths with three or more inference hops and fewer taxonomy-only steps,
  and require the question and reasoning to mention every atom of the proved sequent.
- [ ] **Jev label vocabulary per workflow.** Constrain option labels and question ids per workflow
  (`nonstandard_options` is reported today, not enforced).
- [ ] **Better Jev state extraction.** Fewer truncated entity names and attribute values that contradict the
  evidence.
