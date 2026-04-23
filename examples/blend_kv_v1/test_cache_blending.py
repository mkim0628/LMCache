# SPDX-License-Identifier: Apache-2.0
"""
CacheBlend benchmark: accuracy, latency, and KV-cache hit ratio.

Scenario
--------
Phase 1 – KV-Cache Pre-computation (store)
    Send Doc A, Doc B, and Doc C through the model in their natural order.
    LMCache stores per-chunk KV tensors for each document.

Phase 2 – CacheBlend evaluation (reordered)
    Submit a query whose document order differs from Phase 1 (Doc B first, then
    Doc A).  LMCache detects the chunk-level reuse and blends the cached KVs
    instead of recomputing them from scratch.

Cross-attention accuracy design
    Doc A  – Alice Chen invented the AlphaCache algorithm.
    Doc B  – Project Athena's technical lead is Dr. Alice Chen; AlphaCache was
             cited as the key reason for her selection.
    Doc C  – Unrelated maritime-history filler (distractor).

    Neither document alone can answer:
        "What algorithm was invented by the technical lead of Project Athena,
         and what type of system did it revolutionize?"
    Correct answer: AlphaCache / distributed key-value storage
    (requires cross-document reasoning between Doc A and Doc B)

Metrics
-------
* Accuracy     – whether the generated answer contains the expected keywords
* Latency      – wall-clock generation time per query (seconds)
* KV hit ratio – fraction of input tokens served from cache
                 (computed from per-document chunk counts)

Usage
-----
    python test_cache_blending.py [--model MODEL] [--use-disk] [--max-tokens N]
"""

import argparse
import contextlib
import os
import time
from dataclasses import asdict, dataclass

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


# ---------------------------------------------------------------------------
# Document content
# Each document is written to exceed 256 tokens (= default chunk_size) so
# that at least one complete chunk is stored in LMCache per document.
# ---------------------------------------------------------------------------

DOC_A = (
    "Alice Chen is a distinguished professor of computer science at the Metropolitan "
    "University of Technology. She holds a PhD from Stanford University in distributed "
    "systems and artificial intelligence. Her research focuses on federated learning "
    "and privacy-preserving computation. She joined Metropolitan University in 2015 "
    "after spending eight years at Google Research, where she led the distributed "
    "systems team. Professor Chen has published over 150 peer-reviewed papers in top "
    "venues including SOSP, OSDI, NSDI, and NeurIPS. She is the inventor of the "
    "AlphaCache algorithm, which revolutionized distributed key-value storage. Her "
    "laboratory is known as the Systems and Intelligence Research Group (SIRG). The "
    "SIRG has received over $10 million in funding from NSF, DARPA, and industry "
    "partners. Alice is also the author of the widely-used textbook 'Modern "
    "Distributed Systems: Principles and Practice,' which has been adopted by over "
    "200 universities worldwide. She serves as the program chair of USENIX ATC and "
    "is on the editorial board of ACM TOCS. Professor Chen's most recent breakthrough "
    "involves applying transformer architectures to optimize consensus protocols in "
    "Byzantine fault-tolerant systems. Her PhD advisee Marcus Webb recently won the "
    "Best Paper Award at OSDI for work on cache-coherent memory systems. Professor "
    "Chen is widely regarded as one of the foremost experts in large-scale storage "
    "architecture and has consulted for major technology companies and government "
    "agencies. Her AlphaCache system has been deployed in production environments "
    "handling billions of requests per day across globally distributed infrastructure."
)

DOC_B = (
    "Project Athena is a landmark initiative launched in 2023 by the Federal "
    "Department of Advanced Research and Technology. The project aims to build the "
    "next generation of exascale computing infrastructure for national security and "
    "scientific applications. Project Athena has a budget of $500 million spread "
    "over five years, with the primary objective of achieving 2 exaflop sustained "
    "performance on real-world workloads. The technical lead for Project Athena is "
    "Dr. Alice Chen, whose pioneering work on the AlphaCache algorithm was cited as "
    "the key reason for her selection to lead this critical national program. The "
    "project involves three major components: the Hercules compute cluster, the "
    "Prometheus storage fabric, and the Minerva interconnect. The Hercules cluster "
    "consists of 16,384 nodes each equipped with 8 NVIDIA H100 GPUs and 2 TB of "
    "NVMe storage. The Prometheus storage fabric provides 100 petabytes of "
    "high-performance storage with sub-millisecond latency. The Minerva interconnect "
    "uses a next-generation optical switching fabric with 800 Gbps per port. Project "
    "Athena is currently in Phase 2 of 4, with hardware procurement completed and "
    "software stack integration underway. The project is expected to be fully "
    "operational by Q3 2025 and will be hosted at three geographically distributed "
    "data centers: Denver, Colorado; Raleigh, North Carolina; and Portland, Oregon. "
    "Partner institutions include MIT, Carnegie Mellon University, and the National "
    "Renewable Energy Laboratory. The project is considered a cornerstone of the "
    "national computing strategy for the next decade."
)

DOC_C = (
    "The history of maritime exploration has shaped the modern world in profound "
    "ways. From the Phoenician traders who navigated the Mediterranean Sea three "
    "thousand years ago to the great Age of Discovery in the fifteenth and sixteenth "
    "centuries, seafaring civilizations have driven economic and cultural exchange "
    "across continents. The Portuguese explorer Vasco da Gama opened the sea route "
    "to India in 1498, fundamentally transforming trade between Europe and Asia. "
    "Christopher Columbus, sailing under the Spanish flag in 1492, initiated "
    "sustained contact between Europe and the Americas, leading to the Columbian "
    "Exchange, which introduced crops like potatoes, tomatoes, and maize to the Old "
    "World while bringing horses, cattle, and wheat to the New World. The development "
    "of accurate nautical charts, the astrolabe, and later the chronometer enabled "
    "increasingly precise navigation. Steam-powered vessels gradually replaced sailing "
    "ships during the Industrial Revolution, drastically reducing voyage times and "
    "enabling more reliable freight and passenger transport. The opening of the Suez "
    "Canal in 1869 and the Panama Canal in 1914 further transformed global trade "
    "routes. Today, maritime shipping accounts for approximately ninety percent of "
    "world trade by volume, with container shipping revolutionizing cargo transport "
    "since the 1950s. Modern port facilities handle millions of containers each year "
    "and serve as critical nodes in the global supply chain network."
)

# Cross-attention question: cannot be answered from Doc A or Doc B alone.
#   Doc A → Alice Chen invented AlphaCache, which revolutionized distributed KV storage
#   Doc B → technical lead of Project Athena is Dr. Alice Chen; AlphaCache cited
#   Combined → AlphaCache / distributed key-value storage
CROSS_QUESTION = (
    "Based on the documents above, what is the name of the algorithm invented by the "
    "technical lead of Project Athena, and what type of storage system did that "
    "algorithm revolutionize? Provide a concise answer."
)

EXPECTED_KEYWORDS = ["AlphaCache", "alphacache", "alpha cache", "Alpha Cache"]


# ---------------------------------------------------------------------------
# Metric containers
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    label: str
    generated_text: str
    latency_sec: float
    input_tokens: int
    cached_tokens: int

    @property
    def kv_hit_ratio(self) -> float:
        if self.input_tokens == 0:
            return 0.0
        return self.cached_tokens / self.input_tokens

    @property
    def is_accurate(self) -> bool:
        return any(kw.lower() in self.generated_text.lower() for kw in EXPECTED_KEYWORDS)


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------


def setup_env(use_disk: bool, blend_special_str: str, chunk_size: int) -> None:
    os.environ["LMCACHE_CHUNK_SIZE"] = str(chunk_size)
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = blend_special_str
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"

    if use_disk:
        os.environ["LMCACHE_LOCAL_CPU"] = "False"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "10"
    else:
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "10"


# ---------------------------------------------------------------------------
# LLM builder
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def build_llm(model: str):
    ktc = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
    args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=8192,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    llm = LLM(**asdict(args))
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------


def encode_no_bos(tokenizer, text: str) -> list[int]:
    """Encode text and strip the leading BOS token if present."""
    ids = tokenizer.encode(text)
    # Most HF tokenizers prepend BOS (id=1); drop it so we can manually
    # position it at the very beginning of the full prompt.
    if ids and ids[0] == tokenizer.bos_token_id:
        ids = ids[1:]
    return ids


def count_full_chunk_tokens(n_tokens: int, chunk_size: int) -> int:
    """Number of tokens covered by complete chunks."""
    return (n_tokens // chunk_size) * chunk_size


def build_prompt(
    tokenizer,
    sep_ids: list[int],
    sys_ids: list[int],
    doc_segments: list[list[int]],
    question_ids: list[int],
    eos_ids: list[int],
) -> list[int]:
    """
    Construct a blended prompt:
        sys + SEP + doc0 + SEP + doc1 + ... + SEP + question + eos
    """
    prompt = list(sys_ids)
    for doc in doc_segments:
        prompt += sep_ids + doc
    prompt += sep_ids + question_ids + eos_ids
    return prompt


# ---------------------------------------------------------------------------
# Single generate + measure
# ---------------------------------------------------------------------------


def run_generate(
    llm: LLM,
    prompt_ids: list[int],
    sampling_params: SamplingParams,
    label: str,
    cached_tokens: int,
) -> RunResult:
    print(f"\n{'=' * 60}")
    print(f"  Running: {label}")
    print(f"  Input tokens : {len(prompt_ids)}")
    print(f"  Cached tokens: {cached_tokens}  "
          f"(est. hit ratio: {cached_tokens / max(len(prompt_ids), 1):.1%})")
    print(f"{'=' * 60}")

    t0 = time.perf_counter()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt_ids},
        sampling_params=sampling_params,
    )
    latency = time.perf_counter() - t0

    generated = outputs[0].outputs[0].text if outputs else ""
    result = RunResult(
        label=label,
        generated_text=generated,
        latency_sec=latency,
        input_tokens=len(prompt_ids),
        cached_tokens=cached_tokens,
    )

    print(f"  Generated    : {generated!r}")
    print(f"  Latency      : {latency:.3f}s")
    print(f"  Accurate?    : {'YES ✓' if result.is_accurate else 'NO  ✗'}")
    return result


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------


def benchmark(args) -> None:
    setup_env(args.use_disk, args.blend_special_str, args.chunk_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Encode the separator (strip BOS)
    sep_ids = encode_no_bos(tokenizer, args.blend_special_str)

    # System prompt  (Mistral [INST] wrapper; adapt for other models via --sys-prompt)
    if args.sys_prompt:
        sys_ids = encode_no_bos(tokenizer, args.sys_prompt)
        eos_ids = []
    else:
        # Mistral-style: BOS [INST] text [/INST]
        sys_ids = [tokenizer.bos_token_id] + encode_no_bos(
            tokenizer,
            "[INST] You are a precise question-answering assistant. "
            "Read the provided documents carefully and answer only from their content. "
            "[/INST]",
        )
        eos_ids = []

    # Encode each document
    doc_a_ids = encode_no_bos(tokenizer, DOC_A)
    doc_b_ids = encode_no_bos(tokenizer, DOC_B)
    doc_c_ids = encode_no_bos(tokenizer, DOC_C)
    question_ids = encode_no_bos(tokenizer, CROSS_QUESTION)

    # Report document token counts
    print("\n── Document token counts ────────────────────────────────────")
    for name, ids in [("Doc A", doc_a_ids), ("Doc B", doc_b_ids), ("Doc C", doc_c_ids)]:
        full = count_full_chunk_tokens(len(ids), args.chunk_size)
        print(f"  {name}: {len(ids)} tokens  ({full} in full chunks of {args.chunk_size})")
    print(f"  Question  : {len(question_ids)} tokens")
    print()

    sampling = SamplingParams(temperature=0, top_p=1.0, max_tokens=args.max_tokens)
    results: list[RunResult] = []

    with build_llm(args.model) as llm:
        # ------------------------------------------------------------------
        # Warmup (not measured)
        # ------------------------------------------------------------------
        warmup_ids = encode_no_bos(tokenizer, "Hello, how are you? " * 50)
        print("── Warmup ───────────────────────────────────────────────────")
        llm.generate(
            prompts={"prompt_token_ids": warmup_ids},
            sampling_params=SamplingParams(temperature=0, max_tokens=1),
        )
        print("  Warmup done.\n")

        # ------------------------------------------------------------------
        # Phase 1: Pre-computation (store)
        #
        # Order: Doc A → Doc B → Doc C
        # This populates the LMCache store with KV tensors for each doc.
        # No tokens are in cache yet → hit ratio = 0.
        # ------------------------------------------------------------------
        store_prompt = build_prompt(
            tokenizer, sep_ids, sys_ids,
            [doc_a_ids, doc_b_ids, doc_c_ids],
            question_ids, eos_ids,
        )
        # First call: cold cache → 0 cached tokens
        r_store = run_generate(
            llm, store_prompt, sampling,
            label="Phase 1 – Store (A→B→C, cold cache, baseline accuracy)",
            cached_tokens=0,
        )
        results.append(r_store)
        time.sleep(1)

        # ------------------------------------------------------------------
        # Phase 2a: CacheBlend with reordered docs (Doc B first, then Doc A)
        #
        # The order differs from Phase 1, so vLLM prefix caching cannot help.
        # LMCache detects that doc_b and doc_a chunks are cached and blends
        # their KV tensors to avoid full recomputation.
        #
        # Estimated cached tokens = full chunks from doc_a + full chunks from doc_b
        # (doc_c is not included in this prompt, so its cached KV is unused here)
        # ------------------------------------------------------------------
        blend_prompt = build_prompt(
            tokenizer, sep_ids, sys_ids,
            [doc_b_ids, doc_a_ids],          # reversed: B first, then A
            question_ids, eos_ids,
        )
        cached_ba = (
            count_full_chunk_tokens(len(doc_b_ids), args.chunk_size)
            + count_full_chunk_tokens(len(doc_a_ids), args.chunk_size)
        )
        r_blend_ba = run_generate(
            llm, blend_prompt, sampling,
            label="Phase 2a – CacheBlend (B→A reordered, warm cache)",
            cached_tokens=cached_ba,
        )
        results.append(r_blend_ba)
        time.sleep(1)

        # ------------------------------------------------------------------
        # Phase 2b: CacheBlend repeated (same prompt) – cache fully warm
        #
        # The blended KV from Phase 2a may be stored; this call exercises
        # the fully-warm path.
        # ------------------------------------------------------------------
        r_blend_ba2 = run_generate(
            llm, blend_prompt, sampling,
            label="Phase 2b – CacheBlend (B→A repeat, fully warm cache)",
            cached_tokens=cached_ba,
        )
        results.append(r_blend_ba2)
        time.sleep(1)

        # ------------------------------------------------------------------
        # Phase 3: CacheBlend with Doc B + Doc C (different subset)
        #
        # Doc C contains maritime history – no information about AlphaCache.
        # This run tests that CacheBlend does NOT fabricate a correct answer
        # when the required information (Doc A) is absent from the prompt.
        # Expected: answer should NOT contain AlphaCache keywords.
        # ------------------------------------------------------------------
        blend_bc_prompt = build_prompt(
            tokenizer, sep_ids, sys_ids,
            [doc_b_ids, doc_c_ids],
            question_ids, eos_ids,
        )
        cached_bc = (
            count_full_chunk_tokens(len(doc_b_ids), args.chunk_size)
            + count_full_chunk_tokens(len(doc_c_ids), args.chunk_size)
        )
        r_blend_bc = run_generate(
            llm, blend_bc_prompt, sampling,
            label="Phase 3 – CacheBlend (B→C, Doc A absent – expect no AlphaCache)",
            cached_tokens=cached_bc,
        )
        results.append(r_blend_bc)

    # ------------------------------------------------------------------
    # Summary report
    # ------------------------------------------------------------------
    print("\n\n" + "═" * 70)
    print("  CACHEBLEND BENCHMARK SUMMARY")
    print("═" * 70)
    print(f"  {'Run':<50} {'Latency':>8}  {'Hit%':>6}  {'Accurate':>8}")
    print(f"  {'-'*50} {'-'*8}  {'-'*6}  {'-'*8}")
    for r in results:
        print(
            f"  {r.label:<50} {r.latency_sec:>7.2f}s"
            f"  {r.kv_hit_ratio:>5.1%}  {'YES ✓' if r.is_accurate else 'NO  ✗':>8}"
        )
    print("═" * 70)

    # Latency speedup (Phase 2a vs Phase 1)
    if len(results) >= 2 and results[0].latency_sec > 0:
        speedup = results[0].latency_sec / results[1].latency_sec
        print(f"\n  Speedup  (Phase 2a vs Phase 1): {speedup:.2f}×")

    # Accuracy check
    print()
    correct = sum(1 for r in results[:3] if r.is_accurate)
    print(f"  Accuracy (Phases 1-2b, expected correct): {correct}/3 runs correct")
    print(
        f"  Phase 3 (Doc A absent):  "
        f"{'correct – model did not hallucinate' if not results[3].is_accurate else 'WARNING – hallucinated AlphaCache'}"
        if len(results) >= 4 else ""
    )
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CacheBlend accuracy / latency / hit-ratio benchmark"
    )
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--use-disk",
        action="store_true",
        help="Use local disk as the LMCache backend instead of CPU memory",
    )
    parser.add_argument(
        "--blend-special-str",
        default=" # # ",
        help="Separator string used to delimit document chunks (default: ' # # ')",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="LMCache chunk size in tokens (default: 256)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum tokens to generate per query (default: 64)",
    )
    parser.add_argument(
        "--sys-prompt",
        default="",
        help="Custom system prompt (default: Mistral [INST] format)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    benchmark(parse_args())
