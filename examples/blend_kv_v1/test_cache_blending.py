# SPDX-License-Identifier: Apache-2.0
"""
CacheBlend benchmark: accuracy · latency · KV-cache hit ratio

Follows the same token-building pattern as blend.py (Mistral [INST] tokens,
tokenizer.encode(text)[1:] to strip BOS, llm.generate with prompt_token_ids).

Documents
---------
Doc A  Sarah Kim profile:  12 yrs exp · QuickIndex inventor · joined TechNova
       in 2019 · manages 8 engineers · speaks Korean & English.
Doc B  Project Falcon:     real-time analytics at TechNova · technical lead is
       Sarah Kim · uses QuickIndex for compression · phase 2 of 3 · $8 M budget.
Doc C  Sourdough baking:   unrelated distractor.

5 cross-attention accuracy queries
-----------------------------------
Every query requires combining information from BOTH Doc A and Doc B.
Neither document alone is sufficient to produce a correct answer.

  Q1  How many years of experience does Project Falcon's technical lead have?
      Doc B → lead = Sarah Kim   Doc A → 12 years

  Q2  Who invented the compression algorithm used in Project Falcon?
      Doc B → QuickIndex used   Doc A → Sarah Kim invented QuickIndex

  Q3  What year did Project Falcon's technical lead join TechNova?
      Doc B → lead = Sarah Kim   Doc A → joined 2019

  Q4  How many engineers does Project Falcon's technical lead manage?
      Doc B → lead = Sarah Kim   Doc A → team of 8

  Q5  What languages does Project Falcon's technical lead speak?
      Doc B → lead = Sarah Kim   Doc A → Korean and English

Flow
----
1. Warmup — short dummy request to initialise the engine
2. Store  — run (Doc A → Doc B → Doc C) to populate LMCache with chunk KVs
3. Blend  — run (Doc B → Doc A) for each of the 5 queries; LMCache blends
            the cached KV tensors instead of recomputing from scratch
4. Print summary: accuracy · latency · TTFT · KV-cache hit ratio

Usage
-----
    python test_cache_blending.py
    python test_cache_blending.py --model meta-llama/Llama-3-8B-Instruct
    python test_cache_blending.py --use-disk
"""

# Standard
from dataclasses import asdict, dataclass
import argparse
import contextlib
import os
import time

# Third Party
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


# ---------------------------------------------------------------------------
# Mistral [INST] / [/INST] token IDs
# ---------------------------------------------------------------------------
INST_OPEN = [1, 733, 16289, 28793]      # <s>[INST]
INST_CLOSE = [733, 28748, 16289, 28793]  # [/INST]


# ---------------------------------------------------------------------------
# Documents
# Each document is written to exceed 256 tokens (default chunk_size) so that
# at least one full chunk is stored in LMCache per document.
# ---------------------------------------------------------------------------

# ~270 tokens: personal profile, all facts needed for cross-attention answers
DOC_A = (
    "Sarah Kim is a software engineer with twelve years of professional experience "
    "in backend infrastructure and distributed storage systems. She specializes in "
    "database optimization, query performance tuning, and lossless compression "
    "algorithms for large-scale columnar data pipelines. Sarah is the inventor of "
    "the QuickIndex compression algorithm, a technique that reduces storage overhead "
    "by forty percent in columnar database workloads through a novel block-level "
    "index-encoding scheme. She earned a master's degree in computer science from "
    "Seoul National University in 2012, graduating with highest honors. Before "
    "joining TechNova Inc in 2019 as a senior software engineer, she spent five "
    "years at DataSphere Corp building and maintaining large-scale indexing pipelines "
    "that processed terabytes of structured data daily across distributed clusters. "
    "At TechNova, Sarah was promoted to engineering team lead in 2021 and currently "
    "manages a team of eight engineers focused on storage and retrieval systems. "
    "She is fluent in both Korean and English, having grown up in Seoul and "
    "developed her international career working across North America and Asia. "
    "Sarah holds two patents related to the QuickIndex algorithm and has published "
    "three peer-reviewed papers on database compression. Her QuickIndex system has "
    "been deployed in production at several Fortune 500 companies handling billions "
    "of records per day. Colleagues consistently describe her as methodical, "
    "technically rigorous, and an effective mentor to junior engineers."
)

# ~270 tokens: project description, links Sarah Kim via her role and QuickIndex
DOC_B = (
    "Project Falcon is a real-time analytics platform under active development at "
    "TechNova Inc. The project was initiated in January 2020 with a total approved "
    "budget of eight million dollars and a target public launch in the third quarter "
    "of next year. The primary engineering goal of Project Falcon is to enable "
    "enterprise clients to ingest, process, and visualize high-velocity streaming "
    "data with end-to-end latency consistently below fifty milliseconds. The "
    "technical lead of Project Falcon is Sarah Kim, who was selected for the role "
    "specifically because of her deep expertise in storage optimization and her "
    "invention of the QuickIndex compression algorithm, which forms the core of "
    "Project Falcon's internal data compression layer. The project is structured "
    "into three sequential phases: infrastructure provisioning, core engine "
    "development, and client-facing API integration. The team is currently working "
    "through phase two of three. Project Falcon targets customers in the financial "
    "services and healthcare sectors, where low-latency data visibility is critical. "
    "The platform is architected to scale horizontally and sustain up to one million "
    "ingest events per second under peak load. TechNova plans to open-source the "
    "compression module once the platform reaches general availability, as the module "
    "is built entirely on top of the QuickIndex algorithm. The project has received "
    "strong internal executive sponsorship and is considered a flagship initiative "
    "for TechNova's enterprise data division."
)

# ~270 tokens: unrelated distractor, contains no names, algorithms, or dates
# that could be confused with Doc A or Doc B facts
DOC_C = (
    "Sourdough bread baking has experienced a remarkable revival among home bakers "
    "over the past decade. Unlike breads leavened with commercial yeast, sourdough "
    "relies entirely on a naturally fermented starter culture containing wild yeast "
    "strains and lactic acid bacteria. The starter must be fed with fresh flour and "
    "water on a regular schedule to remain active and healthy. Fermentation time "
    "typically ranges from eight to twenty-four hours depending on the ambient "
    "temperature, the hydration level of the dough, and the maturity of the starter. "
    "A high-hydration dough, sometimes called a wet dough, produces a more open and "
    "irregular crumb structure with larger air pockets. Bakers use a technique known "
    "as the stretch-and-fold method during bulk fermentation to develop the gluten "
    "network without traditional kneading. Scoring the top surface of the shaped loaf "
    "before it enters the oven allows controlled expansion and prevents uneven tearing. "
    "Most experienced bakers recommend baking sourdough inside a preheated Dutch oven "
    "or cast-iron combo cooker to trap steam during the first phase of baking, which "
    "promotes a thin, crackly crust. The Maillard reaction between amino acids and "
    "reducing sugars during baking gives the crust its characteristic deep brown color "
    "and complex, nutty flavor. Extended fermentation also breaks down phytic acid in "
    "the flour, which may improve mineral bioavailability. Many enthusiasts claim that "
    "traditionally fermented sourdough is easier to digest than commercially yeasted "
    "bread. The craft rewards consistent practice, careful observation, and patience."
)


# ---------------------------------------------------------------------------
# 5 cross-attention accuracy queries
# ---------------------------------------------------------------------------

@dataclass
class Query:
    text: str               # question text
    keywords: list[str]     # any one of these in the answer → correct
    description: str        # what cross-doc chain is needed


QUERIES = [
    Query(
        text=(
            "Based only on the documents provided, how many years of professional "
            "experience does the technical lead of Project Falcon have? "
            "Answer with a number or written-out number only."
        ),
        keywords=["twelve", "12"],
        description="Q1  B→lead=Sarah Kim  ·  A→12 years experience",
    ),
    Query(
        text=(
            "Based only on the documents provided, who invented the compression "
            "algorithm that Project Falcon uses? Give only the person's name."
        ),
        keywords=["Sarah", "Kim"],
        description="Q2  B→QuickIndex used  ·  A→Sarah Kim invented QuickIndex",
    ),
    Query(
        text=(
            "Based only on the documents provided, in what year did the technical "
            "lead of Project Falcon join TechNova? Answer with a four-digit year."
        ),
        keywords=["2019"],
        description="Q3  B→lead=Sarah Kim  ·  A→joined TechNova in 2019",
    ),
    Query(
        text=(
            "Based only on the documents provided, how many engineers does the "
            "technical lead of Project Falcon currently manage? "
            "Answer with a number or written-out number only."
        ),
        keywords=["eight", "8"],
        description="Q4  B→lead=Sarah Kim  ·  A→manages 8 engineers",
    ),
    Query(
        text=(
            "Based only on the documents provided, what languages does the technical "
            "lead of Project Falcon speak? List them."
        ),
        keywords=["Korean"],
        description="Q5  B→lead=Sarah Kim  ·  A→speaks Korean and English",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_environment_variables(
    use_disk: bool = False,
    blend_special_str: str = " # # ",
    chunk_size: int = 256,
) -> None:
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


@contextlib.contextmanager
def build_llm_with_lmcache(model: str):
    ktc = KVTransferConfig(kv_connector="LMCacheConnectorV1", kv_role="kv_both")
    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=8192,
        gpu_memory_utilization=0.7,
        enable_prefix_caching=False,
        enforce_eager=True,
    )
    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def count_full_chunk_tokens(n_tokens: int, chunk_size: int) -> int:
    """Tokens covered by complete chunks (i.e. floor(n / chunk_size) * chunk_size)."""
    return (n_tokens // chunk_size) * chunk_size


def get_ttft(output) -> float | None:
    """Extract TTFT from a vLLM RequestOutput if the metrics are available."""
    m = getattr(output, "metrics", None)
    if m is None:
        return None
    t0 = getattr(m, "first_scheduled_time", None)
    t1 = getattr(m, "first_token_time", None)
    if t0 is not None and t1 is not None:
        return t1 - t0
    return None


def measure_generate(
    llm: LLM,
    prompt: list[int],
    sampling_params: SamplingParams,
    label: str,
    cached_tokens: int,
    chunk_size: int,
) -> None:
    """Run a single generate call and print accuracy / latency / hit-ratio."""
    n_input = len(prompt)
    hit_ratio = cached_tokens / n_input if n_input > 0 else 0.0

    print(f"\n{'─' * 62}")
    print(f"  {label}")
    print(f"  input tokens : {n_input}  |  cached tokens (est.): {cached_tokens}"
          f"  |  hit ratio: {hit_ratio:.1%}")
    print(f"{'─' * 62}")

    t_start = time.time()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt},
        sampling_params=sampling_params,
    )
    latency = time.time() - t_start

    generated = outputs[0].outputs[0].text if outputs else ""
    ttft = get_ttft(outputs[0]) if outputs else None

    print(f"  Generated : {generated!r}")
    print(f"  Latency   : {latency:.3f}s"
          + (f"  |  TTFT: {ttft:.3f}s" if ttft is not None else ""))


def run_accuracy_queries(
    llm: LLM,
    sys_prompt: list[int],
    sep: list[int],
    doc_b_ids: list[int],
    doc_a_ids: list[int],
    tokenizer,
    sampling_params: SamplingParams,
    chunk_size: int,
) -> None:
    """Run all 5 cross-attention accuracy queries and print a result table."""
    cached_tokens = (
        count_full_chunk_tokens(len(doc_b_ids), chunk_size)
        + count_full_chunk_tokens(len(doc_a_ids), chunk_size)
    )

    results: list[tuple[str, bool, float, float | None]] = []

    for q in QUERIES:
        q_ids = tokenizer.encode(q.text)[1:]  # strip BOS
        prompt = (
            sys_prompt
            + sep + doc_b_ids
            + sep + doc_a_ids
            + sep + q_ids
            + INST_CLOSE
        )
        n_input = len(prompt)
        hit_ratio = cached_tokens / n_input if n_input > 0 else 0.0

        print(f"\n  ▷ {q.description}")
        t_start = time.time()
        outputs = llm.generate(
            prompts={"prompt_token_ids": prompt},
            sampling_params=sampling_params,
        )
        latency = time.time() - t_start

        generated = outputs[0].outputs[0].text if outputs else ""
        ttft = get_ttft(outputs[0]) if outputs else None

        is_correct = any(kw.lower() in generated.lower() for kw in q.keywords)
        results.append((q.description, is_correct, latency, ttft))

        correct_mark = "✓" if is_correct else "✗"
        ttft_str = f"  TTFT {ttft:.3f}s" if ttft is not None else ""
        print(f"    [{correct_mark}] {generated!r}")
        print(f"        latency {latency:.3f}s{ttft_str}  |  "
              f"hit ratio {hit_ratio:.1%}  (cached {cached_tokens}/{n_input} tokens)")

        time.sleep(0.5)

    # Per-query summary table
    print(f"\n{'═' * 70}")
    print("  ACCURACY SUMMARY  (CacheBlend: B → A order, warm cache)")
    print(f"{'═' * 70}")
    for desc, ok, lat, ttft in results:
        ttft_str = f"  TTFT {ttft:.3f}s" if ttft is not None else ""
        print(f"  {'✓' if ok else '✗'}  {desc:<50}  {lat:.2f}s{ttft_str}")
    n_correct = sum(1 for _, ok, _, _ in results if ok)
    print(f"{'─' * 70}")
    print(f"  Accuracy: {n_correct}/{len(results)}")
    print(f"{'═' * 70}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CacheBlend accuracy / latency / KV-cache hit ratio benchmark"
    )
    parser.add_argument(
        "--model", default="mistralai/Mistral-7B-Instruct-v0.2",
        help="HuggingFace model name or local path"
    )
    parser.add_argument(
        "-d", "--use-disk", action="store_true",
        help="Use local disk backend instead of CPU memory"
    )
    parser.add_argument(
        "-b", "--blend-special-str", default=" # # ",
        help="Chunk separator string (default: ' # # ')"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=256,
        help="LMCache chunk size in tokens (default: 256)"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32,
        help="Max tokens to generate per query (default: 32)"
    )
    args = parser.parse_args()

    setup_environment_variables(args.use_disk, args.blend_special_str, args.chunk_size)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # ── Token encoding (mirrors blend.py exactly) ───────────────────────────
    # sep: strip BOS so it slots cleanly between other segments
    sep = tokenizer.encode(os.getenv("LMCACHE_BLEND_SPECIAL_STR"))[1:]

    # sys_prompt: INST_OPEN already carries the BOS (token 1); system text
    # uses [1:] to avoid a second BOS in the middle of the sequence
    sys_prompt = INST_OPEN + tokenizer.encode(
        "You are a precise question-answering assistant. "
        "Read the documents carefully and answer using only the information they contain."
    )[1:]

    # Documents: strip BOS from each
    doc_a_ids = tokenizer.encode(DOC_A)[1:]
    doc_b_ids = tokenizer.encode(DOC_B)[1:]
    doc_c_ids = tokenizer.encode(DOC_C)[1:]

    # Print token counts so the user can verify chunk coverage
    print("\n── Document token counts ─────────────────────────────────────────")
    for name, ids in [("Doc A", doc_a_ids), ("Doc B", doc_b_ids), ("Doc C", doc_c_ids)]:
        full = count_full_chunk_tokens(len(ids), args.chunk_size)
        print(f"  {name}: {len(ids):4d} tokens  ({full} covered by full chunks "
              f"of {args.chunk_size})")
    print()

    sampling = SamplingParams(temperature=0, top_p=0.95, max_tokens=args.max_tokens)

    with build_llm_with_lmcache(args.model) as llm:

        # ── Warmup ─────────────────────────────────────────────────────────
        warmup_prompt = tokenizer.encode("Nice to meet you. " * 200)[1:]
        print("── Warmup ────────────────────────────────────────────────────────")
        llm.generate(
            prompts={"prompt_token_ids": warmup_prompt},
            sampling_params=SamplingParams(temperature=0, max_tokens=1),
        )
        print("  done.\n")

        # ── Phase 1: Store  (A → B → C) ────────────────────────────────────
        # Run all three docs in order so LMCache stores a KV chunk for each.
        # No tokens in cache yet → hit ratio = 0 %.
        store_prompt = (
            sys_prompt
            + sep + doc_a_ids
            + sep + doc_b_ids
            + sep + doc_c_ids
            + sep + tokenizer.encode("Briefly list the topics covered.")[1:]
            + INST_CLOSE
        )
        measure_generate(
            llm, store_prompt, sampling,
            label="Phase 1 – Store (A → B → C, cold cache)",
            cached_tokens=0,
            chunk_size=args.chunk_size,
        )
        time.sleep(1)

        # ── Phase 2: Blend  (B → A, 5 cross-attention queries) ─────────────
        # Document order is swapped vs. Phase 1.  Prefix caching cannot reuse
        # anything; LMCache blends the per-chunk KVs stored in Phase 1.
        print("\n\n── Phase 2 – CacheBlend accuracy test (B → A order) ─────────────")
        run_accuracy_queries(
            llm=llm,
            sys_prompt=sys_prompt,
            sep=sep,
            doc_b_ids=doc_b_ids,
            doc_a_ids=doc_a_ids,
            tokenizer=tokenizer,
            sampling_params=sampling,
            chunk_size=args.chunk_size,
        )


if __name__ == "__main__":
    main()
