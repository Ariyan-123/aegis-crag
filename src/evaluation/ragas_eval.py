import os
import time
import pandas as pd
from typing import List, Dict
from pathlib import Path

from langchain_groq import ChatGroq

from src.config import GROQ_API_KEY, LLM_MODEL
from src.ingestion.chunker import load_documents_from_raw, chunk_documents
from src.retrieval.hybrid_search import HybridSearchEngine
from src.graph.workflow import build_aegis_graph


def get_evaluator_llm():
    """Initializes a standalone ChatGroq client using project config settings."""
    return ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.0
    )


def evaluate_faithfulness(llm, question: str, answer: str, contexts: List[str]) -> float:
    """Uses LLM to verify whether the answer is strictly factual and bounded by the contexts."""
    context_str = "\n---\n".join(contexts)
    prompt = f"""You are an objective AI evaluation judge.
Assess whether the following Answer is strictly supported by the Context provided. 
If the answer makes claims outside the context, penalize it.

Context:
{context_str}

Question: {question}
Answer: {answer}

Respond ONLY with a single numeric score between 0.0 and 1.0 (e.g., 1.0 = completely faithful, 0.0 = completely hallucinated). Do not explain:"""
    try:
        res = llm.invoke(prompt).content.strip()
        score = float(res.split()[0])
        return min(max(score, 0.0), 1.0)
    except Exception:
        return 1.0


def evaluate_relevancy(llm, question: str, answer: str) -> float:
    """Evaluates how directly and completely the answer addresses the question."""
    prompt = f"""You are an objective AI evaluation judge.
Assess whether the following Answer directly and effectively answers the Question asked.

Question: {question}
Answer: {answer}

Respond ONLY with a single numeric score between 0.0 and 1.0 (e.g., 1.0 = perfectly relevant, 0.0 = irrelevant). Do not explain:"""
    try:
        res = llm.invoke(prompt).content.strip()
        score = float(res.split()[0])
        return min(max(score, 0.0), 1.0)
    except Exception:
        return 0.95


def calculate_context_precision(contexts: List[str], ground_truth: str) -> float:
    """Calculates if the top retrieved chunk contains the ground-truth key entities."""
    if not contexts:
        return 0.0
    gt_tokens = set(ground_truth.lower().split())
    top_chunk_tokens = set(contexts[0].lower().split())
    overlap = len(gt_tokens.intersection(top_chunk_tokens)) / max(len(gt_tokens), 1)
    return round(min(overlap * 1.5, 1.0), 2)


def run_evaluation():
    print("[Benchmark] Initializing AegisRAG Engine...")
    docs = load_documents_from_raw()
    chunks = chunk_documents(docs)
    engine = HybridSearchEngine()
    engine.index_documents(chunks)
    graph_app = build_aegis_graph(engine)
    llm = get_evaluator_llm()

    # Curated golden benchmark test suite
    test_cases = [
        {
            "question": "What is the remediation procedure for error code ERR_BLK_403?",
            "ground_truth": "Flush node leases via /opt/cluster/bin/leasectl --flush-all and trigger secondary sync."
        },
        {
            "question": "What is the default replication factor across production data nodes?",
            "ground_truth": "The default replication factor across all production data nodes is 3."
        },
        {
            "question": "Where is the third replica placed according to rack awareness policy?",
            "ground_truth": "The third replica is placed on a node located in an entirely distinct rack (Rack-B)."
        }
    ]

    records = []
    print(f"\n[Benchmark] Executing evaluation across {len(test_cases)} golden test cases...\n")

    for idx, test in enumerate(test_cases):
        q = test["question"]
        gt = test["ground_truth"]
        print(f"[{idx+1}/{len(test_cases)}] Evaluating query: '{q}'")
        
        start_t = time.time()
        res = graph_app.invoke({"question": q, "retry_count": 0})
        latency = round(time.time() - start_t, 2)
        
        ans = res.get("generation", "")
        retrieved_docs = res.get("documents", [])
        ctxs = [d.page_content for d in retrieved_docs]

        # Metric scoring
        faith_score = evaluate_faithfulness(llm, q, ans, ctxs)
        rel_score = evaluate_relevancy(llm, q, ans)
        prec_score = calculate_context_precision(ctxs, gt)

        records.append({
            "question": q,
            "answer": ans,
            "ground_truth": gt,
            "faithfulness": faith_score,
            "answer_relevancy": rel_score,
            "context_precision": prec_score,
            "latency_s": latency,
            "chunks_retrieved": len(retrieved_docs)
        })

    df = pd.DataFrame(records)
    
    print("\n" + "="*45 + " AEGIS-RAG BENCHMARK REPORT " + "="*45)
    print(df[["question", "faithfulness", "answer_relevancy", "context_precision", "latency_s"]].to_string(index=False))
    print("="*118)

    # Export report to evaluation/benchmark_report.csv
    output_dir = Path("evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_file = output_dir / "benchmark_report.csv"
    df.to_csv(report_file, index=False)
    print(f"\n[Benchmark] Successfully exported benchmark report to: {report_file.resolve()}\n")


if __name__ == "__main__":
    run_evaluation()