import json
import time
import numpy as np
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from query_pipeline import LegalQueryRewriter
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from benchmark.config import (
    GOLD_DATASET_PATH, 
    LLM_MODELS, 
    RESULTS_DIR
)

def compute_semantic_similarity(raw_query, rewritten_query, embedder):
    """Computes cosine similarity between original and rewritten query."""
    v1 = embedder.embed_query(raw_query)
    v2 = embedder.embed_query(rewritten_query)
    return cosine_similarity([v1], [v2])[0][0]

def compute_query_expansion_rate(raw_query, rewritten_query):
    """Rate = len(rewritten_tokens) / len(raw_tokens)."""
    t1 = raw_query.split()
    t2 = rewritten_query.split()
    if len(t1) == 0: return 0.0
    return len(t2) / len(t1)

def compute_legal_term_recall(rewritten_query, gold_key_points):
    """Fallback approximation: Checks if keyword fragments from gold are present in rewritten query."""
    rewritten_lower = rewritten_query.lower()
    
    # Extrapolate keywords from gold points (crude tokenizer)
    found_terms = 0
    total_terms = 0
    for kp in gold_key_points:
        words = [w for w in kp.lower().split() if len(w) > 4] # Keep meaningful words
        for w in words:
            total_terms += 1
            if w in rewritten_lower:
                found_terms += 1
                
    return found_terms / total_terms if total_terms > 0 else 0.0


def run_rewriting_benchmark():
    print(f"Loading Gold Dataset from {GOLD_DATASET_PATH}...")
    with open(GOLD_DATASET_PATH, 'r') as f:
        gold_dataset = json.load(f)
        
    results = {}
    
    # We use a fast fixed embedder to judge semantic distance
    judge_embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    for llm_name in LLM_MODELS:
        print(f"\n{'='*50}\nEvaluating Rewriting (LLM Ablation): {llm_name}\n{'='*50}")
        
        raw_sims = []
        gold_sims = []
        exp_rates = []
        term_recalls = []
        latencies = []
        
        try:
            rewriter = LegalQueryRewriter(model_name=llm_name)
            # If initialization degraded gracefully to None, skip benchmarking it
            if getattr(rewriter, 'llm', None) is None:
                print(f"Skipping {llm_name}, local LLM is missing or unreachable.")
                continue
        except Exception as e:
            print(f"Skipping {llm_name}, failed to load Ollama: {e}")
            continue
        
        for q in gold_dataset:
            raw_q = q["raw_query"]
            gold_q = q["expected_rewritten_query"]
            
            t0 = time.time()
            # Pipeline is rewriting raw to legal
            rewritten_q = rewriter.rewrite(raw_q)
            latencies.append((time.time() - t0) * 1000)
            
            # Semantic Similarity (Rewritten vs Gold) - "How close is rewritten query to gold rewritten query"
            gold_sim = compute_semantic_similarity(gold_q, rewritten_q, judge_embedder)
            gold_sims.append(gold_sim)
            
            # Semantic Similarity (Rewritten vs Raw) - General tracking
            raw_sim = compute_semantic_similarity(raw_q, rewritten_q, judge_embedder)
            raw_sims.append(raw_sim)
            
            # Query Expansion
            exp_rates.append(compute_query_expansion_rate(raw_q, rewritten_q))
            
            # Legal Term Recall
            term_recalls.append(compute_legal_term_recall(rewritten_q, q["gold_answer_key_points"]))
            
        final_metrics = {
            "avg_latency_ms": float(round(np.mean(latencies), 2)) if latencies else 0.0,
            "semantic_similarity_to_gold": float(round(np.mean(gold_sims), 4)) if gold_sims else 0.0,
            "semantic_similarity_to_raw": float(round(np.mean(raw_sims), 4)) if raw_sims else 0.0,
            "query_expansion_rate": float(round(np.mean(exp_rates), 4)) if exp_rates else 0.0,
            "legal_term_recall_approx": float(round(np.mean(term_recalls), 4)) if term_recalls else 0.0
        }
        print(f"Results for {llm_name}:", final_metrics)
        results[llm_name] = final_metrics

    # Save
    out_path = RESULTS_DIR / "rewriting_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[Done] All rewriting ablated metrics written to {out_path}")

if __name__ == "__main__":
    run_rewriting_benchmark()
