import json
import time
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from query_pipeline import LegalQueryPipeline
from benchmark.config import (
    GOLD_DATASET_PATH, 
    LLM_MODELS, 
    EMBEDDING_MODELS,
    RESULTS_DIR,
    CHROMA_DB_BASE_DIR
)

# Optional RAGAS
try:
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from langchain_ollama import OllamaLLM, ChatOllama
    from ragas.llms import LangchainLLMWrapper
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("Warning: `ragas` library not installed. Skipping tier-2 Ragas evaluation. Only computing tier-3 & latencies.")

def compute_tier3_metrics(answer, gold_points):
    """Tier 3 Legal Domain Specific."""
    answer_lower = answer.lower()
    covered = 0
    # Proxy approximation
    for pt in gold_points:
        words = [w for w in pt.lower().split() if len(w) > 4]
        if any(w in answer_lower for w in words):
            covered += 1
    return covered / len(gold_points) if len(gold_points) > 0 else 0.0

def run_e2e_benchmark():
    with open(GOLD_DATASET_PATH, 'r') as f:
        gold_dataset = json.load(f)
        
    results = {}
    
    embedder_name = EMBEDDING_MODELS[0] # Default
    db_folder_name = embedder_name.replace("/", "_")
    persist_dir = CHROMA_DB_BASE_DIR / db_folder_name

    for llm_name in LLM_MODELS:
        print(f"\n{'='*50}\nEvaluating End-To-End (E2E): {llm_name}\n{'='*50}")
        
        queries = []
        answers = []
        contexts = []
        ground_truths = []
        
        tier3_scores = []
        latencies = []
        
        try:
            pipeline = LegalQueryPipeline(
                persist_directory=str(persist_dir),
                llm_model=llm_name,
                k_retrievals=5,
                embedding_model=embedder_name
            )
            if getattr(pipeline, 'llm', None) is None:
                print(f"Skipping {llm_name}, local LLM is missing or unreachable.")
                continue
        except Exception as e:
            print(f"Skipping E2E for {llm_name}: {e}")
            continue
        
        for q in gold_dataset:
            user_profile = q["user_profile"]
            
            # Start query
            t0 = time.time()
            response = pipeline.process_query(
                user_query=q["raw_query"],
                user_profile=user_profile
            )
            latencies.append((time.time() - t0))
            
            # Collect Ragas Formats
            queries.append(q["raw_query"])
            answers.append(response["answer"])
            # Format Contexts
            ctxts = [doc.page_content for doc in response.get("retrieved_documents", [])]
            contexts.append(ctxts)
            
            # Ground truths
            gt = " ".join(q["gold_answer_key_points"])
            ground_truths.append([gt])
            
            tier3_scores.append(compute_tier3_metrics(response["answer"], q["gold_answer_key_points"]))
            
        embedder_results = {
            "avg_pipeline_latency_sec": round(float(sum(latencies)/len(latencies)), 4) if latencies else 0.0,
            "tier3_legal_completeness_approx": round(float(sum(tier3_scores)/len(tier3_scores)), 4) if tier3_scores else 0.0
        }
        
        if RAGAS_AVAILABLE:
            try:
                # Local LLM as Judge
                judge_llm = LangchainLLMWrapper(ChatOllama(model="mistral:7b"))
                
                data = {
                    "question": queries,
                    "answer": answers,
                    "contexts": contexts,
                    "ground_truth": ground_truths
                }
                
                dataset = Dataset.from_dict(data)
                ragas_res = evaluate(
                    dataset=dataset,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                    llm=judge_llm
                )
                embedder_results.update(ragas_res)
            except Exception as e:
                print(f"Ragas evaluation failed: {e}")

        results[llm_name] = embedder_results
        print(f"Results for {llm_name}:", embedder_results)

    out_path = RESULTS_DIR / "e2e_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[Done] E2E metrics dumped to {out_path}")

if __name__ == "__main__":
    run_e2e_benchmark()
