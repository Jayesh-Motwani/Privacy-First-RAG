import json
import time
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from query_pipeline import LegalQueryPipeline
from benchmark.config import GOLD_DATASET_PATH, RESULTS_DIR, CHROMA_DB_BASE_DIR, EMBEDDING_MODELS

def evaluate_ablation_mode(pipeline, dataset, mode_name):
    print(f"\nRunning Component Ablation: {mode_name}")
    latencies = []
    scores = []
    
    for q in dataset:
        user_profile = q["user_profile"]
        raw_query = q["raw_query"]
        gold_points = q["gold_answer_key_points"]
        
        t0 = time.time()
        
        # We need to execute the processing. 
        # For simplicity in this harness, we monkeypatch the pipeline behavior dynamically 
        # before calling process_query, or rely on the pipeline's internal methods directly.
        # But since we use process_query, we'll patch its instances temporarily:
        
        original_rewrite = pipeline.query_rewriter.rewrite
        original_conflicts = pipeline.conflict_detector.detect_conflicts
        
        if mode_name == "No_Query_Rewriting":
            pipeline.query_rewriter.rewrite = lambda x: x # Identity function
            
        if mode_name == "No_Conflict_Detection":
            pipeline.conflict_detector.detect_conflicts = lambda x: [] # Always empty
            
        try:
            # Construct user profile based on ablation mode
            mock_profile = {
                "jurisdiction": user_profile.get("jurisdiction", "central") if mode_name != "No_Applicability_Filtering" else "any",
                "personal_law": user_profile.get("personal_law", "any") if mode_name != "No_Applicability_Filtering" else "any",
                "demographic": user_profile.get("demographic", "any") if mode_name != "No_Applicability_Filtering" else "any"
            }
            
            response = pipeline.process_query(
                user_query=raw_query,
                user_profile=mock_profile
            )
            latencies.append(time.time() - t0)
            
            # Simple Completeness scoring proxy (same as Tier 3)
            answer_lower = response["answer"].lower()
            covered = 0
            for pt in gold_points:
                words = [w for w in pt.lower().split() if len(w) > 4]
                if any(w in answer_lower for w in words):
                    covered += 1
            scores.append(covered / len(gold_points) if len(gold_points) > 0 else 0)
            
        finally:
            # Restore patches
            pipeline.query_rewriter.rewrite = original_rewrite
            pipeline.conflict_detector.detect_conflicts = original_conflicts
            
    return {
        "completeness_score": round(sum(scores)/len(scores) if scores else 0, 4),
        "latency_sec": round(sum(latencies)/len(latencies) if latencies else 0, 4)
    }

def run_ablation_benchmark():
    print("Starting Component Ablation Study...")
    with open(GOLD_DATASET_PATH, 'r') as f:
        gold_dataset = json.load(f)
        
    embedder_name = EMBEDDING_MODELS[0]
    db_folder_name = embedder_name.replace("/", "_")
    persist_dir = CHROMA_DB_BASE_DIR / db_folder_name
    
    # We use Mistral/Qwen as base local LLM for ablation
    base_llm = "mistral:7b" 
    
    try:
        pipeline = LegalQueryPipeline(
            persist_directory=str(persist_dir),
            llm_model=base_llm,
            k_retrievals=5,
            embedding_model=embedder_name
        )
    except Exception as e:
        print(f"Skipping Ablation, failed to init pipeline: {e}")
        return

    results = {}
    
    modes = [
        "Full_Pipeline",
        "No_Query_Rewriting",
        "No_Conflict_Detection",
        "No_Applicability_Filtering"
    ]
    
    for mode in modes:
        res = evaluate_ablation_mode(pipeline, gold_dataset, mode)
        results[mode] = res
        print(f"Results for {mode}:", res)
        
    out_path = RESULTS_DIR / "ablation_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
        
    print(f"\n[Done] Component Ablation metrics dumped to {out_path}")

if __name__ == "__main__":
    run_ablation_benchmark()
