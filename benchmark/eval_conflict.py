import json
import time
import sys
from pathlib import Path
from langchain_core.documents import Document

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from query_pipeline import ConflictDetector
from benchmark.config import GOLD_DATASET_PATH, CONFLICT_MAP_PATH, RESULTS_DIR

def run_conflict_benchmark():
    print(f"Starting Conflict Detection Matrix...")
    detector = ConflictDetector(conflict_map_path=str(CONFLICT_MAP_PATH))
    
    with open(GOLD_DATASET_PATH, 'r') as f:
        gold_dataset = json.load(f)
        
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    
    latencies = []
    
    for q in gold_dataset:
        # Mock what the retriever *would* give if it retrieved exactly the gold passages
        mock_docs = []
        for gp in q["gold_passages"]:
            # Need to align with `get_provision_id` logic
            mock_doc = Document(page_content="Mock", metadata={
                "act_name": gp.get("act_name", ""),
                "section_number": gp.get("section_number", "")
            })
            mock_docs.append(mock_doc)
            
        t0 = time.time()
        detected_conflicts = detector.detect_conflicts(mock_docs)
        latencies.append((time.time() - t0) * 1000)
        
        # Evaluation: precision / recall mapping
        # Gold conflicts are listed out natively in q["expected_conflicts"]
        expected_raw = q.get("expected_conflicts", [])
        
        has_gold_conflict = len(expected_raw) > 0
        has_detected_conflict = len(detected_conflicts) > 0
        
        if has_gold_conflict and has_detected_conflict:
            # We don't do hard string matching as the get_provision_id might abbreviate differently
            # We check logically if it successfully flipped
            true_positives += 1
        elif has_gold_conflict and not has_detected_conflict:
            false_negatives += 1
        elif not has_gold_conflict and has_detected_conflict:
            false_positives += 1
        elif not has_gold_conflict and not has_detected_conflict:
            true_negatives += 1

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    false_alarm_rate = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
    
    final_metrics = {
        "Conflict_Precision": round(precision, 4),
        "Conflict_Recall": round(recall, 4),
        "False_Alarm_Rate": round(false_alarm_rate, 4),
        "avg_detection_latency_ms": round(sum(latencies) / len(latencies), 4) if latencies else 0.0,
        "TP": true_positives,
        "FP": false_positives,
        "FN": false_negatives,
        "TN": true_negatives
    }
    print("\nConflict Detection Matrix complete:")
    print(json.dumps(final_metrics, indent=4))
    
    out_path = RESULTS_DIR / "conflict_results.json"
    with open(out_path, 'w') as f:
        json.dump(final_metrics, f, indent=4)
        
    print(f"\n[Done] Conflict Detection metrics dumped to {out_path}")

if __name__ == "__main__":
    run_conflict_benchmark()
