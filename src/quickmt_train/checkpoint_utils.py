import os
import json
import re

def extract_step(filename):
    """
    Extracts the step number from a filename like model_1000.safetensors or checkpoint_1000.pt
    """
    match = re.search(r'_(\d+)\.', filename)
    if match:
        return int(match.group(1))
    return -1

def get_best_steps(metrics_path, metric_name, lower_is_better, k):
    """
    Returns the step numbers for the top k checkpoints based on the provided metrics.
    """
    if not os.path.exists(metrics_path):
        return []
        
    scored_steps = []
    with open(metrics_path, "r") as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                step = entry.get("step")
                metric_val = entry.get(metric_name)
                if step is not None and metric_val is not None:
                    scored_steps.append((metric_val, step))
            except json.JSONDecodeError:
                continue
                
    # Sort: Primary key is the metric value, secondary key is step (favoring later steps)
    scored_steps.sort(key=lambda x: (x[0] if lower_is_better else -x[0], -x[1]))
    
    return [s for _, s in scored_steps[:k]]
