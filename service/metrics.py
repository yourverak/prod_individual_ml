import numpy as np
import pandas as pd

def calculate_map_at_100(df: pd.DataFrame, target_col: str = 'target', score_col: str = 'score') -> float:

    if 'offer_id' not in df.columns or df.empty:
        return 0.0

    ap_scores = []
    

    for offer_id, group in df.groupby('offer_id'):

        total_positives = group[target_col].sum()

        if total_positives == 0:
            continue
            

        sorted_group = group.sort_values(score_col, ascending=False).head(100)
        targets = sorted_group[target_col].values
        
        relevant_count = 0
        precision_sum = 0.0

        for i, label in enumerate(targets):
            if label == 1:
                relevant_count += 1
                precision_sum += relevant_count / (i + 1)
                

        ap = precision_sum / min(total_positives, 100)
        ap_scores.append(ap)
        
    if not ap_scores:
        return 0.0
        
    return float(np.mean(ap_scores))