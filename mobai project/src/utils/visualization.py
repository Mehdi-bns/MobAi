import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Register pandas converters
pd.plotting.register_matplotlib_converters()

def plot_feature_importance(importance_dict, output_path='output/feature_importance.png'):
    if not importance_dict:
        print("No feature importance data to plot.")
        return

    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    features = [x[0] for x in sorted_features]
    scores = [x[1] for x in sorted_features]

    plt.figure(figsize=(12, 8))
    plt.barh(features, scores, color='skyblue')
    plt.xlabel('Importance Score')
    plt.title('Top 20 Feature Importance (XGBoost)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"Feature importance plot saved to {output_path}")


def generate_decision_flow_diagram():
    flow = """
    Decision Flow Diagram:
    
    [Historical Data] 
           |
           v
    [Preprocessing] --> (Feature Engineering: Lags, Temporal, Entropy)
           |
           v
    [XGBoost Forecast Model] --> (Predict Future Demand)
           |
           v
    [Optimization Engine]
           |
           |--> (Input: Forecast, Product Weight, Warehouse Status)
           |
           |--> [Strategy Selection]
           |       |--> Complexity < 500? -> Canadian (Hungarian) Algorithm
           |       |--> Complexity > 500? -> Greedy Heuristic
           |
           v
    [Pathfinding]
           |--> (Method: A* / CuOpt)
           |--> [Routing Logic] -> (Avoid Congestion, Minimize Distance)
           |
           v
    [Execution Output]
           |--> Operational Instructions (Receipt -> Storage -> Picking -> Delivery)
    """
    with open('output/decision_flow.txt', 'w') as f:
        f.write(flow)
    print("Decision flow diagram saved to output/decision_flow.txt")
