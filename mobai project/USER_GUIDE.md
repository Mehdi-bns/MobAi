# AI WMS Agent User Guide

This guide explains how to run and test the AI WMS Agent, including the manual override interface and output verification.

## 1. Prerequisites
Ensure you have the following Python libraries installed:
```bash
pip install pandas numpy xgboost scikit-learn matplotlib openpyxl pytorch-forecasting pytorch-lightning
```

## 2. Running the Agent
Run the main agent script from the command line:
```bash
python agent.py
```

The agent will execute 4 main stages:
1.  **Data Preprocessing**: Loads Excel data and generates features.
2.  **Demand Forecasting**: Predicts demand for Jan 8 - Feb 8 2026.
3.  **Visualization**: Generates forecast plots and feature importance charts.
4.  **Warehouse Simulation**: Simulates product movements and optimization.

## 3. Manual Override Interface (New Feature)
Before the simulation starts, the agent will present a **Manual Override Interface**.
-   It identifies the **Top 5 High-Demand Products**.
-   For each product, it displays the ID and Forecasted Daily Demand.
-   **Action**:
    -   **To Override**: Type a target slot code (format: `0A-01-01`).
    -   **To Skip**: Press `Enter`.

### Example Interaction:
```text
**************************************************
MANUAL OVERRIDE INTERFACE (Top 5 Products)
**************************************************
Input format: SlotCode (e.g. 0A-01-01). Press Enter to skip.

Product 31554 (Forecast: 1200 unit/day) - Enter Slot: 0A-01-01
  -> Override recorded for 31554 at 0A-01-01

Product 31565 (Forecast: 900 unit/day) - Enter Slot: [ENTER]
...
```

The system validates your input:
-   **REJECTED**: If the slot doesn't exist, is occupied, or weight constraints are violated (e.g. heavy item high up).
-   **WARNING**: If the placement is suboptimal (e.g. high entropy item in a far zone). It will apply the override if it's just a warning.

## 4. Verifying Outputs
Check the `output/` directory for results:

| File | Description |
|------|-------------|
| `eval_forecast_submission.csv` | The generated demand forecast table. |
| `simulation_operations.csv` | **Main Log**: Table of all warehouse operations (Ingoing/Outgoing) matching the requested format. |
| `simulation_operations.json` | JSON version of the operations log. |
| `forecast_plot_X.png` | Comparison plots (History vs Baseline vs Forecast). |
| `feature_importance.png` | Chart showing key drivers of demand. |
| `decision_flow.txt` | Text diagram of the agent's decision logic. |

## 5. Testing Scenarios
### Scenario A: High Demand Optimization
-   Run the agent.
-   When prompted for a high-demand product (e.g. 31554), try placing it in a **far zone** (e.g. `0M-01-01`).
-   Observe the **Comparison** or **Warning** about entropy/distance.

### Scenario B: Weight Constraint
-   Identify a heavy product.
-   Try placing it in a **high level** slot (e.g. `1A-01-01` or `2A...`).
-   Observe the **REJECTION** message.

### Scenario C: Standard Flow
-   Press `Enter` through all overrides.
-   Verify `simulation_operations.csv` contains thousands of realistic operations.
