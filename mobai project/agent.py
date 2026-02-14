import os
import json
import time
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.data.preprocessor import DataPreprocessor
from src.models.forecaster import XGBoostForecaster
from src.simulation.warehouse_sim import WarehouseSimulation
from src.utils.visualization import plot_feature_importance, generate_decision_flow_diagram

EXCEL_FILE = os.path.join("eval", "WMS_Hackathon_DataPack_Templates_FR_FV_B7_ONLY.xlsx")
EVAL_LOC_FILE = os.path.join("eval", "locations_status.csv")

def collect_manual_overrides(sim, products_df, forecast_map, entropy_map):
    overrides = {}
    print("\n" + "*"*50)
    print("MANUAL OVERRIDE INTERFACE (Top 5 Products)")
    print("*"*50)
    print("Input format: SlotCode (e.g. 0A-01-01). Press Enter to skip.")
    
    # Map product weights
    p_weights = {}
    for _, r in products_df.iterrows():
        try: p_weights[int(r['id_produit'])] = float(r['weight_kg'])
        except: pass

    # Identify Top 5 Products by Demand
    # forecast_map is dict {pid: demand_qty}
    top_products = sorted(forecast_map.items(), key=lambda x: x[1], reverse=True)[:5]
    
    if not top_products:
        print("No forecast data available for overrides.")
        return {}

    for pid, demand in top_products:
        try:
            prompt = f"\nProduct {pid} (Forecast: {int(demand)} unit/day) - Enter Slot: "
            user_input = input(prompt).strip()
        except EOFError:
            break
            
        if not user_input:
            continue
            
        # Regex Validation
        pattern = r"^\d[A-Z]-\d{2}-\d{2}$"
        if not re.match(pattern, user_input):
            print("  Invalid format! Expected format: 0A-01-01 (LevelLetter-Side-Block)")
            continue
            
        slot_code = user_input
        weight = p_weights.get(pid, 0)
        entropy = entropy_map.get(pid, 1.0)
        
        # Validation
        is_valid, msg, status = sim.validate_override(pid, slot_code, weight, entropy)
        
        if status == 'IMPOSSIBLE':
            print(f"  REJECTED: {msg}")
        elif status == 'WARNING':
            print(f"  WARNING: {msg}")
            overrides[pid] = slot_code
            print(f"  -> Override recorded for {pid} at {slot_code}")
        elif status == 'OK':
            overrides[pid] = slot_code
            print(f"  -> Override recorded for {pid} at {slot_code}")
            
    return overrides

def main():
    print("="*60)
    print("AI WMS AGENT - EVALUATION MODE")
    print("="*60)
    start_time = time.time()
    
    # 1. Data Preprocessing
    print("\n[Stage 1] Data Preprocessing")
    prep = DataPreprocessor(EXCEL_FILE)
    prep.load_data()
    df_features = prep.engineer_features()
    df_model = prep.prepare_for_modeling(df_features)
    
    # 2. Forecasting
    print("\n[Stage 2] Demand Forecasting (Jan 8 - Feb 8 2026)")
    forecaster = XGBoostForecaster()
    metrics = forecaster.train(df_model)
    print(f"  Model Training Complete.")
    print(f"  WAPE: {metrics['wape']:.2f}% | Bias: {metrics['bias']:+.2f}%")
    
    preds = forecaster.predict(df_model)
    base_preds = dict(zip(preds['id_produit'], preds['prediction']))
    
    forecast_dates = pd.date_range('2026-01-08', '2026-02-08', freq='D')
    days_needed = len(forecast_dates)
    
    forecast_rows = []
    
    print(f"  Generating {days_needed}-day forecast patterns...")
    df_model = df_model.sort_values('date')
    daily_avg = df_model.groupby('day_of_week')['quantite_demande'].mean()
    global_mean = daily_avg.mean() + 1e-6
    seasonality_factors = (daily_avg / global_mean).to_dict()
    volatility = df_model.groupby('id_produit')['quantite_demande'].std().fillna(1.0).to_dict()
    np.random.seed(42)
    
    for pid in df_model['id_produit'].unique():
        pred_level = base_preds.get(pid, 0)
        sigma = max(0.1, volatility.get(pid, 1.0) * 0.4) 
        
        for d in forecast_dates:
            dow = d.dayofweek
            factor = seasonality_factors.get(dow, 1.0)
            ai_mu = pred_level * factor
            ai_val = max(0, np.random.normal(ai_mu, sigma))
            forecast_rows.append({'Date': d.strftime('%d-%m-%Y'), 'id produit': pid, 'quantite demande': round(ai_val)})

    forecast_csv = pd.DataFrame(forecast_rows)
    os.makedirs('output', exist_ok=True)
    forecast_csv.to_csv('output/eval_forecast_submission.csv', index=False)
    print(f"  Forecast submission saved to output/eval_forecast_submission.csv")
    
    # 3. Visualization
    print("\n[Stage 3] Explanation & Visualization")
    importances = forecaster.get_feature_importance()
    plot_feature_importance(importances, 'output/feature_importance.png')
    
    generate_decision_flow_diagram()
    
    # 4. Simulation
    print("\n[Stage 4] Warehouse Simulation")
    try:
        sim = WarehouseSimulation(EXCEL_FILE, EVAL_LOC_FILE)
        xls = pd.ExcelFile(EXCEL_FILE)
        tx_df = pd.read_excel(xls, 'transactions').iloc[2:].reset_index(drop=True)
        tx_df.columns = ['id_transaction', 'type_transaction', 'reference', 'cree_le', 'cree_par', 'statut', 'notes']
        lines_df = pd.read_excel(xls, 'lignes_transaction').iloc[2:].reset_index(drop=True)
        lines_df.columns = ['id_transaction', 'no_ligne', 'id_produit', 'quantite', 'src', 'dst', 'lot', 'motif']
        latest = df_model.sort_values('date').groupby('id_produit').tail(1)
        forecast_map = base_preds
        entropy_map = dict(zip(latest['id_produit'], latest['entropy_30d']))
        
        # Interactive Helper (Top 5 Products)
        overrides = collect_manual_overrides(sim, prep.products_df, forecast_map, entropy_map)
        
        sim.run(tx_df, lines_df, prep.products_df, forecast_map, entropy_map, manual_overrides=overrides)
        results = sim.get_results()
        
        with open('output/simulation_operations.json', 'w') as f:
            json.dump(results['operations'], f, indent=4)
        
        pd.DataFrame(results['operations']).to_csv('output/simulation_operations.csv', index=False)
        print("  Operations log saved to output/simulation_operations.json and .csv")
        
        print(f"  Simulation Complete.")
        print(f"  Total Dist: {results['summary']['total_dist']}")
        
    except Exception as e:
        print(f"Simulation Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print(f"AGENT EXECUTION COMPLETE ({round(time.time() - start_time, 1)}s)")
    print("="*60)

if __name__ == "__main__":
    main()
