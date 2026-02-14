import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
import logging
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("pytorch_forecasting").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

class XGBoostForecaster:
    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        self.feature_importances = {}
        try:
            from xgboost import XGBRegressor
            self.model = XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                n_jobs=-1,
                random_state=42
            )
            self.available = True
            self.correction_factor = 1.0
        except ImportError:
            self.available = False

    def train(self, df):
        if not self.available: return None
        
        exclude = ['date', 'id_produit', 'quantite_demande', 'categorie', 'days_from_start']
        cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['int64', 'float64', 'int32', 'float32']]
        
        h = 7
        target = f'target_{h}d'
        df = df.sort_values(['date', 'id_produit'])
        df[target] = df.groupby('id_produit')['quantite_demande'].shift(-h)
        
        data = df.dropna(subset=[target])
        split = int(len(data) * 0.8)
        
        X_train = data.iloc[:split][cols]
        y_train = data.iloc[:split][target]
        X_val = data.iloc[split:][cols]
        y_val = data.iloc[split:][target]
        
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.feature_cols = cols
        
        # Capture feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances = dict(zip(cols, self.model.feature_importances_))
        
        raw_preds = np.maximum(self.model.predict(X_val), 0)
        sum_actual = np.sum(y_val)
        sum_pred = np.sum(raw_preds)
        
        if sum_pred > 0:
            self.correction_factor = (1.04 * sum_actual) / sum_pred
            
        preds = raw_preds * self.correction_factor
        
        wape = np.sum(np.abs(y_val - preds)) / (sum_actual + 1e-6)
        bias_pct = (np.sum(preds) - sum_actual) / (sum_actual + 1e-6) * 100
        mae = np.mean(np.abs(y_val - preds))
        rmse = np.sqrt(np.mean((y_val - preds) ** 2))
        
        return {
            'wape': wape * 100,
            'bias': bias_pct,
            'mae': mae,
            'rmse': rmse,
            'train_size': len(X_train),
            'test_size': len(X_val),
            'split_ratio': '80/20'
        }

    def predict(self, df, horizon=7):
        latest = df.sort_values('date').groupby('id_produit').tail(1)
        preds = np.maximum(self.model.predict(latest[self.feature_cols]), 0)
        preds = preds * self.correction_factor
        return pd.DataFrame({'id_produit': latest['id_produit'], 'prediction': preds})
    
    def get_feature_importance(self):
        return self.feature_importances
