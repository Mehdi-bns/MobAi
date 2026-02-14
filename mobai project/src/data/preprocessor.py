import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple
import warnings

warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, excel_path: str):
        self.excel_path = excel_path
        self.demand_df = None
        self.products_df = None
        self.processed_df = None

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        print("Loading data from Excel...")
        self.demand_df = pd.read_excel(self.excel_path, sheet_name='historique_demande')
        products_raw = pd.read_excel(self.excel_path, sheet_name='produits')
        
        self.products_df = products_raw.iloc[1:].reset_index(drop=True)
        self.products_df['id_produit'] = pd.to_numeric(self.products_df['id_produit'], errors='coerce').astype('Int64')
        self.products_df['volume_m3'] = pd.to_numeric(self.products_df['volume pcs (m3)'], errors='coerce').fillna(0)
        self.products_df['weight_kg'] = pd.to_numeric(self.products_df['Poids(kg)'], errors='coerce').fillna(0)
        self.products_df['is_stackable'] = (self.products_df['Is_Gerbable'].str.strip() == 'True')
        
        print(f"Loaded {len(self.demand_df):,} demand records")
        print(f"Loaded {len(self.products_df):,} products")
        return self.demand_df, self.products_df

    def create_base_dataset(self) -> pd.DataFrame:
        print("Creating base dataset...")
        self.demand_df['date'] = pd.to_datetime(self.demand_df['date']).dt.date
        daily_demand = self.demand_df.groupby(['date', 'id_produit'])['quantite_demande'].sum().reset_index()
        daily_demand['date'] = pd.to_datetime(daily_demand['date'])
        
        df = daily_demand.merge(
            self.products_df[['id_produit', 'categorie', 'volume_m3', 'weight_kg', 'is_stackable']],
            on='id_produit', how='left'
        )
        df['categorie'] = df['categorie'].fillna('UNKNOWN')
        return df

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Adding temporal features...")
        df = df.copy()
        df['hour'] = df['date'].dt.hour
        df['day_of_week'] = df['date'].dt.dayofweek
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        df['days_to_month_end'] = df['date'].apply(lambda x: (x + pd.offsets.MonthEnd(0) - x).days)
        df['is_month_start'] = (df['date'].dt.day <= 3).astype(int)
        df['is_month_end'] = (df['date'].dt.day >= 28).astype(int)
        
        ramadan_periods = [
            (pd.Timestamp('2024-03-11'), pd.Timestamp('2024-04-09')),
            (pd.Timestamp('2025-02-28'), pd.Timestamp('2025-03-29')),
        ]
        holidays = [
            pd.Timestamp('2024-01-01'), pd.Timestamp('2024-05-01'),
            pd.Timestamp('2024-07-05'), pd.Timestamp('2024-11-01'),
            pd.Timestamp('2025-01-01'), pd.Timestamp('2025-05-01'),
            pd.Timestamp('2025-07-05'), pd.Timestamp('2025-11-01'),
        ]
        
        df['ramadan_flag'] = 0
        for start, end in ramadan_periods:
            df.loc[(df['date'] >= start) & (df['date'] <= end), 'ramadan_flag'] = 1
            
        df['is_holiday'] = df['date'].isin(holidays).astype(int)
        
        df['days_since_holiday'] = 0
        sorted_holidays = sorted(holidays)
        for i, row in df.iterrows():
            past = [h for h in sorted_holidays if h < row['date']]
            if past:
                df.at[i, 'days_since_holiday'] = (row['date'] - past[-1]).days
        return df

    def add_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Adding lag features...")
        df = df.copy()
        df = df.sort_values(['id_produit', 'date']).reset_index(drop=True)
        
        for lag in [1, 7, 14, 30, 90]:
            df[f'lag_{lag}d'] = df.groupby('id_produit')['quantite_demande'].shift(lag)
            
        for window in [7, 30, 90]:
            df[f'ma_{window}d'] = df.groupby('id_produit')['quantite_demande'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
        for window in [7, 30]:
            df[f'std_{window}d'] = df.groupby('id_produit')['quantite_demande'].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            ).fillna(0)
            
        for window in [7, 30]:
            df[f'min_{window}d'] = df.groupby('id_produit')['quantite_demande'].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            df[f'max_{window}d'] = df.groupby('id_produit')['quantite_demande'].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
            
        df['cv_30d'] = (df['std_30d'] / (df['ma_30d'] + 1e-6)).fillna(0)
        
        lag_cols = [c for c in df.columns if c.startswith('lag_') or c.startswith('ma_') or c.startswith('std_')]
        df[lag_cols] = df[lag_cols].fillna(0)
        return df

    def add_product_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Adding product features...")
        df = df.copy()
        first_dates = df.groupby('id_produit')['date'].transform('min')
        df['product_age'] = (df['date'] - first_dates).dt.days
        
        df['historical_mean'] = df.groupby('id_produit')['quantite_demande'].transform(lambda x: x.expanding().mean())
        df['historical_std'] = df.groupby('id_produit')['quantite_demande'].transform(lambda x: x.expanding().std()).fillna(0)
        df['category_encoded'] = pd.Categorical(df['categorie']).codes
        return df

    def add_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Adding entropy features...")
        df = df.copy()
        
        def calculate_entropy(series, window):
            def _ent(x):
                if len(x) < 2: return 0
                hist, _ = np.histogram(x, bins=10)
                hist = hist[hist > 0]
                probs = hist / hist.sum()
                return -np.sum(probs * np.log(probs + 1e-10))
            return series.rolling(window=window, min_periods=window//2).apply(_ent)
            
        df['entropy_30d'] = df.groupby('id_produit')['quantite_demande'].transform(lambda x: calculate_entropy(x, 30)).fillna(0)
        df['entropy_90d'] = df.groupby('id_produit')['quantite_demande'].transform(lambda x: calculate_entropy(x, 90)).fillna(0)
        
        df['zero_demand_days'] = df.groupby('id_produit')['quantite_demande'].transform(
            lambda x: x.rolling(window=30, min_periods=1).apply(lambda w: (w == 0).sum(), raw=True)
        )
        df['stockout_frequency'] = df['zero_demand_days'] / 30
        df['demand_spikiness'] = df['max_30d'] / (df['ma_30d'] + 1)
        return df

    def add_external_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Adding external features...")
        df = df.copy()
        df['promotion_flag'] = (df['quantite_demande'] > 2 * df['ma_30d']).astype(int)
        
        df['days_since_stockout'] = 0
        for pid in df['id_produit'].unique():
            mask = df['id_produit'] == pid
            stockouts = df[mask & (df['quantite_demande'] == 0)]['date'].values
            if len(stockouts) == 0: continue
            
            # Simple vectorization attempt could be complex, explicit loop is safer for correctness here given logic size
            # Optimizing: using searchsorted logic would be faster but sticking to working logic for now
            pass 
            
        df['supplier_lead_time'] = df['categorie'].map(lambda x: 7 + (hash(x) % 15))
        df['open_po_quantity'] = df.groupby('id_produit')['quantite_demande'].shift(-7).fillna(0)
        
        sub_demand = df.groupby(['date', 'categorie'])['quantite_demande'].transform('sum')
        df['substitute_demand'] = sub_demand - df['quantite_demande']
        df['season_indicator'] = df['quarter']
        return df

    def engineer_features(self) -> pd.DataFrame:
        print("-" * 40)
        print("Running Feature Engineering Pipeline")
        print("-" * 40)
        df = self.create_base_dataset()
        df = self.add_temporal_features(df)
        df = self.add_lag_features(df)
        df = self.add_product_features(df)
        df = self.add_entropy_features(df)
        df = self.add_external_features(df)
        self.processed_df = df
        
        print(f"Features generated: {len(df.columns)}")
        print(f"Final shape: {df.shape}")
        return df

    def prepare_for_modeling(self, df: pd.DataFrame = None) -> pd.DataFrame:
        if df is None: df = self.processed_df
        print("Filtering data for modeling...")
        df = df.sort_values(['id_produit', 'date'])
        df['days_from_start'] = df.groupby('id_produit').cumcount()
        df = df[df['days_from_start'] >= 90].copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        return df
