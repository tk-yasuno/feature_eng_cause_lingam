"""
ハザードモデル用データ前処理

labeled_time_series.csv から model_input.npz を生成
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict
from scipy import stats
from .config_hazard import HazardConfig
import warnings
warnings.filterwarnings('ignore')


def prepare_hazard_data(config: HazardConfig) -> dict:
    """
    ハザードモデル用のデータを準備
    
    Parameters:
        config: HazardConfig instance
    
    Returns:
        dict with model input data
    """
    print("\n[ハザードモデル用データ前処理]")
    
    # 時系列データの読み込み
    time_series_path = config.get_absolute_path(config.labeled_time_series)
    print(f"  読込: {time_series_path}")
    
    df = pd.read_csv(time_series_path)
    print(f"  データ点数: {len(df)}")
    print(f"  列: {list(df.columns)}")
    
    # 必須列の確認
    required_cols = ['equipment_id', 'date', 'value']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"必須列 '{col}' が見つかりません")
    
    # date を datetime に変換
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['equipment_id', 'date'])
    
    # ポンプごとにフィルタリング
    print("\n  ポンプフィルタリング...")
    pump_list = []
    
    for equipment_id, group in df.groupby('equipment_id'):
        if len(group) >= config.min_data_points:
            pump_list.append(equipment_id)
    
    print(f"  フィルタ後ポンプ数: {len(pump_list)} (最低{config.min_data_points}点以上)")
    
    df = df[df['equipment_id'].isin(pump_list)]
    
    # ポンプインデックスのマッピング
    pump_to_idx = {pump_id: idx for idx, pump_id in enumerate(pump_list)}
    df['pump_idx'] = df['equipment_id'].map(pump_to_idx)
    
    # 健全度状態の離散化（パーセンタイルベース）
    print(f"\n  健全度状態の離散化（{config.n_states}分割）...")
    
    # 全データの value 分布からパーセンタイルを計算
    percentiles = np.linspace(0, 100, config.n_states + 1)
    thresholds = np.percentile(df['value'].dropna(), percentiles)
    
    def discretize_state(value):
        """値を状態番号に変換（1始まり）"""
        if pd.isna(value):
            return np.nan
        state = np.searchsorted(thresholds[1:-1], value, side='right') + 1
        return min(state, config.n_states)
    
    df['state'] = df['value'].apply(discretize_state)
    df = df.dropna(subset=['state'])
    df['state'] = df['state'].astype(int)
    
    print(f"  状態分布:")
    print(df['state'].value_counts().sort_index())
    
    # トランジションペアの生成
    print("\n  トランジションペアの抽出...")
    
    transitions = []
    
    for pump_id in pump_list:
        pump_df = df[df['equipment_id'] == pump_id].copy()
        pump_df = pump_df.sort_values('date')
        
        for i in range(len(pump_df) - 1):
            row_now = pump_df.iloc[i]
            row_next = pump_df.iloc[i + 1]
            
            delta_t = (row_next['date'] - row_now['date']).days
            
            # 点検間隔のフィルタ
            if delta_t < config.min_delta_t or delta_t > config.max_delta_t:
                continue
            
            state_now = int(row_now['state'])
            state_next = int(row_next['state'])
            
            # 劣化イベント（状態が悪化）
            moved = 1 if state_next > state_now else 0
            
            transitions.append({
                'pump_idx': pump_to_idx[pump_id],
                'equipment_id': pump_id,
                'state_now': state_now,
                'state_next': state_next,
                'delta_t': delta_t,
                'moved': moved,
                'value_now': row_now['value']
            })
    
    trans_df = pd.DataFrame(transitions)
    
    print(f"  トランジションペア数: {len(trans_df)}")
    print(f"  劣化イベント数: {trans_df['moved'].sum()} ({trans_df['moved'].mean()*100:.1f}%)")
    
    # 共変量の生成（簡易版: value_now を正規化）
    print("\n  共変量の生成...")
    
    # 基本的な統計的特徴量
    X_features = []
    
    for _, row in trans_df.iterrows():
        features = [
            row['value_now'],  # 現在値
            row['state_now'],  # 現在状態
            row['delta_t']     # 点検間隔
        ]
        X_features.append(features)
    
    X = np.array(X_features, dtype=np.float64)
    
    # 標準化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    print(f"  共変量次元: {X.shape[1]}")
    
    # モデル入力データの整形
    N_obs = len(trans_df)
    N_pumps = len(pump_list)
    K = config.n_states
    
    model_data = {
        'N_obs': N_obs,
        'N_pumps': N_pumps,
        'K': K,
        'pump_idx': trans_df['pump_idx'].values.astype('int32'),
        'state_now': trans_df['state_now'].values.astype('int32'),
        'state_next': trans_df['state_next'].values.astype('int32'),
        'delta_t': trans_df['delta_t'].values.astype('float64'),
        'moved': trans_df['moved'].values.astype('int32'),
        'X': X,
        'n_cov': X.shape[1],
        'pump_equipment_id': np.array(pump_list),
        'pump_check_item_id': np.zeros(N_pumps)  # ダミー
    }
    
    # 保存
    output_path = config.get_absolute_path(config.model_input)
    np.savez_compressed(output_path, **model_data)
    print(f"\n  保存完了: {output_path}")
    
    print(f"\n  サマリ:")
    print(f"    観測数: {N_obs}")
    print(f"    ポンプ数: {N_pumps}")
    print(f"    状態数: {K}")
    print(f"    共変量次元: {X.shape[1]}")
    
    return model_data
