"""
データ前処理パイプラインモジュール

labeled_time_series.csvから64設備のデータを読み込み、
欠損処理、外れ値処理、時系列の同期化を行います。
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    データ前処理クラス
    
    機能:
    1. 64設備のフィルタリング
    2. 欠損値処理（前方補間、線形補間）
    3. 外れ値処理（±3σ外をNaN化）
    4. 時系列の同期化（日次粒度に統一）
    """
    
    def __init__(
        self,
        outlier_threshold: float = 3.0,
        interpolation_method: str = "linear",
        max_missing_ratio: float = 0.3,
        verbose: bool = True
    ):
        """
        Args:
            outlier_threshold: 外れ値の標準偏差閾値（±Nσ）
            interpolation_method: 補間方法（linear, ffill, bfill）
            max_missing_ratio: 最大欠損率（これを超えるセンサーは除外）
            verbose: 進捗表示
        """
        self.outlier_threshold = outlier_threshold
        self.interpolation_method = interpolation_method
        self.max_missing_ratio = max_missing_ratio
        self.verbose = verbose
    
    def load_time_series_data(
        self,
        time_series_path: str,
        selected_equipment_path: str
    ) -> pd.DataFrame:
        """
        labeled_time_series.csvを読み込み、64設備でフィルタリング
        
        Args:
            time_series_path: labeled_time_series.csvのパス
            selected_equipment_path: selected_64_equipment.jsonのパス
        
        Returns:
            フィルタリング済みデータフレーム
        """
        if self.verbose:
            print("\n[データ読み込み]")
        
        # 時系列データの読み込み
        df = pd.read_csv(time_series_path)
        if self.verbose:
            print(f"  全データ: {len(df)} レコード")
            print(f"  全設備数: {df['equipment_id'].nunique()}")
        
        # 64設備のIDリストを読み込み
        with open(selected_equipment_path, 'r', encoding='utf-8') as f:
            selected_data = json.load(f)
        
        # equipment_idsリストを取得
        if isinstance(selected_data, dict) and 'equipment_ids' in selected_data:
            selected_equipment_ids = selected_data['equipment_ids']
        elif isinstance(selected_data, list):
            selected_equipment_ids = selected_data
        else:
            raise ValueError("selected_64_equipment.jsonの形式が不正です")
        
        if self.verbose:
            print(f"  選定設備数: {len(selected_equipment_ids)}")
        
        # フィルタリング
        df_filtered = df[df['equipment_id'].isin(selected_equipment_ids)].copy()
        
        if self.verbose:
            print(f"  フィルタリング後: {len(df_filtered)} レコード")
            print(f"  対象設備数: {df_filtered['equipment_id'].nunique()}")
            print(f"  チェック項目数: {df_filtered['check_item_id'].nunique()}")
        
        return df_filtered
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        欠損値処理
        
        Args:
            df: 入力データフレーム
        
        Returns:
            欠損値処理済みデータフレーム
        """
        if self.verbose:
            print("\n[欠損値処理]")
        
        # 日付型に変換
        df['date'] = pd.to_datetime(df['date'])
        
        # 欠損率のチェック
        initial_missing = df['value'].isna().sum()
        initial_total = len(df)
        if self.verbose:
            print(f"  初期欠損: {initial_missing} / {initial_total} ({initial_missing/initial_total*100:.2f}%)")
        
        # グループごとに欠損率をチェック
        groups_to_keep = []
        for (eq_id, ch_id), group in df.groupby(['equipment_id', 'check_item_id']):
            missing_ratio = group['value'].isna().sum() / len(group)
            if missing_ratio <= self.max_missing_ratio:
                groups_to_keep.append((eq_id, ch_id))
        
        if self.verbose:
            total_groups = df.groupby(['equipment_id', 'check_item_id']).ngroups
            print(f"  欠損率が{self.max_missing_ratio*100}%以下のグループ: {len(groups_to_keep)} / {total_groups}")
        
        # フィルタリング
        df = df[df.apply(lambda x: (x['equipment_id'], x['check_item_id']) in groups_to_keep, axis=1)].copy()
        
        # 前方補間 → 線形補間
        df_processed = []
        for (eq_id, ch_id), group in df.groupby(['equipment_id', 'check_item_id']):
            group = group.sort_values('date')
            
            # 前方補間
            group['value'] = group['value'].ffill()
            
            # 線形補間（残った欠損値）
            if self.interpolation_method == 'linear':
                group['value'] = group['value'].interpolate(method='linear')
            elif self.interpolation_method == 'ffill':
                group['value'] = group['value'].ffill()
            elif self.interpolation_method == 'bfill':
                group['value'] = group['value'].bfill()
            
            # まだ残っている欠損値は0で埋める
            group['value'] = group['value'].fillna(0.0)
            
            df_processed.append(group)
        
        df = pd.concat(df_processed, ignore_index=True)
        
        final_missing = df['value'].isna().sum()
        if self.verbose:
            print(f"  処理後の欠損: {final_missing} / {len(df)} ({final_missing/len(df)*100:.2f}%)")
        
        return df
    
    def handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        外れ値処理（±3σ外をNaN化してから補間）
        
        Args:
            df: 入力データフレーム
        
        Returns:
            外れ値処理済みデータフレーム
        """
        if self.verbose:
            print("\n[外れ値処理]")
        
        outliers_count = 0
        total_count = 0
        
        df_processed = []
        for (eq_id, ch_id), group in df.groupby(['equipment_id', 'check_item_id']):
            # 統計量の計算
            mean = group['value'].mean()
            std = group['value'].std()
            
            # ±Nσ外を外れ値として検出
            lower_bound = mean - self.outlier_threshold * std
            upper_bound = mean + self.outlier_threshold * std
            
            outlier_mask = (group['value'] < lower_bound) | (group['value'] > upper_bound)
            outliers_count += outlier_mask.sum()
            total_count += len(group)
            
            # 外れ値をNaNに置換
            group.loc[outlier_mask, 'value'] = np.nan
            
            # 線形補間
            group['value'] = group['value'].interpolate(method='linear')
            group['value'] = group['value'].ffill().bfill().fillna(0.0)
            
            df_processed.append(group)
        
        df = pd.concat(df_processed, ignore_index=True)
        
        if self.verbose:
            print(f"  外れ値検出: {outliers_count} / {total_count} ({outliers_count/total_count*100:.2f}%)")
            print(f"  閾値: ±{self.outlier_threshold}σ")
        
        return df
    
    def synchronize_time_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        時系列の同期化（日次粒度に統一）
        
        Args:
            df: 入力データフレーム
        
        Returns:
            同期化済みデータフレーム
        """
        if self.verbose:
            print("\n[時系列の同期化]")
        
        # 日付型に変換
        df['date'] = pd.to_datetime(df['date']).dt.date
        df['date'] = pd.to_datetime(df['date'])
        
        # 同一日に複数レコードがある場合は平均値を取る
        df_sync = df.groupby(['equipment_id', 'check_item_id', 'date']).agg({
            'value': 'mean',
            'value_normalized': 'mean',
            'value_mean': 'first',
            'value_std': 'first',
            'label_current': 'max',  # ラベルは最大値（異常があれば1）
            'label_future_30d': 'max',
            'label_future_60d': 'max',
            'label_future_90d': 'max'
        }).reset_index()
        
        if self.verbose:
            print(f"  同期化前: {len(df)} レコード")
            print(f"  同期化後: {len(df_sync)} レコード")
        
        return df_sync
    
    def preprocess(
        self,
        time_series_path: str,
        selected_equipment_path: str
    ) -> pd.DataFrame:
        """
        前処理パイプライン全体を実行
        
        Args:
            time_series_path: labeled_time_series.csvのパス
            selected_equipment_path: selected_64_equipment.jsonのパス
        
        Returns:
            前処理済みデータフレーム
        """
        # 1. データ読み込みとフィルタリング
        df = self.load_time_series_data(time_series_path, selected_equipment_path)
        
        # 2. 欠損値処理
        df = self.handle_missing_values(df)
        
        # 3. 外れ値処理
        df = self.handle_outliers(df)
        
        # 4. 時系列の同期化
        df = self.synchronize_time_series(df)
        
        if self.verbose:
            print("\n[前処理完了]")
            print(f"  最終レコード数: {len(df)}")
            print(f"  設備数: {df['equipment_id'].nunique()}")
            print(f"  チェック項目数: {df['check_item_id'].nunique()}")
            print(f"  日付範囲: {df['date'].min()} ~ {df['date'].max()}")
        
        return df


def load_and_preprocess_data(
    time_series_path: str,
    selected_equipment_path: str,
    outlier_threshold: float = 3.0,
    interpolation_method: str = "linear",
    max_missing_ratio: float = 0.3,
    verbose: bool = True
) -> pd.DataFrame:
    """
    データ前処理のメイン関数
    
    Args:
        time_series_path: labeled_time_series.csvのパス
        selected_equipment_path: selected_64_equipment.jsonのパス
        outlier_threshold: 外れ値の標準偏差閾値（±Nσ）
        interpolation_method: 補間方法（linear, ffill, bfill）
        max_missing_ratio: 最大欠損率
        verbose: 進捗表示
    
    Returns:
        前処理済みデータフレーム
    """
    preprocessor = DataPreprocessor(
        outlier_threshold=outlier_threshold,
        interpolation_method=interpolation_method,
        max_missing_ratio=max_missing_ratio,
        verbose=verbose
    )
    
    return preprocessor.preprocess(time_series_path, selected_equipment_path)


if __name__ == "__main__":
    # テスト用コード
    from pathlib import Path
    
    base_dir = Path(__file__).parent.parent
    time_series_path = base_dir / "data" / "processed" / "labeled_time_series.csv"
    selected_equipment_path = base_dir / "data" / "processed" / "selected_64_equipment.json"
    
    if time_series_path.exists() and selected_equipment_path.exists():
        print("データ前処理のテスト実行")
        df = load_and_preprocess_data(
            str(time_series_path),
            str(selected_equipment_path)
        )
        print("\n処理完了")
        print(df.head())
    else:
        print("テストデータが見つかりません")
