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
    
    def prepare_var_time_series(
        self,
        processed_time_series_df: pd.DataFrame,
        selected_sensors_df: pd.DataFrame,
        var_order: int = 7,
        min_lookback: int = 30,
        output_dir: str = "output"
    ) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        """
        VARLiNGAM用の時系列データを準備
        
        選定された50センサーの日次時系列データを、VARLiNGAM用の
        行列形式に変換します。
        
        Args:
            processed_time_series_df: processed_time_series.csv データ
            selected_sensors_df: var_selected_sensors.csv データ
                (equipment_id, check_item_id列が必要)
            var_order: VAR次数（ラグ日数）
            min_lookback: 最低必要日数（データ点数を確保）
            output_dir: 出力ディレクトリ
        
        Returns:
            (時系列データ行列, 変数名リスト, 時系列DataFrame)
            - 時系列データ行列: (n_samples, n_sensors) numpy配列
            - 変数名リスト: 各列のセンサー識別子
            - 時系列DataFrame: 保存用のDataFrame
        """
        if self.verbose:
            print(f"\n[VARLiNGAM用時系列データ準備]")
            print(f"  VAR次数: {var_order}日")
            print(f"  最低必要日数: {min_lookback}日")
            print(f"  選定センサー数: {len(selected_sensors_df)}")
        
        # 日付型に変換
        processed_time_series_df['date'] = pd.to_datetime(processed_time_series_df['date'])
        
        # 選定されたセンサーでフィルタリング
        selected_pairs = list(
            zip(
                selected_sensors_df['equipment_id'],
                selected_sensors_df['check_item_id']
            )
        )
        
        filtered_data = processed_time_series_df[
            processed_time_series_df.apply(
                lambda row: (row['equipment_id'], row['check_item_id']) in selected_pairs,
                axis=1
            )
        ].copy()
        
        if self.verbose:
            print(f"  フィルタリング後レコード数: {len(filtered_data)}")
        
        # 日付でソート
        filtered_data = filtered_data.sort_values('date')
        
        # 連続した日付のみを抽出（最小必要日数を満たすもの）
        min_required_days = var_order + min_lookback
        
        # 各センサーグループのデータ範囲を確認
        valid_sensors = []
        sensor_data_dict = {}
        
        for (equipment_id, check_item_id), group in filtered_data.groupby(['equipment_id', 'check_item_id']):
            group = group.sort_values('date')
            
            # 日次データに変換（欠損日を補完）
            date_range = pd.date_range(start=group['date'].min(), end=group['date'].max(), freq='D')
            group_reindexed = group.set_index('date').reindex(date_range)
            
            # 値列を前方補間 + 線形補間
            group_reindexed['value'] = group_reindexed['value'].fillna(method='ffill').fillna(method='bfill')
            
            # 最低必要日数を満たすかチェック
            if len(group_reindexed) >= min_required_days:
                sensor_id = f"{equipment_id}_{check_item_id}"
                valid_sensors.append({
                    'sensor_id': sensor_id,
                    'equipment_id': equipment_id,
                    'check_item_id': check_item_id,
                    'n_days': len(group_reindexed)
                })
                sensor_data_dict[sensor_id] = group_reindexed['value'].values
        
        if self.verbose:
            print(f"  最低日数を満たすセンサー数: {len(valid_sensors)}")
        
        if len(valid_sensors) == 0:
            raise ValueError(f"最低{min_required_days}日分のデータを持つセンサーが見つかりません")
        
        # 全センサーで共通の日付範囲を決定
        # 最も短いセンサーの日数に合わせる
        min_days = min([s['n_days'] for s in valid_sensors])
        
        if self.verbose:
            print(f"  共通日数: {min_days}日")
        
        # 時系列データ行列を作成
        n_samples = min_days
        n_sensors = len(valid_sensors)
        
        time_series_matrix = np.zeros((n_samples, n_sensors))
        variable_names = []
        
        for i, sensor_info in enumerate(valid_sensors):
            sensor_id = sensor_info['sensor_id']
            sensor_values = sensor_data_dict[sensor_id]
            
            # 最新のmin_days分を取得
            time_series_matrix[:, i] = sensor_values[-min_days:]
            variable_names.append(sensor_id)
        
        # 標準化（VARLiNGAMの数値的安定性のため）
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        time_series_matrix = scaler.fit_transform(time_series_matrix)
        
        if self.verbose:
            print(f"  時系列データ行列: {time_series_matrix.shape}")
            print(f"  標準化: 完了（平均0, 標準偏差1）")
            print(f"  変数名数: {len(variable_names)}")
        
        # DataFrameに変換して保存
        time_series_df = pd.DataFrame(
            time_series_matrix,
            columns=variable_names
        )
        
        # 日付インデックスを追加（最新の日付から遡る）
        # 注: 実際の日付は各センサーで異なる可能性があるため、
        # ここでは相対的な日付インデックス（0, 1, 2, ...）を使用
        time_series_df.insert(0, 'time_index', range(n_samples))
        
        # 保存
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "var_time_series.csv")
        time_series_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        if self.verbose:
            print(f"  保存: {output_path}")
        
        return time_series_matrix, variable_names, time_series_df


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
