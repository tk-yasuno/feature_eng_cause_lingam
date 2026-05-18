"""
特徴量エンジニアリングモジュール (並列処理版)

90日間の時系列データから22個の統計的特徴量を生成します。
参考: hetero_advi_discretized_mixture/data_preprocessing.py のFeatureEngineerクラス
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from scipy import stats
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    統計的特徴量エンジニアリングクラス
    
    過去の90日間のデータから22個の特徴量を計算:
    - 統計的特徴量(11個): mean, std, min, max, median, q25, q75, iqr, skewness, kurtosis, cv_90d
    - トレンド特徴量(5個): trend_slope_90d, trend_intercept, recent_vs_past_ratio, recent_vs_past_diff, recent_change_rate
    - 変動性特徴量(6個): diff_mean, diff_abs_mean, rolling_std_7d/14d/30d_mean, max_drawdown, mean_drawdown
    """
    
    @staticmethod
    def extract_all_features(sequence: np.ndarray) -> Dict[str, float]:
        """
        全特徴量の抽出(NaN/Inf安全版)
        
        Args:
            sequence: 時系列データ配列 [seq_len]
            
        Returns:
            全特徴量の辞書(22個)
        """
        features = {}
        
        # 入力データのクリーン
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 最低データ点数チェック
        if len(sequence) < 3:
            return {k: 0.0 for k in [
                'mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'iqr',
                'skewness', 'kurtosis', 'cv_90d', 'trend_slope_90d', 'trend_intercept',
                'recent_vs_past_ratio', 'recent_vs_past_diff', 'recent_change_rate',
                'diff_mean', 'diff_abs_mean', 'rolling_std_7d_mean', 'rolling_std_14d_mean',
                'rolling_std_30d_mean', 'max_drawdown', 'mean_drawdown'
            ]}
        
        # === 統計的特徴量 ===
        features['mean'] = float(np.mean(sequence))
        features['std'] = float(np.std(sequence))
        features['min'] = float(np.min(sequence))
        features['max'] = float(np.max(sequence))
        features['median'] = float(np.median(sequence))
        features['q25'] = float(np.percentile(sequence, 25))
        features['q75'] = float(np.percentile(sequence, 75))
        features['iqr'] = features['q75'] - features['q25']
        
        # 歪度・尖度(NaN対策)
        if len(sequence) > 3:
            try:
                skew_val = stats.skew(sequence)
                kurt_val = stats.kurtosis(sequence)
                features['skewness'] = float(skew_val) if np.isfinite(skew_val) else 0.0
                features['kurtosis'] = float(kurt_val) if np.isfinite(kurt_val) else 0.0
            except:
                features['skewness'] = 0.0
                features['kurtosis'] = 0.0
        else:
            features['skewness'] = 0.0
            features['kurtosis'] = 0.0
        
        # 変動係数(安全な除算)
        mean_abs = abs(features['mean'])
        if mean_abs > 1e-10:
            cv_val = features['std'] / mean_abs
            features['cv_90d'] = float(cv_val) if np.isfinite(cv_val) else 0.0
        else:
            features['cv_90d'] = 0.0
        
        # === トレンド特徴量 ===
        try:
            # 線形回帰による傾き
            coeffs = np.polyfit(np.arange(len(sequence)), sequence, deg=1)
            slope = float(coeffs[0])
            intercept = float(coeffs[1])
            features['trend_slope_90d'] = slope if np.isfinite(slope) else 0.0
            features['trend_intercept'] = intercept if np.isfinite(intercept) else float(sequence[0])
        except:
            features['trend_slope_90d'] = 0.0
            features['trend_intercept'] = float(sequence[0])
        
        # 最近の期間 vs 過去の期間
        if len(sequence) >= 60:
            recent_mean = float(np.mean(sequence[-30:]))
            past_mean = float(np.mean(sequence[-60:-30]))
            
            if abs(past_mean) > 1e-10:
                ratio = recent_mean / past_mean
                features['recent_vs_past_ratio'] = float(ratio) if np.isfinite(ratio) else 1.0
                features['recent_vs_past_diff'] = float(recent_mean - past_mean)
            else:
                features['recent_vs_past_ratio'] = 1.0
                features['recent_vs_past_diff'] = 0.0
        else:
            features['recent_vs_past_ratio'] = 1.0
            features['recent_vs_past_diff'] = 0.0
        
        # 最終値の変化率
        if len(sequence) >= 10:
            recent_slope = float((sequence[-1] - sequence[-10]) / 10)
            features['recent_change_rate'] = recent_slope if np.isfinite(recent_slope) else 0.0
        else:
            features['recent_change_rate'] = 0.0
        
        # === 変動性特徴量 ===
        # 差分系列
        diff = np.diff(sequence)
        diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
        
        features['diff_mean'] = float(np.mean(diff))
        features['diff_abs_mean'] = float(np.mean(np.abs(diff)))
        
        # ローリング統計量(7日、14日、30日)
        for window in [7, 14, 30]:
            if len(sequence) >= window:
                rolling_std = []
                for i in range(len(sequence) - window + 1):
                    window_std = float(np.std(sequence[i:i+window]))
                    if np.isfinite(window_std):
                        rolling_std.append(window_std)
                
                if len(rolling_std) > 0:
                    features[f'rolling_std_{window}d_mean'] = float(np.mean(rolling_std))
                else:
                    features[f'rolling_std_{window}d_mean'] = 0.0
            else:
                features[f'rolling_std_{window}d_mean'] = 0.0
        
        # 最大ドローダウン(ピークからの最大下落)
        cummax = np.maximum.accumulate(sequence)
        drawdown = cummax - sequence
        drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
        features['max_drawdown'] = float(np.max(drawdown))
        features['mean_drawdown'] = float(np.mean(drawdown))
        
        # 最終的なNaN/Infチェック
        for key, value in features.items():
            if not np.isfinite(value):
                features[key] = 0.0
        
        return features
    
    @staticmethod
    def get_feature_names() -> List[str]:
        """
        特徴量名のリストを返す
        
        Returns:
            22個の特徴量名のリスト
        """
        return [
            # 統計的特徴量(11個)
            'mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'iqr',
            'skewness', 'kurtosis', 'cv_90d',
            # トレンド特徴量(5個)
            'trend_slope_90d', 'trend_intercept', 'recent_vs_past_ratio', 
            'recent_vs_past_diff', 'recent_change_rate',
            # 変動性特徴量(6個)
            'diff_mean', 'diff_abs_mean', 'rolling_std_7d_mean', 
            'rolling_std_14d_mean', 'rolling_std_30d_mean', 
            'max_drawdown', 'mean_drawdown'
        ]
    
    @staticmethod
    def create_var_features(
        time_series_df: pd.DataFrame,
        max_sensors: int = 50,
        missing_threshold: float = 0.1,
        output_dir: str = "output",
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        VARLiNGAM用の代表センサーを自動選定
        
        64機器×229センサー(最大14,656変数)から、欠損率と分散に基づいて
        50個の代表センサーを選定します。
        
        選定基準:
        1. 欠損率 < 10% (時系列データの連続性を確保)
        2. 分散上位 (変動が大きいセンサーを優先)
        3. 各機器から最低1センサーは選定 (機器カバレッジを保証)
        
        Args:
            time_series_df: processed_time_series.csv データ
            max_sensors: 選定する最大センサー数
            missing_threshold: 欠損率閾値 (0.1 = 10%)
            output_dir: 出力ディレクトリ
            verbose: 進捗表示
        
        Returns:
            選定されたセンサーの情報を含むDataFrame
            (sensor_id, equipment_id, missing_ratio, variance, rank)
        """
        if verbose:
            print(f"\n[VARLiNGAM用センサー自動選定]")
            print(f"  入力データ: {len(time_series_df)} レコード")
            print(f"  設備数: {time_series_df['equipment_id'].nunique()}")
            print(f"  センサー数: {time_series_df['check_item_id'].nunique()}")
        
        # 日付型に変換
        time_series_df['date'] = pd.to_datetime(time_series_df['date'])
        
        # 各(equipment_id, check_item_id)ごとに統計量を計算
        sensor_stats = []
        
        for (equipment_id, check_item_id), group in time_series_df.groupby(['equipment_id', 'check_item_id']):
            total_count = len(group)
            missing_count = group['value'].isna().sum()
            missing_ratio = missing_count / total_count if total_count > 0 else 1.0
            
            # 分散計算（欠損値を除外）
            valid_values = group['value'].dropna()
            variance = valid_values.var() if len(valid_values) > 1 else 0.0
            
            sensor_stats.append({
                'equipment_id': equipment_id,
                'check_item_id': check_item_id,
                'sensor_id': f"{equipment_id}_{check_item_id}",
                'total_count': total_count,
                'missing_count': missing_count,
                'missing_ratio': missing_ratio,
                'variance': variance,
                'valid_count': len(valid_values)
            })
        
        sensor_stats_df = pd.DataFrame(sensor_stats)
        
        if verbose:
            print(f"  センサー組み合わせ数: {len(sensor_stats_df)}")
        
        # 基準1: 欠損率が閾値未満のセンサーをフィルタリング
        filtered_df = sensor_stats_df[sensor_stats_df['missing_ratio'] < missing_threshold].copy()
        
        if verbose:
            print(f"  欠損率<{missing_threshold*100:.0f}%のセンサー数: {len(filtered_df)}")
        
        # 基準2: 分散でソート（降順）
        filtered_df = filtered_df.sort_values('variance', ascending=False)
        
        # 基準3: max_sensors個選定（全センサーの場合は全て選定）
        if max_sensors >= len(filtered_df):
            # 全センサーを選定
            selected_sensors = filtered_df.to_dict('records')
            if verbose:
                print(f"  全{len(selected_sensors)}センサーを選定")
        else:
            # 分散上位からmax_sensors個選定（同じ機器から複数選ばないように制限）
            selected_sensors = []
            equipment_sensor_count = {}
            max_per_equipment = 2  # 各機器から最大2個まで
            
            for _, row in filtered_df.iterrows():
                equipment_id = row['equipment_id']
                current_count = equipment_sensor_count.get(equipment_id, 0)
                
                # 同じ機器からの選定数が上限に達していなければ選定
                if current_count < max_per_equipment:
                    selected_sensors.append(row.to_dict())
                    equipment_sensor_count[equipment_id] = current_count + 1
                    
                    if len(selected_sensors) >= max_sensors:
                        break
            
            if verbose:
                print(f"  分散上位から{max_sensors}個選定完了")
                print(f"  機器カバレッジ: {len(equipment_sensor_count)}/{filtered_df['equipment_id'].nunique()}機器")
        
        # 選定結果をDataFrameに変換
        selected_df = pd.DataFrame(selected_sensors)
        
        # ランク付け（分散順）
        selected_df = selected_df.sort_values('variance', ascending=False).reset_index(drop=True)
        selected_df['rank'] = range(1, len(selected_df) + 1)
        
        if verbose:
            print(f"  最終選定センサー数: {len(selected_df)}")
            print(f"  欠損率範囲: {selected_df['missing_ratio'].min():.2%} ~ {selected_df['missing_ratio'].max():.2%}")
            print(f"  分散範囲: {selected_df['variance'].min():.2f} ~ {selected_df['variance'].max():.2f}")
        
        # 選定結果を保存
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "var_selected_sensors.csv")
        selected_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        if verbose:
            print(f"  保存: {output_path}")
        
        return selected_df
    
    @staticmethod
    def _process_group(
        group: pd.DataFrame,
        lookback_days: int,
        min_data_points: int,
        value_col: str,
        date_col: str,
        feature_names: List[str]
    ) -> List[Dict]:
        """
        1つのグループ(設備×チェック項目)の特徴量を計算(並列処理用)
        
        Args:
            group: グループのデータフレーム
            lookback_days: ルックバック期間(日数)
            min_data_points: 最低必要データ点数
            value_col: 値の列名
            date_col: 日付の列名
            feature_names: 特徴量名のリスト
        
        Returns:
            特徴量を含む辞書のリスト
        """
        group = group.reset_index(drop=True)
        results = []
        
        # 各時点で特徴量を計算
        for idx in range(len(group)):
            current_date = group.iloc[idx][date_col]
            lookback_start = current_date - pd.Timedelta(days=lookback_days)
            
            # ルックバック期間のデータを抽出
            lookback_data = group[
                (group[date_col] > lookback_start) & 
                (group[date_col] <= current_date)
            ]
            
            # 基本情報をコピー
            row_data = group.iloc[idx].to_dict()
            
            # 特徴量を計算
            if len(lookback_data) >= min_data_points:
                sequence = lookback_data[value_col].values
                features = FeatureEngineer.extract_all_features(sequence)
            else:
                # データ不足時は0埋め
                features = {name: 0.0 for name in feature_names}
            
            # 特徴量を追加
            row_data.update(features)
            results.append(row_data)
        
        return results
    
    @staticmethod
    def extract_windowed_features(
        df: pd.DataFrame,
        lookback_days: int = 90,
        min_data_points: int = 3,
        value_col: str = 'value',
        date_col: str = 'date',
        group_cols: List[str] = None,
        n_jobs: int = -1
    ) -> pd.DataFrame:
        """
        90日ローリング窓で特徴量を抽出(並列処理版)
        
        Args:
            df: 時系列データフレーム(date, equipment_id, check_item_id, value列が必要)
            lookback_days: ルックバック期間(日数)
            min_data_points: 最低必要データ点数
            value_col: 値の列名
            date_col: 日付の列名
            group_cols: グループ化する列名のリスト(デフォルト: ['equipment_id', 'check_item_id'])
            n_jobs: 並列処理のジョブ数(-1で全コア使用、1でシリアル処理)
        
        Returns:
            特徴量が追加されたデータフレーム
        """
        if group_cols is None:
            group_cols = ['equipment_id', 'check_item_id']
        
        # 日付型に変換
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=group_cols + [date_col])
        
        feature_names = FeatureEngineer.get_feature_names()
        
        # グループのリストを作成
        groups = [group for _, group in df.groupby(group_cols)]
        
        # 並列処理でグループごとに特徴量を計算
        if n_jobs == 1:
            # シリアル処理(デバッグ用)
            results_list = [
                FeatureEngineer._process_group(
                    group, lookback_days, min_data_points, 
                    value_col, date_col, feature_names
                )
                for group in groups
            ]
        else:
            # 並列処理(n_jobs=-1で全コア使用)
            results_list = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(FeatureEngineer._process_group)(
                    group, lookback_days, min_data_points, 
                    value_col, date_col, feature_names
                )
                for group in groups
            )
        
        # 結果を統合
        results = []
        for group_results in results_list:
            results.extend(group_results)
        
        return pd.DataFrame(results)


def extract_features_from_time_series(
    time_series_df: pd.DataFrame,
    lookback_days: int = 90,
    min_data_points: int = 3,
    n_jobs: int = -1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    時系列データから特徴量を抽出するメイン関数(並列処理対応)
    
    Args:
        time_series_df: labeled_time_series.csv から読み込んだデータ
        lookback_days: ルックバック期間(日数)
        min_data_points: 最低必要データ点数
        n_jobs: 並列処理のジョブ数(-1で全コア使用、8で8コア使用)
        verbose: 進捗表示
    
    Returns:
        特徴量が追加されたデータフレーム
    """
    if verbose:
        print(f"\n[特徴量抽出(並列処理)] ルックバック期間: {lookback_days}日, 最低データ点数: {min_data_points}")
        print(f"  入力データ: {len(time_series_df)} レコード")
        print(f"  設備数: {time_series_df['equipment_id'].nunique()}")
        print(f"  チェック項目数: {time_series_df['check_item_id'].nunique()}")
        n_groups = time_series_df.groupby(['equipment_id', 'check_item_id']).ngroups
        print(f"  処理グループ数: {n_groups}")
        if n_jobs == -1:
            print(f"  並列処理: 全コア使用")
        else:
            print(f"  並列処理: {n_jobs}コア")
    
    # 特徴量抽出(並列処理)
    features_df = FeatureEngineer.extract_windowed_features(
        df=time_series_df,
        lookback_days=lookback_days,
        min_data_points=min_data_points,
        value_col='value',
        date_col='date',
        group_cols=['equipment_id', 'check_item_id'],
        n_jobs=n_jobs
    )
    
    if verbose:
        print(f"  出力データ: {len(features_df)} レコード")
        print(f"  特徴量列数: {len(FeatureEngineer.get_feature_names())}")
    
    return features_df


if __name__ == "__main__":
    # テスト用コード
    print("FeatureEngineerクラスのテスト")
    
    # サンプルデータ生成
    test_sequence = np.random.randn(90) + 10
    
    # 特徴量抽出
    features = FeatureEngineer.extract_all_features(test_sequence)
    
    print("\n抽出された特徴量:")
    for name, value in features.items():
        print(f"  {name}: {value:.4f}")
    
    print(f"\n合計特徴量数: {len(features)}")
