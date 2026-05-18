"""
因果探索モジュール（LiNGAM）

DirectLiNGAMを用いて特徴量とKPIの因果関係を推定します。
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
from imblearn.over_sampling import SMOTE

warnings.filterwarnings('ignore')


class CausalDiscovery:
    """
    LiNGAM因果探索クラス
    
    DirectLiNGAMを用いて特徴量間・特徴量→KPIの因果構造を推定
    """
    
    def __init__(
        self,
        algorithm: str = "DirectLiNGAM",
        random_state: int = 42,
        max_iter: int = 1000,
        var_order: int = 7,
        verbose: bool = True
    ):
        """
        Args:
            algorithm: LiNGAMアルゴリズム（DirectLiNGAM, ICALiNGAM, VARLiNGAM）
            random_state: 乱数シード
            max_iter: 最大反復回数
            var_order: VARLiNGAMのラグ次数（日数）
            verbose: 進捗表示
        """
        self.algorithm = algorithm
        self.random_state = random_state
        self.max_iter = max_iter
        self.var_order = var_order
        self.verbose = verbose
        self.model = None
        self.feature_names = None
        self.adjacency_matrix = None
        self.adjacency_matrices = None  # VARLiNGAM用（複数ラグ）
        self.causal_order = None
    
    def prepare_data_matrix(
        self,
        features_df: pd.DataFrame,
        feature_cols: List[str],
        kpi_col: str = 'label_future_90d',
        drop_na: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        LiNGAM用のデータ行列を準備
        
        Args:
            features_df: 特徴量データフレーム
            feature_cols: 特徴量の列名リスト
            kpi_col: KPI列名
            drop_na: NaNを含む行を削除するか
        
        Returns:
            (データ行列, 変数名リスト)
        """
        if self.verbose:
            print("\n[データ行列の準備]")
        
        # 必要な列を選択
        all_cols = feature_cols + [kpi_col]
        df_subset = features_df[all_cols].copy()
        
        if self.verbose:
            print(f"  入力レコード数: {len(df_subset)}")
            print(f"  特徴量数: {len(feature_cols)}")
        
        # NaN処理
        if drop_na:
            initial_len = len(df_subset)
            df_subset = df_subset.dropna()
            if self.verbose:
                print(f"  NaN除去後: {len(df_subset)} レコード ({initial_len - len(df_subset)} 削除)")
        else:
            # NaNを0で埋める
            df_subset = df_subset.fillna(0.0)
        
        # 行列に変換
        X = df_subset.values
        variable_names = list(df_subset.columns)
        
        if self.verbose:
            print(f"  データ行列サイズ: {X.shape}")
        
        return X, variable_names
    
    def fit(self, X: np.ndarray, variable_names: List[str]) -> Dict:
        """
        LiNGAMモデルを学習
        
        Args:
            X: データ行列 [n_samples, n_features]
            variable_names: 変数名リスト
        
        Returns:
            推定結果の辞書
        """
        if self.verbose:
            print(f"\n[{self.algorithm}による因果探索]")
            print(f"  サンプル数: {X.shape[0]}")
            print(f"  変数数: {X.shape[1]}")
        
        # lingamのインポート（遅延インポート）
        try:
            import lingam
        except ImportError:
            raise ImportError("lingamパッケージがインストールされていません。pip install lingamを実行してください。")
        
        # モデルの初期化
        if self.algorithm == "DirectLiNGAM":
            self.model = lingam.DirectLiNGAM(random_state=self.random_state)
        elif self.algorithm == "ICALiNGAM":
            self.model = lingam.ICALiNGAM(random_state=self.random_state, max_iter=self.max_iter)
        elif self.algorithm == "VARLiNGAM":
            # prune=Falseで自動ラグ選択を無効化し、指定したラグ次数を固定で使用
            self.model = lingam.VARLiNGAM(
                lags=self.var_order, 
                random_state=self.random_state,
                prune=False  # 自動ラグ選択を無効化
            )
        else:
            raise ValueError(f"未対応のアルゴリズム: {self.algorithm}")
        
        # 学習
        self.model.fit(X)
        
        # 結果の取得
        if self.algorithm == "VARLiNGAM":
            # VARLiNGAMの場合は複数の隣接行列（各ラグごと）
            self.adjacency_matrices = self.model.adjacency_matrices_  # shape: (n_lags, n_features, n_features)
            self.causal_order = self.model.causal_order_ if hasattr(self.model, 'causal_order_') else None
            # 便宜上、lag=1の行列をadjacency_matrixにも格納
            self.adjacency_matrix = self.adjacency_matrices[0] if len(self.adjacency_matrices) > 0 else None
        else:
            # DirectLiNGAM, ICALiNGAMの場合
            self.adjacency_matrix = self.model.adjacency_matrix_
            self.causal_order = self.model.causal_order_
            self.adjacency_matrices = None
        
        self.feature_names = variable_names
        
        if self.verbose:
            if self.algorithm == "VARLiNGAM":
                print(f"  因果順序: {self.causal_order}")
                print(f"  隣接行列数（ラグ数）: {len(self.adjacency_matrices)}")
                print(f"  隣接行列サイズ（各ラグ）: {self.adjacency_matrices[0].shape}")
                total_nonzero = sum([np.count_nonzero(m) for m in self.adjacency_matrices])
                print(f"  非ゼロ要素数（全ラグ合計）: {total_nonzero}")
            else:
                print(f"  因果順序: {self.causal_order}")
                print(f"  隣接行列サイズ: {self.adjacency_matrix.shape}")
                print(f"  非ゼロ要素数: {np.count_nonzero(self.adjacency_matrix)}")
        
        # 結果を辞書にまとめる
        results = {
            'adjacency_matrix': self.adjacency_matrix,
            'adjacency_matrices': self.adjacency_matrices,  # VARLiNGAM用
            'causal_order': self.causal_order,
            'feature_names': self.feature_names,
            'algorithm': self.algorithm,
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }
        
        if self.algorithm == "VARLiNGAM":
            results['var_order'] = self.var_order
        
        return results
    
    def get_effects_to_kpi(
        self,
        kpi_name: str = 'label_future_90d',
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        KPIへの直接効果を取得
        
        Args:
            kpi_name: KPI変数名
            threshold: 効果の絶対値閾値
        
        Returns:
            特徴量名、効果の大きさ、順位を含むデータフレーム
        """
        if self.adjacency_matrix is None or self.feature_names is None:
            raise ValueError("モデルがまだ学習されていません。fit()を先に実行してください。")
        
        # KPIのインデックスを取得
        try:
            kpi_idx = self.feature_names.index(kpi_name)
        except ValueError:
            raise ValueError(f"KPI '{kpi_name}' が変数リストに見つかりません")
        
        # KPIへの効果（列方向）
        effects = self.adjacency_matrix[:, kpi_idx]
        
        # データフレーム化
        effects_df = pd.DataFrame({
            'feature': self.feature_names,
            'effect': effects,
            'abs_effect': np.abs(effects)
        })
        
        # 閾値でフィルタ
        effects_df = effects_df[effects_df['abs_effect'] > threshold].copy()
        
        # 効果の大きさでソート
        effects_df = effects_df.sort_values('abs_effect', ascending=False)
        effects_df['rank'] = range(1, len(effects_df) + 1)
        
        return effects_df
    
    def get_temporal_effects_to_kpi(
        self,
        kpi_name: str = 'label_future_90d',
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        VARLiNGAM用: 各ラグでのKPIへの時間的効果を取得
        
        Args:
            kpi_name: KPI変数名
            threshold: 効果の絶対値閾値
        
        Returns:
            特徴量名、ラグ、効果の大きさを含むデータフレーム
        """
        if self.algorithm != "VARLiNGAM":
            raise ValueError("このメソッドはVARLiNGAMのみで使用できます")
        
        if self.adjacency_matrices is None or self.feature_names is None:
            raise ValueError("モデルがまだ学習されていません。fit()を先に実行してください。")
        
        # KPIのインデックスを取得
        try:
            kpi_idx = self.feature_names.index(kpi_name)
        except ValueError:
            raise ValueError(f"KPI '{kpi_name}' が変数リストに見つかりません")
        
        # 各ラグでの効果を集計
        effects_list = []
        
        for lag in range(len(self.adjacency_matrices)):
            adjacency_matrix = self.adjacency_matrices[lag]
            
            # KPIへの効果（列方向）
            effects = adjacency_matrix[:, kpi_idx]
            
            for feature_idx, effect in enumerate(effects):
                if abs(effect) > threshold:
                    effects_list.append({
                        'feature': self.feature_names[feature_idx],
                        'lag': lag + 1,  # ラグは1から始める（lag=1は1日前）
                        'effect': effect,
                        'abs_effect': abs(effect)
                    })
        
        # データフレーム化
        effects_df = pd.DataFrame(effects_list)
        
        if len(effects_df) > 0:
            # 効果の大きさでソート
            effects_df = effects_df.sort_values('abs_effect', ascending=False).reset_index(drop=True)
            effects_df['rank'] = range(1, len(effects_df) + 1)
        
        return effects_df
    
    def save_results(self, output_path: str):
        """
        推定結果を保存
        
        Args:
            output_path: 出力ファイルパス（.pkl）
        """
        if self.adjacency_matrix is None and self.adjacency_matrices is None:
            raise ValueError("保存する結果がありません。fit()を先に実行してください。")
        
        results = {
            'adjacency_matrix': self.adjacency_matrix,
            'adjacency_matrices': self.adjacency_matrices,  # VARLiNGAM用
            'causal_order': self.causal_order,
            'feature_names': self.feature_names,
            'algorithm': self.algorithm
        }
        
        if self.algorithm == "VARLiNGAM":
            results['var_order'] = self.var_order
        
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)
        
        if self.verbose:
            print(f"\n[結果保存] {output_path}")
    
    @staticmethod
    def load_results(input_path: str) -> Dict:
        """
        保存された結果を読み込み
        
        Args:
            input_path: 入力ファイルパス（.pkl）
        
        Returns:
            結果の辞書
        """
        with open(input_path, 'rb') as f:
            results = pickle.load(f)
        
        return results


def apply_smote_if_needed(
    group_data: Dict[str, pd.DataFrame],
    feature_cols: List[str],
    kpi_col: str,
    config: Dict,
    verbose: bool = True
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, bool]]:
    """
    グループ間のデータ不均衡を検出し、必要に応じてSMOTEで拡張
    
    Args:
        group_data: {group_name: df} のデータ
        feature_cols: 特徴量の列名リスト
        kpi_col: KPI列名
        config: 設定辞書（imbalance_correction セクション）
        verbose: 進捗表示
    
    Returns:
        (拡張後のデータ辞書, {group_name: augmented} の辞書)
    """
    imbalance_config = config.get('imbalance_correction', {})
    enabled = imbalance_config.get('enabled', False)
    
    if not enabled:
        if verbose:
            print("\n[SMOTE拡張] 無効")
        return group_data, {g: False for g in group_data.keys()}
    
    min_ratio = imbalance_config.get('min_ratio_threshold', 10.0)
    target_strategy = imbalance_config.get('target_strategy', 'mean')
    k_neighbors = imbalance_config.get('smote_k_neighbors', 5)
    random_state = imbalance_config.get('random_state', 42)
    
    if verbose:
        print("\n" + "=" * 80)
        print("データ不均衡補正（SMOTE）")
        print("=" * 80)
    
    # グループごとのレコード数を取得
    group_sizes = {g: len(df) for g, df in group_data.items()}
    
    if verbose:
        print(f"\n[グループ別レコード数]")
        for g, size in group_sizes.items():
            print(f"  {g}: {size:,} レコード")
    
    # 最小と最大のグループを特定
    min_group = min(group_sizes, key=group_sizes.get)
    max_group = max(group_sizes, key=group_sizes.get)
    min_size = group_sizes[min_group]
    max_size = group_sizes[max_group]
    
    # 不均衡比率を計算
    imbalance_ratio = max_size / min_size
    
    if verbose:
        print(f"\n[不均衡比率]")
        print(f"  最小グループ: {min_group} ({min_size:,})")
        print(f"  最大グループ: {max_group} ({max_size:,})")
        print(f"  不均衡比率: {imbalance_ratio:.2f}倍")
        print(f"  閾値: {min_ratio}倍")
    
    # 閾値未満の場合は拡張不要
    if imbalance_ratio < min_ratio:
        if verbose:
            print(f"  → 不均衡比率が閾値未満のため、SMOTE拡張をスキップ")
        return group_data, {g: False for g in group_data.keys()}
    
    # 目標レコード数を計算
    if target_strategy == 'mean':
        # 他のグループの平均に合わせる
        other_sizes = [s for g, s in group_sizes.items() if g != min_group]
        target_size = int(np.mean(other_sizes))
    elif target_strategy == 'max':
        # 最大グループに合わせる
        target_size = max_size
    else:
        raise ValueError(f"Unknown target_strategy: {target_strategy}")
    
    if verbose:
        print(f"\n[SMOTE拡張設定]")
        print(f"  対象グループ: {min_group}")
        print(f"  現在のサイズ: {min_size:,}")
        print(f"  目標サイズ: {target_size:,}")
        print(f"  拡張倍率: {target_size / min_size:.2f}倍")
        print(f"  k_neighbors: {k_neighbors}")
    
    # 拡張後のデータを格納
    augmented_data = {}
    augmented_flags = {}
    
    for group, df in group_data.items():
        if group == min_group:
            # 最小グループをSMOTEで拡張
            if verbose:
                print(f"\n[{group} をSMOTE拡張中...]")
            
            # 特徴量とKPIを分離
            X = df[feature_cols].values
            y = df[kpi_col].values
            
            # k_neighborsが利用可能なサンプル数を超えないように調整
            effective_k = min(k_neighbors, len(X) - 1)
            
            if effective_k < 1:
                if verbose:
                    print(f"  ⚠ 警告: サンプル数が少なすぎてSMOTEを適用できません")
                augmented_data[group] = df.copy()
                augmented_flags[group] = False
                continue
            
            try:
                # SMOTEの代わりに手動で近傍サンプリング（連続値対応）
                from sklearn.neighbors import NearestNeighbors
                
                nbrs = NearestNeighbors(n_neighbors=effective_k+1, algorithm='auto').fit(X)
                
                # 合成サンプル生成
                synthetic_X = []
                synthetic_y = []
                
                np.random.seed(random_state)
                
                n_synthetic = target_size - len(X)
                
                for _ in range(n_synthetic):
                    # ランダムに元サンプルを選択
                    idx = np.random.randint(0, len(X))
                    
                    # k近傍を取得
                    distances, indices = nbrs.kneighbors([X[idx]])
                    
                    # 最近傍を除外（自分自身）
                    neighbor_indices = indices[0][1:]
                    
                    # ランダムに近傍を選択
                    neighbor_idx = np.random.choice(neighbor_indices)
                    
                    # 線形補間で合成サンプルを生成
                    alpha = np.random.rand()
                    synthetic_sample = X[idx] + alpha * (X[neighbor_idx] - X[idx])
                    synthetic_target = y[idx] + alpha * (y[neighbor_idx] - y[idx])
                    
                    synthetic_X.append(synthetic_sample)
                    synthetic_y.append(synthetic_target)
                
                # 元データと合成データを結合
                X_resampled = np.vstack([X, np.array(synthetic_X)])
                y_resampled = np.hstack([y, np.array(synthetic_y)])
                
                if verbose:
                    print(f"  元データ: {len(X):,} レコード")
                    print(f"  合成データ: {len(synthetic_X):,} レコード")
                    print(f"  拡張後: {len(X_resampled):,} レコード")
                
                # データフレームに変換
                df_augmented = pd.DataFrame(X_resampled, columns=feature_cols)
                df_augmented[kpi_col] = y_resampled
                
                # u_groupがあれば保持
                if 'u_group' in df.columns:
                    df_augmented['u_group'] = group
                
                augmented_data[group] = df_augmented
                augmented_flags[group] = True
                    
            except Exception as e:
                if verbose:
                    print(f"  ⚠ エラー: SMOTE拡張に失敗しました: {e}")
                augmented_data[group] = df.copy()
                augmented_flags[group] = False
        else:
            # その他のグループはそのまま
            augmented_data[group] = df.copy()
            augmented_flags[group] = False
    
    if verbose:
        print(f"\n[拡張後のレコード数]")
        for g, df in augmented_data.items():
            flag = "（SMOTE拡張済み）" if augmented_flags[g] else ""
            print(f"  {g}: {len(df):,} レコード {flag}")
    
    return augmented_data, augmented_flags


def run_causal_discovery(
    features_df: pd.DataFrame,
    feature_cols: List[str],
    kpi_col: str = 'label_future_90d',
    algorithm: str = "DirectLiNGAM",
    random_state: int = 42,
    output_path: str = None,
    verbose: bool = True
) -> Tuple[Dict, pd.DataFrame]:
    """
    因果探索を実行するメイン関数
    
    Args:
        features_df: 特徴量データフレーム
        feature_cols: 特徴量の列名リスト
        kpi_col: KPI列名
        algorithm: LiNGAMアルゴリズム
        random_state: 乱数シード
        output_path: 結果の保存先（Noneの場合は保存しない）
        verbose: 進捗表示
    
    Returns:
        (因果探索結果, KPIへの効果データフレーム)
    """
    # 因果探索オブジェクトの初期化
    cd = CausalDiscovery(
        algorithm=algorithm,
        random_state=random_state,
        verbose=verbose
    )
    
    # データ行列の準備
    X, variable_names = cd.prepare_data_matrix(
        features_df=features_df,
        feature_cols=feature_cols,
        kpi_col=kpi_col
    )
    
    # 因果探索の実行
    results = cd.fit(X, variable_names)
    
    # KPIへの効果を取得
    effects_df = cd.get_effects_to_kpi(kpi_name=kpi_col)
    
    if verbose:
        print(f"\n[KPIへの直接効果トップ10]")
        print(effects_df.head(10).to_string(index=False))
    
    # 結果の保存
    if output_path:
        cd.save_results(output_path)
    
    return results, effects_df


if __name__ == "__main__":
    # テスト用コード
    print("CausalDiscoveryクラスのテスト")
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 1000
    
    # 因果構造: X1 -> X2 -> X3 -> Y
    X1 = np.random.randn(n_samples)
    X2 = 0.8 * X1 + np.random.randn(n_samples) * 0.5
    X3 = 0.6 * X2 + np.random.randn(n_samples) * 0.5
    Y = 0.7 * X3 + np.random.randn(n_samples) * 0.3
    
    # データフレーム化
    df = pd.DataFrame({
        'feature_1': X1,
        'feature_2': X2,
        'feature_3': X3,
        'label_future_90d': Y
    })
    
    # 因果探索の実行
    results, effects_df = run_causal_discovery(
        features_df=df,
        feature_cols=['feature_1', 'feature_2', 'feature_3'],
        kpi_col='label_future_90d'
    )
    
    print("\n因果探索完了")
