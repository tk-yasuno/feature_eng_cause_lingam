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
        verbose: bool = True
    ):
        """
        Args:
            algorithm: LiNGAMアルゴリズム（DirectLiNGAM, ICALiNGAM）
            random_state: 乱数シード
            max_iter: 最大反復回数
            verbose: 進捗表示
        """
        self.algorithm = algorithm
        self.random_state = random_state
        self.max_iter = max_iter
        self.verbose = verbose
        self.model = None
        self.feature_names = None
        self.adjacency_matrix = None
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
        else:
            raise ValueError(f"未対応のアルゴリズム: {self.algorithm}")
        
        # 学習
        self.model.fit(X)
        
        # 結果の取得
        self.adjacency_matrix = self.model.adjacency_matrix_
        self.causal_order = self.model.causal_order_
        self.feature_names = variable_names
        
        if self.verbose:
            print(f"  因果順序: {self.causal_order}")
            print(f"  隣接行列サイズ: {self.adjacency_matrix.shape}")
            print(f"  非ゼロ要素数: {np.count_nonzero(self.adjacency_matrix)}")
        
        # 結果を辞書にまとめる
        results = {
            'adjacency_matrix': self.adjacency_matrix,
            'causal_order': self.causal_order,
            'feature_names': self.feature_names,
            'algorithm': self.algorithm,
            'n_samples': X.shape[0],
            'n_features': X.shape[1]
        }
        
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
    
    def save_results(self, output_path: str):
        """
        推定結果を保存
        
        Args:
            output_path: 出力ファイルパス（.pkl）
        """
        if self.adjacency_matrix is None:
            raise ValueError("保存する結果がありません。fit()を先に実行してください。")
        
        results = {
            'adjacency_matrix': self.adjacency_matrix,
            'causal_order': self.causal_order,
            'feature_names': self.feature_names,
            'algorithm': self.algorithm
        }
        
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
