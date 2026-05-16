"""
検証モジュール（ブートストラップ）

ブートストラップサンプリングを用いて因果構造の安定性を評価します。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings

warnings.filterwarnings('ignore')


class BootstrapValidator:
    """
    ブートストラップ検証クラス
    
    サブサンプリングを繰り返して因果構造の安定性を評価
    """
    
    def __init__(
        self,
        n_sampling: int = 100,
        sample_ratio: float = 0.8,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: bool = True
    ):
        """
        Args:
            n_sampling: サンプリング回数
            sample_ratio: サブサンプリング比率
            random_state: 乱数シード
            n_jobs: 並列処理のジョブ数(-1で全コア使用)
            verbose: 進捗表示
        """
        self.n_sampling = n_sampling
        self.sample_ratio = sample_ratio
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.adjacency_matrices = []
        self.edge_frequencies = None
    
    @staticmethod
    def _fit_bootstrap_sample(
        X: np.ndarray,
        sample_size: int,
        algorithm: str,
        random_state: int,
        iteration: int
    ) -> np.ndarray:
        """
        1回のブートストラップサンプリングとモデル学習を実行(並列処理用)
        
        Args:
            X: データ行列 [n_samples, n_features]
            sample_size: サンプルサイズ
            algorithm: LiNGAMアルゴリズム
            random_state: 基本乱数シード
            iteration: イテレーション番号
        
        Returns:
            隣接行列、またはエラー時はNone
        """
        try:
            import lingam
            
            # サブサンプリング
            np.random.seed(random_state + iteration)
            indices = np.random.choice(X.shape[0], size=sample_size, replace=True)
            X_sample = X[indices]
            
            # モデルの学習
            if algorithm == "DirectLiNGAM":
                model = lingam.DirectLiNGAM(random_state=random_state + iteration)
            elif algorithm == "ICALiNGAM":
                model = lingam.ICALiNGAM(random_state=random_state + iteration)
            else:
                raise ValueError(f"未対応のアルゴリズム: {algorithm}")
            
            model.fit(X_sample)
            return model.adjacency_matrix_
        
        except Exception as e:
            return None
    
    def bootstrap_causal_discovery(
        self,
        X: np.ndarray,
        variable_names: List[str],
        algorithm: str = "DirectLiNGAM"
    ) -> List[np.ndarray]:
        """
        ブートストラップによる因果探索(並列処理版)
        
        Args:
            X: データ行列 [n_samples, n_features]
            variable_names: 変数名リスト
            algorithm: LiNGAMアルゴリズム
        
        Returns:
            隣接行列のリスト
        """
        if self.verbose:
            print(f"\n[ブートストラップ検証(並列処理)]")
            print(f"  サンプリング回数: {self.n_sampling}")
            print(f"  サブサンプリング比率: {self.sample_ratio}")
            if self.n_jobs == -1:
                print(f"  並列処理: 全コア使用")
            else:
                print(f"  並列処理: {self.n_jobs}コア")
        
        # lingamのインポート確認
        try:
            import lingam
        except ImportError:
            raise ImportError("lingamパッケージがインストールされていません。")
        
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        sample_size = int(n_samples * self.sample_ratio)
        
        # 並列処理でブートストラップ実行
        if self.n_jobs == 1:
            # シリアル処理(デバッグ用)
            adjacency_matrices = []
            iterator = tqdm(range(self.n_sampling), desc="Bootstrap") if self.verbose else range(self.n_sampling)
            for i in iterator:
                adj_matrix = self._fit_bootstrap_sample(
                    X, sample_size, algorithm, self.random_state, i
                )
                if adj_matrix is not None:
                    adjacency_matrices.append(adj_matrix)
        else:
            # 並列処理(n_jobs=-1で全コア使用)
            results = Parallel(n_jobs=self.n_jobs, verbose=1 if self.verbose else 0)(
                delayed(self._fit_bootstrap_sample)(
                    X, sample_size, algorithm, self.random_state, i
                )
                for i in range(self.n_sampling)
            )
            # Noneを除外
            adjacency_matrices = [r for r in results if r is not None]
        
        self.adjacency_matrices = adjacency_matrices
        
        if self.verbose:
            print(f"\n  成功したサンプリング: {len(adjacency_matrices)} / {self.n_sampling}")
        
        return adjacency_matrices
    
    def calculate_edge_frequencies(
        self,
        threshold: float = 0.0
    ) -> np.ndarray:
        """
        エッジ出現頻度を計算
        
        Args:
            threshold: エッジとみなす最小効果の絶対値
        
        Returns:
            エッジ出現頻度の行列 [n_features, n_features]
        """
        if len(self.adjacency_matrices) == 0:
            raise ValueError("ブートストラップが実行されていません。")
        
        n_features = self.adjacency_matrices[0].shape[0]
        edge_count = np.zeros((n_features, n_features))
        
        # 各ブートストラップサンプルでエッジの有無をカウント
        for adj_matrix in self.adjacency_matrices:
            edge_exists = (np.abs(adj_matrix) > threshold).astype(int)
            edge_count += edge_exists
        
        # 頻度に変換（0～1）
        self.edge_frequencies = edge_count / len(self.adjacency_matrices)
        
        return self.edge_frequencies
    
    def get_stable_edges(
        self,
        variable_names: List[str],
        stability_threshold: float = 0.7
    ) -> pd.DataFrame:
        """
        安定したエッジを取得
        
        Args:
            variable_names: 変数名リスト
            stability_threshold: 安定性閾値（この頻度以上のエッジを採用）
        
        Returns:
            安定エッジのデータフレーム
        """
        if self.edge_frequencies is None:
            self.calculate_edge_frequencies()
        
        # 平均効果を計算
        mean_adjacency = np.mean(self.adjacency_matrices, axis=0)
        
        # 安定したエッジを抽出
        stable_edges = []
        n_features = len(variable_names)
        
        for i in range(n_features):
            for j in range(n_features):
                if i != j and self.edge_frequencies[i, j] >= stability_threshold:
                    stable_edges.append({
                        'from': variable_names[i],
                        'to': variable_names[j],
                        'frequency': self.edge_frequencies[i, j],
                        'mean_effect': mean_adjacency[i, j],
                        'abs_mean_effect': abs(mean_adjacency[i, j])
                    })
        
        stable_edges_df = pd.DataFrame(stable_edges)
        
        if len(stable_edges_df) > 0:
            stable_edges_df = stable_edges_df.sort_values('abs_mean_effect', ascending=False)
        
        if self.verbose:
            print(f"\n[安定エッジ（頻度 ≥ {stability_threshold}）]")
            print(f"  安定エッジ数: {len(stable_edges_df)}")
        
        return stable_edges_df
    
    def visualize_stability_heatmap(
        self,
        variable_names: List[str],
        kpi_name: str = 'label_future_90d',
        top_k: int = 15,
        output_path: str = None
    ):
        """
        エッジ出現頻度のヒートマップを可視化
        
        Args:
            variable_names: 変数名リスト
            kpi_name: KPI変数名
            top_k: 表示する特徴量の数
            output_path: 保存先パス
        """
        if self.edge_frequencies is None:
            self.calculate_edge_frequencies()
        
        # KPIのインデックスを取得
        kpi_idx = variable_names.index(kpi_name)
        
        # KPIへのエッジ頻度でソート
        frequencies_to_kpi = self.edge_frequencies[:, kpi_idx]
        top_indices = np.argsort(frequencies_to_kpi)[::-1][:top_k]
        
        # サブ行列を抽出
        top_indices_with_kpi = list(top_indices) + [kpi_idx]
        sub_matrix = self.edge_frequencies[np.ix_(top_indices_with_kpi, top_indices_with_kpi)]
        sub_labels = [variable_names[i] for i in top_indices_with_kpi]
        
        # プロット
        fig, ax = plt.subplots(figsize=(12, 10), dpi=300)
        
        sns.heatmap(
            sub_matrix,
            xticklabels=sub_labels,
            yticklabels=sub_labels,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'Edge Frequency'},
            ax=ax
        )
        
        ax.set_title(
            f'Bootstrap Stability: Edge Frequencies (Top {top_k} → {kpi_name})\n'
            f'n_sampling={self.n_sampling}, sample_ratio={self.sample_ratio}',
            fontsize=14,
            fontweight='bold',
            pad=15
        )
        ax.set_xlabel('To', fontsize=12, fontweight='bold')
        ax.set_ylabel('From', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # 保存または表示
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n[安定性ヒートマップ保存] {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_kpi_stability(
        self,
        variable_names: List[str],
        kpi_name: str = 'label_future_90d',
        top_k: int = 15,
        output_path: str = None
    ):
        """
        KPIへのエッジの安定性を可視化
        
        Args:
            variable_names: 変数名リスト
            kpi_name: KPI変数名
            top_k: 表示する特徴量の数
            output_path: 保存先パス
        """
        if self.edge_frequencies is None:
            self.calculate_edge_frequencies()
        
        # KPIのインデックスを取得
        kpi_idx = variable_names.index(kpi_name)
        
        # KPIへのエッジ頻度を計算
        frequencies_to_kpi = self.edge_frequencies[:, kpi_idx]
        
        # 平均効果を計算
        mean_adjacency = np.mean(self.adjacency_matrices, axis=0)
        mean_effects_to_kpi = mean_adjacency[:, kpi_idx]
        
        # トップK個を選択
        top_indices = np.argsort(frequencies_to_kpi)[::-1][:top_k]
        top_features = [variable_names[i] for i in top_indices if i != kpi_idx]
        top_frequencies = [frequencies_to_kpi[i] for i in top_indices if i != kpi_idx]
        top_effects = [mean_effects_to_kpi[i] for i in top_indices if i != kpi_idx]
        
        # プロット
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, max(6, len(top_features) * 0.3)), dpi=300)
        
        # 左：エッジ出現頻度
        bars1 = ax1.barh(range(len(top_features)), top_frequencies, color='#3498DB', alpha=0.7)
        ax1.set_yticks(range(len(top_features)))
        ax1.set_yticklabels(top_features, fontsize=10)
        ax1.set_xlabel('Edge Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Bootstrap Stability (Edge Frequency)', fontsize=12, fontweight='bold')
        ax1.set_xlim(0, 1)
        ax1.axvline(x=0.7, color='red', linestyle='--', linewidth=1, label='Threshold (0.7)')
        ax1.legend()
        ax1.grid(axis='x', alpha=0.3)
        
        # 値を表示
        for i, (bar, freq) in enumerate(zip(bars1, top_frequencies)):
            width = bar.get_width()
            ax1.text(width + 0.02, i, f'{freq:.2f}', va='center', ha='left', fontsize=9)
        
        # 右：平均因果効果
        colors = ['#2ECC71' if e > 0 else '#E74C3C' for e in top_effects]
        bars2 = ax2.barh(range(len(top_features)), top_effects, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(top_features)))
        ax2.set_yticklabels(top_features, fontsize=10)
        ax2.set_xlabel('Mean Causal Effect', fontsize=12, fontweight='bold')
        ax2.set_title('Average Effect across Bootstrap Samples', fontsize=12, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax2.grid(axis='x', alpha=0.3)
        
        # 値を表示
        for i, (bar, effect) in enumerate(zip(bars2, top_effects)):
            width = bar.get_width()
            label_x = width + (0.02 if width > 0 else -0.02)
            ha = 'left' if width > 0 else 'right'
            ax2.text(label_x, i, f'{effect:.3f}', va='center', ha=ha, fontsize=9)
        
        fig.suptitle(
            f'Bootstrap Validation: Top {len(top_features)} Features → {kpi_name}\n'
            f'(n_sampling={self.n_sampling}, sample_ratio={self.sample_ratio})',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )
        
        plt.tight_layout()
        
        # 保存または表示
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n[KPI安定性可視化保存] {output_path}")
        else:
            plt.show()
        
        plt.close()


def run_bootstrap_validation(
    X: np.ndarray,
    variable_names: List[str],
    kpi_name: str = 'label_future_90d',
    algorithm: str = "DirectLiNGAM",
    n_sampling: int = 100,
    sample_ratio: float = 0.8,
    stability_threshold: float = 0.7,
    random_state: int = 42,
    n_jobs: int = -1,
    output_dir: str = None,
    verbose: bool = True
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    ブートストラップ検証を実行するメイン関数(並列処理対応)
    
    Args:
        X: データ行列 [n_samples, n_features]
        variable_names: 変数名リスト
        kpi_name: KPI変数名
        algorithm: LiNGAMアルゴリズム
        n_sampling: サンプリング回数
        sample_ratio: サブサンプリング比率
        stability_threshold: 安定性閾値
        random_state: 乱数シード
        n_jobs: 並列処理のジョブ数(-1で全コア使用)
        output_dir: 出力ディレクトリ(Noneの場合は保存しない)
        verbose: 進捗表示
    
    Returns:
        (安定エッジデータフレーム, エッジ出現頻度行列)
    """
    from pathlib import Path
    
    # バリデータの初期化
    validator = BootstrapValidator(
        n_sampling=n_sampling,
        sample_ratio=sample_ratio,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    # ブートストラップ実行
    validator.bootstrap_causal_discovery(X, variable_names, algorithm)
    
    # エッジ頻度を計算
    edge_frequencies = validator.calculate_edge_frequencies()
    
    # 安定エッジを取得
    stable_edges_df = validator.get_stable_edges(variable_names, stability_threshold)
    
    if verbose and len(stable_edges_df) > 0:
        print("\n[安定エッジトップ10]")
        print(stable_edges_df.head(10).to_string(index=False))
    
    # 可視化
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ヒートマップ
        validator.visualize_stability_heatmap(
            variable_names=variable_names,
            kpi_name=kpi_name,
            output_path=str(output_dir / "bootstrap_stability_heatmap.png")
        )
        
        # KPI安定性
        validator.visualize_kpi_stability(
            variable_names=variable_names,
            kpi_name=kpi_name,
            output_path=str(output_dir / "bootstrap_kpi_stability.png")
        )
        
        # 安定エッジをCSV保存
        if len(stable_edges_df) > 0:
            stable_edges_df.to_csv(output_dir / "stable_edges.csv", index=False)
            print(f"\n[安定エッジ保存] {output_dir / 'stable_edges.csv'}")
    
    return stable_edges_df, edge_frequencies


if __name__ == "__main__":
    # テスト用コード
    print("BootstrapValidatorクラスのテスト")
    
    # サンプルデータ生成
    np.random.seed(42)
    n_samples = 500
    
    # 因果構造: X1 -> X2 -> X3 -> Y
    X1 = np.random.randn(n_samples)
    X2 = 0.8 * X1 + np.random.randn(n_samples) * 0.5
    X3 = 0.6 * X2 + np.random.randn(n_samples) * 0.5
    Y = 0.7 * X3 + np.random.randn(n_samples) * 0.3
    
    X = np.column_stack([X1, X2, X3, Y])
    variable_names = ['feature_1', 'feature_2', 'feature_3', 'label_future_90d']
    
    # ブートストラップ検証
    stable_edges_df, edge_frequencies = run_bootstrap_validation(
        X=X,
        variable_names=variable_names,
        kpi_name='label_future_90d',
        n_sampling=20,  # テストなので少なめ
        sample_ratio=0.8
    )
    
    print("\nブートストラップ検証完了")
