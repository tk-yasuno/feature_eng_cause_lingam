"""
可視化モジュール

因果グラフの可視化と重要特徴量のレポート生成を行います。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from pathlib import Path
from typing import Dict, List, Tuple
import warnings

warnings.filterwarnings('ignore')


class CausalVisualizer:
    """
    因果グラフ可視化クラス
    """
    
    def __init__(
        self,
        figure_size: Tuple[int, int] = (12, 10),
        dpi: int = 300,
        edge_width_scale: float = 3.0,
        node_size_scale: int = 500,
        layout: str = "spring"
    ):
        """
        Args:
            figure_size: 図のサイズ
            dpi: 解像度
            edge_width_scale: エッジ幅のスケール
            node_size_scale: ノードサイズのスケール
            layout: レイアウトアルゴリズム（spring, circular, kamada_kawai）
        """
        self.figure_size = figure_size
        self.dpi = dpi
        self.edge_width_scale = edge_width_scale
        self.node_size_scale = node_size_scale
        self.layout = layout
        
        # スタイル設定
        sns.set_style("whitegrid")
        plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
        plt.rcParams['font.size'] = 10
    
    def create_causal_graph(
        self,
        adjacency_matrix: np.ndarray,
        feature_names: List[str],
        kpi_name: str = 'label_future_90d',
        threshold: float = 0.1,
        top_k: int = 15
    ) -> nx.DiGraph:
        """
        因果グラフを作成
        
        Args:
            adjacency_matrix: 隣接行列
            feature_names: 変数名リスト
            kpi_name: KPI変数名
            threshold: エッジを表示する最小効果の絶対値
            top_k: KPIへの効果トップK個のノードのみ表示
        
        Returns:
            NetworkX DiGraph
        """
        # KPIのインデックスを取得
        kpi_idx = feature_names.index(kpi_name)
        
        # KPIへの効果を計算
        effects_to_kpi = adjacency_matrix[:, kpi_idx]
        abs_effects = np.abs(effects_to_kpi)
        
        # トップK個の特徴量を選択
        top_indices = np.argsort(abs_effects)[::-1][:top_k]
        
        # グラフの作成
        G = nx.DiGraph()
        
        # KPIノードを追加
        G.add_node(kpi_name, node_type='kpi')
        
        # トップK個の特徴量ノードを追加
        for idx in top_indices:
            if idx != kpi_idx and abs_effects[idx] > threshold:
                feature_name = feature_names[idx]
                G.add_node(feature_name, node_type='feature')
                
                # KPIへのエッジを追加
                effect = effects_to_kpi[idx]
                if abs(effect) > threshold:
                    G.add_edge(feature_name, kpi_name, weight=effect)
        
        # 特徴量間のエッジを追加（閾値以上のもののみ）
        for i in top_indices:
            for j in top_indices:
                if i != j and i != kpi_idx and j != kpi_idx:
                    effect = adjacency_matrix[i, j]
                    if abs(effect) > threshold:
                        G.add_edge(feature_names[i], feature_names[j], weight=effect)
        
        return G
    
    def visualize_causal_graph(
        self,
        adjacency_matrix: np.ndarray,
        feature_names: List[str],
        kpi_name: str = 'label_future_90d',
        threshold: float = 0.1,
        top_k: int = 15,
        output_path: str = None
    ):
        """
        因果グラフを可視化
        
        Args:
            adjacency_matrix: 隣接行列
            feature_names: 変数名リスト
            kpi_name: KPI変数名
            threshold: エッジを表示する最小効果の絶対値
            top_k: KPIへの効果トップK個のノードのみ表示
            output_path: 保存先パス（Noneの場合は表示のみ）
        """
        # グラフ作成
        G = self.create_causal_graph(
            adjacency_matrix=adjacency_matrix,
            feature_names=feature_names,
            kpi_name=kpi_name,
            threshold=threshold,
            top_k=top_k
        )
        
        # レイアウトの計算
        if self.layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        elif self.layout == "circular":
            pos = nx.circular_layout(G)
        elif self.layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)
        
        # KPIを上部に配置
        if kpi_name in pos:
            pos[kpi_name] = np.array([0.5, 0.9])
        
        # プロット
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # ノードの色とサイズ
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if node == kpi_name:
                node_colors.append('#FF6B6B')  # KPIは赤
                node_sizes.append(self.node_size_scale * 2)
            else:
                node_colors.append('#4ECDC4')  # 特徴量は青緑
                node_sizes.append(self.node_size_scale)
        
        # エッジの幅と色
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        edge_widths = [abs(w) * self.edge_width_scale for w in edge_weights]
        edge_colors = ['#2ECC71' if w > 0 else '#E74C3C' for w in edge_weights]
        
        # ノードを描画
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            ax=ax
        )
        
        # エッジを描画
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.6,
            arrows=True,
            arrowsize=20,
            arrowstyle='->',
            connectionstyle='arc3,rad=0.1',
            ax=ax
        )
        
        # ラベルを描画
        nx.draw_networkx_labels(
            G, pos,
            font_size=9,
            font_weight='bold',
            ax=ax
        )
        
        # タイトルと凡例
        ax.set_title(
            f'Causal Graph: Top {top_k} Features → {kpi_name}\n'
            f'(Edge threshold: {threshold})',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        # 凡例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='KPI',
                   markerfacecolor='#FF6B6B', markersize=12),
            Line2D([0], [0], marker='o', color='w', label='Feature',
                   markerfacecolor='#4ECDC4', markersize=10),
            Line2D([0], [0], color='#2ECC71', lw=3, label='Positive Effect'),
            Line2D([0], [0], color='#E74C3C', lw=3, label='Negative Effect')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        ax.axis('off')
        plt.tight_layout()
        
        # 保存または表示
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"\n[因果グラフ保存] {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_effect_heatmap(
        self,
        adjacency_matrix: np.ndarray,
        feature_names: List[str],
        kpi_name: str = 'label_future_90d',
        top_k: int = 20,
        output_path: str = None
    ):
        """
        効果のヒートマップを可視化
        
        Args:
            adjacency_matrix: 隣接行列
            feature_names: 変数名リスト
            kpi_name: KPI変数名
            top_k: 表示する特徴量の数
            output_path: 保存先パス
        """
        # KPIのインデックスを取得
        kpi_idx = feature_names.index(kpi_name)
        
        # KPIへの効果を計算
        effects_to_kpi = adjacency_matrix[:, kpi_idx]
        abs_effects = np.abs(effects_to_kpi)
        
        # トップK個を選択
        top_indices = np.argsort(abs_effects)[::-1][:top_k]
        top_features = [feature_names[i] for i in top_indices if i != kpi_idx]
        top_effects = [effects_to_kpi[i] for i in top_indices if i != kpi_idx]
        
        # プロット
        fig, ax = plt.subplots(figsize=(10, max(6, len(top_features) * 0.3)), dpi=self.dpi)
        
        # 横棒グラフ
        colors = ['#2ECC71' if e > 0 else '#E74C3C' for e in top_effects]
        bars = ax.barh(range(len(top_features)), top_effects, color=colors, alpha=0.7)
        
        # ラベル
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=10)
        ax.set_xlabel('Causal Effect', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {len(top_features)} Features → {kpi_name}', 
                     fontsize=14, fontweight='bold', pad=15)
        
        # グリッド
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        # 値を表示
        for i, (bar, effect) in enumerate(zip(bars, top_effects)):
            width = bar.get_width()
            label_x = width + (0.02 if width > 0 else -0.02)
            ha = 'left' if width > 0 else 'right'
            ax.text(label_x, i, f'{effect:.3f}', va='center', ha=ha, fontsize=9)
        
        plt.tight_layout()
        
        # 保存または表示
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"\n[効果ヒートマップ保存] {output_path}")
        else:
            plt.show()
        
        plt.close()


def generate_interpretation_report(
    effects_df: pd.DataFrame,
    output_path: str = None
) -> str:
    """
    重要特徴量の解釈レポートを生成
    
    Args:
        effects_df: KPIへの効果データフレーム
        output_path: 保存先パス（Noneの場合は返すのみ）
    
    Returns:
        レポートの文字列
    """
    report = "# 因果探索結果レポート\n\n"
    report += "## 概要\n\n"
    report += f"- 分析された特徴量数: {len(effects_df)}\n"
    report += f"- 有意な因果効果を持つ特徴量数: {len(effects_df[effects_df['abs_effect'] > 0.1])}\n\n"
    
    report += "## KPIへの直接効果トップ10\n\n"
    report += "| 順位 | 特徴量 | 効果 | 絶対値 |\n"
    report += "|------|--------|------|--------|\n"
    
    for _, row in effects_df.head(10).iterrows():
        direction = "正" if row['effect'] > 0 else "負"
        report += f"| {int(row['rank'])} | {row['feature']} | {row['effect']:.4f} ({direction}) | {row['abs_effect']:.4f} |\n"
    
    report += "\n## 解釈\n\n"
    report += "### 正の効果（KPIを増加させる特徴量）\n\n"
    positive_effects = effects_df[effects_df['effect'] > 0].head(5)
    if len(positive_effects) > 0:
        for _, row in positive_effects.iterrows():
            report += f"- **{row['feature']}**: 効果 = {row['effect']:.4f}\n"
    else:
        report += "- なし\n"
    
    report += "\n### 負の効果（KPIを減少させる特徴量）\n\n"
    negative_effects = effects_df[effects_df['effect'] < 0].head(5)
    if len(negative_effects) > 0:
        for _, row in negative_effects.iterrows():
            report += f"- **{row['feature']}**: 効果 = {row['effect']:.4f}\n"
    else:
        report += "- なし\n"
    
    report += "\n## 推奨事項\n\n"
    report += "1. 上位の正の効果を持つ特徴量の変動をモニタリング\n"
    report += "2. 負の効果を持つ特徴量の改善策を検討\n"
    report += "3. 物理的な妥当性をドメイン専門家と確認\n"
    
    # 保存
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n[解釈レポート保存] {output_path}")
    
    return report


if __name__ == "__main__":
    # テスト用コード
    print("CausalVisualizerクラスのテスト")
    
    # サンプル隣接行列の生成
    np.random.seed(42)
    n_features = 10
    adjacency_matrix = np.random.randn(n_features, n_features) * 0.3
    adjacency_matrix[adjacency_matrix < 0.1] = 0  # 閾値処理
    
    feature_names = [f'feature_{i}' for i in range(1, n_features)] + ['label_future_90d']
    
    # 可視化
    visualizer = CausalVisualizer()
    visualizer.visualize_causal_graph(
        adjacency_matrix=adjacency_matrix,
        feature_names=feature_names,
        kpi_name='label_future_90d',
        threshold=0.1,
        top_k=8
    )
    
    print("\n可視化テスト完了")
