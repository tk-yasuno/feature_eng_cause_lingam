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
        
        # 値を表示（小さい値は棒の内側、大きい値は外側）
        for i, (bar, effect) in enumerate(zip(bars, top_effects)):
            width = bar.get_width()
            abs_width = abs(width)
            
            # 絶対値が0.01より小さい場合は棒の内側（原点付近）に配置
            if abs_width < 0.01:
                label_x = 0
                ha = 'center'
            else:
                # それ以外は棒の外側に配置
                label_x = width + (0.01 * max(abs_effects) if width > 0 else -0.01 * max(abs_effects))
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
    
    def visualize_var_causal_graph(
        self,
        adjacency_matrices: List[np.ndarray],
        feature_names: List[str],
        kpi_name: str = 'label_future_90d',
        threshold: float = 0.1,
        top_k: int = 15,
        output_path: str = None
    ):
        """
        VARLiNGAM用の時間的因果グラフを可視化
        
        エッジラベルに時間遅延（lag=1, lag=2, ...）を表示する統合グラフ
        
        Args:
            adjacency_matrices: 各ラグの隣接行列のリスト [n_lags, n_features, n_features]
            feature_names: 変数名リスト
            kpi_name: KPI変数名
            threshold: エッジを表示する最小効果の絶対値
            top_k: KPIへの効果トップK個のノードのみ表示
            output_path: 保存先パス
        """
        # KPIのインデックスを取得
        kpi_idx = feature_names.index(kpi_name)
        
        # 全ラグでのKPIへの総合的な効果を計算
        total_effects = np.zeros(len(feature_names))
        for adj_matrix in adjacency_matrices:
            total_effects += np.abs(adj_matrix[:, kpi_idx])
        
        # トップK個の特徴量を選択
        top_indices = np.argsort(total_effects)[::-1][:top_k]
        
        # グラフの作成
        G = nx.DiGraph()
        
        # KPIノードを追加
        G.add_node(kpi_name, node_type='kpi')
        
        # トップK個の特徴量ノードを追加
        selected_features = set()
        for idx in top_indices:
            if idx != kpi_idx and total_effects[idx] > threshold:
                feature_name = feature_names[idx]
                G.add_node(feature_name, node_type='feature')
                selected_features.add(idx)
        
        # 各ラグのエッジを追加
        edge_data = {}  # (from, to) -> [(lag, effect), ...]
        
        for lag, adj_matrix in enumerate(adjacency_matrices, start=1):
            # KPIへのエッジ
            for idx in selected_features:
                effect = adj_matrix[idx, kpi_idx]
                if abs(effect) > threshold:
                    feature_name = feature_names[idx]
                    key = (feature_name, kpi_name)
                    if key not in edge_data:
                        edge_data[key] = []
                    edge_data[key].append((lag, effect))
            
            # 特徴量間のエッジ
            for i in selected_features:
                for j in selected_features:
                    if i != j:
                        effect = adj_matrix[i, j]
                        if abs(effect) > threshold:
                            from_name = feature_names[i]
                            to_name = feature_names[j]
                            key = (from_name, to_name)
                            if key not in edge_data:
                                edge_data[key] = []
                            edge_data[key].append((lag, effect))
        
        # エッジをグラフに追加
        for (from_node, to_node), lag_effects in edge_data.items():
            # 最大絶対値効果を持つラグを代表として使用
            max_lag, max_effect = max(lag_effects, key=lambda x: abs(x[1]))
            total_effect = sum([abs(e) for _, e in lag_effects])
            
            # エッジラベル: 主要ラグと効果
            if len(lag_effects) == 1:
                label = f"lag={max_lag}"
            else:
                label = f"lag={max_lag}*\n({len(lag_effects)} lags)"
            
            G.add_edge(from_node, to_node, weight=max_effect, total_weight=total_effect, label=label)
        
        # レイアウトの計算
        if self.layout == "spring":
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
            # KPIを上部に固定
            if kpi_name in pos:
                pos[kpi_name] = (0.5, 1.0)
        elif self.layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.kamada_kawai_layout(G)
        
        # プロット
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        # ノードの色とサイズ
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            if G.nodes[node]['node_type'] == 'kpi':
                node_colors.append('#E74C3C')  # 赤（KPI）
                node_sizes.append(self.node_size_scale * 2)
            else:
                node_colors.append('#3498DB')  # 青（特徴量）
                node_sizes.append(self.node_size_scale)
        
        # エッジの色と幅（ラグに応じてグラデーション）
        edge_colors = []
        edge_widths = []
        for u, v, data in G.edges(data=True):
            effect = data['weight']
            # 色: 正の効果は緑、負の効果は赤
            if effect > 0:
                edge_colors.append('#2ECC71')
            else:
                edge_colors.append('#E74C3C')
            # 幅: 効果の大きさに比例
            edge_widths.append(abs(effect) * self.edge_width_scale)
        
        # ノードを描画
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, 
                               alpha=0.9, ax=ax)
        
        # エッジを描画
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=edge_widths, 
                               alpha=0.6, arrows=True, arrowsize=20, 
                               arrowstyle='->', connectionstyle='arc3,rad=0.1', ax=ax)
        
        # ノードラベルを描画
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        # エッジラベル（時間遅延）を描画
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, 
                                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7), ax=ax)
        
        ax.set_title('Temporal Causal Graph (VARLiNGAM)', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')
        
        plt.tight_layout()
        
        # 保存または表示
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"\n[VARLiNGAM因果グラフ保存] {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def visualize_temporal_heatmap(
        self,
        adjacency_matrices: List[np.ndarray],
        feature_names: List[str],
        kpi_name: str = 'label_future_90d',
        top_k: int = 20,
        output_path: str = None
    ):
        """
        VARLiNGAM用の時系列ヒートマップを可視化
        
        縦軸=特徴量、横軸=ラグ（1-7日）、色=効果の強さ
        
        Args:
            adjacency_matrices: 各ラグの隣接行列のリスト
            feature_names: 変数名リスト
            kpi_name: KPI変数名
            top_k: 表示する特徴量の数
            output_path: 保存先パス
        """
        # KPIのインデックスを取得
        kpi_idx = feature_names.index(kpi_name)
        
        # 各ラグでのKPIへの効果を集計
        n_lags = len(adjacency_matrices)
        n_features = len(feature_names)
        
        effects_matrix = np.zeros((n_features, n_lags))
        
        for lag, adj_matrix in enumerate(adjacency_matrices):
            effects_matrix[:, lag] = adj_matrix[:, kpi_idx]
        
        # 総合的な効果でソート
        total_effects = np.sum(np.abs(effects_matrix), axis=1)
        top_indices = np.argsort(total_effects)[::-1][:top_k]
        
        # トップK個の特徴量を抽出（KPIを除く）
        top_indices = [idx for idx in top_indices if idx != kpi_idx][:top_k]
        top_features = [feature_names[idx] for idx in top_indices]
        top_effects_matrix = effects_matrix[top_indices, :]
        
        # プロット
        fig, ax = plt.subplots(figsize=(max(8, n_lags), max(6, len(top_features) * 0.4)), dpi=self.dpi)
        
        # ヒートマップ
        im = ax.imshow(top_effects_matrix, cmap='RdBu_r', aspect='auto', 
                       vmin=-np.abs(top_effects_matrix).max(), vmax=np.abs(top_effects_matrix).max())
        
        # カラーバー
        cbar = plt.colorbar(im, ax=ax, label='Causal Effect')
        
        # 軸ラベル
        ax.set_xticks(range(n_lags))
        ax.set_xticklabels([f'lag={i+1}' for i in range(n_lags)], fontsize=10)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features, fontsize=9)
        
        ax.set_xlabel('Time Lag (days)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        ax.set_title(f'Temporal Effects to {kpi_name} (VARLiNGAM)', 
                     fontsize=14, fontweight='bold', pad=15)
        
        # 各セルに値を表示
        for i in range(len(top_features)):
            for j in range(n_lags):
                value = top_effects_matrix[i, j]
                if abs(value) > 0.01:  # 小さい値は表示しない
                    text_color = 'white' if abs(value) > np.abs(top_effects_matrix).max() * 0.5 else 'black'
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                           fontsize=8, color=text_color, fontweight='bold')
        
        plt.tight_layout()
        
        # 保存または表示
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            print(f"\n[時系列ヒートマップ保存] {output_path}")
        else:
            plt.show()
        
        plt.close()


def generate_interpretation_report(
    effects_df: pd.DataFrame,
    output_path: str = None,
    var_effects_df: pd.DataFrame = None
) -> str:
    """
    重要特徴量の解釈レポートを生成
    
    Args:
        effects_df: KPIへの効果データフレーム（DirectLiNGAM用）
        output_path: 保存先パス（Noneの場合は返すのみ）
        var_effects_df: VARLiNGAM用の時間的効果データフレーム（オプション）
    
    Returns:
        レポートの文字列
    """
    report = "# 因果探索結果レポート\n\n"
    
    # DirectLiNGAM結果
    if effects_df is not None and len(effects_df) > 0:
        report += "## DirectLiNGAM: 静的因果関係\n\n"
        report += "### 概要\n\n"
        report += f"- 分析された特徴量数: {len(effects_df)}\n"
        report += f"- 有意な因果効果を持つ特徴量数: {len(effects_df[effects_df['abs_effect'] > 0.1])}\n\n"
        
        report += "### KPIへの直接効果トップ10\n\n"
        report += "| 順位 | 特徴量 | 効果 | 絶対値 |\n"
        report += "|------|--------|------|--------|\n"
        
        for _, row in effects_df.head(10).iterrows():
            direction = "正" if row['effect'] > 0 else "負"
            report += f"| {int(row['rank'])} | {row['feature']} | {row['effect']:.4f} ({direction}) | {row['abs_effect']:.4f} |\n"
        
        report += "\n### 解釈\n\n"
        report += "#### 正の効果（KPIを増加させる特徴量）\n\n"
        positive_effects = effects_df[effects_df['effect'] > 0].head(5)
        if len(positive_effects) > 0:
            for _, row in positive_effects.iterrows():
                report += f"- **{row['feature']}**: 効果 = {row['effect']:.4f}\n"
        else:
            report += "- なし\n"
        
        report += "\n#### 負の効果（KPIを減少させる特徴量）\n\n"
        negative_effects = effects_df[effects_df['effect'] < 0].head(5)
        if len(negative_effects) > 0:
            for _, row in negative_effects.iterrows():
                report += f"- **{row['feature']}**: 効果 = {row['effect']:.4f}\n"
        else:
            report += "- なし\n"
        
        report += "\n---\n\n"
    
    # VARLiNGAM結果
    if var_effects_df is not None and len(var_effects_df) > 0:
        report += "## VARLiNGAM: 時間的因果関係\n\n"
        report += "### 概要\n\n"
        report += f"- 分析されたセンサー数: {var_effects_df['feature'].nunique()}\n"
        report += f"- 分析されたラグ数: {var_effects_df['lag'].nunique()}\n"
        report += f"- 検出された時間的因果効果数: {len(var_effects_df)}\n\n"
        
        # 各ラグでのトップ5効果
        report += "### 各ラグでのトップ5効果\n\n"
        
        for lag in sorted(var_effects_df['lag'].unique()):
            lag_df = var_effects_df[var_effects_df['lag'] == lag].head(5)
            report += f"#### Lag={lag}日 (過去{lag}日の影響)\n\n"
            report += "| 順位 | センサー | 効果 | 絶対値 |\n"
            report += "|------|----------|------|--------|\n"
            
            for idx, row in enumerate(lag_df.iterrows(), start=1):
                _, data = row
                direction = "正" if data['effect'] > 0 else "負"
                report += f"| {idx} | {data['feature']} | {data['effect']:.4f} ({direction}) | {data['abs_effect']:.4f} |\n"
            
            report += "\n"
        
        # 総合的な上位効果
        report += "### 総合上位効果（全ラグ合計）\n\n"
        report += "| 順位 | センサー | ラグ | 効果 | 絶対値 |\n"
        report += "|------|----------|------|------|--------|\n"
        
        for _, row in var_effects_df.head(10).iterrows():
            direction = "正" if row['effect'] > 0 else "負"
            report += f"| {int(row['rank'])} | {row['feature']} | {int(row['lag'])} | {row['effect']:.4f} ({direction}) | {row['abs_effect']:.4f} |\n"
        
        report += "\n### 解釈\n\n"
        report += "#### 短期効果（lag=1-2日）\n\n"
        short_term = var_effects_df[var_effects_df['lag'] <= 2].head(5)
        if len(short_term) > 0:
            for _, row in short_term.iterrows():
                report += f"- **{row['feature']}** (lag={int(row['lag'])}日): 効果 = {row['effect']:.4f}\n"
        else:
            report += "- なし\n"
        
        report += "\n#### 長期効果（lag=5日以上）\n\n"
        long_term = var_effects_df[var_effects_df['lag'] >= 5].head(5)
        if len(long_term) > 0:
            for _, row in long_term.iterrows():
                report += f"- **{row['feature']}** (lag={int(row['lag'])}日): 効果 = {row['effect']:.4f}\n"
        else:
            report += "- なし\n"
        
        report += "\n---\n\n"
    
    # 推奨事項
    report += "## 推奨事項\n\n"
    
    if effects_df is not None and len(effects_df) > 0:
        report += "### DirectLiNGAM結果に基づく推奨\n\n"
        report += "1. 上位の正の効果を持つ特徴量の変動をモニタリング\n"
        report += "2. 負の効果を持つ特徴量の改善策を検討\n"
        report += "3. 物理的な妥当性をドメイン専門家と確認\n\n"
    
    if var_effects_df is not None and len(var_effects_df) > 0:
        report += "### VARLiNGAM結果に基づく推奨\n\n"
        report += "1. 短期効果（lag=1-2日）のセンサーを重点的に監視\n"
        report += "2. 長期効果（lag=5日以上）のセンサーは予防保全の指標として活用\n"
        report += "3. 各ラグでの効果パターンから故障メカニズムを推定\n"
        report += "4. 時間遅延を考慮した早期警告システムの構築を検討\n"
    
    # 保存
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n[解釈レポート保存] {output_path}")
    
    return report


# ========================================================================
# v0.1.2: arXiv論文用の可視化関数
# ========================================================================

def plot_ui_distribution(
    heterogeneity_df: pd.DataFrame,
    groups: List[str] = ['top30', 'bottom30'],
    output_path: str = None,
    dpi: int = 300
):
    """
    u_i分布のヒストグラム（グループ別）
    
    Parameters:
        heterogeneity_df: pump_heterogeneity.csv データ（u_group列を含む）
        groups: 表示するグループ
        output_path: 保存先パス
        dpi: 解像度
    """
    from scipy import stats
    
    print("\n[u_i分布の可視化]")
    
    n_groups = len(groups)
    fig, axes = plt.subplots(1, n_groups, figsize=(6*n_groups, 5), dpi=dpi)
    
    if n_groups == 1:
        axes = [axes]
    
    for ax, group in zip(axes, groups):
        # グループデータ
        group_data = heterogeneity_df[heterogeneity_df['u_group'] == group]
        u_values = group_data['u_mean'].values
        
        print(f"  {group}: N={len(u_values)}, mean={u_values.mean():.4f}, std={u_values.std():.4f}")
        
        # ヒストグラム
        n, bins, patches = ax.hist(
            u_values,
            bins=20,
            density=True,
            alpha=0.7,
            color='steelblue',
            edgecolor='black'
        )
        
        # ガウスカーブのオーバーレイ
        mu, sigma = u_values.mean(), u_values.std()
        x = np.linspace(u_values.min(), u_values.max(), 100)
        y = stats.norm.pdf(x, mu, sigma)
        ax.plot(x, y, 'r-', linewidth=2, label=f'N({mu:.2f}, {sigma:.2f}²)')
        
        # 統計情報の注釈
        textstr = f'N = {len(u_values)}\n' \
                  f'μ = {mu:.3f}\n' \
                  f'σ = {sigma:.3f}\n' \
                  f'95% CI: [{group_data["u_lower"].mean():.3f}, {group_data["u_upper"].mean():.3f}]'
        
        ax.text(
            0.05, 0.95, textstr,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        ax.set_xlabel('Random Effect (u_i)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Density', fontsize=12, fontweight='bold')
        ax.set_title(f'Group: {group.upper()}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"  [保存] {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_group_comparison_graphs(
    results_dict: Dict[str, Dict],
    feature_names: List[str],
    kpi_name: str = 'u_i_target',
    threshold: float = 0.1,
    top_k: int = 10,
    output_path: str = None,
    dpi: int = 300
):
    """
    グループ別因果グラフ（横並び2パネル）
    
    Parameters:
        results_dict: {group_name: causal_results}
        feature_names: 特徴量名のリスト
        kpi_name: KPI名
        threshold: エッジ表示の閾値
        top_k: 表示する上位特徴量数
        output_path: 保存先パス
        dpi: 解像度
    """
    print("\n[グループ別因果グラフの可視化]")
    
    groups = list(results_dict.keys())
    n_groups = len(groups)
    
    fig, axes = plt.subplots(1, n_groups, figsize=(8*n_groups, 8), dpi=dpi)
    
    if n_groups == 1:
        axes = [axes]
    
    for ax, group in zip(axes, groups):
        print(f"  {group}: グラフ生成中...")
        
        results = results_dict[group]
        adj_matrix = results['adjacency_matrix']
        
        # KPIインデックス
        var_names = feature_names + [kpi_name]
        kpi_idx = len(feature_names)
        
        # KPIへの効果でトップK特徴量を選択
        effects_to_kpi = np.abs(adj_matrix[:, kpi_idx])
        top_indices = np.argsort(effects_to_kpi)[::-1][:top_k]
        
        # サブグラフを構築
        selected_indices = list(top_indices) + [kpi_idx]
        sub_adj = adj_matrix[np.ix_(selected_indices, selected_indices)]
        sub_names = [var_names[i] for i in selected_indices]
        
        # NetworkXグラフ
        G = nx.DiGraph()
        
        for i, name_i in enumerate(sub_names):
            G.add_node(name_i)
        
        edge_weights = []
        for i, name_i in enumerate(sub_names):
            for j, name_j in enumerate(sub_names):
                if i != j and abs(sub_adj[i, j]) > threshold:
                    G.add_edge(name_i, name_j, weight=sub_adj[i, j])
                    edge_weights.append(abs(sub_adj[i, j]))
        
        # レイアウト（ノード間距離をさらに広げる）
        # データ点数が少ない場合はkを大きくして分散させる
        k_param = 3.5 if 'bottom' in group.lower() or 'middle' in group.lower() else 2.5
        pos = nx.spring_layout(G, k=k_param, iterations=200, seed=42)
        
        # ノード色（KPIは赤、他は青～緑グラデーション）
        # グラフGのノードリストに基づいて色を設定
        node_colors = []
        for node in G.nodes():
            if node == kpi_name:
                node_colors.append('red')
            else:
                node_colors.append('steelblue')
        
        # エッジの太さと色
        edges = G.edges()
        edge_widths = [abs(G[u][v]['weight']) * 3 for u, v in edges]  # 太さを抑える（5→3）
        edge_colors = ['green' if G[u][v]['weight'] > 0 else 'red' for u, v in edges]
        
        # 描画（ノードサイズを縮小）
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=400,  # 800→400に縮小
            alpha=0.9,
            ax=ax
        )
        
        # ラベル位置を調整（ノードの上に配置）
        label_pos = {k: (v[0], v[1] + 0.08) for k, v in pos.items()}
        nx.draw_networkx_labels(
            G, label_pos,
            font_size=7,  # 9→7に縮小
            font_weight='bold',
            ax=ax
        )
        
        # エッジ描画（透明度を下げる）
        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.4,  # 0.6→0.4に下げる
            arrows=True,
            arrowsize=15,  # 20→15に縮小
            ax=ax,
            connectionstyle='arc3,rad=0.1'  # エッジを曲線にして重なりを回避
        )
        
        ax.set_title(f'Causal Graph: {group.upper()}', fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"  [保存] {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_differential_effects(
    effects_dict: Dict[str, pd.DataFrame],
    feature_names: List[str],
    groups: List[str] = ['top30', 'bottom30'],
    output_path: str = None,
    dpi: int = 300
):
    """
    因果効果の比較図（折れ線グラフ）
    
    Parameters:
        effects_dict: {group_name: effects_df}
        feature_names: 特徴量名のリスト
        groups: 比較するグループ
        output_path: 保存先パス
        dpi: 解像度
    """
    print("\n[因果効果の比較図]")
    
    fig, ax = plt.subplots(figsize=(14, 6), dpi=dpi)
    
    # 各グループの効果を取得
    colors = ['steelblue', 'orange', 'green', 'red']
    
    for i, group in enumerate(groups):
        effects_df = effects_dict[group]
        
        # 特徴量順に効果を並べる
        effects = []
        for feat in feature_names:
            feat_row = effects_df[effects_df['feature'] == feat]
            if len(feat_row) > 0:
                effects.append(feat_row.iloc[0]['effect'])
            else:
                effects.append(0.0)
        
        ax.plot(
            range(len(feature_names)),
            effects,
            marker='o',
            linewidth=2,
            label=group.upper(),
            color=colors[i % len(colors)]
        )
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Causal Effect to u_i', fontsize=12, fontweight='bold')
    ax.set_title('Differential Causal Effects: Group Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=90, ha='right', fontsize=8)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"  [保存] {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_divergent_effects_heatmap(
    differential_df: pd.DataFrame,
    groups: List[str] = ['top30', 'bottom30'],
    output_path: str = None,
    dpi: int = 300
):
    """
    ダイバージェント・エフェクト・ヒートマップ
    
    Parameters:
        differential_df: differential_effects.csv データ
        groups: グループ名
        output_path: 保存先パス
        dpi: 解像度
    """
    print("\n[ダイバージェント・エフェクト・ヒートマップ]")
    
    # データを行列形式に変換
    features = differential_df['feature'].values
    n_features = len(features)
    
    # 各グループの効果を取得
    effect_matrix = np.zeros((n_features, len(groups)))
    
    for i, group in enumerate(groups):
        col_name = f'effect_{group}'
        if col_name in differential_df.columns:
            effect_matrix[:, i] = differential_df[col_name].values
    
    # 差分を計算（2グループの場合）
    if len(groups) == 2:
        diff_values = effect_matrix[:, 0] - effect_matrix[:, 1]
        
        # 差分でソート
        sorted_indices = np.argsort(np.abs(diff_values))[::-1]
        features_sorted = features[sorted_indices]
        diff_values_sorted = diff_values[sorted_indices]
        
        # 上位20個を表示
        n_display = min(20, len(features_sorted))
        features_display = features_sorted[:n_display]
        diff_display = diff_values_sorted[:n_display].reshape(-1, 1)
        
        # ヒートマップ
        fig, ax = plt.subplots(figsize=(8, max(6, n_display * 0.3)), dpi=dpi)
        
        sns.heatmap(
            diff_display,
            yticklabels=features_display,
            xticklabels=['Effect Difference\n(Top30 - Bottom30)'],
            annot=True,
            fmt='.3f',
            cmap='RdBu_r',
            center=0,
            cbar_kws={'label': 'Effect Difference'},
            ax=ax
        )
        
        ax.set_title('Divergent Effects: Top30 vs Bottom30', fontsize=14, fontweight='bold')
        ax.set_ylabel('Features', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"  [保存] {output_path}")
        else:
            plt.show()
        
        plt.close()


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
