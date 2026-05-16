"""
メインパイプライン

ポンプ設備LiNGAM因果探索MVPの統合パイプライン
"""

import argparse
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import warnings

# 自作モジュールのインポート
from src.data_preprocessing import load_and_preprocess_data
from src.feature_engineering import extract_features_from_time_series, FeatureEngineer
from src.causal_discovery import run_causal_discovery
from src.visualization import CausalVisualizer, generate_interpretation_report
from src.validation import run_bootstrap_validation

warnings.filterwarnings('ignore')


class PumpCausalPipeline:
    """
    ポンプ設備因果探索パイプラインクラス
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: 設定ファイルのパス
        """
        # 設定の読み込み
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.base_dir = Path(__file__).parent
        self.output_dir = self.base_dir / self.config['data']['output_dir']
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # データ保持用
        self.df_preprocessed = None
        self.df_features = None
        self.df_scaled = None
        self.causal_results = None
        self.effects_df = None
        
        print("=" * 80)
        print("ポンプ設備LiNGAM因果探索MVP")
        print("=" * 80)
    
    def run_preprocessing(self):
        """
        ステップ1-2: データ前処理
        """
        print("\n" + "=" * 80)
        print("ステップ1-2: データ前処理")
        print("=" * 80)
        
        time_series_path = self.base_dir / self.config['data']['labeled_time_series']
        selected_equipment_path = self.base_dir / self.config['data']['selected_equipment']
        
        self.df_preprocessed = load_and_preprocess_data(
            time_series_path=str(time_series_path),
            selected_equipment_path=str(selected_equipment_path),
            outlier_threshold=self.config['preprocessing']['outlier_threshold'],
            interpolation_method=self.config['preprocessing']['interpolation_method'],
            max_missing_ratio=self.config['preprocessing']['max_missing_ratio'],
            verbose=True
        )
        
        # 保存
        output_path = self.output_dir / "preprocessed_data.csv"
        self.df_preprocessed.to_csv(output_path, index=False)
        print(f"\n[保存] {output_path}")
    
    def run_feature_engineering(self):
        """
        ステップ3-5: 特徴量エンジニアリング
        """
        print("\n" + "=" * 80)
        print("ステップ3-5: 特徴量エンジニアリング")
        print("=" * 80)
        
        if self.df_preprocessed is None:
            raise ValueError("前処理が実行されていません。run_preprocessing()を先に実行してください。")
        
        # n_jobsをconfigから取得（デフォルトは-1で全コア使用）
        n_jobs = self.config.get('misc', {}).get('n_jobs', -1)
        
        self.df_features = extract_features_from_time_series(
            time_series_df=self.df_preprocessed,
            lookback_days=self.config['feature_engineering']['lookback_days'],
            min_data_points=self.config['feature_engineering']['min_data_points'],
            n_jobs=n_jobs,
            verbose=True
        )
        
        # 保存
        output_path = self.output_dir / "features_90d.csv"
        self.df_features.to_csv(output_path, index=False)
        print(f"\n[保存] {output_path}")
    
    def run_scaling_and_correlation_check(self):
        """
        ステップ6: スケーリングと相関チェック
        """
        print("\n" + "=" * 80)
        print("ステップ6: スケーリングと相関チェック")
        print("=" * 80)
        
        if self.df_features is None:
            raise ValueError("特徴量生成が実行されていません。")
        
        # 特徴量列を取得
        feature_names = FeatureEngineer.get_feature_names()
        kpi_col = self.config['kpi']['primary_label']
        
        # 必要な列を選択
        required_cols = feature_names + [kpi_col]
        df_for_scaling = self.df_features[required_cols].copy()
        
        print(f"\n[スケーリング前]")
        print(f"  レコード数: {len(df_for_scaling)}")
        print(f"  特徴量数: {len(feature_names)}")
        print(f"  欠損率: {df_for_scaling.isnull().sum().sum() / (len(df_for_scaling) * len(required_cols)) * 100:.2f}%")
        
        # NaN除去
        df_for_scaling = df_for_scaling.dropna()
        print(f"  NaN除去後: {len(df_for_scaling)} レコード")
        
        # 標準化
        scaler = StandardScaler()
        X_features = df_for_scaling[feature_names].values
        X_scaled = scaler.fit_transform(X_features)
        
        # データフレーム化
        df_scaled = pd.DataFrame(X_scaled, columns=feature_names, index=df_for_scaling.index)
        df_scaled[kpi_col] = df_for_scaling[kpi_col].values
        
        print(f"\n[標準化完了]")
        print(f"  平均（サンプル）: {X_scaled[:, 0].mean():.6f}")
        print(f"  標準偏差（サンプル）: {X_scaled[:, 0].std():.6f}")
        
        # 相関チェック
        corr_threshold = self.config['scaling']['correlation_threshold']
        corr_matrix = df_scaled[feature_names].corr()
        
        # 高相関ペアを検出
        high_corr_pairs = []
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                    high_corr_pairs.append((
                        feature_names[i],
                        feature_names[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        print(f"\n[相関チェック]")
        print(f"  相関閾値: {corr_threshold}")
        print(f"  高相関ペア数: {len(high_corr_pairs)}")
        
        if len(high_corr_pairs) > 0:
            print("\n  高相関ペア（トップ5）:")
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:5]:
                print(f"    {feat1} <-> {feat2}: {corr:.3f}")
        
        # 分散チェック
        variance_threshold = self.config['scaling']['variance_threshold']
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(df_scaled[feature_names])
        
        low_variance_features = [
            name for name, var in zip(feature_names, selector.variances_)
            if var < variance_threshold
        ]
        
        print(f"\n[分散チェック]")
        print(f"  分散閾値: {variance_threshold}")
        print(f"  低分散特徴量数: {len(low_variance_features)}")
        
        if len(low_variance_features) > 0:
            print(f"  低分散特徴量: {low_variance_features}")
        
        self.df_scaled = df_scaled
        
        # 保存
        output_path = self.output_dir / "scaled_features.csv"
        df_scaled.to_csv(output_path, index=False)
        print(f"\n[保存] {output_path}")
    
    def run_causal_discovery(self):
        """
        ステップ7-8: LiNGAM因果探索
        """
        print("\n" + "=" * 80)
        print("ステップ7-8: LiNGAM因果探索")
        print("=" * 80)
        
        if self.df_scaled is None:
            raise ValueError("スケーリングが実行されていません。")
        
        feature_names = FeatureEngineer.get_feature_names()
        kpi_col = self.config['kpi']['primary_label']
        
        # 因果探索の実行
        self.causal_results, self.effects_df = run_causal_discovery(
            features_df=self.df_scaled,
            feature_cols=feature_names,
            kpi_col=kpi_col,
            algorithm=self.config['lingam']['algorithm'],
            random_state=self.config['lingam']['random_state'],
            output_path=str(self.output_dir / "causal_results.pkl"),
            verbose=True
        )
        
        # 効果をCSV保存
        output_path = self.output_dir / "kpi_effects.csv"
        self.effects_df.to_csv(output_path, index=False)
        print(f"\n[保存] {output_path}")
    
    def run_visualization(self):
        """
        ステップ9-10: 可視化と解釈
        """
        print("\n" + "=" * 80)
        print("ステップ9-10: 可視化と解釈")
        print("=" * 80)
        
        if self.causal_results is None or self.effects_df is None:
            raise ValueError("因果探索が実行されていません。")
        
        kpi_col = self.config['kpi']['primary_label']
        
        # 可視化オブジェクトの初期化
        visualizer = CausalVisualizer(
            figure_size=tuple(self.config['visualization']['figure_size']),
            dpi=self.config['visualization']['dpi'],
            edge_width_scale=self.config['visualization']['edge_width_scale'],
            node_size_scale=self.config['visualization']['node_size_scale'],
            layout=self.config['visualization']['layout']
        )
        
        # 因果グラフの可視化
        visualizer.visualize_causal_graph(
            adjacency_matrix=self.causal_results['adjacency_matrix'],
            feature_names=self.causal_results['feature_names'],
            kpi_name=kpi_col,
            threshold=0.1,
            top_k=15,
            output_path=str(self.output_dir / "causal_graph.png")
        )
        
        # 効果ヒートマップの可視化
        visualizer.visualize_effect_heatmap(
            adjacency_matrix=self.causal_results['adjacency_matrix'],
            feature_names=self.causal_results['feature_names'],
            kpi_name=kpi_col,
            top_k=20,
            output_path=str(self.output_dir / "effect_heatmap.png")
        )
        
        # 解釈レポートの生成
        report = generate_interpretation_report(
            effects_df=self.effects_df,
            output_path=str(self.output_dir / "interpretation_report.md")
        )
    
    def run_bootstrap_validation(self):
        """
        ステップ11: ブートストラップ検証
        """
        print("\n" + "=" * 80)
        print("ステップ11: ブートストラップ検証")
        print("=" * 80)
        
        if self.df_scaled is None or self.causal_results is None:
            raise ValueError("因果探索が実行されていません。")
        
        feature_names = FeatureEngineer.get_feature_names()
        kpi_col = self.config['kpi']['primary_label']
        
        # データ行列を準備
        all_cols = feature_names + [kpi_col]
        X = self.df_scaled[all_cols].dropna().values
        variable_names = all_cols
        
        # ブートストラップ検証
        stable_edges_df, edge_frequencies = run_bootstrap_validation(
            X=X,
            variable_names=variable_names,
            kpi_name=kpi_col,
            algorithm=self.config['lingam']['algorithm'],
            n_sampling=self.config['bootstrap']['n_sampling'],
            sample_ratio=self.config['bootstrap']['sample_ratio'],
            stability_threshold=self.config['bootstrap']['stability_threshold'],
            random_state=self.config['lingam']['random_state'],
            n_jobs=self.config['misc']['n_jobs'],
            output_dir=str(self.output_dir),
            verbose=True
        )
    
    def run_all(self):
        """
        全パイプラインを実行
        """
        self.run_preprocessing()
        self.run_feature_engineering()
        self.run_scaling_and_correlation_check()
        self.run_causal_discovery()
        self.run_visualization()
        self.run_bootstrap_validation()
        
        print("\n" + "=" * 80)
        print("全パイプライン完了")
        print("=" * 80)
        print(f"\n出力ディレクトリ: {self.output_dir}")
        print("\n生成されたファイル:")
        for file in sorted(self.output_dir.glob("*")):
            print(f"  - {file.name}")


def main():
    """
    メイン関数
    """
    parser = argparse.ArgumentParser(
        description="ポンプ設備LiNGAM因果探索MVPパイプライン"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='設定ファイルのパス（デフォルト: config.yaml）'
    )
    parser.add_argument(
        '--step',
        type=str,
        choices=['preprocessing', 'feature_engineering', 'scaling', 
                 'causal_discovery', 'visualization', 'validation', 'all'],
        default='all',
        help='実行するステップ（デフォルト: all）'
    )
    
    args = parser.parse_args()
    
    # パイプラインの初期化
    pipeline = PumpCausalPipeline(config_path=args.config)
    
    # ステップの実行
    if args.step == 'all':
        pipeline.run_all()
    elif args.step == 'preprocessing':
        pipeline.run_preprocessing()
    elif args.step == 'feature_engineering':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
    elif args.step == 'scaling':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        pipeline.run_scaling_and_correlation_check()
    elif args.step == 'causal_discovery':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        pipeline.run_scaling_and_correlation_check()
        pipeline.run_causal_discovery()
    elif args.step == 'visualization':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        pipeline.run_scaling_and_correlation_check()
        pipeline.run_causal_discovery()
        pipeline.run_visualization()
    elif args.step == 'validation':
        pipeline.run_preprocessing()
        pipeline.run_feature_engineering()
        pipeline.run_scaling_and_correlation_check()
        pipeline.run_causal_discovery()
        pipeline.run_bootstrap_validation()


if __name__ == "__main__":
    main()
