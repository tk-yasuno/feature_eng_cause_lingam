# ポンプ設備LiNGAM因果探索MVP v0.1.0

## 概要

ポンプ設備の特徴量エンジニアリングと因果探索を行うMVPシステム。90日間の測定データから22の特徴量を生成し、LiNGAM (Linear Non-Gaussian Acyclic Model) を用いて異常予測への因果関係を発見します。

64台のポンプ設備の650日間（2024/3/6〜2025/12/17）の測定データを分析し、90日先の異常予測 (label_future_90d) に対する因果効果を定量化しました。

## プロジェクト構成

```
feature_eng_cause_lingam/
├── data/
│   ├── processed/          # 処理済みデータ
│   │   ├── labeled_time_series.csv
│   │   └── selected_64_equipment.json
│   └── data_source/        # 生データ
├── src/                    # ソースコード
│   ├── data_preprocessing.py      # データ前処理
│   ├── feature_engineering.py     # 特徴量生成
│   ├── causal_discovery.py        # LiNGAM因果探索
│   ├── visualization.py           # 可視化
│   └── validation.py              # ブートストラップ検証
├── output/                 # 出力ファイル
├── config.yaml            # 設定ファイル
├── requirements.txt       # 依存パッケージ
└── main.py               # メインパイプライン

```

## セットアップ

```bash
# 依存パッケージのインストール
pip install -r requirements.txt
```

## 使用方法

```bash
# パイプライン全体を実行
python main.py

# 個別ステップの実行も可能
python main.py --step preprocessing
python main.py --step feature_engineering
python main.py --step causal_discovery
python main.py --step visualization
python main.py --step validation
```

## 出力ファイル

- `output/features_90d.csv` - 生成された特徴量行列
- `output/scaled_features.csv` - スケーリング後の特徴量
- `output/causal_results.pkl` - LiNGAM推定結果
- `output/causal_graph.png` - 因果グラフ可視化
- `output/top_features_report.md` - 重要特徴量レポート
- `output/bootstrap_stability.png` - ブートストラップ安定性ヒートマップ

## 技術仕様

- **対象設備**: 64台のポンプ設備
- **データ期間**: 2024/3/6 ~ 2025/12/17（約650日間、92,861レコード）
- **チェック項目数**: 229項目
- **時間窓**: 90日ローリング窓
- **特徴量数**: 22個（統計的11個、トレンド5個、変動性6個）
- **KPI**: label_future_90d（90日先の異常予測）
- **因果探索**: DirectLiNGAM
- **検証**: ブートストラップ（100回反復、80%サブサンプリング）
- **並列処理**: 10コア並列実行（joblib.Parallel）

## 実行結果

### 因果探索結果

DirectLiNGAMによる因果探索の結果、**139個の因果関係**（非ゼロエッジ）を発見し、ブートストラップ検証により**109個の安定エッジ**（頻度≥70%）を特定しました。

#### KPIへの主要な因果効果

| 順位 | 特徴量 | 因果効果 | 解釈 |
|------|--------|----------|------|
| 1 | skewness | -0.0372 | 分布の歪度が高いほど異常予測が低下 |
| 2 | kurtosis | -0.0270 | 尖度が高いほど異常予測が低下 |
| 3 | min | -0.0097 | 最小値が低いほど異常予測が低下 |
| 4 | diff_abs_mean | -0.0094 | 絶対差分平均が大きいほど異常予測が低下 |
| 5 | std | -0.0023 | 標準偏差が大きいほど異常予測が低下 |
| 6 | rolling_std_7d_mean | +0.0017 | 短期変動が大きいほど異常予測が上昇 |

![因果グラフ](output/causal_graph.png)
*図1: KPIへの上位15特徴量の因果グラフ（エッジ幅は効果の絶対値に比例）*

![効果ヒートマップ](output/effect_heatmap.png)
*図2: 特徴量間およびKPIへの因果効果ヒートマップ*

### ブートストラップ検証結果

100回のブートストラップサンプリング（80%サブサンプリング）により、因果構造の安定性を評価しました。

**安定エッジトップ10**（頻度≥70%）:

| From | To | 頻度 | 平均効果 |
|------|-----|------|----------|
| rolling_std_30d_mean | trend_slope_90d | 0.97 | 1.86 |
| median | mean | 1.00 | 1.75 |
| min | std | 1.00 | 1.07 |
| min | rolling_std_7d_mean | 1.00 | -0.95 |
| trend_slope_90d | recent_vs_past_diff | 1.00 | 0.93 |

![ブートストラップ安定性ヒートマップ](output/bootstrap_stability_heatmap.png)
*図3: ブートストラップ検証による因果関係の安定性（色の濃さは出現頻度）*

![KPI安定性](output/bootstrap_kpi_stability.png)
*図4: KPIへの因果効果の安定性（エラーバーはブートストラップ標準偏差）*

## 考察

### 1. 分布形状の異常性が重要な予測因子

skewnessとkurtosisが最も強い負の因果効果を持つことから、**データ分布の正規性からの逸脱が異常の予兆**である可能性が示唆されます。尖度や歪度が高い場合、設備が通常の動作範囲から外れており、これが将来の異常につながると解釈できます。

### 2. 最小値の低下が異常予測に影響

minの負の効果は、測定値の最小値が低下（例：圧力、流量の異常低下）することが、90日先の異常予測に寄与することを示しています。これは**設備性能の劣化パターン**を捉えている可能性があります。

### 3. 短期変動と長期トレンドの相反する効果

rolling_std_7d_meanは正の効果を持つ一方、stdは負の効果を持ちます。これは、**短期的な変動増加が早期警告信号**として機能する一方、長期的な高変動は異常予測との関連が異なることを示唆します。

### 4. 高い因果構造安定性

ブートストラップ検証により、109個の安定エッジ（頻度≥70%）が特定されました。特に、median→mean、min→stdなどの基本統計量間の関係は100%の頻度で出現しており、**データ駆動で発見された因果構造の頑健性**が確認されました。

### 5. 並列処理による実用性

10コア並列処理により、特徴量エンジニアリング（229グループ）が1.2分、ブートストラップ検証（100回）が28.9分で完了しました。これにより、**実運用環境での定期的な再学習が実現可能**です。

## 今後の展望

1. **異なるKPI期間の比較**: 30日、60日先の異常予測との因果関係を比較し、予測期間による特徴量重要度の変化を分析
2. **物理的解釈の深化**: ドメイン専門家とのレビューにより、発見された因果関係の物理的妥当性を検証
3. **因果効果の活用**: 特定された因果特徴量を用いた異常予測モデルの構築
4. **リアルタイム監視**: 上位因果特徴量のしきい値設定による早期警告システムの開発

## 特徴量リスト

### 統計的特徴（11個）
- mean, std, min, max, median
- q25, q75, iqr
- skewness, kurtosis
- cv_90d（変動係数）

### トレンド特徴（5個）
- trend_slope_90d（90日トレンド傾き）
- trend_intercept（切片）
- recent_vs_past_ratio（直近/過去比）
- recent_vs_past_diff（直近-過去差）
- recent_change_rate（変化率）

### 変動性特徴（6個）
- diff_mean（差分平均）
- diff_abs_mean（絶対差分平均）
- rolling_std_7d/14d/30d_mean（ローリング標準偏差）
- max_drawdown（最大ドローダウン）
- mean_drawdown（平均ドローダウン）

## ライセンス

Apache License 2.0 - 詳細は [LICENSE](LICENSE) ファイルを参照してください。

## バージョン履歴

### v0.1.0 (2026-05-16)
- 初回リリース
- DirectLiNGAMによる因果探索実装
- 90日KPIに対する22特徴量の因果効果分析
- ブートストラップ検証による安定性評価
- 10コア並列処理最適化
