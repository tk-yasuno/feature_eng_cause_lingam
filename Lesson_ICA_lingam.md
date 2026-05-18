# ICALiNGAM検討結果：ポンプ設備データへの適用性

## 結論

**ICALiNGAMはポンプ設備データには推奨しない。DirectLiNGAMを使用すべき。**

## 1. アルゴリズム比較

### DirectLiNGAM（推奨） ✅

- **手法**: 線形回帰ベースの直接法
- **計算量**: O(n × d³) - サンプル数nに線形
- **前提条件**: 
  - 線形性
  - 非巡回性（DAG）
  - 加法的誤差
- **特徴**: 
  - 大規模データに強い
  - 高速・安定
  - ガウス分布でも機能

### ICALiNGAM（非推奨） ❌

- **手法**: 独立成分分析（ICA）ベースの反復法
- **計算量**: O(反復回数 × n × d³) - 10倍以上遅い
- **前提条件**: 
  - 線形性
  - 非巡回性（DAG）
  - 加法的誤差
  - **非ガウス性（重要）**
- **特徴**: 
  - 小規模データ向け
  - 収束不安定
  - 多重共線性に弱い

## 2. ポンプ設備データの特性分析

### データ規模
```
サンプル数: 92,861レコード
変数数: 24（特徴量23 + KPI 1）
設備数: 64
センサー数: 229
期間: 650日（2024/3/6 - 2025/12/17）
```

### 特徴量の性質
```yaml
特徴量タイプ:
  - 統計量: mean, median, std, min, max, q25, q75
  - 傾向: trend_slope, trend_intercept
  - 変動: diff_abs_mean, rolling_std_7d_mean
  - 形状: skewness, kurtosis
```

**問題点**: これらの統計的特徴量は**中心極限定理により正規分布に近づく**

### 多重共線性
```
高相関ペア数: 45組（相関係数 > 0.95）
主な高相関:
  - mean <-> median: 1.000
  - min <-> trend_intercept: 1.000
  - mean <-> q75: 1.000
  - q25 <-> trend_intercept: 1.000
  - median <-> q75: 1.000
```

**問題点**: ICALiNGAMは多重共線性で不安定化

## 3. DirectLiNGAMが最適な理由

### ✅ 実証済みの性能

現在の実行結果（DirectLiNGAM）:
```
実行時間: 約15分（特徴量生成含む）
検出エッジ数: 141
KPI主要効果:
  - skewness: -0.037196
  - kurtosis: -0.026964
  - min: -0.009748
  - diff_abs_mean: -0.009353
```

### ✅ 大規模データ処理能力

- 92,861サンプルを安定処理
- 16コア並列化で高速実行
- メモリ効率が良い

### ✅ 統計的特徴量に適合

- ガウス性の高い特徴量でも正確
- 線形回帰ベースで解釈性が高い
- 多重共線性にロバスト（ある程度）

### ✅ 産業界での実績

- 製造業での因果探索標準手法
- 論文・実装例が豊富
- 安定した収束性

## 4. ICALiNGAMの想定利用ケース

以下の条件を**すべて満たす場合のみ**検討価値あり：

### 適用条件
1. **サンプル数**: 1,000～10,000程度（小～中規模）
2. **非ガウス性**: 
   - 歪度（skewness） > 1 または < -1
   - 尖度（kurtosis） > 3
   - 多峰性分布
3. **低相関**: 変数間相関 < 0.7
4. **生データ**: センサー生値を直接使用（統計量でない）
5. **計算時間許容**: DirectLiNGAMの10倍以上の時間

### 例：適用可能なケース
- **振動センサー生波形**: 強い非ガウス性
- **音響信号**: パルス性・スパース性
- **画像ピクセル値**: 独立成分が明確
- **脳波・心電図**: 非ガウス性の生理信号

### ポンプ設備では不適切な理由
- ✗ 統計的特徴量（正規分布に近い）
- ✗ 大規模データ（92,861サンプル）
- ✗ 高い多重共線性（45ペア > 0.95）
- ✗ 計算コスト過大（推定10時間以上）

## 5. 実験的検証の推奨手順

もしICALiNGAMを試す場合の手順：

### ステップ1: 小規模サブセット作成
```python
# 1,000サンプルをランダム抽出
df_subset = df_scaled.sample(n=1000, random_state=42)
```

### ステップ2: 非ガウス性チェック
```python
from scipy.stats import skew, kurtosis
for col in feature_names:
    sk = skew(df_subset[col])
    ku = kurtosis(df_subset[col])
    print(f"{col}: skew={sk:.2f}, kurtosis={ku:.2f}")
    
# 判定基準:
# |skewness| > 1 かつ kurtosis > 3 なら非ガウス性あり
```

### ステップ3: ICALiNGAM実行（小規模データ）
```python
from lingam import ICALiNGAM
model = ICALiNGAM(random_state=42, max_iter=1000)
model.fit(df_subset[all_cols].values)
```

### ステップ4: 結果比較
- DirectLiNGAMとICALiNGAMの因果グラフを比較
- エッジ一致率を計算
- KPI効果の違いを検証

### 判定基準
- **ICALiNGAMが優れている**: エッジ一致率 < 70% かつ予測性能向上
- **DirectLiNGAMで十分**: エッジ一致率 > 90% または性能同等

## 6. 推奨事項

### 現在の設定を維持
```yaml
# config.yaml
lingam:
  algorithm: "DirectLiNGAM"  # 変更しない
  random_state: 42
  max_iter: 1000
```

### 代替手法の検討

より高度な因果探索が必要な場合：

1. **VARLiNGAM** ✅ 実装済み
   - 時間遅延因果を考慮
   - 別ワークスペースで実装済み: `time_series_varlingam`

2. **Bootstrap DirectLiNGAM** ✅ 実装済み
   - 因果エッジの安定性評価
   - 100回サンプリングで検証中

3. **時間窓の調整**
   - lookback_days: 30/60/90日で比較
   - rolling_windows: 異なる窓サイズで感度分析

4. **非線形手法（将来検討）**
   - Nonlinear ICA
   - Neural Network-based Causal Discovery
   - ただし解釈性が低下

## 7. 学習ポイント

### 手法選択の原則

1. **データ規模で選ぶ**
   - 大規模（>10万）: DirectLiNGAM
   - 中規模（1万～10万）: DirectLiNGAM
   - 小規模（<1万）: DirectLiNGAM or ICALiNGAM

2. **データ性質で選ぶ**
   - ガウス性高い: DirectLiNGAM
   - 非ガウス性高い: ICALiNGAM（小規模のみ）
   - 時系列依存: VARLiNGAM

3. **計算リソースで選ぶ**
   - 時間制約あり: DirectLiNGAM
   - 時間十分: DirectLiNGAM + Bootstrap検証

### DirectLiNGAMが常に優先される理由

- 最も安定・高速
- ガウス・非ガウスの両方で機能
- 大規模データに対応
- 産業界標準

**例外**: 明確な非ガウス性があり、小規模データで、解釈の向上が期待できる場合のみICALiNGAMを試す価値あり

## 8. まとめ

| 項目 | DirectLiNGAM | ICALiNGAM |
|------|--------------|-----------|
| **ポンプ設備適用** | ✅ 最適 | ❌ 不適切 |
| **実行時間** | 15分 | 推定10時間以上 |
| **安定性** | ✅ 高い | ❌ 低い |
| **多重共線性** | ○ やや耐性あり | ❌ 弱い |
| **非ガウス性要求** | ✗ 不要 | ✅ 必須 |
| **実績** | ✅ 豊富 | △ 限定的 |
| **推奨度** | ⭐⭐⭐⭐⭐ | ⭐ |

## 9. 参考文献

1. Shimizu et al. (2006). "A Linear Non-Gaussian Acyclic Model for Causal Discovery"
2. Shimizu et al. (2011). "DirectLiNGAM: A Direct Method for Learning a Linear Non-Gaussian Structural Equation Model"
3. LiNGAM公式ドキュメント: https://github.com/cdt15/lingam

---

**作成日**: 2026年5月17日  
**結論**: ポンプ設備因果探索にはDirectLiNGAMを使用し、ICALiNGAMは使用しない。
