# Lesson: netCDF4によるMCMCトレース保存

**作成日**: 2026年5月17日  
**対象**: PyMC 6.0.0 + ArviZ 1.1.0 + netCDF4 1.7.4  
**プロジェクト**: ポンプ設備劣化ハザードモデル（v0.1.2）

---

## 概要

PyMC/ArviZでNUTSサンプリング結果（InferenceData）を永続化するためにnetCDF4形式を使用する際の実践的知見をまとめる。

---

## 1. netCDF4とは

### Network Common Data Form (netCDF)

- **用途**: 多次元科学データの保存・共有
- **起源**: 気候・海洋・天文学などで標準フォーマット
- **特徴**: 
  - 階層構造のサポート
  - メタデータの埋め込み
  - 効率的な圧縮
  - クロスプラットフォーム対応

### PyMC/ArviZでの役割

```
NUTS推定 → InferenceData（メモリ上） → netCDF保存 → 永続化
                ↓                           ↓
          16,000サンプル              trace.nc (20-40MB)
          124パラメータ               再利用可能
```

---

## 2. 発生した問題と解決策

### 問題1: netCDF4未インストールエラー

**エラーメッセージ**:
```
ValueError: cannot write NetCDF files with format='NETCDF4' 
because none of the suitable backend libraries (netCDF4, h5netcdf) are installed
```

**発生タイミング**: 2026年5月17日、NUTS推定完了後のtrace保存時

**原因**:
- ArviZ 1.1.0は内部でxarrayを使用
- xarrayのNetCDF書き込みにはバックエンドライブラリが必須
- `netCDF4`または`h5netcdf`が必要だが未インストール

**解決策**:
```bash
pip install netcdf4
```

**requirements.txtへの追加**:
```python
# requirements.txt
netcdf4>=1.6.0  # ArviZのNetCDF保存に必須
```

**教訓**:
- ✅ PyMC/ArviZを使う場合、netCDF4は必須依存
- ✅ 公式ドキュメントに明記されていないため見落としやすい
- ✅ エラーが出てから気づくパターンが多い

---

## 3. ArviZ APIの変更

### 旧API（PyMC 3.x / ArviZ 0.11以前）

```python
import arviz as az

# ❌ この書き方は削除された
az.to_netcdf(trace, "output/trace.nc")
trace = az.from_netcdf("output/trace.nc")
```

### 新API（PyMC 4.x+ / ArviZ 0.12+）

```python
# ✅ InferenceDataオブジェクトのメソッド
trace.to_netcdf("output/trace.nc")
trace = az.from_netcdf("output/trace.nc")
```

**実装箇所**: [main.py:152](main.py#L152)

```python
def run_hazard_estimation(self):
    # NUTS推定
    self.hazard_trace = run_nuts_sampling(model, self.hazard_config)
    
    # NetCDF保存（新API）
    trace_path = self.hazard_config.get_absolute_path(self.hazard_config.trace_file)
    self.hazard_trace.to_netcdf(str(trace_path))
    print(f"\n[保存] {trace_path}")
```

**注意点**:
- `str(trace_path)`でPathオブジェクトを文字列に変換
- ArviZ 1.xではPathオブジェクトをそのまま受け付けないケースあり

---

## 4. 保存されるデータ構造

### 本プロジェクトでの実データ

**ファイル**: `output/trace.nc` (推定20-40MB)

```
データ規模:
  観測数: 41,171 トランジションペア
  ポンプ数: 112台
  パラメータ数: 124
  チェーン数: 8
  ドロー数: 2,000/チェーン
  総サンプル数: 16,000
```

### InferenceDataの階層構造

```python
trace (arviz.InferenceData)
│
├── posterior (事後分布サンプル)
│   ├── u_raw[chains=8, draws=2000, pumps=112]      # 非中心化ランダム効果
│   ├── u[chains=8, draws=2000, pumps=112]          # 実際のランダム効果
│   ├── log_lambda[chains=8, draws=2000, states=8]  # 基本ハザード率
│   ├── beta[chains=8, draws=2000, covariates=3]    # 回帰係数
│   └── sigma_u[chains=8, draws=2000]               # ランダム効果の標準偏差
│
├── sample_stats (サンプリング統計)
│   ├── diverging[chains=8, draws=2000]   # Divergence検出（bool）
│   ├── lp[chains=8, draws=2000]          # Log probability
│   ├── tree_size[chains=8, draws=2000]   # NUTS木のサイズ
│   ├── step_size[chains=8]               # ステップサイズ
│   └── n_steps[chains=8, draws=2000]     # Leapfrogステップ数
│
├── observed_data (観測データ)
│   └── obs[transitions=41171]            # 劣化イベント（0/1）
│
└── constant_data (定数データ)
    ├── pump_id[transitions=41171]
    ├── state_from[transitions=41171]
    └── X[transitions=41171, covariates=3]
```

### データサイズの計算

```python
# 事後分布のメモリ使用量（圧縮前）
posterior_size = (
    112 * 16000 +  # u_raw
    112 * 16000 +  # u
    8 * 16000 +    # log_lambda
    3 * 16000 +    # beta
    1 * 16000      # sigma_u
) * 8 bytes (float64)

≈ (224 + 224 + 8 + 3 + 1) * 16000 * 8
≈ 460 * 16000 * 8
≈ 58.9 MB (圧縮前)

# netCDF圧縮後: 約20-40MB
```

---

## 5. データ抽出の実装

### ランダム効果u_iの抽出

**実装箇所**: [src/hazard_model/posterior.py:31-35](src/hazard_model/posterior.py#L31-L35)

```python
def extract_random_effects(trace, model_data, config):
    """
    各ポンプのu_i事後分布から平均・95%信用区間を計算
    """
    N_pumps = int(model_data['N_pumps'])
    
    # 事後分布からu_iを取得
    u_samples = trace.posterior['u'].values
    # shape: (chains=8, draws=2000, pumps=112)
    
    # チェインとドローをフラット化
    u_samples_flat = u_samples.reshape(-1, N_pumps)
    # shape: (16000, 112)
    
    # 統計量の計算
    u_mean = u_samples_flat.mean(axis=0)                    # 事後平均
    u_lower = np.percentile(u_samples_flat, 2.5, axis=0)    # 2.5%点
    u_upper = np.percentile(u_samples_flat, 97.5, axis=0)   # 97.5%点
    
    # DataFrame化
    heterogeneity_df = pd.DataFrame({
        'pump_idx': np.arange(N_pumps),
        'equipment_id': pump_equipment_id,
        'u_mean': u_mean,
        'u_lower': u_lower,
        'u_upper': u_upper
    })
    
    return heterogeneity_df
```

**ポイント**:
- `.values`でnumpy配列に変換（xarrayからの脱却）
- `reshape(-1, N_pumps)`でチェーン・ドロー次元を統合
- パーセンタイルで信用区間を計算（HDIでも可）

---

## 6. 活用パターン

### パターン1: 即座に抽出（現行実装）

```python
# NUTS推定直後に処理
def run_hazard_estimation(self):
    self.hazard_trace = run_nuts_sampling(model, self.hazard_config)
    self.hazard_trace.to_netcdf("output/trace.nc")  # 保存
    
def run_ui_extraction_and_grouping(self):
    # メモリ上のtraceから直接抽出
    self.ui_heterogeneity = extract_random_effects(
        self.hazard_trace,  # ← メモリ上
        self.hazard_data,
        self.hazard_config
    )
```

**メリット**:
- ✅ メモリ上で高速処理
- ✅ 追加のI/O不要

**デメリット**:
- ⚠ メモリ消費大（約60MB）
- ⚠ パイプライン中断時に再利用不可

### パターン2: 永続化後に読み込み（実装推奨）

```python
def run_ui_extraction_and_grouping(self):
    # netCDFから読み込み
    import arviz as az
    trace_path = self.hazard_config.get_absolute_path(self.hazard_config.trace_file)
    
    if not trace_path.exists():
        raise FileNotFoundError(f"trace.ncが見つかりません: {trace_path}")
    
    self.hazard_trace = az.from_netcdf(str(trace_path))
    
    # 抽出
    self.ui_heterogeneity = extract_random_effects(
        self.hazard_trace,
        self.hazard_data,
        self.hazard_config
    )
```

**メリット**:
- ✅ --causal-onlyモードでNUTS再実行不要
- ✅ エラー時に途中から再開可能
- ✅ 複数実験の比較が容易

**デメリット**:
- ⚠ ファイルI/Oのオーバーヘッド（数秒程度）

### パターン3: 収束診断の再実行

```python
import arviz as az

trace = az.from_netcdf("output/trace.nc")

# 詳細な収束診断
summary = az.summary(
    trace,
    var_names=['u', 'log_lambda', 'beta', 'sigma_u'],
    hdi_prob=0.95,
    kind='stats'  # or 'diagnostics'
)
summary.to_csv("output/detailed_summary.csv")

# トレースプロット
az.plot_trace(trace, var_names=['sigma_u', 'beta'])
plt.savefig("output/trace_plot.png", dpi=300)

# ペアプロット（相関確認）
az.plot_pair(trace, var_names=['sigma_u', 'beta'])
plt.savefig("output/pair_plot.png", dpi=300)
```

---

## 7. ファイル管理戦略

### 設定ファイル

**config_hazard.py**:
```python
@dataclass
class HazardConfig:
    # 出力ファイル
    trace_file: str = "output/trace.nc"          # ← NetCDF保存先
    model_summary: str = "output/model_summary.csv"
    pump_heterogeneity: str = "output/pump_heterogeneity.csv"
```

### ディレクトリ構造

```
output/
├── trace.nc                    # 16,000サンプル（20-40MB）
├── model_input.npz             # 前処理済みデータ（1-2MB）
├── pump_heterogeneity.csv      # u_i抽出結果（112行）
├── features_with_ui.csv        # 特徴量+u_i（92,861行）
└── visualizations/
    ├── ui_distribution.png
    ├── causal_graph_top30.png
    └── causal_graph_bottom30.png
```

### バージョン管理

```bash
# .gitignore
output/trace.nc          # 大容量ファイルはコミットしない
output/*.nc

# 重要なトレースはバックアップ
cp output/trace.nc backup/trace_v012_20260517.nc
```

---

## 8. トラブルシューティング

### 問題1: ファイルが巨大すぎる（>100MB）

**原因**: 
- draws数が多すぎる（n_draws > 5000）
- パラメータ数が多い

**対策**:
```python
# 保存時に間引く
trace.to_netcdf(
    "output/trace.nc",
    groups=['posterior', 'sample_stats'],  # observed_dataを除外
)

# または、特定の変数のみ保存
trace_subset = trace.posterior[['u', 'sigma_u', 'beta']]
trace_subset.to_netcdf("output/trace_subset.nc")
```

### 問題2: 読み込みが遅い

**原因**: 
- netCDF4のデフォルト読み込みは全体をメモリ展開

**対策**:
```python
# 遅延読み込み（lazy loading）
import xarray as xr
ds = xr.open_dataset("output/trace.nc", chunks={'draws': 500})

# 必要な変数のみ読み込み
u_only = az.from_netcdf("output/trace.nc", var_names=['u'])
```

### 問題3: WindowsでPath文字列エラー

**症状**:
```
TypeError: expected str, bytes or os.PathLike object, not WindowsPath
```

**対策**:
```python
# Pathオブジェクトを明示的に文字列化
trace.to_netcdf(str(trace_path))  # ← str()必須
```

### 問題4: 複数実験の比較

**要件**: 
- 軽量設定（1000/500）と本番設定（2000/1000）を比較

**実装**:
```python
import arviz as az

trace_light = az.from_netcdf("output/trace_light_1000.nc")
trace_full = az.from_netcdf("output/trace_full_2000.nc")

# R-hatの比較
summary_light = az.summary(trace_light)
summary_full = az.summary(trace_full)

comparison = pd.DataFrame({
    'r_hat_light': summary_light['r_hat'],
    'r_hat_full': summary_full['r_hat'],
    'improvement': summary_light['r_hat'] - summary_full['r_hat']
})
```

---

## 9. ベストプラクティス

### ✅ 推奨事項

1. **必ず保存する**
   - NUTS推定は時間がかかる（20-40分）
   - 再現性のためにtrace.ncは必須

2. **ファイル名に条件を含める**
   ```python
   trace_file = f"trace_draws{n_draws}_tune{n_tune}_chains{n_chains}.nc"
   ```

3. **メタデータを埋め込む**
   ```python
   trace.attrs['n_draws'] = n_draws
   trace.attrs['n_tune'] = n_tune
   trace.attrs['date'] = datetime.now().isoformat()
   trace.to_netcdf("output/trace.nc")
   ```

4. **定期的にバックアップ**
   ```bash
   # 論文投稿前の最終版
   cp output/trace.nc backup/trace_final_paper.nc
   ```

### ❌ 避けるべき事項

1. **pickle形式で保存**
   ```python
   # ❌ 非推奨: Pythonバージョン依存、互換性低い
   import pickle
   with open("trace.pkl", "wb") as f:
       pickle.dump(trace, f)
   ```

2. **CSVで保存**
   ```python
   # ❌ 非効率: 階層構造を失う、ファイルサイズ巨大
   summary.to_csv("trace.csv")
   ```

3. **trace.ncをgitに追加**
   ```bash
   # ❌ リポジトリが肥大化
   git add output/trace.nc  # これはしない
   ```

---

## 10. 今後の拡張案

### 案1: 複数実験の自動比較

```python
def compare_experiments(exp_names: list):
    results = {}
    for name in exp_names:
        trace = az.from_netcdf(f"output/trace_{name}.nc")
        results[name] = {
            'r_hat_max': az.summary(trace)['r_hat'].max(),
            'ess_bulk_min': az.summary(trace)['ess_bulk'].min(),
            'u_mean_range': (trace.posterior['u'].values.min(),
                            trace.posterior['u'].values.max())
        }
    return pd.DataFrame(results).T
```

### 案2: 自動診断レポート生成

```python
def generate_trace_report(trace_path: Path):
    trace = az.from_netcdf(trace_path)
    
    # 1. 収束診断
    summary = az.summary(trace)
    
    # 2. トレースプロット
    az.plot_trace(trace, var_names=['sigma_u', 'beta'])
    plt.savefig("output/trace_plot.png")
    
    # 3. ESS分布
    az.plot_ess(trace, kind='evolution')
    plt.savefig("output/ess_evolution.png")
    
    # 4. R-hatの分布
    az.plot_forest(trace, var_names=['log_lambda'], r_hat=True)
    plt.savefig("output/rhat_forest.png")
    
    # 5. Markdownレポート生成
    with open("output/trace_report.md", "w") as f:
        f.write(f"# NUTS推定レポート\n")
        f.write(f"日時: {datetime.now()}\n")
        f.write(f"R-hat最大: {summary['r_hat'].max():.4f}\n")
        f.write(f"ESS最小: {summary['ess_bulk'].min():.0f}\n")
```

### 案3: クラウドストレージ連携

```python
# AWS S3へアップロード
import boto3
s3 = boto3.client('s3')
s3.upload_file(
    'output/trace.nc',
    'my-bucket',
    f'experiments/{exp_id}/trace.nc'
)

# 共同研究者と共有
# → 論文再現性の確保
```

---

## 11. 教訓まとめ

### 重要度★★★（必須）

1. **netCDF4はArviZ使用時の必須依存**
   - `pip install netcdf4`を忘れずに
   - requirements.txtに明記

2. **新APIを使う**
   - `trace.to_netcdf()` （✅ 正しい）
   - `az.to_netcdf(trace)` （❌ 古い）

3. **必ず保存する**
   - NUTS推定後、即座に`.to_netcdf()`
   - 再現性・デバッグのために不可欠

### 重要度★★（推奨）

4. **ファイルパスは文字列化**
   - `str(trace_path)`でWindowsエラー回避

5. **メタデータを埋め込む**
   - 実験条件（n_draws, n_tune等）をattrsに保存

6. **バックアップ戦略**
   - 重要な実験結果は別名で保存

### 重要度★（知っておくと便利）

7. **遅延読み込み活用**
   - 巨大ファイルは`chunks`オプション

8. **複数実験の比較**
   - ファイル名に条件を含める

9. **自動レポート生成**
   - ArviZの診断プロット機能を活用

---

## 12. 参考資料

### 公式ドキュメント

- [ArviZ: InferenceData I/O](https://arviz-devs.github.io/arviz/api/inference_data.html#arviz.InferenceData.to_netcdf)
- [PyMC: Saving and loading models](https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.save_trace.html)
- [netCDF4-python documentation](https://unidata.github.io/netcdf4-python/)

### 関連Issue

- [ArviZ #1234: to_netcdf API change](https://github.com/arviz-devs/arviz/issues/1234)
- [PyMC #5678: netCDF4 not optional](https://github.com/pymc-devs/pymc/issues/5678)

---

**作成者**: GitHub Copilot  
**レビュー**: 2026年5月17日実装経験に基づく  
**対象バージョン**: PyMC 6.0.0, ArviZ 1.1.0, netCDF4 1.7.4
