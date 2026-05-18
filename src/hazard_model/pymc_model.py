"""
PyMCランダム効果ハザードモデル

ランダム効果付きマルコフ劣化ハザードモデル:
  λ_ik(t) = λ_0k * exp(β^T x_it + u_i)
  u_i ~ N(0, σ²)
"""

import numpy as np
import pymc as pm
import arviz as az
from pathlib import Path
from .config_hazard import HazardConfig
import warnings
warnings.filterwarnings('ignore')


def build_hazard_model(data: dict):
    """
    ランダム効果付きマルコフ劣化ハザードモデルの構築
    
    Parameters:
        data: prepared hazard data dict with keys:
            N_obs, N_pumps, K, pump_idx, state_now, delta_t, X, moved, n_cov
    
    Returns:
        PyMC model
    """
    print("\n[モデル構築]")
    
    N_obs = data['N_obs']
    N_pumps = data['N_pumps']
    K = data['K']
    n_cov = data['n_cov']
    
    pump_idx_arr = data['pump_idx']
    state_now_arr = data['state_now']
    delta_t_arr = data['delta_t']
    X_arr = data['X']
    moved_arr = data['moved']
    
    print(f"  観測数: {N_obs}")
    print(f"  ポンプ数: {N_pumps}")
    print(f"  状態数: {K}")
    print(f"  共変量次元: {n_cov}")
    
    with pm.Model() as model:
        
        # 1. 基準ハザード（状態ごと）
        log_lambda = pm.Normal(
            "log_lambda", 
            mu=-5.0, 
            sigma=2.0, 
            shape=K
        )
        
        # 2. 共変量係数
        beta = pm.Normal(
            "beta", 
            mu=0.0, 
            sigma=1.0, 
            shape=n_cov
        )
        
        # 3. ランダム効果（ポンプごと）- 非中心化パラメータ化
        sigma_u = pm.HalfNormal("sigma_u", sigma=1.0)
        u_raw = pm.Normal(
            "u_raw", 
            mu=0.0, 
            sigma=1.0, 
            shape=N_pumps
        )
        u = pm.Deterministic("u", u_raw * sigma_u)
        
        # 4. 観測ごとのハザード計算
        log_lambda_now = log_lambda[state_now_arr - 1]
        lin_cov = pm.math.dot(X_arr, beta)
        u_obs = u[pump_idx_arr]
        
        log_hazard = log_lambda_now + lin_cov + u_obs
        hazard = pm.math.exp(log_hazard)
        
        # 5. 遷移確率
        p_move = 1.0 - pm.math.exp(-hazard * delta_t_arr)
        p_move = pm.math.clip(p_move, 1e-6, 1 - 1e-6)
        
        # 6. 尤度（Bernoulli）
        obs = pm.Bernoulli("obs", p=p_move, observed=moved_arr)
    
    print(f"  パラメータ数: {K + n_cov + 1 + N_pumps}")
    print("  モデル構築完了")
    
    return model


def run_nuts_sampling(model, config: HazardConfig):
    """
    NUTS推定の実行
    
    Parameters:
        model: PyMC model
        config: HazardConfig instance
    
    Returns:
        InferenceData (arviz)
    """
    print("\n[NUTS推定開始]")
    print(f"  draws: {config.n_draws}")
    print(f"  tune: {config.n_tune}")
    print(f"  chains: {config.n_chains}")
    print(f"  cores: {config.n_cores}")
    print(f"  target_accept: {config.target_accept}")
    print("  推定中... (60-120分程度かかります)")
    
    with model:
        trace = pm.sample(
            draws=config.n_draws,
            tune=config.n_tune,
            target_accept=config.target_accept,
            chains=config.n_chains,
            cores=config.n_cores,
            random_seed=config.random_seed,
            return_inferencedata=True,
            progressbar=True
        )
    
    print("\n  サンプリング完了")
    
    # 収束診断
    check_convergence(trace)
    
    return trace


def check_convergence(trace):
    """
    収束診断の実行
    
    Parameters:
        trace: InferenceData
    """
    print("\n[収束診断]")
    
    # サマリ統計の計算
    summary = az.summary(
        trace,
        var_names=['log_lambda', 'beta', 'sigma_u']
    )
    
    print("\n主要パラメータのサマリ:")
    print(summary)
    
    # R-hat のチェック
    rhat_values = summary['r_hat'].astype(float)
    rhat_max = rhat_values.max()
    print(f"\n  最大 R-hat: {rhat_max:.4f}")
    
    if rhat_max < 1.01:
        print("  ✓ 収束判定: 良好 (R-hat < 1.01)")
    elif rhat_max < 1.05:
        print("  ⚠ 収束判定: 注意 (1.01 <= R-hat < 1.05)")
    else:
        print("  ✗ 収束判定: 不良 (R-hat >= 1.05) - 再実行を推奨")
    
    # ESS のチェック
    ess_bulk_values = summary['ess_bulk'].astype(float)
    ess_tail_values = summary['ess_tail'].astype(float)
    ess_bulk_min = ess_bulk_values.min()
    ess_tail_min = ess_tail_values.min()
    
    print(f"  最小 ESS (bulk): {ess_bulk_min:.0f}")
    print(f"  最小 ESS (tail): {ess_tail_min:.0f}")
    
    if ess_bulk_min < 400:
        print("  ⚠ 警告: ESS (bulk) が400未満です")
    if ess_tail_min < 400:
        print("  ⚠ 警告: ESS (tail) が400未満です")
    
    # Divergence チェック
    if hasattr(trace, 'sample_stats') and 'diverging' in trace.sample_stats:
        divergences = trace.sample_stats.diverging.sum().item()
        total_samples = trace.sample_stats.diverging.size
        divergence_rate = divergences / total_samples
        
        print(f"  Divergences: {divergences} / {total_samples} ({divergence_rate*100:.2f}%)")
        
        if divergence_rate > 0.01:
            print("  ⚠ 警告: Divergence rate > 1% - target_accept を上げることを推奨")
