"""
pump_heterogeneity.csvにu_groupカラムを追加する補助スクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path

def add_u_group(heterogeneity_path: str, 
                top_percentile: float = 0.70, 
                bottom_percentile: float = 0.30):
    """
    pump_heterogeneity.csvにu_groupカラムを追加
    
    Parameters:
        heterogeneity_path: pump_heterogeneity.csvのパス
        top_percentile: 上位グループの閾値（デフォルト: 0.70 = 70パーセンタイル）
        bottom_percentile: 下位グループの閾値（デフォルト: 0.30 = 30パーセンタイル）
    """
    # CSVを読み込み
    df = pd.read_csv(heterogeneity_path)
    print(f"[読込] {heterogeneity_path}")
    print(f"  レコード数: {len(df)}")
    print(f"  列: {list(df.columns)}")
    
    # パーセンタイル計算
    u_top_threshold = df['u_mean'].quantile(top_percentile)
    u_bottom_threshold = df['u_mean'].quantile(bottom_percentile)
    
    print(f"\n[グループ分け閾値]")
    print(f"  上位30%閾値（{top_percentile*100:.0f}パーセンタイル）: {u_top_threshold:.4f}")
    print(f"  下位30%閾値（{bottom_percentile*100:.0f}パーセンタイル）: {u_bottom_threshold:.4f}")
    
    # グループ割り当て
    def assign_group(u_val):
        if u_val >= u_top_threshold:
            return 'top30'
        elif u_val <= u_bottom_threshold:
            return 'bottom30'
        else:
            return 'middle'
    
    df['u_group'] = df['u_mean'].apply(assign_group)
    
    # グループ統計
    group_counts = df['u_group'].value_counts()
    print(f"\n[グループ別ポンプ数]")
    for group in ['top30', 'middle', 'bottom30']:
        count = group_counts.get(group, 0)
        pct = count / len(df) * 100
        print(f"  {group}: {count} ({pct:.1f}%)")
    
    # 上書き保存
    df.to_csv(heterogeneity_path, index=False)
    print(f"\n[保存完了] {heterogeneity_path}")
    print(f"  新しい列: {list(df.columns)}")


if __name__ == "__main__":
    heterogeneity_path = "output/pump_heterogeneity.csv"
    add_u_group(heterogeneity_path)
