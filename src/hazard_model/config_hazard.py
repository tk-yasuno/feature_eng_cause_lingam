"""
ハザードモデル設定クラス
"""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class HazardConfig:
    """ハザードモデルの設定"""
    
    # ディレクトリ設定
    base_dir: Path = Path(__file__).parent.parent.parent
    output_dir: Path = base_dir / "output"
    
    # 入力データ
    labeled_time_series: str = "data/processed/labeled_time_series.csv"
    selected_equipment: str = "data/processed/selected_64_equipment.json"
    
    # 出力ファイル
    model_input: str = "output/model_input.npz"
    trace_file: str = "output/trace.nc"
    model_summary: str = "output/model_summary.csv"
    pump_heterogeneity: str = "output/pump_heterogeneity.csv"
    
    # モデル設定
    n_states: int = 8  # 健全度状態数（8分割）
    min_data_points: int = 30  # 最低データ点数
    min_delta_t: int = 1  # 最小点検間隔（日）
    max_delta_t: int = 365  # 最大点検間隔（日）
    
    # NUTS推定パラメータ
    n_draws: int = 2000
    n_tune: int = 1000
    n_chains: int = 8
    n_cores: int = 8
    target_accept: float = 0.95
    random_seed: int = 42
    
    # グループ分け設定
    top_percentile: float = 0.70  # 上位30%（70パーセンタイル以上）
    bottom_percentile: float = 0.30  # 下位30%（30パーセンタイル以下）
    
    def __post_init__(self):
        """出力ディレクトリの作成"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """相対パスを絶対パスに変換"""
        return self.base_dir / relative_path
