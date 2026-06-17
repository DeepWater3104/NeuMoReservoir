import numpy as np

def get_spike_timings(time, Vm_trace, threshold):
    spike_timings = []
    for current_time_idx, current_time in enumerate(time[1:]):
        if Vm_trace[current_time_idx-1] < threshold and threshold < Vm_trace[current_time_idx]:
            spike_timings.append(current_time)

    return spike_timings

def get_firing_rate(spike_timings, duration):
    # spike_timings: list of spike timings
    # duration: Temporal duration recorded membrane potential in sec
    return len(spike_timings) / duration

def get_spike_indices(vm, t, v_threshold=-20):
    """
    vm: (N_time_steps, N_compartments)
    """
    dt = t[1] - t[0]
    mean_vm = np.mean(vm, axis=1) # calculate mean membrane potential for each time steps
    
    # bAP
    spike_cond = (mean_vm[:-1] < v_threshold) & (mean_vm[1:] >= v_threshold)
    bap_indices = np.where(spike_cond)[0]
    
    # dAP
    all_local_spikes = []
    for i in range(vm.shape[1]): # iteration about each compartments
        spks = np.where((vm[:-1, i] < v_threshold) & (vm[1:, i] >= v_threshold))[0]
        all_local_spikes.extend(spks)
    all_local_spikes = np.unique(all_local_spikes)

    dap_indices = []
    exclusion_window = int(10.0 / dt) 
    for d_idx in all_local_spikes:
        if not np.any(np.abs(bap_indices - d_idx) < exclusion_window):
            dap_indices.append(d_idx)
                
    return bap_indices, np.array(dap_indices)


# ==============================================
# Effective Rank defined in Roy & Vetterli, 2007
# ==============================================
def calculate_effective_rank(sub_data):
    """
    sub_data: (N_time_steps, N_compartments)
    """
    if np.all(sub_data == 0) or sub_data.size == 0:
        return 1.0
        
    try:
        # 特異値分解
        # 形状が (T, N) の場合、s は min(T, N) 個の特異値を返す
        _, s, _ = np.linalg.svd(sub_data, full_matrices=False)
    except np.linalg.LinAlgError:
        return 1.0

    sigma_sum = np.sum(s)
    if sigma_sum == 0:
        return 1.0
        
    p = s / sigma_sum
    p = p[p > 0]
    entropy = -np.sum(p * np.log(p))
    return np.exp(entropy)

# neuron_analyzer.py
import datetime
import glob
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from tqdm import tqdm

# 解析用カスタムモジュールのパスを通す（環境に応じて調整）
current_dir = Path.cwd()
parent_dir = str(current_dir.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from Analysis import calculate_effective_rank, get_spike_indices


class NeuMoAnalyzer:
    """NeuMoReservoirのマルチランデータを効率的に解析するためのフレームワーク"""

    def __init__(self, base_path):
        self.base_dir = Path(base_path)
        if not self.base_dir.exists():
            raise FileNotFoundError(f"Base path not found: {base_path}")

        # ディレクトリ構造からjob_idsを取得
        self.job_dirs = sorted(
            [d for d in self.base_dir.iterdir() if d.is_dir() and d.name.isdigit()],
            key=lambda x: int(x.name),
        )

    def _load_config(self, job_dir):
        """Hydraのconfigからパラメータを抽出"""
        cfg_path = job_dir / ".hydra" / "config.yaml"
        if not cfg_path.exists():
            return None
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        return {
            "syn_loc_mean": cfg.get("syn_loc_mean"),
            "syn_loc_std": cfg.get("syn_loc_std"),
            "seed": cfg.get("seed"),
        }

    def extract_and_save_spikes(self, output_dir="extracted_data"):
        """Stage 1: すべてのジョブからスパイクインデックス/時刻を抽出し、軽量なCSVとして保存する（高速化の肝）"""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        all_spikes = []

        print("=== Stage 1: Extracting Spike Indices ===")
        for job_dir in tqdm(self.job_dirs, desc="Jobs"):
            cfg_meta = self._load_config(job_dir)
            if not cfg_meta:
                continue

            buffer_files = sorted(glob.glob(str(job_dir / "data" / "buffer*.npz")))

            for bf in buffer_files:
                file_name = Path(bf).name
                data_ts = np.load(bf, allow_pickle=True)
                t = data_ts["t_rec"]
                vm = data_ts["variables"][1]  # Vm (T, N)
                duration_s = (t[-1] - t[0]) / 1000.0

                # スパイク抽出 (get_spike_indicesのインターフェースに依存)
                bap_idxs, dspike_idxs = get_spike_indices(vm, t, v_threshold=-30)

                # 各タイプ（bAP, dSpike）のインデックスを保存
                # カンマ区切りの文字列としてインデックスをシリアライズして軽量化
                for s_type, idxs in [("bAP", bap_idxs), ("dSpike", dspike_idxs)]:
                    all_spikes.append(
                        {
                            "job_id": job_dir.name,
                            "file_name": file_name,
                            "syn_loc_mean": cfg_meta["syn_loc_mean"],
                            "syn_loc_std": cfg_meta["syn_loc_std"],
                            "seed": cfg_meta["seed"],
                            "spike_type": s_type,
                            "spike_count": len(idxs),
                            "spike_indices": ",".join(map(str, idxs)),
                            "duration_sec": duration_s,
                            "t_min": t[0],
                            "t_max": t[-1],
                        }
                    )

        df_spikes = pd.DataFrame(all_spikes)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        spike_file = out_path / f"df_spikes_meta_{timestamp}.pkl"

        # インデックス文字列を保持するため、型変化のないPickleで保存
        df_spikes.to_pickle(spike_file)
        print(f"Spike metadata saved to: {spike_file}")
        return df_spikes

    def generate_firing_rate_report(self, df_spikes, output_dir="extracted_data"):
        """Stage 2-A: スパイクメタデータのみから発火率およびISI統計量を計算（再ロード不要のため超高速）"""
        print("=== Stage 2-A: Generating Firing Rate Report ===")
        rate_results = []

        for _, row in df_spikes.iterrows():
            # 文字列からインデックスの復元
            idxs = (
                [int(x) for x in row["spike_indices"].split(",")]
                if row["spike_indices"]
                else []
            )
            num_spikes = row["spike_count"]
            duration_s = row["duration_sec"]

            firing_rate = num_spikes / duration_s if duration_s > 0 else 0

            # 擬似的にISIを計算（インデックス間隔から秒換算する場合はt_recが必要だが、
            # ここでは簡易統計、あるいは必要ならt_recをStage1で引く設計にする）
            # 今回は元のロジックを継承（厳密なISIが必要な場合は、Stage1側でt[idxs]を文字列化して持たせること）
            cv_isi = np.nan  # 必要に応じてStage1で拡張可能

            rate_results.append(
                {
                    "job_id": row["job_id"],
                    "syn_loc_mean": row["syn_loc_mean"],
                    "syn_loc_std": row["syn_loc_std"],
                    "seed": row["seed"],
                    "spike_type": row["spike_type"],
                    "spike_count": num_spikes,
                    "firing_rate_hz": firing_rate,
                    "cv_isi": cv_isi,
                    "duration_sec": duration_s,
                }
            )

        df_rate = pd.DataFrame(rate_results)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path(output_dir) / f"df_rate_{timestamp}.csv"
        df_rate.to_csv(csv_path, index=False)
        print(f"Firing rate report saved to: {csv_path}")
        return df_rate

    def generate_effective_rank_report(
        self, df_spikes, window_ms=5.0, output_dir="extracted_data"
    ):
        """Stage 2-B: 必要なスパイク周辺のデータのみをピンポイントでロードしてランク計算（I/Oを極小化）"""
        print("=== Stage 2-B: Generating Effective Rank Report ===")
        rank_results = []

        # ファイル単位でグルーピングして、同一ファイルの再オープンを1回に抑える
        grouped = df_spikes.groupby(["job_id", "file_name"])

        for (job_id, file_name), group in tqdm(
            grouped, desc="Processing Ranks by File"
        ):
            job_dir = self.base_dir / job_id
            bf_path = job_dir / "data" / file_name

            if not bf_path.exists():
                continue

            # ここで初めてファイルをロード（1ファイルにつき1回のみ）
            data_ts = np.load(bf_path, allow_pickle=True)
            t = data_ts["t_rec"]
            ca = data_ts["variables"][0]  # Ca (T, N)
            dt = t[1] - t[0]
            window_size = int(window_ms / dt)

            # PCAの事前計算（ファイル全体に対して1回だけ行う）
            pca = PCA(n_components=1)
            ca_pca = pca.fit_transform(ca)
            ca_proj = pca.inverse_transform(ca_pca)
            ca_res = ca - ca_proj

            for _, row in group.iterrows():
                idxs = (
                    [int(x) for x in row["spike_indices"].split(",")]
                    if row["spike_indices"]
                    else []
                )
                s_type = row["spike_type"]

                for idx in idxs:
                    if (
                        idx < window_size // 2
                        or idx > ca.shape[0] - window_size // 2
                    ):
                        continue

                    w_start = idx - window_size // 2
                    w_end = idx + window_size // 2

                    er_orig = calculate_effective_rank(ca[w_start:w_end, :])
                    er_res = calculate_effective_rank(ca_res[w_start:w_end, :])

                    rank_results.append(
                        {
                            "job_id": job_id,
                            "syn_loc_mean": row["syn_loc_mean"],
                            "syn_loc_std": row["syn_loc_std"],
                            "seed": row["seed"],
                            "spike_type": s_type,
                            "er_orig": er_orig,
                            "er_res": er_res,
                        }
                    )

        df_rank = pd.DataFrame(rank_results)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path(output_dir) / f"df_rank_{timestamp}.csv"
        df_rank.to_csv(csv_path, index=False)
        print(f"Effective rank report saved to: {csv_path}")
        return df_rank
