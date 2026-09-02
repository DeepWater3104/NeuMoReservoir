import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import os
import sys
import json
import subprocess

# Configure logging to output information to the console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# __file__ は chdir 後も実行時に固定された絶対パスを指すため、ワーカーの場所を安全に解決できる
WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "neuron_worker.py")


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for the reservoir simulation.
    Hydra handles the configuration loading, output directory management, and
    (via hydra-joblib-launcher) the sweep's parallel job scheduling / concurrency cap.

    シミュレーション本体（NEURON を触る処理）はここでは実行せず、`neuron_worker.py` を
    ジョブごとに使い捨てのサブプロセスとして起動する。NEURON はプロセス内に
    C言語側のグローバル状態を溜め込む性質があり、joblib の並列ワーカープロセスを
    使い回すとジョブを重ねるほど実行時間が線形に悪化するため、ワーカーの
    使い回し自体は許容しつつ、NEURON に触れる処理だけを毎回フレッシュな
    OSプロセスに追い出すことで状態の持ち越しを断つ。
    """
    from hydra.core.hydra_config import HydraConfig
    from hydra.utils import get_original_cwd
    hydra_cfg = HydraConfig.get()
    is_multirun = hydra_cfg.mode.name == "MULTIRUN"

    # Convert DictConfig to a standard Python dictionary for compatibility with existing classes
    params = OmegaConf.to_container(cfg, resolve=True)

    # Ensure necessary directories exist for outputs
    os.makedirs("figure", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Log the resolved configuration for verification
    logger.info("--- Simulation Configuration ---")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info(f"Task name: {params['task']['name']}")
    logger.info(f"Random seed: {params.get('seed', 1234)}")

    payload = {
        "params": params,
        "original_cwd": get_original_cwd(),
        "is_multirun": is_multirun,
    }
    payload_path = os.path.join(os.getcwd(), ".worker_params.json")
    with open(payload_path, "w") as f:
        json.dump(payload, f)

    try:
        # stdout/stderr は継承させ、tqdm の進捗バー表示や main.log への追記を
        # 元の挙動のまま維持する（ワーカー側で main.log に追記ハンドラを追加する）。
        result = subprocess.run(
            [sys.executable, WORKER_SCRIPT, "--payload", payload_path],
            cwd=os.getcwd(),
        )
    finally:
        os.remove(payload_path)

    if result.returncode != 0:
        raise RuntimeError(f"neuron_worker.py failed with exit code {result.returncode}")


if __name__ == "__main__":
    main()
