import numpy as np
import os
import argparse
import json

# Arrays #

def is_binary_array(arr: np.ndarray) -> bool:
    """Checks whether `arr` is a binary array."""

    return np.isin(arr, [0, 1]).all()


# Configuration #

def get_config() -> dict:
    """Loads in the configuration file."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = json.load(f)
    
    return config


def run_in_conda_env(env_name: str, script_path: str, config_path: str) -> None:
    """Run `script_path` in `env_name` with the configuration at `config_path`."""

    command = f'bash -c "source ~/.bashrc && conda activate {env_name} && python {script_path} --config {config_path}"'
    result = subprocess.run(command, shell=True)
