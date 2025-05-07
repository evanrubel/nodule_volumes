from segment.scripts import segment
from register.scripts import register

import argparse
import json
import os
from pprint import pprint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run task with the specified dataset")

    parser.add_argument("-t", "--task", required=True, help="Task name ('segment' or 'register')")
    parser.add_argument("-d", "--dataset", required=True, help="Dataset name (e.g., 'toy')")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()
    task = args.task
    dataset = args.dataset
    verbose = args.verbose

    assert task in {"segment", "register"}

    assert os.path.isdir(f"../data/{dataset}"), "Expected the dataset directory to exist."
    assert os.path.isdir(f"../data/{dataset}/images") and os.path.isfile(f"../data/{dataset}/config.json"), "Expected the dataset directory to be well-formed."

    with open(f"../data/{dataset}/config.json") as f:
        config = json.load(f) | {"debug": verbose}
    
    assert "device" in config and "debug" in config, "Expected a well-formed configuration file."

    if config["debug"]:
        pprint(config)
     
    if task == "segment":
        segment.main(config)

    elif task == "register":
        raise NotImplementedError