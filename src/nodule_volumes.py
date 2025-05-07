from segment.scripts import segment
from register.scripts import register
from utils import preprocessing

import argparse
from datetime import datetime
import json
import os
from pprint import pprint
import sys

if __name__ == "__main__":
    # read in command-line arguments

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

    # configuration

    with open(f"../data/{dataset}/config.json") as f:
        config = json.load(f) | {
            "debug": verbose,
            "output_dir": f"../data/{dataset}/results/{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "nifti_dir": f"../data/{dataset}/images",
        }
    
    assert "device" in config and "debug" in config, "Expected a well-formed configuration file."

    if config["debug"]:
        print("\n")
        pprint(config)
        print("\n")
    
    os.makedirs(config["output_dir"])

    # before we proceed, we check that all of the files in the images directory are well-formed NIfTI files

    for dirpath, dirnames, filenames in os.walk(config["nifti_dir"]):
        # TODO: test this functionality
        for i, dirname in enumerate(dirnames):
            assert all([sub_filename.endswith(".dcm") for sub_filename in os.listdir(os.path.join(config["nifti_dir"], dirname))]), "Expected all DICOM files in the directory."

            # TODO: maybe clean up the file name?
            dicom_to_nifti(dirname, os.path.join(config["nifti_dir"], f"{dirname}.nii.gz"))

        for filename in filenames:
            assert filename.endswith(".nii.gz"), "We only support .nii.gz files for now!"

    # now, we can proceed with the remainder of our pipeline
     
    if task == "segment":
        segment.main(config)

    elif task == "register":
        raise NotImplementedError
        register.main(config)