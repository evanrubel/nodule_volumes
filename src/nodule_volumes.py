# example: python nodule_volumes.py -t segment -d toy -v

from segment.scripts import segment
from register.scripts import register
from utils import preprocessing

import argparse
from datetime import datetime
import json
import os
from pprint import pprint

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

    assert task in {"segment", "register", "full"}

    assert os.path.isdir(f"../data/{dataset}"), "Expected the dataset directory to exist."
    assert os.path.isdir(f"../data/{dataset}/images") and os.path.isfile(f"../data/{dataset}/config.json"), "Expected the dataset directory to be well-formed."

    # configuration
    with open(f"../data/{dataset}/config.json") as f:
        config = json.load(f) | {
            "debug": verbose,
            "dataset_dir": os.path.abspath(f"../data/{dataset}"),
            "dataset_name": dataset,
            "output_dir": os.path.abspath(f"../data/{dataset}/results/{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
            "nifti_dir": os.path.abspath(f"../data/{dataset}/images"),
            "transforms_dir": os.path.abspath(f"../data/{dataset}/transforms"),
            "registered_masks_dir": os.path.abspath(f"../data/{dataset}/results/{datetime.now().strftime('%Y%m%d_%H%M%S')}/registered_masks"),
        }
    
    for k in ["device", "debug", "p_f_threshold", "lung_vessel_overlap_threshold", "lung_mask_mode", "prompt_type", "prompt_subset_type"]:
        assert k in config, "Expected a well-formed configuration file."

    # BiomedParse++    
    assert isinstance(config["p_f_threshold"], (float, int)) and 0 <= config["p_f_threshold"] <= 1, "Expected a valid threshold for p_f."

    # Postprocessing
    assert (isinstance(config["lung_vessel_overlap_threshold"], (float, int)) and 0 <= config["lung_vessel_overlap_threshold"] <= 1) or config["lung_vessel_overlap_threshold"] is None, "Expected a valid threshold for lung_vessel_overlap."
    assert config["lung_mask_mode"] in {"mask", "range", False}

    # nnInteractive
    assert config["prompt_type"] in {"bbox", "mask", "pos_point"}
    assert config["prompt_subset_type"] in {"all", "maximum", "median"}

    if config["debug"]:
        print("\n")
        pprint(config)
        print("\n")
    
    # should not already exist
    os.makedirs(config["output_dir"])
    
    # can already exist
    os.makedirs(os.path.join(config["dataset_dir"], "lung_masks"), exist_ok=True)
    os.makedirs(os.path.join(config["dataset_dir"], "lung_vessel_masks"), exist_ok=True)
    os.makedirs(config["transforms_dir"], exist_ok=True)

    with open(os.path.join(config["output_dir"], "experiment_config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # before we proceed, we check that all of the files in the images directory are well-formed NIfTI files

    for dirpath, dirnames, filenames in os.walk(config["nifti_dir"]):
        # TODO: test this functionality
        for i, dirname in enumerate(dirnames):
            assert all([sub_filename.endswith(".dcm") for sub_filename in os.listdir(os.path.join(config["nifti_dir"], dirname))]), "Expected all DICOM files in the directory."

            # TODO: maybe clean up the file name?
            dicom_to_nifti(dirname, os.path.join(config["nifti_dir"], f"{dirname}_0000.nii.gz")) # all input images must end in _0000

        for filename in filenames:
            assert filename.endswith("_0000.nii.gz"), "We only support .nii.gz files for now, and their filenames must end in _0000."

    # now, we can proceed with the remainder of our pipeline
     
    if task == "segment":
        segment.main(config)
    elif task == "register":
        register.main(config)
    elif task == "full":
        segment.main(config)
        input("Enter here when ready!")
        register.main(config)
