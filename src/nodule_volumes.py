# example: python nodule_volumes.py -t segment -d toy -v

from segment.scripts import segment
from register.scripts import register

import argparse
from datetime import datetime
import json
import os
from pprint import pprint
import SimpleITK as sitk


def dicom_to_nifti(dicom_folder_path: str, nifti_path: str) -> None:
    """
    Converts the DICOM slices in `dicom_folder_path` to a single NIfTI file.
    Writes the NIfTI to `nifti_path`.
    """

    assert nifti_path.endswith(".nii.gz"), "Expects a NIfTI file format."

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, nifti_path)


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

    # postprocessing
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

    # before we proceed, we check that all of the files in the images directory are DICOM directories or well-formed NIfTI files
    for dirpath, dirnames, _ in os.walk(config["nifti_dir"]):
        for i, dirname in enumerate(dirnames):
            assert all([sub_filename.endswith(".dcm") for sub_filename in os.listdir(os.path.join(config["nifti_dir"], dirname))]), "Expected all DICOM files in the directory."
            
            # was not already exported to NIFTI
            nifti_path = os.path.join(config["nifti_dir"], f"{dirname.replace('_0000', '')}_0000.nii.gz") # all input images must end in _0000
            if os.path.isfile(nifti_path):
                dicom_to_nifti(os.path.join(config["nifti_dir"], dirname), nifti_path)

    # only this directory
    for fname in os.listdir(config["nifti_dir"]):
        if os.path.isfile(os.path.join(config["nifti_dir"], fname)):
            assert fname.endswith("_0000.nii.gz"), "We only support .nii.gz files for now, and their filenames must end in _0000."

    # now, we can proceed with the remainder of our pipeline
     
    if task == "segment":
        segment.main(config)
    elif task == "register":
        register.main(config)
    elif task == "full":
        segment.main(config)
        register.main(config)
