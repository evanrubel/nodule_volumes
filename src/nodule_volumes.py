# example: python nodule_volumes.py -t segment -d toy -v

from segment.scripts import segment
from register.scripts import register

import argparse
from datetime import datetime
from huggingface_hub import hf_hub_download
import json
import nibabel as nib
from nibabel.orientations import axcodes2ornt, io_orientation, inv_ornt_aff, apply_orientation
import os
import pandas as pd
import pickle
from pprint import pprint
import shutil
import SimpleITK as sitk


def reorient_to_RAS(nifti_path: str, output_path: str) -> None:
    """Reorient the NIFTI at `nifti_path` to be in the RAS orientation and save the new NIFTI at `output_path`."""

    img = nib.load(nifti_path)
    orig_ornt = io_orientation(img.affine)
    ras_ornt = axcodes2ornt(('R', 'A', 'S'))
    transform = nib.orientations.ornt_transform(orig_ornt, ras_ornt)
    
    data = apply_orientation(img.get_fdata(), transform)
    new_affine = img.affine @ inv_ornt_aff(transform, img.shape)
    new_img = nib.Nifti1Image(data, new_affine)
    nib.save(new_img, output_path)


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

    # overwrite it with the proper orientation
    reorient_to_RAS(nifti_path, nifti_path)


def is_in_ras_orientation(nifti_path: str) -> bool:
    """Returns whether the NIFTI at `nifti_path` is in the correct RAS orientation."""

    img = nib.load(nifti_path)
    current_ornt = io_orientation(img.affine)
    ras_ornt = axcodes2ornt(('R', 'A', 'S'))

    return (current_ornt == ras_ornt).all()


def download_checkpoint(checkpoint_type: str) -> None:
    """Download the checkpoint of type `checkpoint_type` from Hugging Face."""

    if checkpoint_type == "biomedparse++":
        repo_id = "microsoft/BiomedParse"
        filename = "biomedparse_v1.pt"
        local_dir = "segment/checkpoints"
        cache_name = "models--microsoft--BiomedParse"

    elif checkpoint_type == "nnInteractive":
        repo_id = "nnInteractive/nnInteractive"
        filename = "nnInteractive_v1.0/fold_0/checkpoint_final.pth"
        local_dir = "segment/checkpoints/nnInteractive/fold_0"
        cache_name = "models--nnInteractive--nnInteractive"

        os.makedirs(local_dir, exist_ok=True) # just in case, we create its enclosing directory
    
    local_path = os.path.join(local_dir, os.path.basename(filename))

    # only download if it does not yet exist
    if not os.path.exists(local_path):
        if config["debug"]:
            print(f"Downloading {filename} from Hugging Face...")

        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=local_dir,
            local_dir_use_symlinks=False,  # download actual file, not a symlink
            cache_dir=local_dir, # do all of the caching in the target directory
        )

        if checkpoint_type == "nnInteractive": # move it to the proper place and clean up
            shutil.move(os.path.join(local_dir, filename), os.path.join(local_dir, "checkpoint_final.pth"))
            shutil.rmtree(os.path.join(local_dir, "nnInteractive_v1.0"))

        if config["debug"]:
            print(f"Downloaded to {downloaded_path}")
        
        # clean up cache directory
        shutil.rmtree(os.path.join(local_dir, cache_name))

    else:
        if config["debug"]:
            print(f"Checkpoint already exists at {local_path}")
    

# def import_nlst_files(file_path: str, config: dict) -> None:
#     """Based on the PIDs in the JSON at `file_path`, we save them all of their longitudinal scans as NIFTIs."""

#     with open(file_path, "rb") as f:
#         nlst_pids = json.load(f)
    
#     assert isinstance(nlst_pids, list) and all([isinstance(elem, str) for elem in nlst_pids]), "Expected a well-formed input."

#     if len(nlst_pids) == 0:
#         return

#     with open("/data/rbg/users/erubel/datasets/nlst/nlst_dataset.p", "rb") as f:
#         nlst_dataset = pickle.load(f)

#     pid_to_dicom_directory = pickle.load(open("/data/rbg/shared/datasets/NLST/NLST/all_nlst_dicoms_pid2directory.p", "rb"))

#     for pid, tp, dir_name in [(elem["pid"], elem["screen_timepoint"], os.path.join(pid_to_dicom_directory[elem["pid"]].replace("/Mounts/rbg-storage1/datasets", "/data/rbg/shared/datasets/NLST"), elem["pid"], "/".join(elem["paths"][0].split("/")[-3:-1]))) for elem in nlst_dataset if elem["pid"] in nlst_pids]:
#         dicom_to_nifti(dir_name, os.path.join(config["nifti_dir"], f"nlst_{pid}T{tp}_0000.nii.gz"))


def export_niftis_from_csv(file_path: str, config: dict) -> None:
    """Based on the PIDs in the csv at `file_path`, we save them all of their longitudinal scans as NIFTIs."""

    # TODO: fix this when we can extract the correct series

    df = pd.read_csv(file_path)

    for _, row in list(df.iterrows())[:1]:
        pid = row["PID"]
        print(pid)
        for tp in range(2):
            if not pd.isna(row[f'Series_{tp}']):
                print(row[f'Series_{tp}'])
                dicom_to_nifti(row[f'Series_{tp}'], os.path.join(config["nifti_dir"], f"nlst_{pid}T{tp}_0000.nii.gz"))


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
    
    for k in ["device", "detection_model", "debug", "p_f_threshold", "lung_vessel_overlap_threshold", "lung_mask_mode", "prompt_type", "prompt_subset_type"]:
        assert k in config, "Expected a well-formed configuration file."

    assert config["detection_model"] in {"biomedparse++", "total_segmentator"}, "Expected a support detection model."

    if config["detection_model"] == "biomedparse++":
        assert isinstance(config["p_f_threshold"], (float, int)) and 0 <= config["p_f_threshold"] <= 1, "Expected a valid threshold for p_f."
    elif config["detection_model"] == "total_segmentator":
        assert config["p_f_threshold"] is None    

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

    # # check if we need to import the NLST scans from their PIDs
    # if os.path.isfile(os.path.join(config["nifti_dir"], "nlst_pids.json")):
    #     import_nlst_files(os.path.join(config["nifti_dir"], "nlst_pids.json"), config)

    # check that all NIFTIs are in the RAS orientation
    for fname in os.listdir(config["nifti_dir"]):
        fname_path = os.path.join(config["nifti_dir"], fname)
        if os.path.isfile(fname_path):
            assert fname.endswith(".csv") or (fname_path.endswith("_0000.nii.gz") and is_in_ras_orientation(fname_path)), "We only support .nii.gz files. Their filenames must end in _0000, and their corresponding images must be in the RAS orientation."

    # before we proceed, we check that all of the files in the images directory are DICOM directories or well-formed NIfTI files
    for dirpath, dirnames, _ in os.walk(config["nifti_dir"]):
        for i, dirname in enumerate(dirnames):
            assert all([sub_filename.endswith(".dcm") for sub_filename in os.listdir(os.path.join(config["nifti_dir"], dirname))]), "Expected all DICOM files in the directory."
            
            # we export the DICOMs to NIFTIs if they do not already exist
            nifti_path = os.path.join(config["nifti_dir"], f"{dirname.replace('_0000', '')}_0000.nii.gz") # all input images must end in _0000
            if not os.path.isfile(nifti_path):
                dicom_to_nifti(os.path.join(config["nifti_dir"], dirname), nifti_path)
    
    # export the entries in any csv
    for fname in os.listdir(config["nifti_dir"]):
        continue
        # TODO: fix this!
        fname_path = os.path.join(config["nifti_dir"], fname)
        if os.path.isfile(fname_path) and fname.endswith(".csv"):
            export_niftis_from_csv(fname_path, config)
        
    # we next download any required checkpoints (we use local checkpoints if they were already downloaded)
    if config["detection_model"] == "biomedparse++":
        download_checkpoint("biomedparse++")
    
    download_checkpoint("nnInteractive") # we always need this
    
    # now, we can proceed with the remainder of our pipeline
    if task == "segment":
        segment.main(config)
    elif task == "register":
        register.main(config)
    elif task == "full":
        segment.main(config)
        register.main(config)
