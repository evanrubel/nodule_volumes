# conda activate total

import os
import nibabel as nib
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm
import json
import numpy as np


def get_nifti_nib(series_id: str, config: dict):
    """Checks all subfolders and returns the matching Nibabel object (for `series_id`) within those subfolders."""
    
    if any(["group" in f for f in os.listdir(config["nifti_dir"])]):
        # look in subdivided groups
        for f in os.listdir(config["nifti_dir"]):
            if "group" in f and os.path.isdir(os.path.join(config["nifti_dir"], f)):
                for dir_fname in os.listdir(os.path.join(config["nifti_dir"], f)):
                    if dir_fname == f"{series_id}_0000.nii.gz":
                        return nib.load(os.path.join(config["nifti_dir"], f, dir_fname))
    elif len([f for f in os.listdir(config["nifti_dir"]) if f.endswith(".nii.gz")]) >= 1:
        # in one directory
        return nib.load(os.path.join(config["nifti_dir"], f"{series_id}_0000.nii.gz"))
    
    raise FileNotFoundError


def get_sorted_filenames(config: dict) -> list[str]:
    """Returns the sorted filenames to process based on `config.`"""

    all_filenames = []

    if any(["group" in f for f in os.listdir(config["nifti_dir"])]):
        # look in subdivided groups
        for f in os.listdir(config["nifti_dir"]):
            if "group" in f and os.path.isdir(os.path.join(config["nifti_dir"], f)):
                all_filenames.extend(os.listdir(os.path.join(config["nifti_dir"], f)))
    elif len([f for f in os.listdir(config["nifti_dir"]) if f.endswith(".nii.gz")]) >= 1:
        # in one directory
        all_filenames.extend([fname for fname in os.listdir(os.path.join(config["nifti_dir"])) if fname.endswith(".nii.gz")])
    else:
        raise NotImplementedError
        
    all_filenames_sorted = sorted(all_filenames)

    if config["group_ix"] == "all":
        return all_filenames_sorted
    else:
        assert isinstance(config["group_ix"], int) and isinstance(config["total_num_groups"], int)
        assert config["group_ix"] in range(config["total_num_groups"])

        start_ix = (len(all_filenames_sorted) // config["total_num_groups"]) * config["group_ix"]
        stop_ix = (len(all_filenames_sorted) // config["total_num_groups"]) * (config["group_ix"] + 1)

        if config["group_ix"] == config["total_num_groups"] - 1: # last one
            print(f"Going from {start_ix} to {len(all_filenames_sorted) - 1}...")
            return all_filenames_sorted[start_ix:]
        else:
            print(f"Going from {start_ix} to {stop_ix}...")
            return all_filenames_sorted[start_ix:stop_ix]

LUNG_LABEL = 1
NODULE_LABEL = 2

def main(config):
    skipped = []

    for fname in tqdm(get_sorted_filenames(config)):
        if fname.endswith(".nii.gz"):
            try:
                series_id = fname.replace("_0000.nii.gz", "")

                image = get_nifti_nib(series_id, config)

                mask = totalsegmentator(image, task=config["task"], device=f"gpu:{config['device']}").get_fdata()

                if config["task"] == "lung_nodules":
                    # only include nodules, not the lungs
                    mask = (mask == NODULE_LABEL).astype(np.uint8)

                if config["debug"]:
                    print(f"Mask Shape: {mask.shape}")
                
                nib.save(
                    nib.Nifti1Image(mask, affine=image.affine, header=image.header),
                    os.path.join(config["output_dir"], f"{series_id}.nii.gz"),
                )

            except Exception as e:
                print(f"Skipping {series_id} due to error {str(e)}...")
                skipped.append(series_id)
    
        with open(os.path.join(config["output_dir"], f"skipped_group_{config['group_ix']}.json"), "w") as f:
            json.dump(skipped, f, indent=4)
    
if __name__ == "__main__":
    config = {
        # "dataset": "utc",
        "dataset": "nlst",
        
        "debug": False,

        # "device": 1,
        # "device": 2,
        # "device": 3,
        "device": 4,

        # "group_ix": 4,
        # "group_ix": 5,
        # "group_ix": 6,
        "group_ix": 7,

        "total_num_groups": 8,
        "task": "lung_nodules",
    }

    assert config["task"] in {"lung_nodules", "lung_vessels"}

    if config["dataset"] == "nlst":
        config["nifti_dir"] = "/data/scratch/erubel/nlst/niftis"
        config["output_dir"] = os.path.join("/data/scratch/erubel/nlst", f"{config['task']}_total_segmentator")
    elif config["dataset"] == "utc":
        config["nifti_dir"] = "/data/scratch/erubel/external_val/utc/data"
        config["output_dir"] = os.path.join("/data/scratch/erubel/external_val/utc/experiments", f"{config['task']}_total_segmentator")

    os.makedirs(config["output_dir"], exist_ok=True)

    main(config)

