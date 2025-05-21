import os
import sys

sys.path.append(os.path.join(os.getcwd(), "segment", "scripts"))

from utils.postprocessing import generate_single_lung_mask

from totalsegmentator.python_api import totalsegmentator

import json
import nibabel as nib
import os
from tqdm import tqdm
import subprocess
from multiprocessing import Pool
from functools import partial

num_workers = 16


def generate_single_lung_mask_wrapper(fname, config):
    """Wrapper that handles try-except and returns skipped series_id if any."""
    series_id = fname.replace("_0000", "").replace(".nii.gz", "")

    try:
        if fname.endswith(".nii.gz"):
            if not os.path.exists(os.path.join(config["dataset_dir"], "lung_masks", f"{series_id}.nii.gz")):
                generate_single_lung_mask(fname, series_id, config)
        return None  # not skipped
    except Exception as e:
        print(f"Skipping {series_id} due to error: {str(e)}")
        return series_id


def generate_lung_masks(config: dict) -> None:
    """Generates all of the lung masks for the files in the input directory using multiprocessing."""

    print("\n\nGenerating lung masks...")

    all_files = [f for f in sorted(os.listdir(config["nifti_dir"])) if f.endswith(".nii.gz")]

    with Pool(processes=num_workers) as pool:
        func = partial(generate_single_lung_mask_wrapper, config=config)
        results = list(tqdm(pool.imap(func, all_files), total=len(all_files)))

    skipped = [r for r in results if r is not None]

    with open(os.path.join(config["dataset_dir"], "lung_masks", "skipped.json"), "w") as f:
        json.dump(skipped, f, indent=4)


def generate_lung_vessel_masks(config: dict) -> None:
    """Generates all of the lung vessel masks for the files in the input directory (except if the masks already exist.)"""

    skipped = []

    print("\n\nGenerating lung vessel masks...")

    for fname in tqdm(sorted(os.listdir(config["nifti_dir"]))):
        if fname.endswith(".nii.gz"):
            try:
                series_id = fname.replace("_0000", "").replace(".nii.gz", "")

                # only generate a lung vessel mask if it does not yet already exist
                if not os.path.exists(os.path.join(config["dataset_dir"], "lung_vessel_masks", f"{series_id}.nii.gz")):
                    image = nib.load(os.path.join(config["nifti_dir"], fname))

                    mask = totalsegmentator(image, task="lung_vessels", device=f"gpu:{config['device']}", quiet=(not config["debug"])).get_fdata()

                    nib.save(
                        nib.Nifti1Image(mask, affine=image.affine, header=image.header),
                        os.path.join(os.path.join(config["dataset_dir"], "lung_vessel_masks", f"{series_id}.nii.gz"))
                    )
            except Exception as e:
                print(f"Skipping {series_id} due to error {str(e)}...")
                skipped.append(series_id)
    
    with open(os.path.join(config["dataset_dir"], "lung_vessel_masks", "skipped.json"), "w") as f:
        json.dump(skipped, f, indent=4)
    
    print("\n\n")


def run_in_conda_env(env_name, script_path, config_path):
    command = f'bash -c "source ~/.bashrc && conda activate {env_name} && python {script_path} --config {config_path}"'
    result = subprocess.run(command, shell=True)


def main(config: dict) -> None:
    """The entry point for the segmentation task."""

    # Generate the lung masks if required and they do not already exist
    if config["lung_mask_mode"]:
        generate_lung_masks(config)

    # Generate the lung vessel masks if required and they do not already exist
    if config["lung_vessel_overlap_threshold"] is not None:
        generate_lung_vessel_masks(config)
    
    # sys.path.append(os.path.join(os.getcwd(), "segment", "scripts"))

    # # import biomedparse_plus_plus
    # # import nn_interactive # make sure not to conflict with the nnInteractive pip package

    # # Initial "detection" step with BiomedParse++
    # print("\n(1) Running BiomedParse++...\n\n")
    # # biomedparse_plus_plus.main(config)

    # exit()
    # # input("Enter to continue...")

    # # Segmentation step where we smooth the outputs with nnInteractive
    # print("\n(2) Running nnInteractive...\n\n")
    # # nn_interactive.main(config)

    config_file_path = os.path.join(config["output_dir"], "experiment_config.json")

    print("\n(1) Running BiomedParse++ in `biomedparse`...\n")
    run_in_conda_env("biomedparse", "segment/scripts/biomedparse_plus_plus.py", config_file_path)

    # print("\n(2) Running nnInteractive in `nnInteractive`...\n")
    # run_in_conda_env("nnInteractive", "segment/scripts/nn_interactive.py", config_file_path)
