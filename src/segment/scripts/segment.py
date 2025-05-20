from totalsegmentator.python_api import totalsegmentator
from utils.postprocessing import generate_single_lung_mask

import nibabel as nib
import os
import sys


def generate_lung_masks(config: dict) -> None:
    """Generates all of the lung masks for the files in the input directory (except if the masks already exist.)"""

    skipped = []

    for fname in tqdm(sorted(os.listdir(config["nifti_dir"]))):
        if fname.endswith(".nii.gz"):
            try:
                series_id = fname.replace("_0000", "").replace(".nii.gz", "")

                # only generate a lung mask if it does not yet already exist
                if not os.path.exists(os.path.join(config["dataset_dir"], "lung_masks", f"{series_id}.npy")):
                    generate_single_lung_mask(fname, series_id, config)
                
            except Exception as e:
                print(f"Skipping {series_id} due to error {str(e)}...")
                skipped.append(series_id)
    
    with open(os.path.join(config["dataset_dir"], "lung_masks", "skipped.json"), "w") as f:
        json.dump(skipped, f, indent=4)


def generate_lung_vessel_masks(config: dict) -> None:
    """Generates all of the lung vessel masks for the files in the input directory (except if the masks already exist.)"""

    skipped = []

    for fname in tqdm(sorted(os.listdir(config["nifti_dir"]))):
        if fname.endswith(".nii.gz"):
            try:
                series_id = fname.replace("_0000", "").replace(".nii.gz", "")

                # only generate a lung vessel mask if it does not yet already exist
                if not os.path.exists(os.path.join(config["dataset_dir"], "lung_vessel_masks", f"{series_id}.nii.gz")):
                    image = nib.load(os.path.join(config["nifti_dir"], fname))

                    mask = totalsegmentator(image, task="lung_vessels", device=f"gpu:{config['device']}").get_fdata()

                    nib.save(
                        nib.Nifti1Image(mask, affine=image.affine, header=image.header),
                        os.path.join(os.path.join(config["dataset_dir"], "lung_vessel_masks", f"{series_id}.nii.gz"))
                    )
                
            except Exception as e:
                print(f"Skipping {series_id} due to error {str(e)}...")
                skipped.append(series_id)
    
    with open(os.path.join(config["dataset_dir"], "lung_vessel_masks", "skipped.json"), "w") as f:
        json.dump(skipped, f, indent=4)


def main(config: dict) -> None:
    """The entry point for the segmentation task."""

    sys.path.append(os.path.join(os.getcwd(), "segment", "scripts"))

    import biomedparse_plus_plus
    # import nn_interactive # make sure not to conflict with the nnInteractive pip package

    # Generate the lung masks if they do not already exist
    if config["lung_mask_mode"]:
        generate_lung_masks(config)

    # Generate the lung vessel masks if required and they do not already exist
    if config["lung_vessel_overlap_threshold"] is not None:
        generate_lung_vessel_masks(config)

    # Initial "detection" step with BiomedParse++
    print("\n(1) Running BiomedParse++...\n\n")
    biomedparse_plus_plus.main(config)

    exit()

    # Segmentation step where we smooth the outputs with nnInteractive
    print("\n(1) Running nnInteractive...\n\n")
    nn_interactive.main(config)
