import torch
from argparse import Namespace
import pickle
import pandas as pd
import json
import math
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
import time
import os
import SimpleITK as sitk
import nibabel as nib
import ants
import multiprocessing as mp

from utils import get_pid_to_timepoints


def apply_to_mask(
    mask_nifti_dir,
    target_pid,
    fixed_timepoint,
    moving_timepoint,
    fixed_mask_arr,
    moving_mask_arr,
    dataset_name,
):
    """Transform the moving mask to be in the same space as the fixed mask."""

    assert isinstance(fixed_mask_arr, np.ndarray) and isinstance(moving_mask_arr, np.ndarray)

    # handle the fixed image first
    fixed_image_path = get_image_path(target_pid, fixed_timepoint)
    fixed_image = nib.load(fixed_image_path)

    fixed_mask_path = os.path.join(mask_nifti_dir, f"{dataset_name}_{target_pid}T{fixed_timepoint}.nii.gz")

    # save fixed mask in same space as fixed image
    nib.save(
        nib.Nifti1Image(fixed_mask_arr.astype(np.uint8), affine=fixed_image.affine, header=fixed_image.header),
        fixed_mask_path
    )

    ####

    # now, handle the moving image
    moving_image = nib.load(get_image_path(target_pid, moving_timepoint))

    moving_mask_path = os.path.join(mask_nifti_dir, f"{dataset_name}_{target_pid}T{moving_timepoint}.nii.gz")

    # save moving mask in same space as moving image
    nib.save(
        nib.Nifti1Image(moving_mask_arr.astype(np.uint8), affine=moving_image.affine, header=moving_image.header),
        moving_mask_path
    )

    ####

    # finally, transform the mask!
    
    output_path = os.path.join(outputs_dir, f"{dataset_name}_{target_pid}T{moving_timepoint}.nii.gz")

    affine_path = os.path.join(transforms_dir, f"{target_pid}_T{moving_timepoint}_registered_to_T{fixed_timepoint}_0GenericAffine.mat")
    warp_path = os.path.join(transforms_dir, f"{target_pid}_T{moving_timepoint}_registered_to_T{fixed_timepoint}_1Warp.nii.gz")

    mask = ants.image_read(moving_mask_path)
    ref_image = ants.image_read(fixed_image_path)
    tx = [warp_path, affine_path] # warp (1) then generic affine (0)

    transformed_mask = ants.apply_transforms(
        fixed=ref_image,  # The reference image in the fixed space
        moving=mask,  # The segmentation mask in the moving space
        transformlist=tx,
        interpolator='nearestNeighbor'  # suggested
    )

    ants.image_write(transformed_mask, output_path)
    print(f"Transformed mask saved to {output_path}")


def get_mask_array(model, target_pid, timepoint, config) -> np.ndarray:
    """Returns the array corresponding to the mask."""

    return nib.load(os.path.join(config["output_dir"], f"{config['dataset_name']}_{target_pid}T{timepoint}_0000.nii.gz")).get_fdata()
    

def get_image_path(pid: int, timepoint: int, config: dict) -> str:
    """Returns the path for an image for a given PID and timepoint."""

    return os.path.join(config["nifti_dir"], f"{config['dataset_name']}_{pid}T{timepoint}_0000.nii.gz")


def process_pid(pid: int, num_timepoints: int) -> None:
    """Transform all of the patient masks based on the CT registration."""

    if num_timepoints == 1: # no need to do any registration, but we should still save the original mask for later
        keys = list(pid_to_timepoints[pid].keys())
        assert len(keys) == 1
        timepoint = keys[0]

        image = nib.load(get_image_path(pid, timepoint))

        mask_arr = get_mask_array(model, pid, timepoint)
        mask_path = os.path.join(mask_nifti_dir, f"{dataset_name}_{pid}T{timepoint}.nii.gz")
        
        nib.save(
            nib.Nifti1Image(mask_arr.astype(np.uint8), affine=image.affine, header=image.header),
            mask_path,
        )    
    elif num_timepoints == 2:
        fixed_timepoint = max(pid_to_timepoints[pid]) # fix the last one
        moving_timepoint = min(pid_to_timepoints[pid])

        fixed_mask_arr = get_mask_array(model, pid, fixed_timepoint)
        moving_mask_arr = get_mask_array(model, pid, moving_timepoint)

        apply_to_mask(mask_nifti_dir, pid, fixed_timepoint=fixed_timepoint, moving_timepoint=moving_timepoint, fixed_mask_arr=fixed_mask_arr, moving_mask_arr=moving_mask_arr)
    elif num_timepoints == 3:
        mask_arr0 = get_mask_array(model, pid, 0)
        mask_arr1 = get_mask_array(model, pid, 1)
        mask_arr2 = get_mask_array(model, pid, 2)

        apply_to_mask(mask_nifti_dir, pid, fixed_timepoint=2, moving_timepoint=1, fixed_mask_arr=mask_arr2, moving_mask_arr=mask_arr1)
        apply_to_mask(mask_nifti_dir, pid, fixed_timepoint=2, moving_timepoint=0, fixed_mask_arr=mask_arr2, moving_mask_arr=mask_arr0)
    

def process_wrapper(args):
    """For multiprocessing"""

    pid, num_timepoints = args
    try:
        process_pid(pid, num_timepoints)
    except Exception as e:  # catch-all for now
        print(f"Error {type(e).__name__}, {str(e)} with {pid} -- skipping it!")
        return pid
    return None


def main(config: dict) -> None:  
    # nn_interactive_outputs_dir = f"/data/scratch/erubel/nlst/nnInteractive/{model}"
    # niftis_dir = "/data/scratch/erubel/nlst/niftis"
    # transforms_dir = "/data/rbg/scratch/nlst_nodules/v1/transforms" # we use the cached transforms!
    # outputs_dir = f"/data/rbg/scratch/nlst_nodules/v2/registered_masks/{model}"
    # mask_nifti_dir = f"/data/rbg/scratch/nlst_nodules/v2/masks/{model}"

    niftis_dir = config["nifti_dir"]
    transforms_dir = config["transforms_dir"]
    outputs_dir = config["registered_masks_dir"]
    mask_nifti_dir = config["output_dir"] # do we need this?

    dataset_name = config["dataset_name"]

    pid_to_timepoints = get_pid_to_timepoints(config)

    skipped = []

    pids = sorted(pid_to_timepoints.keys())
    tasks = [(pid, len(pid_to_timepoints[pid])) for pid in pids]

    assert all(num_timepoints in {1, 2, 3} for _, num_timepoints in tasks)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_wrapper, tasks), total=len(tasks)):
            if result is not None:
                skipped.append(result)

    with open(os.path.join(outputs_dir, "mask_transformations_skipped.json"), "w") as f:
        json.dump(skipped, f, indent=4)
