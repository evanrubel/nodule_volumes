# Use sybil2 as conda env

# Step 3

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

import sys
sys.path.append('/data/rbg/users/erubel/sybil/SybilX')
from sybilx.utils.registry import get_object

def apply_to_mask(
    mask_nifti_dir,
    target_pid,
    fixed_timepoint,
    moving_timepoint,
    fixed_exam,
    moving_exam,
    fixed_mask_arr,
    moving_mask_arr,
):
    """Transform the moving mask to be in the same space as the fixed mask."""

    assert isinstance(fixed_mask_arr, np.ndarray) and isinstance(moving_mask_arr, np.ndarray)

    # handle the fixed image first
    # fixed_image_path = os.path.join(niftis_dir, f"nlst_{target_pid}T{fixed_timepoint}_0000.nii.gz")
    fixed_image_path = get_image_path(target_pid, fixed_timepoint)
    fixed_image = nib.load(fixed_image_path)

    fixed_mask_path = os.path.join(mask_nifti_dir, f"nlst_{target_pid}T{fixed_timepoint}.nii.gz")

    # save fixed mask in same space as fixed image
    nib.save(
        nib.Nifti1Image(fixed_mask_arr.astype(np.uint8), affine=fixed_image.affine, header=fixed_image.header),
        fixed_mask_path
    )

    ####

    # now, handle the moving image
    # moving_image = nib.load(os.path.join(niftis_dir, f"nlst_{target_pid}T{moving_timepoint}_0000.nii.gz"))
    moving_image = nib.load(get_image_path(target_pid, moving_timepoint))

    moving_mask_path = os.path.join(mask_nifti_dir, f"nlst_{target_pid}T{moving_timepoint}.nii.gz")

    # save moving mask in same space as moving image
    nib.save(
        nib.Nifti1Image(moving_mask_arr.astype(np.uint8), affine=moving_image.affine, header=moving_image.header),
        moving_mask_path
    )


    ####

    # finally, transform the mask!
    
    output_path = os.path.join(outputs_dir, f"nlst_{target_pid}T{moving_timepoint}.nii.gz")

    affine_path = os.path.join(transforms_dir, f"{target_pid}_T{moving_timepoint}_registered_to_T{fixed_timepoint}0GenericAffine.mat")
    warp_path = os.path.join(transforms_dir, f"{target_pid}_T{moving_timepoint}_registered_to_T{fixed_timepoint}1Warp.nii.gz")

    mask = ants.image_read(moving_mask_path)
    ref_image = ants.image_read(fixed_image_path)
    tx = [warp_path, affine_path] # warp (1) then generic affine (0)

    transformed_mask = ants.apply_transforms(
        fixed=ref_image,  # The reference image in the fixed space
        moving=mask,  # The segmentation mask in the moving space
        transformlist=tx,
        interpolator='nearestNeighbor'  # suggested
        # interpolator='genericLabel' # use for label images according to docs
    )

    ants.image_write(transformed_mask, output_path)
    print(f"Transformed mask saved to {output_path}")


def get_mask_array(model, exam, target_pid, timepoint) -> np.ndarray:
    if "bmp" in model:
        # out = pickle.load(open(os.path.join(BMP_PATH, f"sample_{exam}.hiddens"), "rb")) # bmp3d
        return pickle.load(open(os.path.join(BMP_PATH, f"sample_{exam}.hiddens"), "rb"))["hidden"].numpy()[0] # bmp2d_finetuned

    elif "unet" in model:
        if model == "unet_combined":
            mask_dir = "/data/scratch/erubel/nlst/combined_inference_evan"
        elif model == "unet_luna":
            mask_dir = "/data/scratch/erubel/nlst/luna_inference"
        elif model == "unet_lndb":
            mask_dir = "/data/scratch/erubel/nlst/lndb_inference"
        
        return nib.load(os.path.join(mask_dir, f"nlst_{target_pid}T{timepoint}.nii.gz")).get_fdata()

    elif "nn" in model or "tsm" in model:
        for group_dir_name in os.listdir(nn_interactive_outputs_dir):
            proposed_path = os.path.join(nn_interactive_outputs_dir, group_dir_name, f"nlst_{target_pid}T{timepoint}_output.nii.gz")

            if os.path.isfile(proposed_path):
                return nib.load(proposed_path).get_fdata()

        raise FileNotFoundError


def get_image_path(pid: int, timepoint: int) -> str:
    """Returns the path for an image for a given PID and timepoint."""

    # no group divisions!
    if len([fname for fname in os.listdir(niftis_dir) if fname.endswith(".nii.gz")]) > 0:
        return os.path.join(niftis_dir, f"nlst_{pid}T{timepoint}_0000.nii.gz")

    for group_dir_name in os.listdir(niftis_dir):
        proposed_path = os.path.join(niftis_dir, group_dir_name, f"nlst_{pid}T{timepoint}_0000.nii.gz")
        if os.path.isfile(proposed_path):
            return proposed_path

    raise FileNotFoundError


def process_pid(pid: int, num_timepoints: int) -> None:
    """Transform all of the patient masks based on the CT registration."""

    if num_timepoints == 1: # no need to do any registration, but we should still save the original mask for later
        keys = list(pid_to_exam_by_timepoint[pid].keys())
        assert len(keys) == 1
        timepoint = keys[0]
        exam = pid_to_exam_by_timepoint[pid][timepoint]["exam"]

        image = nib.load(get_image_path(pid, timepoint))

        mask_arr = get_mask_array(model, exam, pid, timepoint)
        mask_path = os.path.join(mask_nifti_dir, f"nlst_{pid}T{timepoint}.nii.gz")
        
        nib.save(
            nib.Nifti1Image(mask_arr.astype(np.uint8), affine=image.affine, header=image.header),
            mask_path,
        )    
    elif num_timepoints == 2:
        fixed_timepoint = max(pid_to_exam_by_timepoint[pid]) # fix the last one
        moving_timepoint = min(pid_to_exam_by_timepoint[pid])

        fixed_exam = pid_to_exam_by_timepoint[pid][fixed_timepoint]["exam"]
        moving_exam = pid_to_exam_by_timepoint[pid][moving_timepoint]["exam"]

        fixed_mask_arr = get_mask_array(model, fixed_exam, pid, fixed_timepoint)
        moving_mask_arr = get_mask_array(model, moving_exam, pid, moving_timepoint)

        apply_to_mask(mask_nifti_dir, pid, fixed_timepoint=fixed_timepoint, moving_timepoint=moving_timepoint, fixed_exam=fixed_exam, moving_exam=moving_exam, fixed_mask_arr=fixed_mask_arr, moving_mask_arr=moving_mask_arr)
    elif num_timepoints == 3:
        exam0 = pid_to_exam_by_timepoint[pid][0]["exam"]
        exam1 = pid_to_exam_by_timepoint[pid][1]["exam"]
        exam2 = pid_to_exam_by_timepoint[pid][2]["exam"]

        mask_arr0 = get_mask_array(model, exam0, pid, 0)
        mask_arr1 = get_mask_array(model, exam1, pid, 1)
        mask_arr2 = get_mask_array(model, exam2, pid, 2)

        apply_to_mask(mask_nifti_dir, pid, fixed_timepoint=2, moving_timepoint=1, fixed_exam=exam2, moving_exam=exam1, fixed_mask_arr=mask_arr2, moving_mask_arr=mask_arr1)
        apply_to_mask(mask_nifti_dir, pid, fixed_timepoint=2, moving_timepoint=0, fixed_exam=exam2, moving_exam=exam0, fixed_mask_arr=mask_arr2, moving_mask_arr=mask_arr0)
    

def process_wrapper(args):
    """For multiprocessing"""

    pid, num_timepoints = args
    try:
        process_pid(pid, num_timepoints)
    except Exception as e:  # catch-all for now
        print(f"Error {type(e).__name__}, {str(e)} with {pid} -- skipping it!")
        return pid
    return None


def get_pid_to_exam_by_timepoint(nodule_dataset, dataset) -> dict:
    """Returns a mapping between PID and exams."""

    pid_to_exam_by_timepoint = {}
    for i, row in tqdm(enumerate(dataset.dataset), total=len(dataset.dataset), ncols=100):
        exam = row['exam']

        nodule_row = nodule_dataset[nodule_dataset['PID'] == int(row['pid'])]
        tp = row['screen_timepoint']

        paths = row['paths']
        pid = int(row['pid'])

        if pid in pid_to_exam_by_timepoint:
            pid_to_exam_by_timepoint[pid][int(tp)] = {'paths': paths, 'exam': exam}
        else:
            pid_to_exam_by_timepoint[pid] = {int(tp): {'paths': paths, 'exam': exam}}

    print(len(pid_to_exam_by_timepoint))

    return pid_to_exam_by_timepoint


if __name__ == "__main__":
    annotations = json.load(open("/data/rbg/shared/datasets/NLST/NLST/annotations_122020.json", "r"))
    args = Namespace(**pickle.load(open('/data/rbg/users/pgmikhael/current/SybilX/logs/c32cb085afbe045d58a7c83dcb71398c.args', 'rb')))
    
    is_cancer = False
    if is_cancer:
        nodule_dataset = pd.read_csv('/data/rbg/users/pgmikhael/current/SybilX/notebooks/NoduleGrowth/nlst_cancer_nodules.csv')
    else:
        nodule_dataset = pd.read_csv('/data/rbg/users/pgmikhael/current/SybilX/notebooks/NoduleGrowth/nlst_benign_nodules_subset.csv')

    dataset = get_object(args.dataset if is_cancer else 'nlst_benign_nodules', 'dataset')(args, "test")

    pid_to_exam_by_timepoint = get_pid_to_exam_by_timepoint(nodule_dataset, dataset)
    
    # model = "nnInteractive_all_mask_min_1"
    # model = "nnInteractive_all_mask_min_2"
    # model = "nnInteractive_all_mask_min_3"
    # model = "tsm_25dfb675_nn_all_mask_min_1_no_mask"
    # model = "tsm_25dfb675_nn_all_mask_min_1"
    model = "tsm_25dfb675"

    if is_cancer:
        # nn_interactive_outputs_dir = "/data/scratch/erubel/nlst/nnInteractive/bmp2d_nn_all_mask_min_1"
        # nn_interactive_outputs_dir = "/data/scratch/erubel/nlst/nnInteractive/bmp2d_nn_all_mask_min_2"
        # nn_interactive_outputs_dir = "/data/scratch/erubel/nlst/nnInteractive/bmp2d_nn_all_mask_min_3"
        # nn_interactive_outputs_dir = "/data/scratch/erubel/nlst/nnInteractive/tsm_25dfb675_nn_all_mask_min_1_no_mask"
        # nn_interactive_outputs_dir = "/data/scratch/erubel/nlst/nnInteractive/tsm_25dfb675_nn_all_mask_min_1"

        nn_interactive_outputs_dir = f"/data/scratch/erubel/nlst/nnInteractive/{model}"

        niftis_dir = "/data/scratch/erubel/nlst/niftis"
        # TODO: move these transforms to a different dir
        transforms_dir = "/data/rbg/scratch/nlst_nodules/v1/transforms" # we use the cached transforms!
        outputs_dir = f"/data/rbg/scratch/nlst_nodules/v2/registered_masks/{model}"
        mask_nifti_dir = f"/data/rbg/scratch/nlst_nodules/v2/masks/{model}"
    else:
        nn_interactive_outputs_dir = f"/data/scratch/erubel/nlst_benign/nnInteractive/{model}"

        niftis_dir = "/data/rbg/scratch/nlst_benign_nodules/niftis"
        # TODO: move these transforms to a different dir
        transforms_dir = "/data/rbg/scratch/nlst_benign_nodules/v1/transforms" # we use the cached transforms!
        outputs_dir = f"/data/rbg/scratch/nlst_benign_nodules/v2/registered_masks/{model}"
        mask_nifti_dir = f"/data/rbg/scratch/nlst_benign_nodules/v2/masks/{model}"
    
    # group_num = 0
    # group_num = 1
    # group_num = 2
    # group_num = 3
    # group_num = 4
    # group_num = 5
    group_num = 6
    # group_num = 7

    total_num_groups = 8

    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(mask_nifti_dir, exist_ok=True)

    skipped = []
    pids = sorted(pid_to_exam_by_timepoint.keys())
    tasks = [(pid, len(pid_to_exam_by_timepoint[pid])) for pid in pids]

    start_ix = group_num * (len(tasks) // total_num_groups)
    stop_ix = ((group_num + 1) * (len(tasks) // total_num_groups)) if group_num != total_num_groups - 1 else len(tasks)

    tasks = tasks[start_ix:stop_ix]

    print(f"From {start_ix} to {stop_ix} (exclusive)...\n")
    print(tasks[0])
    print(tasks[-1])

    assert all(num_timepoints in {1, 2, 3} for _, num_timepoints in tasks)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_wrapper, tasks), total=len(tasks)):
            if result is not None:
                skipped.append(result)

    with open(os.path.join(outputs_dir, f"mask_transformations_skipped_{model}_group_{group_num}.json"), "w") as f:
        json.dump(skipped, f, indent=4)
