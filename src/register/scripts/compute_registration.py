import torch
from argparse import Namespace
import pickle
import pandas as pd
import json
import math
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
from fast_hdbscan import HDBSCAN
import time
import os
import SimpleITK as sitk
import os
import ants

from utils import get_pid_to_timepoints


def register_scans(nifti_dir, transforms_dir, target_pid, fixed_timepoint, moving_timepoint, dataset_name):
    fixed_image = ants.image_read(os.path.join(nifti_dir, f"{dataset_name}_{target_pid}T{fixed_timepoint}_0000.nii.gz"))
    moving_image = ants.image_read(os.path.join(nifti_dir, f"{dataset_name}_{target_pid}T{moving_timepoint}_0000.nii.gz"))

    result = ants.registration(
        fixed_image,
        moving_image,
        type_of_transform='SyN',
        outprefix=os.path.join(
            transforms_dir,
            f"{target_pid}_T{moving_timepoint}_registered_to_T{fixed_timepoint}_")
    )

    for k in ["warpedmovout", "warpedfixout"]:
        ants.image_write(
            result[k],
            os.path.join(
                transforms_dir, 
                f"{target_pid}_T{moving_timepoint}_registered_to_T{fixed_timepoint}_{k}.nii.gz"
            )
        )


def main(config: dict):
    nifti_dir = config["nifti_dir"]

    transforms_dir = config["transforms_dir"]
    dataset_name = config["dataset_name"]

    pid_to_timepoints = get_pid_to_timepoints(config)
    
    print(pid_to_timepoints)

    skipped = []

    for pid in tqdm(sorted(list(pid_to_timepoints.keys()))):
        if any([str(pid) in fname for fname in os.listdir(transforms_dir)]):
            print(f"Skipping {pid} because it's already there!")
            continue

        try:
            num_timepoints = len(pid_to_timepoints[pid])

            assert num_timepoints in {1, 2, 3}

            if num_timepoints == 1: # no need to do anything
                continue
            elif num_timepoints == 2:
                fixed_timepoint = max(pid_to_timepoints[pid]) # fix the last one
                moving_timepoint = min(pid_to_timepoints[pid])

                register_scans(nifti_dir, transforms_dir, pid, fixed_timepoint, moving_timepoint, dataset_name)
            elif num_timepoints == 3:
                register_scans(nifti_dir, transforms_dir, pid, fixed_timepoint=2, moving_timepoint=1, dataset_name=dataset_name)
                register_scans(nifti_dir, transforms_dir, pid, fixed_timepoint=2, moving_timepoint=0, dataset_name=dataset_name)
        except Exception as e: # catch-all for now
            print(f"Error {type(e).__name__ }, {str(e)} with {pid} -- skipping it!")
            skipped.append(pid)

    with open(os.path.join(transforms_dir, "skipped.json"), "w") as f:
        json.dump(skipped, f, indent=4)