# Step 2

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

import sys
sys.path.append('/data/rbg/users/erubel/sybil/SybilX')
from sybilx.utils.registry import get_object

annotations = json.load(open("/data/rbg/shared/datasets/NLST/NLST/annotations_122020.json", "r"))
args = Namespace(**pickle.load(open('/data/rbg/users/pgmikhael/current/SybilX/logs/c32cb085afbe045d58a7c83dcb71398c.args', 'rb')))

is_cancer = False

if is_cancer:
    nodule_dataset = pd.read_csv('/data/rbg/users/pgmikhael/current/SybilX/notebooks/NoduleGrowth/nlst_cancer_nodules.csv')
else:
    nodule_dataset = pd.read_csv('/data/rbg/users/pgmikhael/current/SybilX/notebooks/NoduleGrowth/nlst_benign_nodules_subset.csv')

dataset = get_object(args.dataset if is_cancer else 'nlst_benign_nodules', 'dataset')(args, "test")

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


if is_cancer:
    assert len(pid_to_exam_by_timepoint) == 788
    nifti_dir = "/data/scratch/erubel/nlst/niftis"
    transforms_dir = "/data/rbg/scratch/nlst_nodules/v1/transforms"
    outputs_dir = "/data/rbg/scratch/nlst_nodules/v1/register_outputs"
else:
    nifti_dir = "/data/rbg/scratch/nlst_benign_nodules/niftis"
    transforms_dir = "/data/rbg/scratch/nlst_benign_nodules/v1/transforms"
    outputs_dir = "/data/rbg/scratch/nlst_benign_nodules/v1/register_outputs"

    with open("/data/rbg/scratch/nlst_benign_nodules/v1/pid_to_exam_by_timepoint.p", "wb") as f:
        pickle.dump(pid_to_exam_by_timepoint, f)

group_num = 15 # TODO: change here!

total_num_groups = 16

start_ix = group_num * (len(pid_to_exam_by_timepoint) // total_num_groups)
stop_ix = ((group_num + 1) * (len(pid_to_exam_by_timepoint) // total_num_groups)) if group_num != total_num_groups - 1 else len(pid_to_exam_by_timepoint)

print(f"From {start_ix} to {stop_ix} (exclusive)...\n")

def register_scans(nifti_dir, transforms_dir, outputs_dir, target_pid, fixed_timepoint, moving_timepoint):
    fixed_image = ants.image_read(os.path.join(nifti_dir, f"nlst_{target_pid}T{fixed_timepoint}_0000.nii.gz"))
    moving_image = ants.image_read(os.path.join(nifti_dir, f"nlst_{target_pid}T{moving_timepoint}_0000.nii.gz"))

    result = ants.registration(
        fixed_image,
        moving_image,
        type_of_transform='SyN',
        outprefix=os.path.join(
            transforms_dir,
            f"{target_pid}_T{moving_timepoint}_registered_to_T{fixed_timepoint}")
    )

    for k in ["warpedmovout", "warpedfixout"]:
        ants.image_write(
            result[k],
            os.path.join(
                outputs_dir, 
                f"{target_pid}_T{moving_timepoint}_registered_to_T{fixed_timepoint}_{k}.nii.gz"
            )
        )
skipped = []

for pid in tqdm(sorted(list(pid_to_exam_by_timepoint.keys()))[start_ix: stop_ix]):
    if any([str(pid) in fname for fname in os.listdir(transforms_dir)]):
        print(f"Skipping {pid} because it's already there!")
        continue

    try:
        num_timepoints = len(pid_to_exam_by_timepoint[pid])

        assert num_timepoints in {1, 2, 3}

        if num_timepoints == 1: # no need to do anything
            continue
        elif num_timepoints == 2:
            fixed_timepoint = max(pid_to_exam_by_timepoint[pid]) # fix the last one
            moving_timepoint = min(pid_to_exam_by_timepoint[pid])

            register_scans(nifti_dir, transforms_dir, outputs_dir, pid, fixed_timepoint, moving_timepoint)
        elif num_timepoints == 3:
            register_scans(nifti_dir, transforms_dir, outputs_dir, pid, fixed_timepoint=2, moving_timepoint=1)
            register_scans(nifti_dir, transforms_dir, outputs_dir, pid, fixed_timepoint=2, moving_timepoint=0)
    except Exception as e: # catch-all for now
        print(f"Error {type(e).__name__ }, {str(e)} with {pid} -- skipping it!")
        skipped.append(pid)

with open(os.path.join(outputs_dir, f"registrations_skipped_group_{group_num}.json"), "w") as f:
    json.dump(skipped, f, indent=4)