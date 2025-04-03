# Step 1

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
import multiprocessing as mp

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
        pid_to_exam_by_timepoint[pid][int(tp)] = {'paths': paths}
    else:
        pid_to_exam_by_timepoint[pid] = {int(tp): {'paths': paths}}

skipped = []

print(len(pid_to_exam_by_timepoint))

if is_cancer:
    output_dir = "/data/scratch/erubel/nlst/niftis"
else:
    output_dir = "/data/rbg/scratch/nlst_benign_nodules/niftis"

os.makedirs(output_dir, exist_ok=True)

def process_series(args):
    pid, timepoint, dicom_folder, output_dir = args
    series_id = f"nlst_{pid}T{timepoint}_0000"
    try:
        # Reorients to LAS orientation
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)
        reader.SetFileNames(dicom_names)
        image = reader.Execute()

        # Change the axes of the data
        image = sitk.PermuteAxes(image, [2, 1, 0])
        
        sitk.WriteImage(image, os.path.join(output_dir, f'{series_id}.nii.gz'))
    except Exception as e:
        return f"Error {type(e).__name__}, {str(e)} with {series_id} -- skipping it!"
    return None

tasks = []
for pid in sorted(list(pid_to_exam_by_timepoint.keys())):
    if int(pid) >= 211980:
        # TODO: remove this!
        for timepoint in pid_to_exam_by_timepoint[pid]:
            dicom_folder = os.path.dirname(pid_to_exam_by_timepoint[pid][timepoint]["paths"][0])
            tasks.append((pid, timepoint, dicom_folder, output_dir))

skipped = []
with mp.Pool(mp.cpu_count()) as pool:
    for result in tqdm(pool.imap_unordered(process_series, tasks), total=len(tasks)):
        if result is not None:
            skipped.append(result)

with open(os.path.join(output_dir, "nlst_dicoms_skipped.json"), "w") as f:
    json.dump(skipped, f, indent=4)
