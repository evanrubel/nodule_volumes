# Use `sybil2` as a conda environment.

import torch
from argparse import Namespace
import pickle
import pandas as pd
import json
import math
from tqdm import tqdm
import sys
import numpy as np
from segmentation_evaluator_new import NoduleSegmentEvaluator
# sys.path.append('/data/rbg/users/pgmikhael/current/SybilX')
sys.path.append('/data/rbg/users/erubel/sybil/SybilX')
from sybilx.utils.registry import get_object


def get_annotations_mask(row, annots, shape):
    slice_ids = [p.split('/')[-1].split('.dcm')[0] for p in row['paths']]
    mask = torch.zeros(shape)
    W, H = mask.shape[1:]
    for i, slice in enumerate(slice_ids):
        for bbox in annots.get(slice, []):
            x_left, y_top = bbox["x"] * W, bbox["y"] * H
            x_right, y_bottom = x_left + bbox["width"] * W, y_top + bbox["height"] * H
            x_left, y_top = math.floor(x_left), math.floor(y_top)
            x_right, y_bottom = math.ceil(x_right), math.ceil(y_bottom)
            mask[i,y_top:y_bottom, x_left:x_right] = 1
    return mask


def main():
    annotations = json.load(open("/data/rbg/shared/datasets/NLST/NLST/annotations_122020.json", "r"))
    args = Namespace(**pickle.load(open('/data/rbg/users/pgmikhael/current/SybilX/logs/c32cb085afbe045d58a7c83dcb71398c.args', 'rb')))
    nodule_dataset = pd.read_csv('/data/rbg/users/pgmikhael/current/SybilX/notebooks/NoduleGrowth/nlst_cancer_nodules.csv')
    dataset = get_object(args.dataset, 'dataset')(args, "test")

    evaluator = NoduleSegmentEvaluator(min_cluster_size=25)

    # mask_dir = "/data/rbg/scratch/lung_ct/e31840e7efe14a10472d817f8a14b27f" # BMP3D finetuned

    # mask_dir = "/data/rbg/scratch/lung_ct/epoch=7" # BMP2D finetuned
    # mask_dir = "/data/rbg/scratch/lung_ct/0f18c617a2f6b5a768d81c7465e206f2epoch=12" # TSM 1
    mask_dir = "/data/rbg/scratch/lung_ct/aeec028d12497e8dcd29cdf025dfb675epoch=0" # TSM 2

    print(f"\n\nProcessing for {mask_dir}...\n")

    nodule_identification = []
    dices = []
    row_to_id = {}
    for i, row in tqdm(enumerate(dataset.dataset), total=len(dataset.dataset), ncols=100):
        exam = row['exam']

        nodule_row = nodule_dataset[nodule_dataset['PID'] == int(row['pid'])]
        tp = row['screen_timepoint']

        if isinstance(nodule_row[f"Annotated_{tp}"].iloc[0], str): # has annotation
            annotated_sid = [s for s in nodule_row[f"Annotated_{tp}"].iloc[0].split(';') if s == row['series']]

            if len(annotated_sid) == 0: continue

            annots = annotations[annotated_sid[0]]

            segmentation = pickle.load(open(f"{mask_dir}/sample_{exam}.hiddens", "rb"))["hidden"][0]

            # nodule identification according to export annotation
            mask1 = get_annotations_mask(row, annots, segmentation.shape)

            nodule_identification.append(((mask1 * segmentation).sum() > 0).item())
            dices.append(
                evaluator.get_scan_wise_dice(mask1[None].numpy(), segmentation[None].numpy()).item()
            )
            row_to_id[f"{row['pid']}_{row['screen_timepoint']}"] = ((mask1 * segmentation).sum() > 0).item()
    
        with open("output.json", "w") as f:
            json.dump(row_to_id, f)

    print(f"Nodule Recall: {len([val for val in nodule_identification if val]) / len(nodule_identification)}")
    print(f"Mean Dice: {np.mean(dices).item()}\nMedian Dice: {np.median(dices).item()}\n")


if __name__ == "__main__":
    main()
