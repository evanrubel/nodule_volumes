# Step 4

import torch
from argparse import Namespace
import pickle
import pandas as pd
import json
import math
from tqdm import tqdm
from typing import List, Tuple, Union
import numpy as np
from fast_hdbscan import HDBSCAN
import time
import os
import SimpleITK as sitk
import nibabel as nib
import ants
from scipy.spatial.distance import cdist
import sys
import copy


# from segmentation_evaluator import NoduleSegmentEvaluator, compute_volume_voxel_count

# sys.path.append('/data/rbg/users/erubel/sybil/SybilX')
# from sybilx.utils.registry import get_object


def get_mask_array(mask_dir: str, exam: str, target_pid: int, timepoint: int) -> np.ndarray:
    """Returns the corresponding NLST mask array in `mask_dir`."""

    return nib.load(os.path.join(mask_dir, f"nlst_{target_pid}T{timepoint}.nii.gz")).get_fdata()


def get_nonzero_slices(mask_arr: np.ndarray) -> list[int]:
    """Returns the slice numbers in `mask_arr` that have at least one non-zero pixel on them."""

    return sorted(np.where(np.any(mask_arr, axis=(1, 2)))[0].tolist())


def get_annotations_mask(paths, annots, shape) -> np.ndarray:
    """Returns a mask that has "1-pixels" within the annotation boxes."""

    slice_ids = [p.split('/')[-1].split('.dcm')[0] for p in paths]
    mask = torch.zeros(shape)

    if annots is not None:
        _, W, H = mask.shape
        for i, slice_ in enumerate(slice_ids):
            for bbox in annots.get(slice_, []):
                x_left, y_top = bbox["x"] * W, bbox["y"] * H
                x_right, y_bottom = x_left + bbox["width"] * W, y_top + bbox["height"] * H
                x_left, y_top = math.floor(x_left), math.floor(y_top)
                x_right, y_bottom = math.ceil(x_right), math.ceil(y_bottom)
                mask[i,y_top:y_bottom, x_left:x_right] = 1
    return mask.numpy()


def deregister_moving_mask(
    niftis_dir: str,
    registered_moving_masks_dir: str,
    deregistered_mask_nifti_dir: str,
    transforms_dir: str,
    moving_mask_instance_arr_registered: np.ndarray,
    target_pid: int,
    fixed_timepoint: int,
    moving_timepoint: int
) -> np.ndarray:
    """Reverts a registered moving instance mask back to its original de-registered state, while preserving the instance labels, and saves it."""

    # fixed_image_path = os.path.join(niftis_dir, f"nlst_{target_pid}T{fixed_timepoint}_0000.nii.gz")
    fixed_image_path = get_image_path(target_pid, fixed_timepoint, niftis_dir)
    # moving_image_path = os.path.join(niftis_dir, f"nlst_{target_pid}T{moving_timepoint}_0000.nii.gz")
    moving_image_path = get_image_path(target_pid, moving_timepoint, niftis_dir)

    # binary version (registered)
    moving_mask_image_registered_ants = ants.image_read(os.path.join(registered_moving_masks_dir, f"nlst_{target_pid}T{moving_timepoint}.nii.gz"))

    # instance version (registered) from clustering with metadata from binary registered version
    moving_mask_instance_arr_registered_image = ants.from_numpy(
        moving_mask_instance_arr_registered,
        origin=moving_mask_image_registered_ants.origin,
        spacing=moving_mask_image_registered_ants.spacing,
        direction=moving_mask_image_registered_ants.direction
    )

    affine_path = os.path.join(transforms_dir, f"{target_pid}_T{moving_timepoint}_registered_to_T{fixed_timepoint}_0GenericAffine.mat")
    inverse_warp_path = os.path.join(transforms_dir, f"{target_pid}_T{moving_timepoint}_registered_to_T{fixed_timepoint}_1InverseWarp.nii.gz")

    mask_arr = ants.apply_transforms(
        fixed=ants.image_read(moving_image_path),  # reference (original *moving* image)
        moving=moving_mask_instance_arr_registered_image,  # input (registered moving mask) with instance labels!
        transformlist=[affine_path, inverse_warp_path], # reverses the order of the original list
        whichtoinvert=[True, False],  # invert only the affine transform
        interpolator="nearestNeighbor",
    ).numpy().astype(np.uint8)

    # save it
    # moving_image = nib.load(os.path.join(niftis_dir, f"nlst_{target_pid}T{moving_timepoint}_0000.nii.gz"))
    moving_image = nib.load(get_image_path(target_pid, moving_timepoint, niftis_dir))
    nib.save(
        nib.Nifti1Image(mask_arr, affine=moving_image.affine, header=moving_image.header),
        os.path.join(deregistered_mask_nifti_dir, f"nlst_{target_pid}T{moving_timepoint}.nii.gz"),
    )   

    return mask_arr


def register_mask(target_pid: int, moving_timepoint: int, fixed_timepoint: int, moving_mask_arr: np.ndarray, niftis_dir: str, original_mask_nifti_dir: str, transforms_dir: str) -> np.ndarray:
    """We use this to check whether bounding boxes overlap after registration."""

    fixed_image_path = get_image_path(target_pid, fixed_timepoint, niftis_dir)

    affine_path = os.path.join(transforms_dir, f"{target_pid}_T{moving_timepoint}_registered_to_T{fixed_timepoint}_0GenericAffine.mat")
    warp_path = os.path.join(transforms_dir, f"{target_pid}_T{moving_timepoint}_registered_to_T{fixed_timepoint}_1Warp.nii.gz")

    moving_mask_image_ants = ants.image_read(os.path.join(original_mask_nifti_dir, f"nlst_{target_pid}T{moving_timepoint}.nii.gz"))
    
    moving_mask = ants.from_numpy(
        moving_mask_arr,
        origin=moving_mask_image_ants.origin,
        spacing=moving_mask_image_ants.spacing,
        direction=moving_mask_image_ants.direction
    )

    ref_image = ants.image_read(fixed_image_path)
    tx = [warp_path, affine_path] # warp (1) then generic affine (0)

    return ants.apply_transforms(
        fixed=ref_image,  # The reference image in the fixed space
        moving=moving_mask,  # The segmentation mask in the moving space
        transformlist=tx,
        interpolator='nearestNeighbor'  # suggested
    ).numpy().astype(np.uint8)



def get_scan_spacing(
    image_dir: str,
    mask_dir: str,
    exam: str,
    pid: int,
    timepoint: int,
) -> tuple[list[float], float, tuple[int]]:
    """Returns the spacing for the given scan."""

    print("Replace nlst with dataset_name everywhere")
    raise NotImplementedError

    # image = nib.load(os.path.join(image_dir, f"nlst_{pid}T{timepoint}_0000.nii.gz"))
    image = nib.load(get_image_path(pid, timepoint, image_dir))
    mask = nib.load(os.path.join(mask_dir, f"nlst_{pid}T{timepoint}.nii.gz"))
    
    assert image.get_fdata().shape == mask.get_fdata().shape, "Expected shapes to match."
    assert image.header.get_zooms() == mask.header.get_zooms(), "Expected spacing values to match."

    sx, sy, sz = image.header.get_zooms()

    pixel_spacing = [sx, sy]
    slice_thickness = sz
    shape = image.get_fdata().shape

    return pixel_spacing, slice_thickness, shape


def compute_centroid(mask: np.ndarray, label: int) -> Union[np.ndarray, None]:
    """Computes the centroid (center of mass) of a given cluster in a 3D mask."""
    indices = np.argwhere(mask == label)  # Get voxel (z, y, x) positions
    return np.mean(indices, axis=0) if indices.size > 0 else None


def compute_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Returns the IOU between two masks."""

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


def compute_min_mask_distance(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute the minimum distance between any point in `mask1` and any point in `mask2`."""

    # coordinates of nonzero points
    coords1 = np.column_stack(np.where(mask1))
    coords2 = np.column_stack(np.where(mask2))
    
    if coords1.size == 0 or coords2.size == 0:
        return float('inf')
    
    distances = cdist(coords1, coords2)
    return distances.min()


def has_clusters(mask: np.ndarray, segmentation_evaluator: NoduleSegmentEvaluator) -> bool:
    """Returns whether the semantic `mask` has any clusters that we could potentially match with other scan masks."""

    if np.count_nonzero(mask) == 0: # nothing to cluster
        return False

    instance_mask = segmentation_evaluator.get_instance_segmentation(mask)

    return len(np.unique(instance_mask).tolist()) >= 2 # has more than zero values!


def match_clusters(
    labels_src: np.ndarray,
    centroids_src: np.ndarray,
    mask_src: np.ndarray,
    labels_tgt: np.ndarray,
    centroids_tgt: np.ndarray,
    mask_tgt: np.ndarray,
    dist_thresh: float
) -> tuple[list, list, list]:
    """Matches clusters from the source mask to the target mask by minimum centroid distance."""

    labels_src = labels_src.tolist()
    labels_tgt = labels_tgt.tolist()

    # if not labels_src or not labels_tgt:
    #     return []

    # distance_matrix = cdist(centroids_src, centroids_tgt)
    
    # matches = []
    # matched_labels_tgt = set()  # track assigned labels in target mask

    # for i, label_src in enumerate(labels_src):
    #     j = np.argmin(distance_matrix[i, :])  # index of closest cluster in target mask
    #     min_centroid_distance = float(distance_matrix[i, j])
    #     label_tgt = labels_tgt[j]

    #     if min_centroid_distance < dist_thresh and label_tgt not in matched_labels_tgt:
    #         binary_instance_src_mask = (mask_src == label_src).astype(np.uint8)
    #         binary_instance_tgt_mask = (mask_tgt == label_tgt).astype(np.uint8)
    #         iou_val = compute_iou(binary_instance_src_mask, binary_instance_tgt_mask)
    #         min_distance = compute_min_mask_distance(binary_instance_src_mask, binary_instance_tgt_mask) # between any two points
    #         matches.append((label_src, label_tgt, min_centroid_distance, min_distance, iou_val))
    #         matched_labels_tgt.add(label_tgt)  # mark cluster as assigned

    # return matches

    if not labels_src:
        return [], [], labels_tgt  # No source labels, all target labels are unmatched
    if not labels_tgt:
        return [], labels_src, []  # No target labels, all source labels are unmatched

    distance_matrix = cdist(centroids_src, centroids_tgt)
    
    matches = []
    matched_labels_src = set()
    matched_labels_tgt = set()

    for i, label_src in enumerate(labels_src):
        j = np.argmin(distance_matrix[i, :])  # index of closest cluster in target mask
        min_centroid_distance = float(distance_matrix[i, j])
        label_tgt = labels_tgt[j]

        if min_centroid_distance < dist_thresh and label_tgt not in matched_labels_tgt:
            binary_instance_src_mask = (mask_src == label_src).astype(np.uint8)
            binary_instance_tgt_mask = (mask_tgt == label_tgt).astype(np.uint8)
            iou_val = compute_iou(binary_instance_src_mask, binary_instance_tgt_mask)
            min_distance = compute_min_mask_distance(binary_instance_src_mask, binary_instance_tgt_mask)
            matches.append((label_src, label_tgt, min_centroid_distance, min_distance, iou_val))
            matched_labels_src.add(label_src)
            matched_labels_tgt.add(label_tgt)
    
    unmatched_src = [label for label in labels_src if label not in matched_labels_src]
    unmatched_tgt = [label for label in labels_tgt if label not in matched_labels_tgt]

    return matches, unmatched_src, unmatched_tgt


def volume(mask: np.ndarray, label: int, pixel_spacing: tuple[float], slice_thickness: float) -> float:
    """Wrapper function for computing the volume via a voxel count."""

    return compute_volume_voxel_count((mask == label).astype(np.uint8), pixel_spacing, slice_thickness)


def run_matching(
    mask1: np.ndarray,
    mask2: np.ndarray,
    mask3: Union[np.ndarray, None] = None,
    dist_thresh: int = 5,
) -> tuple[list, list, int]:
    """
    Matches clusters either:
    - Between two masks (`mask1 ↔ mask2`), or
    - Across three masks (`mask3 → mask2 → mask1` if `mask3` is provided).

    The arguments are in *reverse-chronological* order.

    Returns the matches, the unmatched nodules (in the same order as passed in), and the size of each grouping.
    """

    labels1 = np.unique(mask1[mask1 > 0]).astype(np.uint8)
    labels2 = np.unique(mask2[mask2 > 0]).astype(np.uint8)
    centroids1 = np.array([compute_centroid(mask1, l) for l in labels1])
    centroids2 = np.array([compute_centroid(mask2, l) for l in labels2])

    # print("MaskFixed", labels1)
    # print("MaskMoving", labels2)

    # Step 1: Match mask2 → mask1
    match_21, unmatched2_from_21, unmatched1_from_21 = match_clusters(labels2, centroids2, mask2, labels1, centroids1, mask1, dist_thresh)

    # print("UnmatchedFixed", unmatched1_from_21)
    # print("UnmatchedMoving", unmatched2_from_21)
    
    if mask3 is None:
        return [(label1, label2, centroid_dist21, min_dist21, iou21) for (label2, label1, centroid_dist21, min_dist21, iou21) in match_21], [unmatched1_from_21, unmatched2_from_21, []], 2

    # If mask3 is provided, match mask3 → mask2
    labels3 = np.unique(mask3[mask3 > 0]).astype(np.uint8)
    centroids3 = np.array([compute_centroid(mask3, l) for l in labels3])
    match_32, unmatched3_from_32, unmatched2_from_32 = match_clusters(labels3, centroids3, mask3, labels2, centroids2, mask2, dist_thresh)

    # print("MATCH32")
    # print(match_32)
    # print(unmatched2_from_32)
    # print(unmatched3_from_32)

    # Step 3: Chain match_32 and match_21 to get mask3 → mask2 → mask1
    results = []
    for (label3, label2, centroid_dist32, min_dist32, iou32) in match_32:
        match_21_entry = next((m for m in match_21 if m[0] == label2), None)
        if match_21_entry:
            label1, centroid_dist21, min_dist21, iou21 = match_21_entry[1], match_21_entry[2], match_21_entry[3], match_21_entry[4]
        else:
            label1, centroid_dist21, min_dist21, iou21 = None, None, None, None  # placeholder if no match in mask1

        results.append((label1, label2, label3, centroid_dist21, centroid_dist32, min_dist21, min_dist32, iou21, iou32))

    # include unmatched mask2 → mask1 matches with placeholder for mask3
    matched_label2s = {m[1] for m in match_32}  # labels from mask2 that got a match in mask3
    for (label2, label1, centroid_dist21, min_dist21, iou21) in match_21:
        if label2 not in matched_label2s:
            results.append((label1, label2, None, centroid_dist21, None, min_dist21, None, iou21, None))  # no match in mask3
    
    if len(results) == 0:
        # try to match directly mask3 → mask1
        match_31, unmatched3_from_31, unmatched1_from_31 = match_clusters(labels3, centroids3, mask3, labels1, centroids1, mask1, dist_thresh)
        # print("MATCH31")
        # print(match_31)
        # print(unmatched1_from_31)
        # print(unmatched3_from_31)
        # return [(label1, label3, centroid_dist31, min_dist31, iou31) for (label3, label1, centroid_dist31, min_dist31, iou31) in match_31], [unmatched1_from_31, labels2.astype(int).tolist(), unmatched3_from_31], 2 # return all labels2 since there are no matches
        return [(label1, label3, centroid_dist31, min_dist31, iou31) for (label3, label1, centroid_dist31, min_dist31, iou31) in match_31], [unmatched1_from_31, labels2.astype(int).tolist(), list(set(unmatched3_from_31) | set(unmatched3_from_32))], 2 # return all labels2 since there are no matches
    else:
        matched_labels2_from_21 = {label2 for (label2, _, _, _, _) in match_21}
        matched_labels2_from_32 = {label2 for (_, label2, _, _, _) in match_32}
        # return results, [unmatched1_from_21, list((set(unmatched2_from_21) | set(unmatched2_from_32)) - matched_labels_from_21), unmatched3_from_32]  # format: (label1, label2, label3, centroid_dist21, centroid_dist32, min_dist21, min_dist32, iou21, iou32)
        return results, [unmatched1_from_21, list((set(unmatched2_from_21) | set(unmatched2_from_32)) - matched_labels2_from_21 - matched_labels2_from_32), unmatched3_from_32], 3  # format: (label1, label2, label3, centroid_dist21, centroid_dist32, min_dist21, min_dist32, iou21, iou32)


# def run_matching(
#     mask1: np.ndarray,
#     mask2: np.ndarray,
#     mask3: Union[np.ndarray, None] = None,
#     dist_thresh: int = 5,
# ) -> tuple[list, list, int]:
#     """
#     Matches clusters either:
#     - Between two masks (`mask1 ↔ mask2`), or
#     - Across three masks (`mask3 → mask2 → mask1` if `mask3` is provided).

#     The arguments are in *reverse-chronological* order.

#     Returns the matches, the unmatched nodules (in the same order as passed in), and the size of each grouping.
#     """

#     labels1 = np.unique(mask1[mask1 > 0]).astype(np.uint8)
#     labels2 = np.unique(mask2[mask2 > 0]).astype(np.uint8)
#     centroids1 = np.array([compute_centroid(mask1, l) for l in labels1])
#     centroids2 = np.array([compute_centroid(mask2, l) for l in labels2])

#     # Step 1: Match mask2 → mask1
#     match_21, unmatched2_from_21, unmatched1_from_21 = match_clusters(labels2, centroids2, mask2, labels1, centroids1, mask1, dist_thresh)

#     if mask3 is None:
#         return [(label1, label2, centroid_dist21, min_dist21, iou21) for (label2, label1, centroid_dist21, min_dist21, iou21) in match_21], [unmatched1_from_21, unmatched2_from_21, []], 2

#     # If mask3 is provided, match mask3 → mask2
#     labels3 = np.unique(mask3[mask3 > 0]).astype(np.uint8)
#     centroids3 = np.array([compute_centroid(mask3, l) for l in labels3])
#     match_32, unmatched3_from_32, unmatched2_from_32 = match_clusters(labels3, centroids3, mask3, labels2, centroids2, mask2, dist_thresh)

#     # Step 3: Chain match_32 and match_21 to get mask3 → mask2 → mask1
#     results = []
#     matched_label2s = set()
    
#     for (label3, label2, centroid_dist32, min_dist32, iou32) in match_32:
#         match_21_entry = next((m for m in match_21 if m[0] == label2), None)
#         if match_21_entry:
#             label1, centroid_dist21, min_dist21, iou21 = match_21_entry[1], match_21_entry[2], match_21_entry[3], match_21_entry[4]
#         else:
#             label1, centroid_dist21, min_dist21, iou21 = None, None, None, None  # placeholder if no match in mask1

#         results.append((label1, label2, label3, centroid_dist21, centroid_dist32, min_dist21, min_dist32, iou21, iou32))
#         matched_label2s.add(label2)

#     # Include unmatched mask2 → mask1 matches with placeholder for mask3
#     for (label2, label1, centroid_dist21, min_dist21, iou21) in match_21:
#         if label2 not in matched_label2s:
#             results.append((label1, label2, None, centroid_dist21, None, min_dist21, None, iou21, None))

#     # Step 4: Compute unmatched nodules correctly
#     if len(results) == 0:
#         match_31, unmatched3_from_31, unmatched1_from_31 = match_clusters(labels3, centroids3, mask3, labels1, centroids1, mask1, dist_thresh)
#         return [(label1, label3, centroid_dist31, min_dist31, iou31) for (label3, label1, centroid_dist31, min_dist31, iou31) in match_31], [
#             unmatched1_from_31, labels2.astype(int).tolist(), list(set(unmatched3_from_31) | set(unmatched3_from_32))
#             # list(set(unmatched1_from_31) | set(unmatched1_from_21)), labels2.astype(int).tolist(), list(set(unmatched3_from_31) | set(unmatched3_from_32))
#         ], 2

#     # Remove matched label2s from unmatched2 lists
#     unmatched2 = list(set(unmatched2_from_21) | set(unmatched2_from_32) - matched_label2s)
#     unmatched3 = list(set(unmatched3_from_32) - {label3 for (label1, label2, label3, _, _, _, _, _, _) in results if label3 is not None})

#     return results, [unmatched1_from_21, unmatched2, unmatched3], 3


def run_one_scan(
    pid: int,
    timepoint: int,
    pid_to_timepoints: dict,
    evaluator_new: NoduleSegmentEvaluator,
    niftis_dir: str,
    original_mask_nifti_dir: str,
    instance_mask_nifti_dir: str,
) -> Union[dict, str]:

    exam = pid_to_timepoints[pid][timepoint]["exam"]
    paths = pid_to_timepoints[pid][timepoint]["paths"]

    mask_instance_arr = evaluator_new.get_instance_segmentation(get_mask_array(original_mask_nifti_dir, exam, pid, timepoint))

    # save it
    # image = nib.load(os.path.join(niftis_dir, f"nlst_{pid}T{timepoint}_0000.nii.gz"))
    image = nib.load(get_image_path(pid, timepoint, niftis_dir))
    nib.save(
        nib.Nifti1Image(mask_instance_arr, affine=image.affine, header=image.header),
        os.path.join(instance_mask_nifti_dir, f"nlst_{pid}T{timepoint}.nii.gz"),
    )   

    pixel_spacing, slice_thickness, shape = get_scan_spacing(niftis_dir, original_mask_nifti_dir, exam, pid, timepoint)

    bbox_annotations = pid_to_timepoints[pid][timepoint]["bbox_annotations"]
    bbox_mask = get_annotations_mask(paths, bbox_annotations, shape)

    cluster_ids = np.unique(mask_instance_arr).astype(int).tolist()

    outputs = {
        "matched_nodules": {}, # nodule_idx (0...n) -> {}
        "unmatched_nodules": {}, # nodule_idx (n+1...m) -> {} [include tmp]
        "bboxes": {
            f"T{timepoint}_has_bboxes": np.count_nonzero(bbox_mask) > 0,
        },
        "reg_metrics": {},
        "total_num_voxels": {
            f"T{timepoint}": np.count_nonzero(mask_instance_arr),
        }, # Tx -> int
        "total_num_nodules": {
            f"T{timepoint}": len([id_ for id_ in cluster_ids if id_ != 0]),
        }, # Tx -> int
        "num_scans": 1,
    }

    i = 0
    for label in sorted(cluster_ids):
        if label != 0: # exclude the background
            outputs["unmatched_nodules"][i] = { # nodule_idx is the nodule label here
                "mask_volume": volume(mask_instance_arr, label, pixel_spacing, slice_thickness),
                "instance_label": label,
                "has_bbox_overlap": ((bbox_mask * (mask_instance_arr == label).astype(np.uint8)).sum() > 0).item(),
                "mask_slice_range": get_nonzero_slices((mask_instance_arr == label).astype(np.uint8)),
                "bbox_slice_range": get_nonzero_slices(bbox_mask * (mask_instance_arr == label).astype(np.uint8)),
                "timepoint":  f"T{timepoint}",
            }
            i += 1

    return outputs


def run_two_scans(
    pid: int,
    pid_to_timepoints: dict,
    evaluator_new: NoduleSegmentEvaluator,
    niftis_dir: str,
    original_mask_nifti_dir: str,
    instance_mask_nifti_dir: str,
    registered_mask_nifti_dir: str,
    registered_instance_mask_nifti_dir: str,
    deregistered_mask_nifti_dir: str,
    transforms_dir: str,
    dist_thresh: float,
) -> Union[dict, str]:
    # TODO: change fixed to second, moving to first (for clarity)
    fixed_timepoint = max(pid_to_timepoints[pid]) # fix the last one
    moving_timepoint = min(pid_to_timepoints[pid])

    fixed_exam = pid_to_timepoints[pid][fixed_timepoint]["exam"]
    moving_exam = pid_to_timepoints[pid][moving_timepoint]["exam"]

    fixed_paths = pid_to_timepoints[pid][fixed_timepoint]["paths"]
    moving_paths = pid_to_timepoints[pid][moving_timepoint]["paths"]

    fixed_mask_instance_arr = evaluator_new.get_instance_segmentation(get_mask_array(original_mask_nifti_dir, fixed_exam, pid, fixed_timepoint))
    # save it
    # fixed_image = nib.load(os.path.join(niftis_dir, f"nlst_{pid}T{fixed_timepoint}_0000.nii.gz"))
    fixed_image = nib.load(get_image_path(pid, fixed_timepoint, niftis_dir))
    nib.save(
        nib.Nifti1Image(fixed_mask_instance_arr, affine=fixed_image.affine, header=fixed_image.header),
        os.path.join(instance_mask_nifti_dir, f"nlst_{pid}T{fixed_timepoint}.nii.gz"),
    )  

    fixed_pixel_spacing, fixed_slice_thickness, fixed_shape = get_scan_spacing(niftis_dir, original_mask_nifti_dir, fixed_exam, pid, fixed_timepoint)

    fixed_bbox_annotations = pid_to_timepoints[pid][fixed_timepoint]["bbox_annotations"]
    fixed_bbox_mask = get_annotations_mask(fixed_paths, fixed_bbox_annotations, fixed_shape)
    
    moving_mask_instance_arr_registered = evaluator_new.get_instance_segmentation(get_mask_array(registered_mask_nifti_dir, moving_exam, pid, moving_timepoint))
    # save it
    # moving_image = nib.load(os.path.join(niftis_dir, f"nlst_{pid}T{moving_timepoint}_0000.nii.gz"))
    moving_image = nib.load(get_image_path(pid, moving_timepoint, niftis_dir))
    nib.save(
        nib.Nifti1Image(moving_mask_instance_arr_registered, affine=moving_image.affine, header=moving_image.header),
        os.path.join(registered_instance_mask_nifti_dir, f"nlst_{pid}T{moving_timepoint}.nii.gz"),
    )
    
    # the spacing is from the unregistered (original) moving image!
    moving_pixel_spacing, moving_slice_thickness, moving_shape = get_scan_spacing(niftis_dir, original_mask_nifti_dir, moving_exam, pid, moving_timepoint)

    moving_bbox_annotations = pid_to_timepoints[pid][moving_timepoint]["bbox_annotations"]
    moving_bbox_mask = get_annotations_mask(moving_paths, moving_bbox_annotations, moving_shape)
    
    matches, unmatched, match_length = run_matching(
        fixed_mask_instance_arr,
        moving_mask_instance_arr_registered,
        dist_thresh=dist_thresh,
    )

    assert match_length == 2

    # this needs to have the same labels as the registered moving mask (instance)
    moving_mask_instance_arr_deregistered = deregister_moving_mask(
        niftis_dir,
        registered_mask_nifti_dir,
        deregistered_mask_nifti_dir,
        transforms_dir,
        moving_mask_instance_arr_registered,
        pid,
        fixed_timepoint=fixed_timepoint,
        moving_timepoint=moving_timepoint,
    )

    outputs = {
        "matched_nodules": {}, # nodule_idx (0...n) -> {}
        "unmatched_nodules": {}, # nodule_idx (n+1...m) -> {} [include tmp]
        "bboxes": {},
        "reg_metrics": {},
        "total_num_voxels": {
            f"T{moving_timepoint}": np.count_nonzero(moving_mask_instance_arr_registered),
            f"T{fixed_timepoint}": np.count_nonzero(fixed_mask_instance_arr),
        },
        "total_num_nodules": {
            f"T{moving_timepoint}": len([id_ for id_ in np.unique(moving_mask_instance_arr_registered).astype(int).tolist() if id_ != 0]),
            f"T{fixed_timepoint}": len([id_ for id_ in np.unique(fixed_mask_instance_arr).astype(int).tolist() if id_ != 0]),
        },
        "num_scans": 2,
    }

    # print("MATCHES")
    # print(matches)

    # print("UNMATCHED")
    # print(unmatched)

    # handle matches
    for nodule_idx, (fixed_label, moving_label, centroid_distance, min_distance, iou_val) in enumerate(matches):
        outputs["matched_nodules"][nodule_idx] = {}

        moving_volume = volume(moving_mask_instance_arr_deregistered, moving_label, moving_pixel_spacing, moving_slice_thickness)
        fixed_volume = volume(fixed_mask_instance_arr, fixed_label, fixed_pixel_spacing, fixed_slice_thickness)

        if moving_volume > 0:
            outputs["matched_nodules"][nodule_idx][f"T{moving_timepoint}"] = {
                # use the unregistered spacing
                "mask_volume": moving_volume,
                "instance_label": moving_label,
                f"centroid_distance_to_T{fixed_timepoint}_nodule": centroid_distance,
                f"min_distance_to_T{fixed_timepoint}_nodule": min_distance,
                f"iou_with_T{fixed_timepoint}_nodule_after_reg": iou_val,
                "has_bbox_overlap": ((moving_bbox_mask * (moving_mask_instance_arr_deregistered == moving_label).astype(np.uint8)).sum() > 0).item(),
                "mask_slice_range": get_nonzero_slices((moving_mask_instance_arr_deregistered == moving_label).astype(np.uint8)),
                "bbox_slice_range": get_nonzero_slices(moving_bbox_mask * (moving_mask_instance_arr_deregistered == moving_label).astype(np.uint8)),
            }

        if fixed_volume > 0:
            outputs["matched_nodules"][nodule_idx][f"T{fixed_timepoint}"] = {
                "mask_volume": fixed_volume,
                "instance_label": fixed_label,
                f"centroid_distance_to_T{moving_timepoint}_nodule": centroid_distance,
                f"min_distance_to_T{moving_timepoint}_nodule": min_distance,
                f"iou_with_registered_T{moving_timepoint}_nodule": iou_val,
                "has_bbox_overlap": ((fixed_bbox_mask * (fixed_mask_instance_arr == fixed_label).astype(np.uint8)).sum() > 0).item(),
                "mask_slice_range": get_nonzero_slices((fixed_mask_instance_arr == fixed_label).astype(np.uint8)),
                "bbox_slice_range": get_nonzero_slices(fixed_bbox_mask * (fixed_mask_instance_arr == fixed_label).astype(np.uint8)),
            }
        
    fixed_unmatched, moving_unmatched, _ = unmatched

    # handle unmatched nodules
    i = 0
    for moving_label in moving_unmatched:
        moving_volume = volume(moving_mask_instance_arr_deregistered, moving_label, moving_pixel_spacing, moving_slice_thickness)

        if moving_volume > 0:
            outputs["unmatched_nodules"][i] = {
                # use the unregistered spacing
                "mask_volume": moving_volume,
                "instance_label": moving_label,
                "has_bbox_overlap": ((moving_bbox_mask * (moving_mask_instance_arr_deregistered == moving_label).astype(np.uint8)).sum() > 0).item(),
                "mask_slice_range": get_nonzero_slices((moving_mask_instance_arr_deregistered == moving_label).astype(np.uint8)),
                "bbox_slice_range": get_nonzero_slices(moving_bbox_mask * (moving_mask_instance_arr_deregistered == moving_label).astype(np.uint8)),
                "timepoint": f"T{moving_timepoint}"
            }
            i += 1
    
    for fixed_label in fixed_unmatched:
        fixed_volume = volume(fixed_mask_instance_arr, fixed_label, fixed_pixel_spacing, fixed_slice_thickness)

        if fixed_volume > 0:
            outputs["unmatched_nodules"][i] = {
                "mask_volume": fixed_volume,
                "instance_label": fixed_label,
                "has_bbox_overlap": ((fixed_bbox_mask * (fixed_mask_instance_arr == fixed_label).astype(np.uint8)).sum() > 0).item(),
                "mask_slice_range": get_nonzero_slices((fixed_mask_instance_arr == fixed_label).astype(np.uint8)),
                "bbox_slice_range": get_nonzero_slices(fixed_bbox_mask * (fixed_mask_instance_arr == fixed_label).astype(np.uint8)),
                "timepoint": f"T{fixed_timepoint}"
            }
            i += 1



        # # now, we want to store all nodules that are not matched
        # moving_nodules = list(run_one_scan( 
        #     pid,
        #     moving_timepoint,
        #     pid_to_timepoints,
        #     evaluator_new,
        #     niftis_dir,
        #     original_mask_nifti_dir,
        #     instance_mask_nifti_dir,    
        # )["unmatched_nodules"].values())

        # fixed_nodules = list(run_one_scan(
        #     pid,
        #     fixed_timepoint,
        #     pid_to_timepoints,
        #     evaluator_new,
        #     niftis_dir,
        #     original_mask_nifti_dir,
        #     instance_mask_nifti_dir,
        # )["unmatched_nodules"].values())

        # all_unmatched_nodules = []

        # for tp, single_nodule_entries in [(moving_timepoint, moving_nodules), (fixed_timepoint, fixed_nodules)]:
        #     for single_nodule_entry in single_nodule_entries:
        #         is_already_matched = False
        #         for nodule_idx in outputs["matched_nodules"]:
        #             matched_nodule_entry = outputs["matched_nodules"][nodule_idx].get(f"T{tp}", {})

        #             # print(matched_nodule_entry)
        #             # print(single_nodule_entry)

        #             # don't check timepoint because we do not store that in the matched entry
        #             if all([matched_nodule_entry[k] == single_nodule_entry[k] for k in single_nodule_entry if k != "timepoint"]):
        #                 is_already_matched = True
                
        #         if not is_already_matched:
        #             all_unmatched_nodules.append(single_nodule_entry)

        # for nodule_idx, nodule_entry in enumerate(all_unmatched_nodules):
        #     outputs["unmatched_nodules"][nodule_idx] = nodule_entry

    # # handle unmatched
    # if len(matches) == 0:
    #     moving_nodules = list(run_one_scan( 
    #         pid,
    #         moving_timepoint,
    #         pid_to_timepoints,
    #         evaluator_new,
    #         niftis_dir,
    #         original_mask_nifti_dir,
    #         instance_mask_nifti_dir,    
    #     )["unmatched_nodules"].values())

    #     fixed_nodules = list(run_one_scan(
    #         pid,
    #         fixed_timepoint,
    #         pid_to_timepoints,
    #         evaluator_new,
    #         niftis_dir,
    #         original_mask_nifti_dir,
    #         instance_mask_nifti_dir,
    #     )["unmatched_nodules"].values())

    #     for nodule_idx, nodule_entry in enumerate(moving_nodules + fixed_nodules): # we ignore the key (original nodule idx)
    #         outputs["unmatched_nodules"][nodule_idx] = nodule_entry

    # else:
    #     for nodule_idx, (fixed_label, moving_label, centroid_distance, min_distance, iou_val) in enumerate(matches):
    #         outputs["matched_nodules"][nodule_idx] = {
    #             f"T{moving_timepoint}": {
    #                 # use the unregistered spacing
    #                 "mask_volume": volume(moving_mask_instance_arr_deregistered, moving_label, moving_pixel_spacing, moving_slice_thickness),
    #                 "instance_label": moving_label,
    #                 f"centroid_distance_to_T{fixed_timepoint}_nodule": centroid_distance,
    #                 f"min_distance_to_T{fixed_timepoint}_nodule": min_distance,
    #                 f"iou_with_T{fixed_timepoint}_nodule_after_reg": iou_val,
    #                 "has_bbox_overlap": ((moving_bbox_mask * (moving_mask_instance_arr_deregistered == moving_label).astype(np.uint8)).sum() > 0).item(),
    #                 "mask_slice_range": get_nonzero_slices((moving_mask_instance_arr_deregistered == moving_label).astype(np.uint8)),
    #                 "bbox_slice_range": get_nonzero_slices(moving_bbox_mask * (moving_mask_instance_arr_deregistered == moving_label).astype(np.uint8)),
    #             },
    #             f"T{fixed_timepoint}": {
    #                 "mask_volume": volume(fixed_mask_instance_arr, fixed_label, fixed_pixel_spacing, fixed_slice_thickness),
    #                 "instance_label": fixed_label,
    #                 f"centroid_distance_to_T{moving_timepoint}_nodule": centroid_distance,
    #                 f"min_distance_to_T{moving_timepoint}_nodule": min_distance,
    #                 f"iou_with_registered_T{moving_timepoint}_nodule": iou_val,
    #                 "has_bbox_overlap": ((fixed_bbox_mask * (fixed_mask_instance_arr == fixed_label).astype(np.uint8)).sum() > 0).item(),
    #                 "mask_slice_range": get_nonzero_slices((fixed_mask_instance_arr == fixed_label).astype(np.uint8)),
    #                 "bbox_slice_range": get_nonzero_slices(fixed_bbox_mask * (fixed_mask_instance_arr == fixed_label).astype(np.uint8)),
    #             },
    #         }
        

    #     # now, we want to store all nodules that are not matched
    #     moving_nodules = list(run_one_scan( 
    #         pid,
    #         moving_timepoint,
    #         pid_to_timepoints,
    #         evaluator_new,
    #         niftis_dir,
    #         original_mask_nifti_dir,
    #         instance_mask_nifti_dir,    
    #     )["unmatched_nodules"].values())

    #     fixed_nodules = list(run_one_scan(
    #         pid,
    #         fixed_timepoint,
    #         pid_to_timepoints,
    #         evaluator_new,
    #         niftis_dir,
    #         original_mask_nifti_dir,
    #         instance_mask_nifti_dir,
    #     )["unmatched_nodules"].values())

    #     all_unmatched_nodules = []

    #     for tp, single_nodule_entries in [(moving_timepoint, moving_nodules), (fixed_timepoint, fixed_nodules)]:
    #         for single_nodule_entry in single_nodule_entries:
    #             is_already_matched = False
    #             for nodule_idx in outputs["matched_nodules"]:
    #                 matched_nodule_entry = outputs["matched_nodules"][nodule_idx].get(f"T{tp}", {})

    #                 # print(matched_nodule_entry)
    #                 # print(single_nodule_entry)

    #                 # don't check timepoint because we do not store that in the matched entry
    #                 if all([matched_nodule_entry[k] == single_nodule_entry[k] for k in single_nodule_entry if k != "timepoint"]):
    #                     is_already_matched = True
                
    #             if not is_already_matched:
    #                 all_unmatched_nodules.append(single_nodule_entry)

    #     for nodule_idx, nodule_entry in enumerate(all_unmatched_nodules):
    #         outputs["unmatched_nodules"][nodule_idx] = nodule_entry
    
    # compute the IOU between all bounding boxes (if they exist)
    # TODO: maybe split by individual bounding boxes?
    outputs["bboxes"] = {
        f"T{moving_timepoint}_has_bboxes": np.count_nonzero(moving_bbox_mask) > 0,
        f"T{fixed_timepoint}_has_bboxes": np.count_nonzero(fixed_bbox_mask) > 0,
    }

    if outputs["bboxes"][f"T{moving_timepoint}_has_bboxes"] and outputs["bboxes"][f"T{fixed_timepoint}_has_bboxes"]:
        registered_moving_bbox_mask = register_mask(
            pid,
            moving_timepoint,
            fixed_timepoint,
            moving_bbox_mask,
            niftis_dir,
            original_mask_nifti_dir,
            transforms_dir
        )    

        outputs["reg_metrics"][f"bbox_mask_iou_for_T{moving_timepoint}_registered_and_T{fixed_timepoint}"] = compute_iou(
                registered_moving_bbox_mask, fixed_bbox_mask
            )
    else:
        outputs["reg_metrics"][f"bbox_mask_iou_for_T{moving_timepoint}_registered_and_T{fixed_timepoint}"] = None
    
    return outputs


def run_three_scans(
    pid: int,
    pid_to_timepoints: dict,
    evaluator_new: NoduleSegmentEvaluator,
    niftis_dir: str,
    original_mask_nifti_dir: str,
    instance_mask_nifti_dir: str,
    registered_mask_nifti_dir: str,
    registered_instance_mask_nifti_dir: str,
    deregistered_mask_nifti_dir: str,
    transforms_dir: str,
    dist_thresh: float,
) -> dict:
    exam0 = pid_to_timepoints[pid][0]["exam"]
    exam1 = pid_to_timepoints[pid][1]["exam"]
    exam2 = pid_to_timepoints[pid][2]["exam"]

    paths0 = pid_to_timepoints[pid][0]["paths"]
    paths1 = pid_to_timepoints[pid][1]["paths"]
    paths2 = pid_to_timepoints[pid][2]["paths"]

    # note that we use `registered_mask_nifti_dir` for masks 0 and 1 (moving and then registered) and `original_mask_nifti_dir` for mask 2 (fixed, not registered)

    mask_instance_arr0_registered = evaluator_new.get_instance_segmentation(get_mask_array(registered_mask_nifti_dir, exam0, pid, 0))
    mask_instance_arr1_registered = evaluator_new.get_instance_segmentation(get_mask_array(registered_mask_nifti_dir, exam1, pid, 1))
    mask_instance_arr2 = evaluator_new.get_instance_segmentation(get_mask_array(original_mask_nifti_dir, exam2, pid, 2))

    # save them!
    # image0 = nib.load(os.path.join(niftis_dir, f"nlst_{pid}T0_0000.nii.gz"))
    # image1 = nib.load(os.path.join(niftis_dir, f"nlst_{pid}T1_0000.nii.gz"))
    # image2 = nib.load(os.path.join(niftis_dir, f"nlst_{pid}T2_0000.nii.gz"))

    image0 = nib.load(get_image_path(pid, 0, niftis_dir))
    image1 = nib.load(get_image_path(pid, 1, niftis_dir))
    image2 = nib.load(get_image_path(pid, 2, niftis_dir))

    nib.save(
        nib.Nifti1Image(mask_instance_arr0_registered, affine=image0.affine, header=image0.header),
        os.path.join(registered_instance_mask_nifti_dir, f"nlst_{pid}T0.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(mask_instance_arr1_registered, affine=image1.affine, header=image1.header),
        os.path.join(registered_instance_mask_nifti_dir, f"nlst_{pid}T1.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(mask_instance_arr2, affine=image2.affine, header=image2.header),
        os.path.join(instance_mask_nifti_dir, f"nlst_{pid}T2.nii.gz"),
    )

    pixel_spacing0, slice_thickness0, shape0 = get_scan_spacing(niftis_dir, original_mask_nifti_dir, exam0, pid, 0)
    pixel_spacing1, slice_thickness1, shape1 = get_scan_spacing(niftis_dir, original_mask_nifti_dir, exam1, pid, 1)
    pixel_spacing2, slice_thickness2, shape2 = get_scan_spacing(niftis_dir, original_mask_nifti_dir, exam2, pid, 2)

    # all bbox masks and annotations are in the original images (NOT registered)
    bbox_annotations0 = pid_to_timepoints[pid][0]["bbox_annotations"]
    bbox_mask0 = get_annotations_mask(paths0, bbox_annotations0, shape0)

    bbox_annotations1 = pid_to_timepoints[pid][1]["bbox_annotations"]
    bbox_mask1 = get_annotations_mask(paths1, bbox_annotations1, shape1)

    bbox_annotations2 = pid_to_timepoints[pid][2]["bbox_annotations"]
    bbox_mask2 = get_annotations_mask(paths2, bbox_annotations2, shape2)

    matches, unmatched, match_length = run_matching(
        mask_instance_arr2,
        mask_instance_arr1_registered,
        mask_instance_arr0_registered,
        dist_thresh=dist_thresh,
    )

    assert match_length in {2, 3}

    outputs = {
        "matched_nodules": {}, # nodule_idx (0...n) -> {}
        "unmatched_nodules": {}, # nodule_idx (n+1...m) -> {} [include tmp]
        "bboxes": {},
        "reg_metrics": {},
        "total_num_voxels": {
            "T0": np.count_nonzero(mask_instance_arr0_registered),
            "T1": np.count_nonzero(mask_instance_arr1_registered),
            "T2": np.count_nonzero(mask_instance_arr2),
        },
        "total_num_nodules": {
            "T0": len([id_ for id_ in np.unique(mask_instance_arr0_registered).astype(int).tolist() if id_ != 0]),
            "T1": len([id_ for id_ in np.unique(mask_instance_arr1_registered).astype(int).tolist() if id_ != 0]),
            "T2": len([id_ for id_ in np.unique(mask_instance_arr2).astype(int).tolist() if id_ != 0]),
        },
        "num_scans": 3,
    }

    # matches orders of arguments above
    # this needs to have the same labels as the registered moving mask (instance)
    mask_instance_arr0_deregistered = deregister_moving_mask(
        niftis_dir,
        registered_mask_nifti_dir,
        deregistered_mask_nifti_dir,
        transforms_dir,
        mask_instance_arr0_registered,
        pid,
        fixed_timepoint=2,
        moving_timepoint=0,
    )
    mask_instance_arr1_deregistered = deregister_moving_mask(
        niftis_dir,
        registered_mask_nifti_dir,
        deregistered_mask_nifti_dir,
        transforms_dir,
        mask_instance_arr1_registered,
        pid,
        fixed_timepoint=2,
        moving_timepoint=1,
    )

    # if len(matches) == 0: # no matches
    #     nodules0 = list(run_one_scan( 
    #         pid,
    #         0,
    #         pid_to_timepoints,
    #         evaluator_new,
    #         niftis_dir,
    #         original_mask_nifti_dir,
    #         instance_mask_nifti_dir,    
    #     )["unmatched_nodules"].values())

    #     nodules1 = list(run_one_scan(
    #         pid,
    #         1,
    #         pid_to_timepoints,
    #         evaluator_new,
    #         niftis_dir,
    #         original_mask_nifti_dir,
    #         instance_mask_nifti_dir,
    #     )["unmatched_nodules"].values())

    #     nodules2 = list(run_one_scan(
    #         pid,
    #         2,
    #         pid_to_timepoints,
    #         evaluator_new,
    #         niftis_dir,
    #         original_mask_nifti_dir,
    #         instance_mask_nifti_dir,
    #     )["unmatched_nodules"].values())

    #     for nodule_idx, nodule_entry in enumerate(nodules0 + nodules1 + nodules2): # we ignore the key (original nodule idx)
    #         outputs["unmatched_nodules"][nodule_idx] = nodule_entry
            
    # else:

    # handle matches
    # if len(unmatched) == 2: # matches of two
    if match_length == 2: # matches of two
        for nodule_idx, (label2, label0, centroid_distance02, min_distance02, iou02) in enumerate(matches):
            outputs["matched_nodules"][nodule_idx] = {}

            volume0 = volume(mask_instance_arr0_deregistered, label0, pixel_spacing0, slice_thickness0)
            volume2 = volume(mask_instance_arr2, label2, pixel_spacing2, slice_thickness2)

            if volume0 > 0:
                outputs["matched_nodules"][nodule_idx]["T0"] = {
                        # use the deregistered mask and the unregistered spacing
                        "mask_volume": volume0,
                        "instance_label": label0,
                        "centroid_distance_to_T2_nodule": centroid_distance02,
                        "min_distance_to_T2_nodule": min_distance02,
                        "iou_with_T2_nodule_after_reg": iou02,
                        "has_bbox_overlap": ((bbox_mask0 * (mask_instance_arr0_deregistered == label0).astype(np.uint8)).sum() > 0).item(),
                        "mask_slice_range": get_nonzero_slices((mask_instance_arr0_deregistered == label0).astype(np.uint8)),
                        "bbox_slice_range": get_nonzero_slices(bbox_mask0 * (mask_instance_arr0_deregistered == label0).astype(np.uint8)),
                    }

            if volume2 > 0:
                outputs["matched_nodules"][nodule_idx]["T2"] = {
                    "mask_volume": volume2,
                    "instance_label": label2,
                    "centroid_distance_to_T0_nodule": centroid_distance02,
                    "min_distance_to_T0_nodule": min_distance02,
                    "iou_with_registered_T0_nodule": iou02,
                    "has_bbox_overlap": ((bbox_mask2 * (mask_instance_arr2 == label2).astype(np.uint8)).sum() > 0).item(),
                    "mask_slice_range": get_nonzero_slices((mask_instance_arr2 == label2).astype(np.uint8)),
                    "bbox_slice_range": get_nonzero_slices(bbox_mask2 * (mask_instance_arr2 == label2).astype(np.uint8)),
                }

            # # handle the missed middle one
            # if has_clusters(mask_instance_arr1_deregistered, evaluator_new):
            #     outputs |= run_one_scan( # treat as a single case if no matches
            #                             pid,
            #                             1,
            #                             pid_to_timepoints,
            #                             evaluator_new,
            #                             niftis_dir,
            #                             original_mask_nifti_dir,
            #                             instance_mask_nifti_dir,
            #                             )
                        
            # else:
            #     outputs["T1"] = {"info": f"no_clusters ({np.count_nonzero(mask_instance_arr1_deregistered)} total voxels)"}

        # nodules1 = list(run_one_scan(
        #     pid,
        #     1,
        #     pid_to_timepoints,
        #     evaluator_new,
        #     niftis_dir,
        #     original_mask_nifti_dir,
        #     instance_mask_nifti_dir,
        # )["unmatched_nodules"].values())

        # for nodule_idx, nodule_entry in enumerate(nodules1): # we ignore the key (original nodule idx)
        #     outputs["unmatched_nodules"][nodule_idx] = nodule_entry
        
    # elif len(unmatched) == 3: # matches of three
    elif match_length == 3: # matches of three
        for nodule_idx, (label2, label1, label0, centroid_distance12, centroid_distance01, min_distance12, min_distance01, iou12, iou01) in enumerate(matches):
            outputs["matched_nodules"][nodule_idx] = {}

            volume0 = volume(mask_instance_arr0_deregistered, label0, pixel_spacing0, slice_thickness0)
            volume1 = volume(mask_instance_arr1_deregistered, label1, pixel_spacing1, slice_thickness1)
            volume2 = volume(mask_instance_arr2, label2, pixel_spacing2, slice_thickness2)

            if volume0 > 0:
                outputs["matched_nodules"][nodule_idx]["T0"] = {
                    # use the deregistered mask and the unregistered spacing
                    "mask_volume": volume0,
                    "instance_label": label0,
                    "centroid_distance_to_T1_nodule": centroid_distance01,
                    "min_distance_to_T1_nodule": min_distance01,
                    "iou_with_T1_nodule_after_reg": iou01,
                    "has_bbox_overlap": ((bbox_mask0 * (mask_instance_arr0_deregistered == label0).astype(np.uint8)).sum() > 0).item(),
                    "mask_slice_range": get_nonzero_slices((mask_instance_arr0_deregistered == label0).astype(np.uint8)),
                    "bbox_slice_range": get_nonzero_slices(bbox_mask0 * (mask_instance_arr0_deregistered == label0).astype(np.uint8)),
                }
            if volume1 > 0:
                outputs["matched_nodules"][nodule_idx]["T1"] = {
                    # use the deregistered mask and the unregistered spacing
                    "mask_volume": volume1,
                    "instance_label": label1,
                    "centroid_distance_to_T0_nodule": centroid_distance01,
                    "min_distance_to_T0_nodule": min_distance01,
                    "iou_with_T0_nodule_after_reg": iou01,
                    "centroid_distance_to_T2_nodule": centroid_distance12,
                    "min_distance_to_T2_nodule": min_distance12,
                    "iou_with_T2_nodule_after_reg": iou12,
                    "has_bbox_overlap": ((bbox_mask1 * (mask_instance_arr1_deregistered == label1).astype(np.uint8)).sum() > 0).item(),
                    "mask_slice_range": get_nonzero_slices((mask_instance_arr1_deregistered == label1).astype(np.uint8)),
                    "bbox_slice_range": get_nonzero_slices(bbox_mask1 * (mask_instance_arr1_deregistered == label1).astype(np.uint8)),
                }
            if volume2 > 0:
                outputs["matched_nodules"][nodule_idx]["T2"] = {
                    "mask_volume": volume2,
                    "instance_label": label2,
                    "centroid_distance_to_T1_nodule": centroid_distance12,
                    "min_distance_to_T1_nodule": min_distance12,
                    "iou_with_registered_T1_nodule": iou12,
                    "has_bbox_overlap": ((bbox_mask2 * (mask_instance_arr2 == label2).astype(np.uint8)).sum() > 0).item(),
                    "mask_slice_range": get_nonzero_slices((mask_instance_arr2 == label2).astype(np.uint8)),
                    "bbox_slice_range": get_nonzero_slices(bbox_mask2 * (mask_instance_arr2 == label2).astype(np.uint8)),
                }
    else:
        raise Exception("We should only have either 2 (missed middle scan) or 3 sets!")


    # handle unmatched nodules
    # if len(unmatched) == 2:
    # if match_length == 2:
    #     unmatched2, unmatched0 = unmatched
    # # if len(unmatched) == 3:
    # elif match_length == 3:

    unmatched2, unmatched1, unmatched0 = unmatched
    
    # print(len(unmatched))
    # print(unmatched)
    # print(match_length)
    
    i = 0
    for label0 in unmatched0:
        volume0 = volume(mask_instance_arr0_deregistered, label0, pixel_spacing0, slice_thickness0)

        # print("volume0", volume0)

        if volume0 > 0:
            outputs["unmatched_nodules"][i] = {
                # use the deregistered mask and the unregistered spacing
                "mask_volume": volume0,
                "instance_label": label0,
                "has_bbox_overlap": ((bbox_mask0 * (mask_instance_arr0_deregistered == label0).astype(np.uint8)).sum() > 0).item(),
                "mask_slice_range": get_nonzero_slices((mask_instance_arr0_deregistered == label0).astype(np.uint8)),
                "bbox_slice_range": get_nonzero_slices(bbox_mask0 * (mask_instance_arr0_deregistered == label0).astype(np.uint8)),
                "timepoint": "T0",
            }
            i += 1
    
    # if len(unmatched) == 3:
    for label1 in unmatched1:
        volume1 = volume(mask_instance_arr1_deregistered, label1, pixel_spacing1, slice_thickness1)

        # print("volume1", volume1)

        if volume1 > 0:
            outputs["unmatched_nodules"][i] = {
                # use the deregistered mask and the unregistered spacing
                "mask_volume": volume1,
                "instance_label": label1,
                "has_bbox_overlap": ((bbox_mask1 * (mask_instance_arr1_deregistered == label1).astype(np.uint8)).sum() > 0).item(),
                "mask_slice_range": get_nonzero_slices((mask_instance_arr1_deregistered == label1).astype(np.uint8)),
                "bbox_slice_range": get_nonzero_slices(bbox_mask1 * (mask_instance_arr1_deregistered == label1).astype(np.uint8)),
                "timepoint": "T1",
            }
            i += 1
    
    for label2 in unmatched2:
        volume2 = volume(mask_instance_arr2, label2, pixel_spacing2, slice_thickness2)

        # print("volume2", volume2)

        if volume2 > 0:
            outputs["unmatched_nodules"][i] = {
                "mask_volume": volume2,
                "instance_label": label2,
                "has_bbox_overlap": ((bbox_mask2 * (mask_instance_arr2 == label2).astype(np.uint8)).sum() > 0).item(),
                "mask_slice_range": get_nonzero_slices((mask_instance_arr2 == label2).astype(np.uint8)),
                "bbox_slice_range": get_nonzero_slices(bbox_mask2 * (mask_instance_arr2 == label2).astype(np.uint8)),
                "timepoint": "T2",
            }
            i += 1


        # # now, we want to store all nodules that are not matched
        # nodules0 = list(run_one_scan( 
        #     pid,
        #     0,
        #     pid_to_timepoints,
        #     evaluator_new,
        #     niftis_dir,
        #     original_mask_nifti_dir,
        #     instance_mask_nifti_dir,    
        # )["unmatched_nodules"].values())

        # nodules1 = list(run_one_scan(
        #     pid,
        #     1,
        #     pid_to_timepoints,
        #     evaluator_new,
        #     niftis_dir,
        #     original_mask_nifti_dir,
        #     instance_mask_nifti_dir,
        # )["unmatched_nodules"].values())

        # nodules2 = list(run_one_scan(
        #     pid,
        #     2,
        #     pid_to_timepoints,
        #     evaluator_new,
        #     niftis_dir,
        #     original_mask_nifti_dir,
        #     instance_mask_nifti_dir,
        # )["unmatched_nodules"].values())

        # all_unmatched_nodules = []

        # for tp, single_nodule_entries in [(0, nodules0), (1, nodules1), (2, nodules2)]:
        #     for single_nodule_entry in single_nodule_entries:
        #         is_already_matched = False
        #         for nodule_idx in outputs["matched_nodules"]:
        #             matched_nodule_entry = outputs["matched_nodules"][nodule_idx].get(f"T{tp}", {})

        #             # print(matched_nodule_entry)
        #             # print(single_nodule_entry)

        #             # don't check timepoint because we do not store that in the matched entry
        #             if all([matched_nodule_entry[k] == single_nodule_entry[k] for k in single_nodule_entry if k != "timepoint"]):
        #                 is_already_matched = True
                
        #         if not is_already_matched:
        #             all_unmatched_nodules.append(single_nodule_entry)

        # for nodule_idx, nodule_entry in enumerate(all_unmatched_nodules):
        #     outputs["unmatched_nodules"][nodule_idx] = nodule_entry


    # both T0 and T2 and T1 and T2

    # compute the IOU between all bounding boxes (if they exist)
    # TODO: maybe split by individual bounding boxes?
    outputs["bboxes"] = {
        f"T0_has_bboxes": np.count_nonzero(bbox_mask0) > 0,
        f"T1_has_bboxes": np.count_nonzero(bbox_mask1) > 0,
        f"T2_has_bboxes": np.count_nonzero(bbox_mask2) > 0,
    }

    if outputs["bboxes"]["T0_has_bboxes"] and outputs["bboxes"]["T2_has_bboxes"]:
        registered_bbox_mask0 = register_mask(
            pid,
            0,
            2,
            bbox_mask0,
            niftis_dir,
            original_mask_nifti_dir,
            transforms_dir
        )    

        outputs["reg_metrics"]["bbox_mask_iou_for_T0_registered_and_T2"] = compute_iou(
                registered_bbox_mask0, bbox_mask2
            )
    else:
        outputs["reg_metrics"]["bbox_mask_iou_for_T0_registered_and_T2"] = None
    
    if outputs["bboxes"]["T1_has_bboxes"] and outputs["bboxes"]["T2_has_bboxes"]:
        registered_bbox_mask1 = register_mask(
            pid,
            1,
            2,
            bbox_mask1,
            niftis_dir,
            original_mask_nifti_dir,
            transforms_dir
        )    

        outputs["reg_metrics"]["bbox_mask_iou_for_T1_registered_and_T2"] = compute_iou(
                registered_bbox_mask1, bbox_mask2
            )
    else:
        outputs["reg_metrics"]["bbox_mask_iou_for_T1_registered_and_T2"] = None
    
    return outputs


def remove_duplicate_unmatched_nodules(patient_data):
    out = copy.deepcopy(patient_data)

    matched_nodules = out.get("matched_nodules", {})
    unmatched_nodules = out.get("unmatched_nodules", {})
    
    matched_set = {
        (nodule["mask_volume"], tuple(nodule["mask_slice_range"]), tuple(nodule["bbox_slice_range"]), nodule["instance_label"], nodule["has_bbox_overlap"], tp)
        for match_group in matched_nodules.values()
        for tp, nodule in match_group.items()
    }
    
    to_delete = []
    
    for nodule_id, nodule in unmatched_nodules.items():
        nodule_key = (
            nodule["mask_volume"], 
            tuple(nodule["mask_slice_range"]), 
            tuple(nodule["bbox_slice_range"]), 
            nodule["instance_label"], 
            nodule["has_bbox_overlap"],
            nodule["timepoint"],
        )
        if nodule_key in matched_set:
            to_delete.append(nodule_id)
    
    for nodule_id in to_delete:
        del unmatched_nodules[nodule_id]
    
    return out


def get_image_path(pid: int, timepoint: int, niftis_dir: str) -> str:
    """Returns the path for an image for a given PID and timepoint."""

    # no group divisions!
    if len([fname for fname in os.listdir(niftis_dir) if fname.endswith(".nii.gz")]) > 0:
        return os.path.join(niftis_dir, f"nlst_{pid}T{timepoint}_0000.nii.gz")

    for group_dir_name in os.listdir(niftis_dir):
        proposed_path = os.path.join(niftis_dir, group_dir_name, f"nlst_{pid}T{timepoint}_0000.nii.gz")

        if os.path.isfile(proposed_path):
            return proposed_path

    raise FileNotFoundError


def main(config: dict):
    # model = "nnInteractive_all_mask_min_1"
    # model = "nnInteractive_all_mask_min_2"
    # model = "nnInteractive_all_mask_min_3"
    # model = "tsm_25dfb675_nn_all_mask_min_1_no_mask"
    # model = "tsm_25dfb675"

    # model = "frozen" # latest BMP-based
    # model = "frozen_nnunet_b3_after_lung_mask" # latest nnU-Net-based
    # model = "frozen4" # latest nnU-Net-based
    # model = "frozen5" # latest frozen

    # assert model == "frozen5"

    # is_cancer = True

    min_cluster_size = 25 # (can also do 15)

    dist_thresh = 20 # considered the same cluster (changed from 10 to be more generous with registration, we still take the minimum)

    # group_num = 0
    # group_num = 1
    # group_num = 2
    # group_num = 3

    # group_num = 4
    # group_num = 5
    # group_num = 6
    # group_num = 7

    # group_num = 8
    # group_num = 9
    # group_num = 10
    # group_num = 11

    # group_num = 12
    # group_num = 13
    # group_num = 14
    # group_num = 15

    # start_offset = 0

    # total_num_groups = 16

    # assert group_num in range(total_num_groups)

    # print(f"Processing {model} outputs with cluster size of {min_cluster_size} and distance thresh of {dist_thresh}...")

    # if is_cancer:
    #     niftis_dir = "/data/scratch/erubel/nlst/niftis"
    #     registered_niftis_dir = "/data/rbg/scratch/nlst_nodules/v2/register_outputs"
    #     transforms_dir = "/data/rbg/scratch/nlst_nodules/v1/transforms" # use cached outputs!
    #     outputs_dir = f"/data/rbg/scratch/nlst_nodules/v2/matching/{model}"
    #     original_mask_nifti_dir = f"/data/rbg/scratch/nlst_nodules/v2/masks/{model}"
    #     # so many directories...
    #     registered_mask_nifti_dir = f"/data/rbg/scratch/nlst_nodules/v2/registered_masks/{model}"
    #     deregistered_mask_nifti_dir = f"/data/rbg/scratch/nlst_nodules/v2/deregistered_instance_masks/{model}"
        
    #     registered_instance_mask_nifti_dir = f"/data/rbg/scratch/nlst_nodules/v2/registered_instance_masks/{model}"
    #     instance_mask_nifti_dir = f"/data/rbg/scratch/nlst_nodules/v2/instance_masks/{model}"
    # else:
    #     niftis_dir = "/data/rbg/scratch/nlst_benign_nodules/niftis"
    #     registered_niftis_dir = "/data/rbg/scratch/nlst_benign_nodules/v2/register_outputs"
    #     transforms_dir = "/data/rbg/scratch/nlst_benign_nodules/v1/transforms" # use cached outputs!
    #     outputs_dir = f"/data/rbg/scratch/nlst_benign_nodules/v2/matching/{model}"
    #     original_mask_nifti_dir = f"/data/rbg/scratch/nlst_benign_nodules/v2/masks/{model}"
    #     # so many directories...
    #     registered_mask_nifti_dir = f"/data/rbg/scratch/nlst_benign_nodules/v2/registered_masks/{model}"
    #     deregistered_mask_nifti_dir = f"/data/rbg/scratch/nlst_benign_nodules/v2/deregistered_instance_masks/{model}"
        
    #     registered_instance_mask_nifti_dir = f"/data/rbg/scratch/nlst_benign_nodules/v2/registered_instance_masks/{model}"
    #     instance_mask_nifti_dir = f"/data/rbg/scratch/nlst_benign_nodules/v2/instance_masks/{model}"

    niftis_dir = config["nifti_dir"]
    transforms_dir = config["transforms_dir"]
    outputs_dir = config["registered_masks_dir"]
    mask_nifti_dir = config["output_dir"]

    # TODO: do rest of directories
    
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(deregistered_mask_nifti_dir, exist_ok=True)
    os.makedirs(registered_instance_mask_nifti_dir, exist_ok=True)
    os.makedirs(instance_mask_nifti_dir, exist_ok=True)

    ## Load Data

    # annotations = json.load(open("/data/rbg/shared/datasets/NLST/NLST/annotations_122020.json", "r"))
    # args = Namespace(**pickle.load(open('/data/rbg/users/pgmikhael/current/SybilX/logs/c32cb085afbe045d58a7c83dcb71398c.args', 'rb')))
    
    # if is_cancer:
    #     nodule_dataset = pd.read_csv('/data/rbg/users/pgmikhael/current/SybilX/notebooks/NoduleGrowth/nlst_cancer_nodules.csv')
    # else:
    #     nodule_dataset = pd.read_csv('/data/rbg/users/pgmikhael/current/SybilX/notebooks/NoduleGrowth/nlst_benign_nodules_subset.csv')

    # dataset = get_object(args.dataset if is_cancer else 'nlst_benign_nodules', 'dataset')(args, "test")

    # pid_to_timepoints = {}

    # for i, row in tqdm(enumerate(dataset.dataset), total=len(dataset.dataset), ncols=100):
    #     exam = row['exam']

    #     nodule_row = nodule_dataset[nodule_dataset['PID'] == int(row['pid'])]
    #     tp = row['screen_timepoint']

    #     paths = row['paths']
    #     pid = int(row['pid'])

    #     bbox_annotations = None

    #     if isinstance(nodule_row[f"Annotated_{tp}"].iloc[0], str):
    #         annotated_sid = [s for s in nodule_row[f"Annotated_{tp}"].iloc[0].split(';') if s == row['series']]

    #         if len(annotated_sid) > 0:
    #             bbox_annotations = annotations[annotated_sid[0]]

    #     if pid in pid_to_timepoints:
    #         pid_to_timepoints[pid][int(tp)] = {'paths': paths, 'exam': exam, 'bbox_annotations': bbox_annotations}
    #     else:
    #         pid_to_timepoints[pid] = {int(tp): {'paths': paths, 'exam': exam, 'bbox_annotations': bbox_annotations}}

    # print(len(pid_to_timepoints))

    pid_to_timepoints = get_pid_to_timepoints(config)

    ## Run Evaluation

    evaluator_new = NoduleSegmentEvaluator(min_cluster_size)

    # don't recompute!
    if os.path.isfile(os.path.join(outputs_dir, "matching_outputs.json")):
        with open(os.path.join(outputs_dir, "matching_outputs.json"), "r") as f:
            output = json.load(f)
    else:
        output = {}

    if os.path.isfile(os.path.join(outputs_dir, "matching_skipped.json")):
        with open(os.path.join(outputs_dir, "matching_skipped.json"), "r") as f:
            skipped = json.load(f)
    else:
        skipped = []


    for pid in tqdm(sorted(list(pid_to_timepoints.keys()))):
        if pid in output:
            print(f"Skipping {pid} since it was already computed...")
            continue
    
        num_timepoints = len(pid_to_timepoints[pid])

        assert num_timepoints in {1, 2, 3}
        assert pid not in output # each iteration is a unique patient ID

        try:
            if num_timepoints == 1: # no matching?
                # TODO: maybe don't need to do this?

                keys = list(pid_to_timepoints[pid].keys())
                assert len(keys) == 1

                timepoint = keys[0]

                output[pid] = run_one_scan(
                    pid,
                    timepoint,
                    pid_to_timepoints,
                    evaluator_new,
                    niftis_dir,
                    original_mask_nifti_dir,
                    instance_mask_nifti_dir,
                )
        
            elif num_timepoints == 2:
                output[pid] = run_two_scans(
                    pid,
                    pid_to_timepoints,
                    evaluator_new,
                    niftis_dir,
                    original_mask_nifti_dir,
                    instance_mask_nifti_dir,
                    registered_mask_nifti_dir,
                    registered_instance_mask_nifti_dir,
                    deregistered_mask_nifti_dir,
                    transforms_dir,
                    dist_thresh,
                )
            
            elif num_timepoints == 3:
                output[pid] = run_three_scans(
                    pid,
                    pid_to_timepoints,
                    evaluator_new,
                    niftis_dir,
                    original_mask_nifti_dir,
                    instance_mask_nifti_dir,
                    registered_mask_nifti_dir,
                    registered_instance_mask_nifti_dir,
                    deregistered_mask_nifti_dir,
                    transforms_dir,
                    dist_thresh,
                )
            
            # sanity check (total = matched + unmatched)
            for tp in output[pid]["total_num_nodules"]:
                total_num_nodules = output[pid]["total_num_nodules"][tp]

                num_matched_nodules = 0
                for nodule_idx in output[pid]["matched_nodules"]:
                    matched_entry = output[pid]["matched_nodules"][nodule_idx].get(tp, {})

                    if matched_entry and matched_entry["mask_volume"] > 0:
                        num_matched_nodules += 1
                
                num_unmatched_nodules = 0
                for entry in output[pid]["unmatched_nodules"].values():
                    if entry["timepoint"] == tp and entry["mask_volume"] > 0:
                        num_unmatched_nodules += 1

                if total_num_nodules > num_matched_nodules + num_unmatched_nodules:
                    assert total_num_nodules == num_matched_nodules + num_unmatched_nodules, "Failed even after deleting"         
                elif total_num_nodules < num_matched_nodules + num_unmatched_nodules:
                    output[pid] = remove_duplicate_unmatched_nodules(output[pid])

                    # TODO: copied code
                    total_num_nodules = output[pid]["total_num_nodules"][tp]

                    num_matched_nodules = 0
                    for nodule_idx in output[pid]["matched_nodules"]:
                        matched_entry = output[pid]["matched_nodules"][nodule_idx].get(tp, {})

                        if matched_entry and matched_entry["mask_volume"] > 0:
                            num_matched_nodules += 1
                    
                    num_unmatched_nodules = 0
                    for entry in output[pid]["unmatched_nodules"].values():
                        if entry["timepoint"] == tp and entry["mask_volume"] > 0:
                            num_unmatched_nodules += 1
                    
                assert total_num_nodules == num_matched_nodules + num_unmatched_nodules, "Failed even after deleting"                     
                    
        except Exception as e: # catch-all for now
            print(f"Error {type(e).__name__ }, {str(e)} with {pid} -- skipping it!")
            skipped.append(pid)

        # save at every iteration
        with open(os.path.join(outputs_dir, "matching_outputs.json"), "w") as f:
            json.dump(output, f, indent=4)

        with open(os.path.join(outputs_dir, "matching_skipped.json"), "w") as f:
            json.dump(skipped, f, indent=4)
