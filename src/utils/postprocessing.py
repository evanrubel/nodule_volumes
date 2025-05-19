import numpy as np
import os
from scipy.ndimage import label


def get_lung_slice_range(lung_mask: np.ndarray) -> tuple[int, int]:
    """
    Given a 3D segmentation mask of the lungs, determine the range of slice numbers that the lungs cover.
    
    Parameters:
        lung_mask (np.ndarray): A 3D numpy array (shape: [slices, height, width]) where lung regions are nonzero.
        
    Returns:
        tuple: (min_slice, max_slice) indicating the range of slices containing lung regions.
    """
    lung_slices = np.any(lung_mask > 0, axis=(1, 2))
    
    if not np.any(lung_slices):
        return None  # no lungs
    
    min_slice = np.argmax(lung_slices)  # first occurrence of True
    max_slice = len(lung_slices) - np.argmax(lung_slices[::-1]) - 1  # last slice with lungs
    
    return min_slice, max_slice


def get_lung_mask(series_id: str, config: dict, exam_id=None) -> np.ndarray:
    """Returns the lung mask as specified in `config`.
    
    We either return the entire lung mask or a mask of all ones on the slices that the lung covers.
    """

    # TOOD: check this with NLST dataset, not just UTC

    lung_mask_arr = np.load(os.path.join(config["dataset_dir"], "lung_masks", f"{series_id}.npy")).astype(np.uint8)

    # if config["dataset"] == "utc":
    lung_mask_arr = np.rot90(np.flip(lung_mask_arr, axis=2), k=-2, axes=(1,2))
    
    if config["lung_mask_mode"] == "mask":
        return lung_mask_arr
    elif config["lung_mask_mode"] == "range":
        min_lung_slice, max_lung_slice = get_lung_slice_range(lung_mask_arr)

        final_lung_mask_arr = np.ones(lung_mask_arr.shape)
        
        # zero out everything beyond the range!
        final_lung_mask_arr[0:min_lung_slice] = 0
        final_lung_mask_arr[max_lung_slice+1:] = 0

        return final_lung_mask_arr.astype(np.uint8)
    else:
        raise NotImplementedError


def apply_lung_mask_and_retain_whole_instances(instance_mask: np.ndarray, lung_mask: np.ndarray, config: dict) -> np.ndarray:
    """
    Args:
        instance_mask (np.ndarray): 3D array (D, H, W) with instance IDs.
        lung_mask (np.ndarray): 3D binary array (D, H, W) where 1 = lung region.
    
    Returns:
        np.ndarray: Binary mask that retains whole instances from `instance_mask` if any of the instance overlaps with `lung_mask`.
    """

    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids != 0] # exclude background

    instances_to_keep = []
    
    for instance_id in instance_ids:
        instance_id_only = (instance_mask == instance_id)

        # any overlap with lung mask
        if np.any(instance_id_only & lung_mask):
            instances_to_keep.append(instance_id)
        
        else:
            if config["debug"]:
                print(f"Removing instance {int(instance_id)} on slices {np.where(np.any(instance_id_only, axis=(1, 2)))[0].tolist()}...")

    return np.isin(instance_mask, instances_to_keep)


def remove_flashing_entities(mask: np.ndarray, config: dict, k: int = 10) -> np.ndarray:
    """
    Removes any entities that "flash" when scrolling through the slices (i.e., do not overlap with any 
    other part of the mask).
    """

    assert k % 2 == 0, "Expected k to be even."

    mask = np.transpose(mask, (2, 1, 0))

    num_slices = mask.shape[0]
    output_mask = np.zeros_like(mask)

    # to track persistence of objects across slices
    prev_labels = None

    for i in range(num_slices):
        slice_mask = mask[i]
        labeled_slice, num_features = label(slice_mask)

        # count how often each label appears in adjacent slices
        for label_id in range(1, num_features + 1):
            current_component = (labeled_slice == label_id)
            appears_elsewhere = False

            # check if it appears in adjacent slices
            for j in range(i - (k // 2), i + (k // 2) + 1): # look at k // 2 adjacent slices in both directions
                if 0 <= j < num_slices and j != i:
                    overlap = current_component & mask[j]
                    if np.any(overlap):
                        appears_elsewhere = True
                        break

            if appears_elsewhere:
                output_mask[i][current_component] = 1
    
    return np.transpose(output_mask, (2, 1, 0))
