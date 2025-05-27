import numpy as np
import os
import sys
import pickle
import torch
import matplotlib.pyplot as plt
import cv2
import nibabel as nib
from tqdm import tqdm
from PIL import Image
from pprint import pprint
import json

from utils.segment_evaluator import NoduleSegmentEvaluator
from utils.postprocessing import get_lung_mask, apply_lung_mask_and_retain_whole_instances
from utils.misc import get_config, run_in_conda_env

sys.path.append(os.path.join(os.getcwd(), "segment", "models"))

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession


def get_instance_bboxes(
    mask: np.ndarray, config: dict, padding: int = 0,
) -> dict[int, dict[int, tuple[int, int, int, int]]]:
    """
    Extracts bounding boxes for each instance in a 3D instance segmentation mask.
    
    Args:
        mask (np.ndarray): A 3D array of shape (depth, height, width) where each pixel
                           has an integer instance ID (0 for background).
        config (dict): Configuration settings
        padding (int): Number of pixels which we add to each corner of the bounding box

    Returns:
        dict[int, dict[int, tuple[int, int, int, int]]]:
            A dictionary mapping instance ID to a dictionary mapping slice index
            to a bounding box (x1, y1, x2, y2).
    """
    instance_bboxes = {}
    assert len(mask.shape) == 3

    depth, height, width = mask.shape
    assert height == width
    
    unique_instances = np.unique(mask)
    unique_instances = unique_instances[unique_instances != 0]
    
    for instance_id in unique_instances:
        instance_bboxes[instance_id] = {}
        
        for z in range(depth):
            slice_mask = mask[z] == instance_id
            if np.any(slice_mask):
                y_indices, x_indices = np.where(slice_mask)
                x1, x2 = max(x_indices.min() - padding, 0), min(x_indices.max() + padding, width - 1)
                y1, y2 = max(y_indices.min() - padding, 0), min(y_indices.max() + padding, height - 1)
                instance_bboxes[instance_id][z] = (x1.item(), y1.item(), x2.item(), y2.item())
                
    return instance_bboxes


def bbox_center(x1, y1, x2, y2) -> tuple[int, int]:
    """
    Returns the center of the bounding box defined by [`x1`, `y1`, `x2`, `y2`].
    
    Rounds down for any non-integer values.
    """
    return int((x1 + x2) // 2), int((y1 + y2) // 2)


def get_lower_median(samples: list[int]) -> int:
    """Returns the lower median, in case we have an even number of samples."""

    return np.quantile(samples, 0.5, method="lower").item()


def prompt_nnInteractive(
    session,
    instance_id: int,
    instance_input_mask: np.ndarray,
    instance_input_bboxes: dict,
    config: dict,
) -> tuple[int, int]:
    """
    Prompts the nnInterctive model based on the input prompts as specified in `config`.

    Only tracks a single object referred to by its `instance_id`.

    - `instance_input_mask` only has values corresponding to `instance_id`.
    - `instance_input_bboxes` is a dictionary mapping slice numbers to a corresponding
    bounding box for `instance_id`.

    Returns `min_slice_num_with_prompt` and `max_slice_num_with_prompt`.
    """

    if config["prompt_type"] == "bbox":
        if config["prompt_subset_type"] == "median":
            median_slice_num = get_lower_median(list(instance_input_bboxes.keys()))
            median_bbox = instance_input_bboxes[median_slice_num]

            bbox_input = [
                [median_bbox[0], median_bbox[2]],
                [median_bbox[1], median_bbox[3]],
                [median_slice_num, median_slice_num + 1], # at `median_slice_num`
            ]

            if config["debug"]:
                print(f"Prompting bbox at median slice {median_slice_num} with instance {instance_id} and box {bbox_input}...")

            session.add_bbox_interaction(bbox_input, include_interaction=True)

            return median_slice_num, median_slice_num

        elif config["prompt_subset_type"] == "maximum":
            max_slice_num = np.argmax(np.sum(instance_input_mask, axis=(1, 2))).item()
            max_bbox = instance_input_bboxes[max_slice_num]

            bbox_input = [
                [max_bbox[0], max_bbox[2]],
                [max_bbox[1], max_bbox[3]],
                [max_slice_num, max_slice_num + 1], # at `max_slice_num`
            ]
        
            if config["debug"]:
                print(f"Prompting at maximum-pixel slice {max_slice_num} with {instance_id} with {bbox_input}...")

            session.add_bbox_interaction(bbox_input, include_interaction=True)

            return max_slice_num, max_slice_num

        elif config["prompt_subset_type"] == "all":
            for slice_num, bbox in instance_input_bboxes.items():
                bbox_input = [
                    [bbox[0], bbox[2]],
                    [bbox[1], bbox[3]],
                    [slice_num, slice_num + 1], # at `slice_num`
                ]

                if config["debug"]:
                    print(f"Prompting bbox at slice {slice_num} with {instance_id} with {bbox_input}...")

                # as recommended, we run the prediction after each interaction and then only use the prediction after the final prompt
                session.add_bbox_interaction(bbox_input, include_interaction=True, run_prediction=True)
            
            return min(list(instance_input_bboxes.keys())), max(list(instance_input_bboxes.keys()))
    
    elif config["prompt_type"] == "mask":
        # we can only prompt with a single mask (can be 3D for multiple slices)
        # this may not be officially supported...

        if config["prompt_subset_type"] == "median":
            median_slice_num = get_lower_median(list(instance_entries.keys()))
            median_mask_slice = instance_input_mask[:, :, median_slice_num]

            if config["debug"]:
                print(f"Prompting mask at median slice {median_slice_num} with instance {instance_id}...")
            
            input_mask_3d = np.zeros(instance_input_mask.shape).astype(np.uint8)
            input_mask_3d[:, :, median_slice_num] = median_mask_slice
           
            if config["debug"]:
                print(f"3D Prompt Mask Shape: {input_mask_3d.shape}")

            # we run the prediction here since we only add a single (3D) mask
            
            session.add_initial_seg_interaction(input_mask_3d, run_prediction=True)
                
            return median_slice_num, median_slice_num
        elif config["prompt_subset_type"] == "maximum":
            max_slice_num = np.argmax(np.sum(instance_input_mask, axis=(0, 1))).item()
            max_mask_slice = instance_input_mask[:, :, max_slice_num]

            if config["debug"]:
                print(f"Prompting at maximum-pixel slice {max_slice_num} with {instance_id}...")
            
            input_mask_3d = np.zeros(instance_input_mask.shape).astype(np.uint8)
            
            input_mask_3d[:, :, max_slice_num] = max_mask_slice
           
            if config["debug"]:
                print(f"3D Prompt Mask Shape: {input_mask_3d.shape}")

            # we run the prediction here since we only add a single (3D) mask
        
            session.add_initial_seg_interaction(input_mask_3d, run_prediction=True)
                
            return max_slice_num, max_slice_num
        elif config["prompt_subset_type"] == "all":
            # we can only prompt once with a single 3D mask

            input_mask_3d = np.zeros(instance_input_mask.shape).astype(np.uint8)
            
            for slice_num in instance_input_bboxes:
                if config["debug"]:
                    print(f"Prompting mask at slice {slice_num} with instance {instance_id}...")

                input_mask_3d[:, :, slice_num] = instance_input_mask[:, :, slice_num]
            
            if config["debug"]:
                print(f"3D Prompt Mask Shape: {input_mask_3d.shape}")

            # we run the prediction here since we only add a single (3D) mask
    
            session.add_initial_seg_interaction(input_mask_3d, run_prediction=True)
                
            return min(list(instance_input_bboxes.keys())), max(list(instance_input_bboxes.keys()))
    
    elif "point" in config["prompt_type"]:
        assert config["prompt_type"] in {"pos_point", "neg_point"}

        is_positive = (config["prompt_type"] == "pos_point")

        if config["prompt_subset_type"] == "median":
            median_slice_num = get_lower_median(list(instance_input_bboxes.keys()))
            median_bbox = instance_input_bboxes[median_slice_num]

            point_x, point_y = bbox_center(*median_bbox)
            point_input = (point_x, point_y, median_slice_num)

            session.add_point_interaction(point_input, include_interaction=is_positive)

            if config["debug"]:
                print(f"Prompting median point at {point_input} with instance {instance_id}...")

            return median_slice_num, median_slice_num

        elif config["prompt_subset_type"] == "maximum":
            max_slice_num = np.argmax(np.sum(instance_input_mask, axis=(1, 2))).item()
            max_bbox = instance_input_bboxes[max_slice_num]

            point_x, point_y = bbox_center(*max_bbox)
            point_input = (point_x, point_y, max_slice_num)

            session.add_point_interaction(point_input, include_interaction=is_positive)

            if config["debug"]:
                print(f"Prompting maximum point at {point_input} with instance {instance_id}...")

            return max_slice_num, max_slice_num

        elif config["prompt_subset_type"] == "all":
            for slice_num, bbox in instance_input_bboxes.items():
                point_x, point_y = bbox_center(*bbox)
                point_input = (point_x, point_y, slice_num)

                # as recommended, we run the prediction after each interaction and then only use the prediction after the final prompt
                session.add_point_interaction(point_input, include_interaction=is_positive, run_prediction=True)

                if config["debug"]:
                    print(f"Prompting median point at {point_input} with instance {instance_id}...")

            return min(list(instance_input_bboxes.keys())), max(list(instance_input_bboxes.keys()))
    elif config["prompt_type"] == "lasso":
        raise NotImplementedError
        # can do positive and negative
    elif config["prompt_type"] == "scribble":
        raise NotImplementedError
        # can do positive and negative
    else:
        raise NotImplementedError


def main(config: dict) -> None:
    """Runs the nnInteractive inference pipeline."""
    
    np.random.seed(42)

    session = nnInteractiveInferenceSession(
        device=torch.device(f"cuda:{config['device']}"),
        use_torch_compile=False,  # experimental: not tested yet
        verbose=False,
        torch_n_threads=os.cpu_count(),
        do_autozoom=True,  # enables AutoZoom for better patching
        use_pinned_memory=True,  # optimizes GPU memory transfers
    )

    session.initialize_from_trained_model_folder(os.path.abspath("segment/checkpoints/nnInteractive"))

    segment_evaluator = NoduleSegmentEvaluator()

    skipped = []

    for fname in sorted(os.listdir(config["nifti_dir"])):
        if fname.endswith(".nii.gz"):
            try:
                series_id = fname.replace("_0000", "").replace(".nii.gz", "")

                image = nib.load(os.path.join(config["nifti_dir"], fname))

                sx, sy, sz = image.header.get_zooms()

                pixel_spacing = [sx, sy]
                slice_thickness = sz

                input_mask = np.expand_dims(np.transpose((nib.load(os.path.join(config["output_dir"], f"{series_id}_initial.nii.gz")).get_fdata() > 0).astype(np.uint8), (2, 1, 0)), axis=0)

                if config["lung_mask_mode"]:
                    lung_mask = get_lung_mask(series_id, config)
                    assert input_mask[0].shape == lung_mask.shape

                if config["debug"]:
                    print(f"Input Mask Type: {input_mask.dtype}")

                    if config["lung_mask_mode"]:
                        print(f"Lung Mask Type: {lung_mask.dtype}")

                if config["debug"]:
                    print(f"Input Mask Shape: {input_mask.shape}")

                assert len(input_mask.shape) == 4

                instance_input_mask = segment_evaluator.get_instance_segmentation(input_mask)[0] # excludes channel

                mask_shape = instance_input_mask.shape # excludes channel
                assert len(mask_shape) == 3

                if config["debug"]:
                    print(f"Instance Input Mask Shape: {mask_shape}")

                instance_input_bboxes = get_instance_bboxes(instance_input_mask, config)
                
                if config["debug"]:
                    print("Number of Slices per Instance:", sorted([(f"Instance {k}", len(v)) for k, v in instance_input_bboxes.items()], reverse=True, key=lambda x:x[1]))  

                    print(f"\nPrompting with {len(instance_input_bboxes)} instances...\n\n")

                all_instance_ids = list(instance_input_bboxes.keys())

                all_output_masks = []
                
                # MUST BE x then y then z (not an arbitrary order!)
                # we do all of the processing as (x, y, z) and then convert to (z, y, x) at the very end
                img = image.get_fdata()[None]
                instance_input_mask = np.transpose(instance_input_mask, (2, 1, 0)) # send to (x, y, z)

                assert img.shape[0] == 1

                # validate input dimensions
                if img.ndim != 4:
                    raise ValueError("Input image must be 4D with shape (1, x, y, z)")

                if config["debug"]:
                    print("Input Shape", img.shape)
                    print("Instance Mask Shape", instance_input_mask.shape)
                
                # track every object/nodule separately
                for instance_id in all_instance_ids:
                    session.set_image(img)
                    session.set_target_buffer(torch.zeros(img.shape[1:], dtype=torch.uint8))  # Must be 3D (x, y, z)

                    min_slice_num_with_prompt, max_slice_num_with_prompt = prompt_nnInteractive(
                        session,
                        instance_id,
                        (instance_input_mask == instance_id).astype(np.uint8),
                        instance_input_bboxes[instance_id],
                        config,
                    )

                    instance_output_mask = session.target_buffer.clone().cpu().numpy() > 0

                    assert instance_output_mask.dtype == "bool"

                    if config["debug"]:
                        print("Instance Output Shape", instance_output_mask.shape)

                    all_output_masks.append(instance_output_mask)

                    # clears the target buffer and resets interactions
                    session.reset_interactions()

                if all_output_masks:
                    output_mask = np.logical_or.reduce(all_output_masks).astype(np.uint8)
                else:
                    output_mask = np.transpose(np.zeros(mask_shape), (2, 1, 0)).astype(np.uint8)

                # send from (x, y, z) to (z, y, x)
                output_mask = np.transpose(output_mask, (2, 1, 0))

                if config["lung_mask_mode"]: # we clip again here in case of over-segmentation
                    final_lung_mask = apply_lung_mask_and_retain_whole_instances(segment_evaluator.get_instance_segmentation(np.expand_dims(output_mask, axis=0))[0], lung_mask, config)
                    output_mask *= final_lung_mask
                    
                if config["debug"]:
                    print("Final Output Shape", np.expand_dims(output_mask, axis=0).shape)

                nib.save(
                    nib.Nifti1Image(
                        np.transpose(output_mask, (2, 1, 0)),
                        affine=image.affine,
                        header=image.header,
                    ),
                    os.path.join(config["output_dir"], f"{series_id}.nii.gz"),
                )
            except Exception as e:
                print(f"Skipping {series_id} due to error {str(e)}...")
                skipped.append(series_id)
    
    with open(os.path.join(config["output_dir"], "nnInteractive_skipped.json"), "w") as f:
        json.dump(skipped, f, indent=4)


if __name__ == "__main__":
    config = get_config()
    main(config)
