# use `biomedparse`

import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib
import os
from tqdm import tqdm
import torch.nn.functional as F
import json
from PIL import Image
from pprint import pprint
import pickle

sys.path.append("/data/rbg/users/erubel/volumetry/models/BiomedParse")

from modeling.BaseModel import BaseModel
from modeling import build_model

from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES

from inference_utils.inference import interactive_infer_image, interactive_infer_image_all
from inference_utils.processing_utils import process_intensity_image


def get_nifti_nib(series_id: str, config: dict):
    """Checks all subfolders and returns the matching Nibabel object (for `series_id`) within those subfolders."""
    
    if any(["group" in f for f in os.listdir(config["nifti_dir"])]):
        # look in subdivided groups
        for f in os.listdir(config["nifti_dir"]):
            if "group" in f and os.path.isdir(os.path.join(config["nifti_dir"], f)):
                for dir_fname in os.listdir(os.path.join(config["nifti_dir"], f)):
                    if dir_fname == f"{series_id}_0000.nii.gz":
                        return nib.load(os.path.join(config["nifti_dir"], f, dir_fname))
    elif len([f for f in os.listdir(config["nifti_dir"]) if f.endswith(".nii.gz")]) >= 1:
        # in one directory
        return nib.load(os.path.join(config["nifti_dir"], f"{series_id}_0000.nii.gz"))
    
    raise FileNotFoundError


def get_sorted_filenames(config: dict) -> list[str]:
    """Returns the sorted filenames to process based on `config.`"""

    all_filenames = []

    if any(["group" in f for f in os.listdir(config["nifti_dir"])]):
        # look in subdivided groups
        for f in os.listdir(config["nifti_dir"]):
            if "group" in f and os.path.isdir(os.path.join(config["nifti_dir"], f)):
                all_filenames.extend(os.listdir(os.path.join(config["nifti_dir"], f)))
    elif len([f for f in os.listdir(config["nifti_dir"]) if f.endswith(".nii.gz")]) >= 1:
        # in one directory
        all_filenames.extend([fname for fname in os.listdir(os.path.join(config["nifti_dir"])) if fname.endswith(".nii.gz")])
    else:
        raise NotImplementedError
        
    all_filenames_sorted = sorted(all_filenames)

    if config["group_ix"] == "all":
        return all_filenames_sorted
    else:
        assert isinstance(config["group_ix"], int) and isinstance(config["total_num_groups"], int)
        assert config["group_ix"] in range(config["total_num_groups"])

        start_ix = (len(all_filenames_sorted) // config["total_num_groups"]) * config["group_ix"]
        stop_ix = (len(all_filenames_sorted) // config["total_num_groups"]) * (config["group_ix"] + 1)

        if config["group_ix"] == config["total_num_groups"] - 1: # last one
            print(f"Going from {start_ix} to {len(all_filenames_sorted) - 1}...")
            return all_filenames_sorted[start_ix:]
        else:
            print(f"Going from {start_ix} to {stop_ix}...")
            return all_filenames_sorted[start_ix:stop_ix]


def inference_nifti(
    model,
    raw_image_array: np.ndarray,
    text_prompts: list[str],
    is_CT: bool,
    site: str = None,
    show: bool = True,
):
    assert len(raw_image_array.shape) == 2
    
    image = process_intensity_image(raw_image_array, is_CT, site)

    pred_mask = interactive_infer_image(model, Image.fromarray(image), text_prompts)

    if show:
        plot_segmentation_masks(image, pred_mask, text_prompts, rotate=rotate)
    
    return image, pred_mask


def inference_nifti_recognition(
    model,
    raw_image_array: np.ndarray,
    image_type: str,
    is_CT: bool,
    targets_to_include: list[str],
    site: str = None,
    p_value_threshold: float = None,
    show: bool = True,
    verbose: bool = False,
):
    """
    Runs the object recognition pipeline for a given `image_type`, including
    `targets_to_include` only.

    If `p_value_threshold` is not None, we filter out any targets that do not
    meet `p_value_threshold`.
    
    We use this over the other `inference_nifti` function because this uses a more
    intelligent p-value filter over all possible targets in the given `image_type`
    (e.g., for the `image_type` of "CT-Chest', the targets are "nodule", "tumor" and
    "COVID-19 infection").
    """

    assert len(raw_image_array.shape) == 2
    
    image = process_intensity_image(raw_image_array, is_CT, site)
    
    predictions, p_values = interactive_infer_image_all(model, Image.fromarray(image), image_type, p_value_threshold, batch_size=2, verbose=verbose)

    targets = [t for t in list(predictions.keys()) if t in targets_to_include]
    pred_masks = [predictions[t] for t in targets]

    if show:
        plot_segmentation_masks_recognition(image, pred_masks, targets)
    
    return image, pred_masks, p_values, targets


def get_model(config: dict):
    """Returns the BiomedParse model."""

    opt = load_opt_from_config_files(["/data/rbg/users/erubel/volumetry/models/BiomedParse/configs/biomedparse_inference.yaml"])
    opt = init_distributed(opt)
    opt["device"] = torch.device("cuda", config["device"])
    torch.cuda.set_device(config["device"])

    # load model from pretrained weights
    pretrained_pth = "/data/rbg/users/erubel/volumetry_old/src/BiomedParse/pretrained/biomed_parse.pt"

    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().to(f"cuda:{config['device']}")
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ["background"], is_eval=True)

    return model


def insert_p_values(model, image_array_slice: np.ndarray, image_shape: tuple, config: dict) -> np.ndarray:
    """Set all of the pixel values to be its confidence values (slice-wise p-values)."""

    assert len(image_array_slice.shape) == 2

    # `pred_masks` matches the order of `targets`
    _, pred_masks, p_values, targets = inference_nifti_recognition(
        model=model,
        raw_image_array=image_array_slice,
        image_type='CT-Chest',
        is_CT=True,
        targets_to_include=["nodule", "tumor"], # do not include the third target in this image type ("COVID-19 infection")
        site='lung',
        p_value_threshold=config["p_value_threshold"],
        show=False,
        verbose=config["debug"],
    )

    assert len(pred_masks) == len(targets)

    assert all([(np.all(pred_mask == 0) or np.unique(pred_mask).tolist()) == [0, 1] for pred_mask in pred_masks]), "Expected all binary masks."

    assert len(pred_masks) <= 2, "Either two masks (both 'nodule' and 'tumor' are sufficiently confident), one mask, or no mask at all."
    
    if config["debug"]:
        print(f"There are {len(pred_masks)} masks")

    if len(pred_masks) == 0:
        return None
    elif len(pred_masks) == 1:
        # set all of its pixel values to be its confidence values
        return F.interpolate(
            torch.from_numpy(pred_masks[0] * p_values[targets[0]]).unsqueeze(0).unsqueeze(0),
            image_shape[1:],
        ).numpy()
    elif len(pred_masks) == 2:
        # set the two masks with their corresponding confidence values and take the element-wise maximum of them in case of overlap
        return F.interpolate(
            torch.from_numpy(np.maximum(pred_masks[0] * p_values[targets[0]], pred_masks[1] * p_values[targets[1]])).unsqueeze(0).unsqueeze(0),
            image_shape[1:],
        ).numpy()
    else:
        raise Exception


def get_probabilities(model, image_array_slice: np.ndarray, image_shape: tuple, prompts: list[str], config: dict) -> np.ndarray:
    """Returns the raw probabilities from the BiomedParse model."""

    _, pred_masks = inference_nifti(
        model=model,
        raw_image_array=image_array_slice,
        text_prompts=["nodule"], # can also change to ["tumor"]
        is_CT=True,
        site="lung",
        show=False,
    )

    assert len(pred_masks) == 1, "Expected a single mask corresponding to a single prompt."

    # note we do interpolate here, so the probabilities are smoothed

    return F.interpolate(
        torch.from_numpy(pred_masks[0]).unsqueeze(0).unsqueeze(0),
        image_shape[1:],
    ).numpy()


def main(config: dict) -> None:
    """Runs the BiomedParse++ inference pipeline."""

    with open(os.path.join(config["output_dir"], f"config_{config['group_ix']}.json"), "w") as f:
        json.dump(config, f, indent=4)

    skipped = []

    model = get_model(config)

    # TODO: clean this up by moving to the same directory
    if config["dataset"] == "nlst":
        with open("/data/rbg/users/erubel/volumetry/notebooks/sam2/pid_to_exam_by_timepoint.p", "rb") as f:
            pid_to_exam_by_timepoint = pickle.load(f)
    elif config["dataset"] == "nlst_benign":
        with open("/data/rbg/scratch/nlst_benign_nodules/v1/pid_to_exam_by_timepoint.p", "rb") as f:
            pid_to_exam_by_timepoint = pickle.load(f)
    ###

    for fname in tqdm(get_sorted_filenames(config)):
        if fname.endswith(".nii.gz"):
            try:
                series_id = fname.replace("_0000.nii.gz", "")

                # load image
                image = get_nifti_nib(series_id, config)
                image_array = image.get_fdata()

                if "nlst" not in config["dataset"]:
                    image_array = np.transpose(image_array, (2, 1, 0))

                image_shape = image_array.shape

                if config["debug"]:
                    print(f"Image Shape: {image_shape}")
                
                assert image_shape[1] == image_shape[2], "Expected a square image."

                output_mask = np.zeros(image_shape)
                
                # run inference
                for slice_num in tqdm(range(image_shape[0])):
                    if config["insert_p_values"]:
                        candidate_slice_mask = insert_p_values(model, image_array[slice_num], image_shape, config)

                        if candidate_slice_mask is not None:
                            output_mask[slice_num] = candidate_slice_mask
                    else:
                        output_mask[slice_num] = get_probabilities(model, image_array[slice_num], image_shape, config)

                if "nlst" not in config["dataset"]:
                    final_mask = np.transpose(output_mask, (2, 1, 0))
                else:
                    final_mask = output_mask

                if config["debug"]:
                    print("Mask values:", np.unique(final_mask).tolist())
                    print("Final before saving", final_mask.shape)
                
                # save final output
                nib.save(
                    nib.Nifti1Image(final_mask, affine=image.affine, header=image.header),
                    os.path.join(config["output_dir"], f"{series_id.replace('_0000', '')}.nii.gz"),
                )
            except Exception as e:
                print(f"Skipping {series_id} due to error {str(e)}...")
                skipped.append(series_id)
    
        with open(os.path.join(config["output_dir"], f"skipped_group_{config['group_ix']}.json"), "w") as f:
            json.dump(skipped, f, indent=4)
    

if __name__ == "__main__":
    config = {
        "dataset": "nlst",

        "device": 0,
        "group_ix": 0,

        "use_lung_mask": False,
        "use_vessel_mask": False,

        "p_value_threshold": 0.0, # it is best to keep this as 0.0 and then filter in downstream scripts (e.g., postprocess.py, utc_inference_evaluation.py, etc.)
    
        "total_num_groups": 12,

        "insert_p_values": True, # this is for slice-wise p-values (otherwise, False means we do instance-wise [experimental])

        "debug": False,
    }
    
    assert config["dataset"] in {"nlst", "utc"}
    assert 0 <= config["p_value_threshold"] <= 1
    assert 0 <= config["group_ix"] < config["total_num_groups"]

    config["experiment_name"] = f"p_{config['p_value_threshold']}"

    if config["use_lung_mask"]:
        config["experiment_name"] += "_with_lung_mask"
    else:
        config["experiment_name"] += "_no_lung_mask"

    if config["use_vessel_mask"]:
        config["experiment_name"] += "_with_vessel_mask"
    else:
        config["experiment_name"] += "_no_vessel_mask"
    

    print("\n\n")
    pprint(config)
    print("\n\n")

    if config["dataset"] == "nlst":
        config["nifti_dir"] = "/data/scratch/erubel/nlst/niftis"
        config["output_dir"] = f"/data/scratch/erubel/nlst/biomedparse++/{config['experiment_name']}"
        config["vessel_mask_dir"] = "/data/scratch/erubel/nlst/lung_vessels"
    elif config["dataset"] == "utc":
        config["nifti_dir"] = "/data/scratch/erubel/external_val/utc/data"
        config["output_dir"] = f"/data/scratch/erubel/external_val/utc/experiments/biomedparse++/{config['experiment_name']}"
        config["vessel_mask_dir"] = "/data/scratch/erubel/external_val/utc/experiments/lung_vessels"
    
    config["lung_mask_dir"] = f"/data/rbg/scratch/lung_ct/{config['dataset'].replace('_benign', '')}_lung_mask"
    
    os.makedirs(config["output_dir"], exist_ok=True)

    main(config)
