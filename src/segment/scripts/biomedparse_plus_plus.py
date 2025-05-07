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

sys.path.append(os.path.join(os.getcwd(), "segment", "models", "BiomedParse"))

from modeling.BaseModel import BaseModel
from modeling import build_model

from utilities.distributed import init_distributed
from utilities.arguments import load_opt_from_config_files
from utilities.constants import BIOMED_CLASSES

from inference_utils.inference import interactive_infer_image, interactive_infer_image_all
from inference_utils.processing_utils import process_intensity_image


def inference_nifti_recognition(
    model,
    raw_image_array: np.ndarray,
    image_type: str,
    is_CT: bool,
    targets_to_include: list[str],
    site: str = None,
    show: bool = True,
    verbose: bool = False,
):
    """
    Runs the object recognition pipeline for a given `image_type`, including
    `targets_to_include` only.
    
    We use this over the other `inference_nifti` function because this uses a more
    intelligent p-value filter over all possible targets in the given `image_type`
    (e.g., for the `image_type` of "CT-Chest', the targets are "nodule", "tumor" and
    "COVID-19 infection").
    """

    assert len(raw_image_array.shape) == 2
    
    image = process_intensity_image(raw_image_array, is_CT, site)
    
    predictions, p_values = interactive_infer_image_all(model, Image.fromarray(image), image_type, batch_size=2)

    targets = [t for t in list(predictions.keys()) if t in targets_to_include]
    pred_masks = [predictions[t] for t in targets]

    if show:
        plot_segmentation_masks_recognition(image, pred_masks, targets)
    
    return image, pred_masks, p_values, targets


def get_model(config: dict):
    """Returns the BiomedParse model."""

    opt = load_opt_from_config_files([os.path.join(os.getcwd(), "segment", "models", "BiomedParse", "configs", "biomedparse_inference.yaml")])
    opt = init_distributed(opt)
    opt["device"] = torch.device("cuda", config["device"])
    torch.cuda.set_device(config["device"])

    model = BaseModel(opt, build_model(opt)).from_pretrained(os.path.join(os.getcwd(), "segment", "checkpoints", "biomedparse_v1.pt")).eval().to(f"cuda:{config['device']}")
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


def main(config: dict) -> None:
    """Runs the BiomedParse++ inference pipeline."""

    with open(os.path.join(config["output_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    skipped = []

    model = get_model(config)

    for fname in tqdm(sorted(os.listdir(config["nifti_dir"]))):
        if fname.endswith(".nii.gz"):
            # try:
            print("bring back try except")
            series_id = fname.replace("_0000.nii.gz", "")

            # load image
            image = nib.load(os.path.join(config["nifti_dir"], fname))
            image_array = np.transpose(image.get_fdata(), (2, 1, 0)) # send to (z, y, x)

            image_shape = image_array.shape

            if config["debug"]:
                print(f"Image Shape: {image_shape}")
            
            assert image_shape[1] == image_shape[2], "Expected a square image."

            output_mask = np.zeros(image_shape)
            
            # run inference
            for slice_num in tqdm(range(image_shape[0])[10:20]):
                candidate_slice_mask = insert_p_values(model, image_array[slice_num], image_shape, config)

                if candidate_slice_mask is not None:
                    output_mask[slice_num] = candidate_slice_mask

            final_mask = np.transpose(output_mask, (2, 1, 0))

            if config["debug"]:
                print("Mask values:", np.unique(final_mask).tolist())
                print("Final before saving", final_mask.shape)
            
            # save final output
            nib.save(
                nib.Nifti1Image(final_mask, affine=image.affine, header=image.header),
                os.path.join(config["output_dir"], f"{series_id.replace('_0000', '')}.nii.gz"),
            )
            # except Exception as e:
            #     print(f"Skipping {series_id} due to error {str(e)}...")
            #     skipped.append(series_id)
    
        with open(os.path.join(config["output_dir"], "skipped.json"), "w") as f:
            json.dump(skipped, f, indent=4)
