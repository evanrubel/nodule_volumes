# use `sybil2`

from segmentation_evaluator import NoduleSegmentEvaluator
import os
from tqdm import tqdm
import nibabel as nib
import json
import numpy as np
import pickle
from pprint import pprint


def main(config: dict):
    evaluator = NoduleSegmentEvaluator(config["min_cluster_size"])

    output = {}

    true_vols = []
    pred_vols = []
    
    for fname in tqdm(sorted(os.listdir(config["fname_src_path"]))):
        if fname.endswith(".nii.gz"): # assumes we are dealing with all NIFTIs
            scan_id = fname.replace(".nii.gz", "").replace("_0000", "")

            # (1) get masks

            try:
                # both should be 1xNxHxW
                if config["use_niftis"]:
                    if config["p_value_threshold"] is None:
                        raise NotImplementedError

                        if config["pred_mask_path"] == "/data/scratch/erubel/external_val/utc/experiments/biomedparse_2d":
                            predicted_segmentation = nib.load(os.path.join(config["pred_mask_path"], f"{scan_id}_mask.nii.gz"))
                        elif "nnunet_b3" in config["pred_mask_path"] or "lung_nodules_total_segmentator_postprocessed" in config["pred_mask_path"]:
                            predicted_segmentation = nib.load(os.path.join(config["pred_mask_path"], f"{scan_id}_output_mask_binary.nii.gz")) # this is > 0, not > 0.5
                        else:
                            predicted_segmentation = nib.load(os.path.join(config["pred_mask_path"], f"{scan_id}.nii.gz"))
                    else:
                        assert config["p_value_threshold"] == 0.5 # change what's loaded in below

                        # predicted_segmentation = nib.load(os.path.join(config["pred_mask_path"], f"{scan_id}_output_mask.nii.gz" if "postprocessed" in config["pred_mask_path"] else f"{scan_id}.nii.gz"))
                        # # some have "_mask" in the filename, so we remove it

                        # predicted_segmentation_mask_arr = np.expand_dims(np.transpose(predicted_segmentation.get_fdata() > config["p_value_threshold"], (2, 1, 0)), axis=0).astype(np.uint8)

                        predicted_segmentation = nib.load(os.path.join(config["pred_mask_path"], f"{scan_id}_output_mask_p_0.5_cleaned.nii.gz"))
                    predicted_segmentation_mask_arr = np.expand_dims(np.transpose(predicted_segmentation.get_fdata(), (2, 1, 0)), axis=0).astype(np.uint8)
                else:
                    # note that we must flip about the horizontal axis!
                    # logits
                    predicted_segmentation_mask_arr = np.flip((pickle.load(open(os.path.join(config["pred_mask_path"], f"sample_{scan_id}.hiddens"), "rb"))["hidden"].numpy() > 0.5).astype(np.uint8), axis=2)

                true_segmentation = nib.load(os.path.join(config["true_mask_path"], f"{scan_id}.nii.gz"))
                        
            except FileNotFoundError:
                continue

            true_segmentation_mask_arr = np.expand_dims(np.transpose(true_segmentation.get_fdata(), (2, 1, 0)), axis=0).astype(np.uint8)

            # some sanity checks

            assert predicted_segmentation_mask_arr.shape == true_segmentation_mask_arr.shape
            assert len(predicted_segmentation_mask_arr.shape) == 4 and true_segmentation_mask_arr.shape[0] == 1

            assert np.issubdtype(predicted_segmentation_mask_arr.dtype, np.integer), "Array must have an integer dtype."
            assert np.issubdtype(true_segmentation_mask_arr.dtype, np.integer), "Array must have an integer dtype."

            # (2) get pixel spacing and slice thickness
            
            if config["use_niftis"]: # otherwise, we assume it's well-formed!
                assert predicted_segmentation.header.get_zooms() == true_segmentation.header.get_zooms(), "Expected spacing values to match."

            sx, sy, sz = true_segmentation.header.get_zooms()

            pixel_spacing = [sx, sy]
            slice_thickness = sz

            output[scan_id] = evaluator.evaluate(
                predicted_segmentation_mask_arr,
                true_segmentation_mask_arr,
                pixel_spacing,
                slice_thickness,
            )

            # write every scan!
            with open(os.path.join(
                config["output_dir"],
                f"utc_results_experiment_adjusted_p_{config['p_value_threshold'] if config['p_value_threshold'] is not None else 'NONE'}_{os.path.basename(config['pred_mask_path'])}_group_{config['group_ix']}.json"
            ), "w") as f:
                json.dump(output, f, indent=4)


if __name__ == "__main__":
    # nothing
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/p_0.0_no_lung_mask"

    # lung mask only
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/p_0.0_no_lung_mask_postprocessed_with_lung_mask_no_vessel_mask"
    
    # lung range only
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/p_0.0_no_lung_mask_postprocessed_with_lung_range_no_vessel_mask"
    
    # lung mask + vessel (0.5)
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/p_0.0_no_lung_mask_postprocessed_with_lung_mask_with_vessel_mask_overlap_0.5"

    # lung range + vessel (0.5)
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/p_0.0_no_lung_mask_postprocessed_with_lung_range_with_vessel_mask_overlap_0.5"

    # lung range + vessel (0.8)
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/p_0.0_no_lung_mask_postprocessed_with_lung_mask_with_vessel_mask_overlap_0.8"

    # lung range + vessel (0.5)
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/p_0.0_no_lung_mask_postprocessed_with_lung_range_with_vessel_mask_overlap_0.9"

    # FROZEN
    # lung range + clean extreme slices + sweep adjacent 5 slices + p = 0.5
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/p_0.0_no_lung_mask_postprocessed_with_lung_range_no_vessel_mask_frozen"
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/p_0.0_no_lung_mask_postprocessed_with_lung_range_no_vessel_mask_frozen2"

    ### NEW ###

    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/a5"

    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/b3"
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse_2d"
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/b_latest"
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/combined_all"

    # b3, lung mask
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/b3_postprocessed_with_lung_mask_no_vessel_mask_frozen2_nnunet_b3"
    
    # b3, lung range
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/b3_postprocessed_with_lung_range_no_vessel_mask_frozen2_nnunet_b3"
    
    # total segmentator
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/lung_nodules_total_segmentator"

    # total segmentator, lung mask
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/lung_nodules_total_segmentator_postprocessed_with_lung_mask_no_vessel_mask"

    # total segmentator, lung range
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/lung_nodules_total_segmentator_postprocessed_with_lung_range_no_vessel_mask"


    # FROZEN 5

    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/p_0.0_no_lung_mask_postprocessed_with_lung_range_no_vessel_mask_frozen3"
    # mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/p_0.0_no_lung_mask_postprocessed_with_lung_mask_no_vessel_mask_frozen5"
    
    mask_path = "/data/scratch/erubel/external_val/utc/experiments/biomedparse++/p_0.0_no_lung_mask_postprocessed_with_lung_mask_no_vessel_mask_frozen5_lots"

    config = {
        "min_cluster_size": 25,

        "true_mask_path": "/data/rbg/users/erubel/segmentation/data/nnUNet/nnUNet_raw/Dataset083_Utc/labelsTr",

        # "group_ix": 0,
        # "group_ix": 1,
        # "group_ix": 2,
        "group_ix": 3,

        "total_num_groups": 4,

        "use_niftis": True, # False for masks that we do not have NIFTIs for...

        "p_value_threshold": 0.5,
    }

    config["pred_mask_path"] = mask_path

    if config["p_value_threshold"] is None:
        config["output_dir"] = os.path.join("/data/scratch/erubel/external_val/utc/experiments/others", os.path.basename(mask_path))
    else:
        config["output_dir"] = os.path.join(mask_path, f"p_{config['p_value_threshold']}")
    

    os.makedirs(config["output_dir"], exist_ok=True)

    assert config["group_ix"] in list(range(config["total_num_groups"])), "Expected a valid group number."
    assert isinstance(config["group_ix"], int) and isinstance(config["total_num_groups"], int)
    assert config["group_ix"] in range(config["total_num_groups"])

    print(f"Processing {config['group_ix']}...\n")

    config["fname_src_path"] = f"/data/scratch/erubel/external_val/utc/data/group{config['group_ix']}"

    print("\n\n")
    pprint(config)
    print("\n\n")

    main(config)
