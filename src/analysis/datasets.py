import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from segmentation_evaluator import NoduleSegmentEvaluator, compute_volume_voxel_count
from tqdm import tqdm
import json

def preprocess(img_arr):
    """From https://github.com/microsoft/BiomedParse/blob/main/inference_utils/processing_utils.py."""
    lower_bound, upper_bound = -1000, 1000
    image_data_pre = np.clip(img_arr, lower_bound, upper_bound)
    image_data_pre = (
        (image_data_pre - image_data_pre.min())
        / (image_data_pre.max() - image_data_pre.min())
        * 255.0
    )

    return image_data_pre


def get_sorted_filenames(nifti_dir) -> list[str]:
    """Returns the sorted filenames in `nifti_dir` to process."""

    all_filenames = []

    if any(["group" in f for f in os.listdir(nifti_dir)]):
        # look in subdivided groups
        for f in os.listdir(nifti_dir):
            if "group" in f and os.path.isdir(os.path.join(nifti_dir, f)):
                all_filenames.extend(os.listdir(os.path.join(nifti_dir, f)))
    elif len([f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")]) >= 1:
        # in one directory
        all_filenames.extend([fname for fname in os.listdir(os.path.join(nifti_dir)) if fname.endswith(".nii.gz")])
    else:
        raise NotImplementedError

    return sorted(all_filenames)

def get_nifti_nib(series_id: str, nifti_dir: str):
    """Checks all subfolders and returns the matching Nibabel object (for `series_id`) within those subfolders."""
    
    if any(["group" in f for f in os.listdir(nifti_dir)]):
        # look in subdivided groups
        for f in os.listdir(nifti_dir):
            if "group" in f and os.path.isdir(os.path.join(nifti_dir, f)):
                for dir_fname in os.listdir(os.path.join(nifti_dir, f)):
                    if dir_fname == f"{series_id}_0000.nii.gz":
                        return nib.load(os.path.join(nifti_dir, f, dir_fname))
    elif len([f for f in os.listdir(nifti_dir) if f.endswith(".nii.gz")]) >= 1:
        # in one directory
        return nib.load(os.path.join(nifti_dir, f"{series_id}_0000.nii.gz"))
    
    raise FileNotFoundError

def analyze_nodule_properties(nifti_dir, mask_dir):
    all_mean_intensity_vals = []
    mean_intensity_vals_with_mask = []
    volumes = []
    num_nodules_per_scan = []
    slice_thicknesses = []
    slice_counts = []

    evaluator = NoduleSegmentEvaluator()

    for dir_ in nifti_dir:
        for fname in tqdm(get_sorted_filenames(dir_)):
            if not fname.endswith(".nii.gz"):
                continue
            nifti_path = os.path.join(dir_, fname)

            # img = nib.load(nifti_path)
            img = get_nifti_nib(fname.replace("_0000", "").replace(".nii.gz", ""), dir_)
            img_data = img.get_fdata()

            if mask_dir is not None:
                mask_path = os.path.join(mask_dir, fname.replace("_0000", ""))
                if not os.path.exists(mask_path):
                    continue
                mask = nib.load(mask_path)
                mask_data = mask.get_fdata()

                assert img_data.shape == mask_data.shape, "Expected shapes to match."
                # assert img.header.get_zooms() == mask.header.get_zooms(), "Expected spacing values to match."

                sx, sy, sz = mask.header.get_zooms()
            else: # for NLST 
                img_data = np.transpose(img_data, (2, 1, 0))
                sz, sy, sx = img.header.get_zooms()

            pixel_spacing = [sx, sy]
            slice_thickness = sz

            preprocessed_img_data = preprocess(img_data)

            all_mean_intensity_vals.append(np.mean(preprocessed_img_data.flatten()).item())

            if mask_dir is not None:
                mean_intensity_vals_with_mask.append(np.mean(preprocessed_img_data[mask_data > 0].flatten()).item())
        
                instance_mask = evaluator.get_instance_segmentation(np.expand_dims(mask_data, axis=0))[0]
                vols = [
                        compute_volume_voxel_count((instance_mask == i).astype(np.uint8), pixel_spacing, slice_thickness)
                        for i in np.unique(instance_mask) if i != 0 # exclude the background
                ]
                volumes.extend(vols)

                num_nodules_per_scan.append(len(vols))
            
            slice_thicknesses.append(float(slice_thickness))
            slice_counts.append(img_data.shape[2])

    return all_mean_intensity_vals, mean_intensity_vals_with_mask, volumes, num_nodules_per_scan, slice_thicknesses, slice_counts


def plot_distributions(all_intensities, mask_intensities, volumes, num_nodules_per_scan, thicknesses, num_slices, dataset_name):
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    axs = axs.flatten()

    axs[0].hist(all_intensities, bins=100, color='blue', alpha=0.7)
    axs[0].set_title(f"Mean Normalized Pixel Intensities [N = {len(all_intensities)}]")

    axs[1].hist(mask_intensities, bins=100, color='blue', alpha=0.7)
    axs[1].set_title(f"Mean Normalized Pixel Intensities of Annotated Regions [N = {len(mask_intensities)}]")

    axs[2].hist(volumes, bins=50, color='green', alpha=0.7)
    axs[2].set_title(f"True Nodule Volumes (mm³) [N = {len(volumes)}]")

    axs[3].hist(num_nodules_per_scan, bins=50, color='orange', alpha=0.7)
    axs[3].set_title(f"Number of True Nodules per Scan [N = {len(num_nodules_per_scan)}]")

    axs[4].hist(thicknesses, bins=30, color='purple', alpha=0.7)
    axs[4].set_title(f"Slice Thicknesses (mm) [N = {len(thicknesses)}]")

    axs[5].hist(num_slices, bins=30, color='red', alpha=0.7)
    axs[5].set_title(f"Number of Slices per Scan [N = {len(num_slices)}]")

    # axs[6].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join("/data/scratch/erubel/nlst/v2/figures/datasets", dataset_name))

# LUNA16
# base_dir = "/data/rbg/users/erubel/segmentation/data/nnUNet/nnUNet_raw/Dataset082_Luna"

# LNDb
# base_dir = "/data/rbg/users/erubel/segmentation/data/nnUNet/nnUNet_raw/Dataset081_Lndb"

# UTC
# base_dir = "/data/rbg/users/erubel/segmentation/data/nnUNet/nnUNet_raw/Dataset083_Utc"

# nifti_dir = os.path.join(base_dir, "imagesTr")
# mask_dir = os.path.join(base_dir, "labelsTr")

###

# NLST
nifti_dir = ["/data/scratch/erubel/nlst/niftis", "/data/rbg/scratch/nlst_benign_nodules/niftis"]
mask_dir = None

all_ints, mask_ints, vols, num_nodules_per_scan, thick, slices = analyze_nodule_properties(nifti_dir, mask_dir)

with open(os.path.join("/data/scratch/erubel/nlst/v2/figures/datasets", f"{os.path.basename(base_dir if mask_dir is not None else nifti_dir[1])}_stats.json"), "w") as f:
    json.dump({
        "all_ints": all_ints,
        "mask_ints": mask_ints,
        "num_nodules_per_scan": num_nodules_per_scan,
        "vols": vols,
        "thick": thick,
        "slices": slices,
    }, f)


plot_distributions(all_ints, mask_ints, vols, num_nodules_per_scan, thick, slices, os.path.basename(base_dir if mask_dir is not None else nifti_dir[1]))
