import numpy as np
import os
from scipy.ndimage import label
import numpy as np
import scipy.ndimage
from skimage import measure
from skimage.morphology import convex_hull_image
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
import pydicom
from tqdm import tqdm
from argparse import ArgumentParser
import pickle
import nibabel as nib

# Lung Masks #

# def get_pixels_hu(img):
#     image = np.stack([s.pixel_array for s in slices])
#     # Convert to int16 (from sometimes int16),
#     # should be possible as values should always be low enough (<32k)
#     image = image.astype(np.int16)

#     # Convert to Hounsfield units (HU)
#     for slice_number in range(len(slices)):
#         intercept = slices[slice_number].RescaleIntercept
#         slope = slices[slice_number].RescaleSlope

#         if slope != 1:
#             image[slice_number] = slope * image[slice_number].astype(np.float64)
#             image[slice_number] = image[slice_number].astype(np.int16)

#         image[slice_number] += np.int16(intercept)

#     return np.array(image, dtype=np.int16), np.array(
#         [slices[0].SliceThickness] + list(slices[0].PixelSpacing), dtype=np.float32
#     )

def get_pixels_hu_nifti(img):
    # we transpose here to go from (x, y, z) -> (z, y, x)
    img_arr = np.transpose(img.get_fdata(dtype=np.float32), (2, 1, 0))  # ensures float representation if scaling is needed

    # apply scaling if necessary (slope and intercept)
    slope = img.header.get_slope_inter()  # returns (slope, intercept)
    if slope is not None:
        rescale_slope, rescale_intercept = slope
        if rescale_slope is not None:
            img_arr = img_arr * rescale_slope
        if rescale_intercept is not None:
            img_arr = img_arr + rescale_intercept

    # get spacing information: [slice_thickness, pixel_spacing_row, pixel_spacing_col]
    header = img.header
    zooms = header.get_zooms()

    # as above, we reverse order of zooms [which is of size (3,)] to go from (x, y, z) -> (z, y, x)
    spacing = np.array(zooms, dtype=np.float32)[::-1]

    return img_arr.astype(np.int16), spacing


def binarize_per_slice(
    image,
    spacing,
    intensity_th=-600,
    sigma=1,
    area_th=30,
    eccen_th=0.99,
    bg_patch_size=10,
):
    bw = np.zeros(image.shape, dtype=bool)

    # prepare a mask, with all corner values set to nan
    image_size = image.shape[1]
    grid_axis = np.linspace(-image_size / 2 + 0.5, image_size / 2 - 0.5, image_size)
    x, y = np.meshgrid(grid_axis, grid_axis)
    d = (x**2 + y**2) ** 0.5
    nan_mask = (d < image_size / 2).astype(float)
    nan_mask[nan_mask == 0] = np.nan
    for i in range(image.shape[0]):
        # Check if corner pixels are identical, if so the slice  before Gaussian filtering
        if len(np.unique(image[i, 0:bg_patch_size, 0:bg_patch_size])) == 1:
            current_bw = (
                scipy.ndimage.filters.gaussian_filter(
                    np.multiply(image[i].astype("float32"), nan_mask),
                    sigma,
                    truncate=2.0,
                )
                < intensity_th
            )
        else:
            current_bw = (
                scipy.ndimage.filters.gaussian_filter(
                    image[i].astype("float32"), sigma, truncate=2.0
                )
                < intensity_th
            )

        # select proper components
        label = measure.label(current_bw)
        properties = measure.regionprops(label)
        valid_label = set()
        for prop in properties:
            if (
                prop.area * spacing[1] * spacing[2] > area_th
                and prop.eccentricity < eccen_th
            ):
                valid_label.add(prop.label)
        current_bw = np.in1d(label, list(valid_label)).reshape(label.shape)
        bw[i] = current_bw

    return bw


def all_slice_analysis(
    bw, spacing, cut_num=0, vol_limit=[0.68, 8.2], area_th=6e3, dist_th=62
):
    # in some cases, several top layers need to be removed first
    if cut_num > 0:
        bw0 = np.copy(bw)
        bw[-cut_num:] = False
    label = measure.label(bw, connectivity=1)
    # remove components access to corners
    mid = int(label.shape[2] / 2)
    bg_label = set(
        [
            label[0, 0, 0],
            label[0, 0, -1],
            label[0, -1, 0],
            label[0, -1, -1],
            label[-1 - cut_num, 0, 0],
            label[-1 - cut_num, 0, -1],
            label[-1 - cut_num, -1, 0],
            label[-1 - cut_num, -1, -1],
            label[0, 0, mid],
            label[0, -1, mid],
            label[-1 - cut_num, 0, mid],
            label[-1 - cut_num, -1, mid],
        ]
    )
    for l in bg_label:
        label[label == l] = 0

    # select components based on volume
    properties = measure.regionprops(label)
    for prop in properties:
        if (
            prop.area * spacing.prod() < vol_limit[0] * 1e6
            or prop.area * spacing.prod() > vol_limit[1] * 1e6
        ):
            label[label == prop.label] = 0

    # prepare a distance map for further analysis
    x_axis = (
        np.linspace(-label.shape[1] / 2 + 0.5, label.shape[1] / 2 - 0.5, label.shape[1])
        * spacing[1]
    )
    y_axis = (
        np.linspace(-label.shape[2] / 2 + 0.5, label.shape[2] / 2 - 0.5, label.shape[2])
        * spacing[2]
    )
    x, y = np.meshgrid(x_axis, y_axis)
    d = (x**2 + y**2) ** 0.5
    vols = measure.regionprops(label)
    valid_label = set()
    # select components based on their area and distance to center axis on all slices
    for vol in vols:
        single_vol = label == vol.label
        slice_area = np.zeros(label.shape[0])
        min_distance = np.zeros(label.shape[0])
        for i in range(label.shape[0]):
            slice_area[i] = np.sum(single_vol[i]) * np.prod(spacing[1:3])
            min_distance[i] = np.min(
                single_vol[i] * d + (1 - single_vol[i]) * np.max(d)
            )

        if (
            np.average(
                [
                    min_distance[i]
                    for i in range(label.shape[0])
                    if slice_area[i] > area_th
                ]
            )
            < dist_th
        ):
            valid_label.add(vol.label)

    bw = np.in1d(label, list(valid_label)).reshape(label.shape)

    # fill back the parts removed earlier
    if cut_num > 0:
        # bw1 is bw with removed slices, bw2 is a dilated version of bw, part of their intersection is returned as final mask
        bw1 = np.copy(bw)
        bw1[-cut_num:] = bw0[-cut_num:]
        bw2 = np.copy(bw)
        bw2 = scipy.ndimage.binary_dilation(bw2, iterations=cut_num)
        bw3 = bw1 & bw2
        label = measure.label(bw, connectivity=1)
        label3 = measure.label(bw3, connectivity=1)
        l_list = list(set(np.unique(label)) - {0})
        valid_l3 = set()
        for elem in l_list:
            indices = np.nonzero(label == elem)
            l3 = label3[indices[0][0], indices[1][0], indices[2][0]]
            if l3 > 0:
                valid_l3.add(l3)
        bw = np.in1d(label3, list(valid_l3)).reshape(label3.shape)

    return bw, len(valid_label)


def fill_hole(bw):
    # fill 3d holes
    label = measure.label(~bw)
    # idendify corner components
    bg_label = set(
        [
            label[0, 0, 0],
            label[0, 0, -1],
            label[0, -1, 0],
            label[0, -1, -1],
            label[-1, 0, 0],
            label[-1, 0, -1],
            label[-1, -1, 0],
            label[-1, -1, -1],
        ]
    )
    bw = ~np.in1d(label, list(bg_label)).reshape(label.shape)

    return bw


def two_lung_only(bw, spacing, max_iter=22, max_ratio=4.8):
    def extract_main(bw, cover=0.95):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            properties.sort(key=lambda x: x.area, reverse=True)
            area = [prop.area for prop in properties]
            count = 0
            sum = 0
            while sum < np.sum(area) * cover:
                sum = sum + area[count]
                count = count + 1
            filter = np.zeros(current_slice.shape, dtype=bool)
            for j in range(count):
                bb = properties[j].bbox
                filter[bb[0] : bb[2], bb[1] : bb[3]] = (
                    filter[bb[0] : bb[2], bb[1] : bb[3]] | properties[j].convex_image
                )
            bw[i] = bw[i] & filter

        label = measure.label(bw)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        bw = label == properties[0].label
        return bw

    def fill_2d_hole(bw):
        for i in range(bw.shape[0]):
            current_slice = bw[i]
            label = measure.label(current_slice)
            properties = measure.regionprops(label)
            for prop in properties:
                bb = prop.bbox
                current_slice[bb[0] : bb[2], bb[1] : bb[3]] = (
                    current_slice[bb[0] : bb[2], bb[1] : bb[3]] | prop.filled_image
                )
            bw[i] = current_slice
        return bw

    found_flag = False
    iter_count = 0
    bw0 = np.copy(bw)
    while not found_flag and iter_count < max_iter:
        label = measure.label(bw, connectivity=2)
        properties = measure.regionprops(label)
        properties.sort(key=lambda x: x.area, reverse=True)
        if len(properties) > 1 and properties[0].area / properties[1].area < max_ratio:
            found_flag = True
            bw1 = label == properties[0].label
            bw2 = label == properties[1].label
        else:
            bw = scipy.ndimage.binary_erosion(bw)
            iter_count = iter_count + 1

    if found_flag:
        d1 = scipy.ndimage.morphology.distance_transform_edt(
            bw1 == False, sampling=spacing
        )
        d2 = scipy.ndimage.morphology.distance_transform_edt(
            bw2 == False, sampling=spacing
        )
        bw1 = bw0 & (d1 < d2)
        bw2 = bw0 & (d1 > d2)

        bw1 = extract_main(bw1)
        bw2 = extract_main(bw2)

    else:
        bw1 = bw0
        bw2 = np.zeros(bw.shape).astype("bool")

    bw1 = fill_2d_hole(bw1)
    bw2 = fill_2d_hole(bw2)
    bw = bw1 | bw2

    return bw1, bw2, bw

def process_mask(mask):
    convex_mask = np.copy(mask)
    for i_layer in range(convex_mask.shape[0]):
        mask1 = np.ascontiguousarray(mask[i_layer])
        if np.sum(mask1) > 0:
            mask2 = convex_hull_image(mask1)
            if np.sum(mask2) > 1.5 * np.sum(mask1):
                mask2 = mask1
        else:
            mask2 = mask1
        convex_mask[i_layer] = mask2
    struct = generate_binary_structure(3, 1)
    dilatedMask = binary_dilation(convex_mask, structure=struct, iterations=10)
    return dilatedMask


def generate_single_lung_mask(fname: str, series_id: str, config: dict) -> None:
    # load dicoms or read volume
    # slices = [pydicom.dcmread(p) for p in exam["paths"]]
    # case pixels: N, H, W
    # spacing: [thickness, pixel_spacing_x, pixel_spacing_y]
    # case_pixels, spacing = get_pixels_hu(slices)

    img = nib.load(os.path.join(config["nifti_dir"], fname))
    case_pixels, spacing = get_pixels_hu_nifti(img)
    bw = binarize_per_slice(case_pixels, spacing)
    flag = 0
    cut_num = 0
    cut_step = 2
    bw0 = np.copy(bw)
    while flag == 0 and cut_num < bw.shape[0]:
        bw = np.copy(bw0)
        bw, flag = all_slice_analysis(
            bw, spacing, cut_num=cut_num, vol_limit=[0.68, 7.5]
        )
        cut_num = cut_num + cut_step

    bw = fill_hole(bw)
    bw1, bw2, bw = two_lung_only(bw, spacing)
    m1, m2 = bw1, bw2
    dm1 = process_mask(m1)
    dm2 = process_mask(m2)
    dilatedMask = dm1 + dm2
    final_mask = m1 + m2

    # transpose from (z, y, x) to (x, y, z)
    nib.save(
        nib.Nifti1Image(np.transpose(final_mask, (2, 1, 0)), affine=img.affine, header=img.header),
        os.path.join(config["dataset_dir"], "lung_masks", f"{series_id}.nii.gz"),
    )


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

# Lung Vessel Mask

def apply_vessel_mask_and_remove_whole_instances(instance_mask: np.ndarray, vessel_mask: np.ndarray, config: dict) -> np.ndarray:
    """
    Removes instances from the binarized `instance_mask` that have an overlap >= `config["overlap_threshold"]` with `vessel_mask` since they are likely to be vessels.

    Args:
        instance_mask (np.ndarray): 3D array (D, H, W) with instance IDs.
        vessel_mask (np.ndarray): 3D binary array (D, H, W) where 1 = vessel region.
        config (dict): Dictionary with optional "debug" flag.
    
    Returns:
        np.ndarray: Binary mask with high-overlap instances removed.
    """

    overlap_threshold = config["overlap_threshold"]

    instance_ids = np.unique(instance_mask)
    instance_ids = instance_ids[instance_ids != 0] # exclude background

    instances_to_keep = []
    
    for instance_id in instance_ids:
        instance_mask_only = (instance_mask == instance_id)
        intersection = np.sum(instance_mask_only & vessel_mask)
        instance_area = np.sum(instance_mask_only)

        overlap_ratio = intersection / instance_area if instance_area > 0 else 0.0

        if config["debug"]:
            print(f"Instance {instance_id}: Overlap Ratio {overlap_ratio}")

        if overlap_ratio < overlap_threshold:
            instances_to_keep.append(instance_id)
        else:
            if config["debug"]:
                print(f"Removing instance {int(instance_id)} on slices {np.where(np.any(instance_mask_only, axis=(1, 2)))[0].tolist()}...")

    return np.isin(instance_mask, instances_to_keep)