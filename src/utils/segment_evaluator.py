from typing import List, Tuple
import numpy as np
from fast_hdbscan import HDBSCAN
import time
from pprint import pprint


def get_tp_fp_fn_tn(net_output, y_onehot, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """

    x_, y_ = 1 - net_output, 1 - y_onehot
    tp = net_output * y_onehot
    fp = net_output * y_
    fn = x_ * y_onehot
    tn = x_ * y_

    if square:
        tp = tp**2
        fp = fp**2
        fn = fn**2
        tn = tn**2

    tp = np.sum(tp)
    fp = np.sum(fp)
    fn = np.sum(fn)
    tn = np.sum(tn)

    return tp, fp, fn, tn


def compute_volume_voxel_count(
    processed_mask_3d, pixel_spacing, slice_thickness
) -> float:
    """
    Returns volume in mm^3 by counting voxels in a given 3D mask.

    We assume that we have a single connected component.

    To convert to mL, multiply by 0.001.

    See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8899957/#:~:text=Voxel%2Dcounting-,method,-Nodule%20volume%20(.
    """
    num_voxels = np.sum(processed_mask_3d > 0)
    mm3_per_voxel = pixel_spacing[0] * pixel_spacing[1] * slice_thickness
    return (num_voxels * mm3_per_voxel).item()


def get_nonzero_slices(segmentation: np.ndarray) -> list[int]:
    """Returns a sorted list of slice indices in which `segmentation` has at least one non-zero pixel."""

    return sorted(np.where(np.any(segmentation > 0, axis=(0, 2, 3)))[0].tolist())


class NoduleSegmentEvaluator:
    def __init__(
        self,
        min_cluster_size: int = 25,
        nodule_identification_threshold: float = 0,
    ) -> None:
        """
        Initialize the evaluator.

        Parameters
        ----------
        min_cluster_size : int
            minimum cluster size for clustering
        nodule_identification_threshold : float, optional
            percent overlap needed to consider a nodule identified, by default 0
        """
        self.scanner = HDBSCAN(
            min_cluster_size=min_cluster_size, allow_single_cluster=True
        )
        self.nodule_identification_threshold = nodule_identification_threshold

    def evaluate(
        self,
        predicted_segmentation: np.ndarray,
        true_segmentation: np.ndarray,
        pixel_spacing: List[float],
        slice_thickness: float,
    ) -> dict[str, float]:
        """
        Evaluate model predictions.

        Parameters
        ----------
        predicted_segmentation : np.ndarray 1xNxHxW
            model prediction
        true_segmentation : np.ndarray 1xNxHxW
            ground truth
         pixel_spacing : List[float]
            spacing in x-y plane
        slice_thickness : float
            spacing in z direction

        Returns
        -------
        dict
            evaluation metrics
        """

        assert predicted_segmentation.shape == true_segmentation.shape
        assert len(predicted_segmentation.shape) == 4 and predicted_segmentation.shape[0] == 1
        
        assert np.issubdtype(predicted_segmentation.dtype, np.integer), "Array must have an integer dtype."
        assert np.issubdtype(true_segmentation.dtype, np.integer), "Array must have an integer dtype."

        metrics = {}

        predicted_segmentation = (predicted_segmentation > 0.5) * 1
        scan_wise_dice = self.get_scan_wise_dice(
            predicted_segmentation, true_segmentation
        )
        instance_segmentation = self.get_instance_segmentation(true_segmentation)
        predicted_instance_segmentation = self.get_instance_segmentation(
            predicted_segmentation
        )

        slice_identification, nodule_identification, nodule_dices, matched_volumes = (
            self.get_nodule_wise_metrics(
                predicted_instance_segmentation, instance_segmentation, pixel_spacing, slice_thickness,
            )
        )

        metrics = {
            "scan_wise_dice": scan_wise_dice,
            "slice_identification": slice_identification,
            "nodule_identification": nodule_identification,
            "nodule_dices": nodule_dices,
            "matched_volumes": matched_volumes,
            "num_true_volumes": len(nodule_dices),
            # number of predicted instances - number of matched volumes
            "num_unmatched_pred_volumes": len([val for val in np.unique(predicted_instance_segmentation).tolist() if val > 0]) - len([vol for vol in matched_volumes if vol["predicted_volume"] != 0]),
        }

        # some regression testing to ensure things are matched correctly
        assert metrics["num_true_volumes"] >= 0
        assert metrics["num_unmatched_pred_volumes"] >= 0
        assert len([val for val in metrics["nodule_identification"] if val]) >= 0

        return metrics

    def get_instance_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Get instance segmentation from a binary segmentation. This is done by clustering the binary segmentation.

        Parameters
        ----------
        segmentation : np.ndarray
            binary segmentation

        Returns
        -------
        np.ndarray
            instance segmentation
        """
        # cluster the segmentation
        instance_segmentation = np.zeros_like(segmentation)
        nodule_coords = np.where(segmentation)
        nodule_coords_ = np.stack(nodule_coords, 1)

        if len(nodule_coords_) < 2:
            return segmentation

        labels = self.scanner.fit_predict(nodule_coords_)
        if np.all(labels == -1):
            instance_segmentation[nodule_coords] = np.ones_like(labels) # treat as a single cluster
        else:
            instance_segmentation[nodule_coords] = labels + 1
        return instance_segmentation

    def get_scan_wise_dice(
        self, predicted_segmentation: np.ndarray, true_segmentation: np.ndarray
    ) -> float:
        """
        Calculate Dice coefficient for a 3D scan.

        Parameters
        ----------
        predicted_segmentation : np.ndarray
            model prediction
        true_segmentation : np.ndarray
            ground truth

        Returns
        -------
        float
            Dice coefficient
        """
        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation, true_segmentation)
        nominator = 2 * tp
        denominator = 2 * tp + fp + fn
        dc = nominator / (denominator + 1e-8)
        return dc

    
    def get_nodule_wise_metrics(
        self, predicted_segmentation: np.ndarray, true_segmentation: np.ndarray, pixel_spacing: float, slice_thickness: float,
    ) -> Tuple[List[bool], List[bool], List[float], List[dict]]:
        """
        Calculate metrics for each nodule in a 3D scan.

        Parameters
        ----------
        predicted_segmentation : np.ndarray
            model prediction
        true_segmentation : np.ndarray
            ground truth

        Returns
        -------
        tuple
            slice_identification: whether the nodule slice is identified
            nodule_identification: whether the nodule is identified at some non-zero IoU
            nodule_dices: dice coefficient for each nodule, excluding other predicted nodules
            matched_volumes: list of dictionaries with match details
        """

        nodule_ids = np.unique(true_segmentation)
        nodule_ids = nodule_ids[nodule_ids != 0] # exclude background

        used_predicted_ids = set()
        nodule_dices = []
        nodule_identification = []
        slice_identification = []
        matched_volumes = []

        for nodule_id in nodule_ids:
            nodule_mask = true_segmentation == nodule_id

            # check if any predicted segmentation overlaps slice-wise
            is_frames_overlap = self.get_slice_identification(predicted_segmentation, nodule_mask)
            slice_identification.append(is_frames_overlap)

            # get predicted IDs overlapping with this true nodule
            overlapping_preds = predicted_segmentation[nodule_mask]
            pred_ids = np.unique(overlapping_preds[overlapping_preds > 0])

            best_iou = 0.0
            best_pred_id = None

            for pred_id in pred_ids:
                if pred_id in used_predicted_ids:
                    continue

                pred_mask = predicted_segmentation == pred_id
                intersection = np.logical_and(pred_mask, nodule_mask).sum().item()
                union = np.logical_or(pred_mask, nodule_mask).sum().item()
                iou = intersection / union if union > 0 else 0

                if iou > best_iou:
                    best_iou = iou
                    best_pred_id = pred_id

            if best_pred_id is not None:
                used_predicted_ids.add(best_pred_id)
                predicted_nodule_mask = predicted_segmentation == best_pred_id

                nodule_identification.append(best_iou > self.nodule_identification_threshold)
                dice = self.get_scan_wise_dice(predicted_nodule_mask, nodule_mask)
                nodule_dices.append(dice)
            else:
                predicted_nodule_mask = np.zeros_like(predicted_segmentation, dtype=bool)
                nodule_identification.append(False)
                nodule_dices.append(0.0)

            true_volume = compute_volume_voxel_count(
                nodule_mask, pixel_spacing, slice_thickness,
            )
            predicted_volume = compute_volume_voxel_count(
                predicted_nodule_mask, pixel_spacing, slice_thickness,
            )

            vol_data = {
                    "true_volume": true_volume,
                    "predicted_volume": predicted_volume,
                    "true_slice_range": get_nonzero_slices(nodule_mask),
                    "predicted_slice_range": get_nonzero_slices(predicted_nodule_mask),
                    "nodule_id": int(nodule_id),
                    "predicted_nodule_id": int(best_pred_id) if best_pred_id is not None else None,
                    "iou": float(best_iou),
                }

            matched_volumes.append(vol_data)

        return slice_identification, nodule_identification, nodule_dices, matched_volumes

    def get_slice_identification(
        self, predicted_segmentation: np.ndarray, true_segmentation: np.ndarray
    ) -> bool:
        """
        Check if the nodule slice is identified by the model.

        Parameters
        ----------
        predicted_segmentation : np.ndarray
            model prediction
        true_segmentation : np.ndarray
            ground truth

        Returns
        -------
        bool
            whether the nodule slice is identified
        """
        nodule_frames = np.where(true_segmentation.sum((0,2,3)) > 0)[0]
        predicted_frames = np.where(predicted_segmentation.sum((0,2,3)) > 0)[0]
        return np.intersect1d(nodule_frames, predicted_frames).size > 0
