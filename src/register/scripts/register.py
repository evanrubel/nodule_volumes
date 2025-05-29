import os
import sys

def get_pid_to_timepoints(config: dict) -> dict:
    """Collects all of the scans that have multiple timepoints."""

    pid_to_timepoints = {}

    for fname in os.listdir(config["nifti_dir"]):
        if fname.endswith(".nii.gz") and "T" in fname:
            series_id = fname.replace(f"{config['dataset_name']}_", "").replace(".nii.gz", "").replace("_0000", "")
            curr_pid, curr_tp = series_id.split("T")

            if curr_pid in pid_to_timepoints:
                pid_to_timepoints[curr_pid].add(int(curr_tp))
            else:
                pid_to_timepoints[curr_pid] = {int(curr_tp)}

    return pid_to_timepoints


def main(config: dict) -> None:
    """The entry point for the registration task."""

    sys.path.append(os.path.join(os.getcwd(), "register", "scripts"))

    import compute_registration
    import transform_mask
    import match_nodules

    pid_to_timepoints = get_pid_to_timepoints(config)

    # First, find the transforms
    compute_registration.main(pid_to_timepoints, config)

    # Then, apply them to the masks
    transform_mask.main(pid_to_timepoints, config)

    # Finally, match nodules across the registered scans and compute the nodule growth metrics
    match_nodules.main(pid_to_timepoints, config)
