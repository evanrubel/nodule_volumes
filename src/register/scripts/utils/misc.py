import os

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
