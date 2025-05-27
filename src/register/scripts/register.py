import os
import sys


def main(config: dict) -> None:
    """The entry point for the registration task."""

    print(os.getcwd())

    sys.path.append(os.path.join(os.getcwd(), "register", "scripts"))

    # from utils.misc import get_pid_to_timepoints
    import utils
    import compute_registration
    import transform_mask
    import match_nodules

    pid_to_timepoints = get_pid_to_timepoints(config)    

    # First, find the transforms
    # compute_registration.main(pid_to_timepoints, config)

    # # Then, apply them to the masks
    # transform_mask.main(pid_to_timepoints, config)

    # # Finally, match nodules across the registered scans and compute the nodule growth metrics
    # match_nodules.main(pid_to_timepoints, config)
