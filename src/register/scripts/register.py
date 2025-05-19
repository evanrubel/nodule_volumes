import os
import sys


def main(config: dict) -> None:
    """The entry point for the registration task."""

    sys.path.append(os.path.join(os.getcwd(), "register", "scripts"))

    import compute_registration
    # import transform_mask
    # import match_nodules

    # First, find the transforms
    compute_registration.main(config)

    # Then, apply them to the masks
    transform_mask.main(config)

    # Finally, match nodules across the registered scans
    # match_nodules.main(config)
