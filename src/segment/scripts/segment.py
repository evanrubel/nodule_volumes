import os
import sys

def main(config: dict) -> None:
    """The entry point for the segmentation task."""

    sys.path.append(os.path.join(os.getcwd(), "segment", "scripts"))

    import biomedparse_plus_plus
    import nnInteractive

    # Initial "detection" step with BiomedParse++
    biomedparse_plus_plus.main(config)

    # # Segmentation step where we smooth the outputs with nnInteractive
    nnInteractive.main(config)
