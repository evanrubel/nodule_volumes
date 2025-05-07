import biomedparse_plus_plus
import nnInteractive

def main():
    """The entry point for the segmentation task."""

    # Initial "detection" step with BiomedParse++
    biomedparse_plus_plus.main()

    # Segmentation step where we smooth the outputs with nnInteractive
    nnInteractive.main()

