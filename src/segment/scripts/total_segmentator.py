import numpy as np
import nibabel as nib
import os
import json

from totalsegmentator.python_api import totalsegmentator

from utils.segment_evaluator import NoduleSegmentEvaluator
from utils.misc import get_config
from utils.postprocessing import postprocess_initial


def main(config: dict) -> None:
    """Runs the Total Segmentator inference pipeline for lung nodules."""

    skipped = []

    evaluator = NoduleSegmentEvaluator()

    for fname in sorted(os.listdir(config["nifti_dir"])):
        if fname.endswith(".nii.gz"):
            try:
                series_id = fname.replace("_0000", "").replace(".nii.gz", "")

                image = nib.load(os.path.join(config["nifti_dir"], fname))

                output_mask = totalsegmentator(image, task="lung_nodules", device=f"gpu:{config['device']}", quiet=(not config["debug"])).get_fdata()

                postprocessed_mask = postprocess_initial(np.transpose(output_mask, (2, 1, 0)), series_id, image, evaluator, config)
                final_mask = np.transpose(postprocessed_mask, (2, 1, 0))

                nib.save(
                    nib.Nifti1Image(final_mask, affine=image.affine, header=image.header),
                    os.path.join(config["output_dir"], f"{series_id}_initial.nii.gz"),
                )
            except Exception as e:
                print(f"Skipping {series_id} due to error {str(e)}...")
                skipped.append(series_id)
    
    with open(os.path.join(config["output_dir"], "total_segmentator_skipped.json"), "w") as f:
        json.dump(skipped, f, indent=4)
    

if __name__ == "__main__":
    config = get_config()
    main(config)
