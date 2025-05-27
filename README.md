# nodule_volumes

Note that we have a synced fork for the [BiomedParse](https://github.com/evanrubel/BiomedParse) repository at `src/segment/models/BiomedParse`.

See additional documentation [here](https://docs.google.com/document/d/1My76WuBxeqBuQXIBevDKrWPAox0fJdXXWl1wikzfgds/edit?usp=sharing).


## Installation

1. `git clone git@github.com:evanrubel/nodule_volumes.git`
2. `cd src/segment/models && git clone git@github.com:evanrubel/BiomedParse.git`


## Example Usage

### Setting up the Data

All inputs must be folders of DICOMS (name the directory the desired series ID) or NIFTIs. The NIFTIs must end with the suffix `_0000.nii.gz` to indicate that they are input images. We also accept an array of PIDs stored in a JSON entitled `nlst_pids.json`. We then ingest all of the timepoints for each PID.

### Configuration

{
    "device": 0,
    "detection_model": "biomedparse++",
    "p_f_threshold": 0.2,
    "lung_vessel_overlap_threshold": null,
    "lung_mask_mode": "mask",
    "prompt_type": "mask",
    "prompt_subset_type": "all"
}

### Execution

`cd src && python nodule_volumes.py -t segment -d toy --v`


## TODOs
[] Figure out the conda environment `nodule_volumes` --> freeze the requirements for reproducibility
