# nodule_volumes

## Description

We present `nodule_volumes`, an open-source repository to segment and track lung nodules in longitudinal CT scans in order to assess nodule growth over time.

## TODOs
[] Figure out the conda environment `nodule_volumes` --> freeze the requirements for reproducibility
[] Change CSV to JSON (nlst_10000T0 -> full path to dicom series directory)

Note that we have a synced fork for the [BiomedParse](https://github.com/evanrubel/BiomedParse) repository at `src/segment/models/BiomedParse`.

See additional documentation [here](https://docs.google.com/document/d/1My76WuBxeqBuQXIBevDKrWPAox0fJdXXWl1wikzfgds/edit?usp=sharing).


## Installation

1. `git clone git@github.com:evanrubel/nodule_volumes.git`
2. `cd src/segment/models && git clone git@github.com:evanrubel/BiomedParse.git`
3. Configure a conda environment called `biomedparse` as per [these](https://github.com/microsoft/BiomedParse?tab=readme-ov-file#installation) instructions.
4. Configure a conda environment called `nnInteractive` with Python 3.10 by `conda create -n nnInteractive python=3.10 -y && conda activate nnInteractive && pip install nninteractive`.


## Example Usage

### Setting up the Data

Once you have cloned this repository, create the directory for the input datasets by running `mkdir nodule_volumes/data`. We treat each directory in `nodule_volumes/data` as a standalone dataset which will contain the input data to process as well as any outputs. Each directory will have the following structure:

<pre>
```
nodule_volumes
    data
        my_dataset
            images # a directory for the input data
            config.json # see Configuration for details

```
</pre>

In the `images` directory, we accept three valid input formats:

1. A NIFTI ending in `_0000.nii.gz` and in the RAS orientation
2. A directory of DICOM slices (the name of the directory will be the corresponding scan ID)
3. A CSV with columns `pid`, `Series_0`, `Series_1`, and `Series_2` (the latter three correspond to the directory for the scans at TP 0, 1, and 2, respectively.)

You can mix and match input data of these three types, and the pipeline allows for an arbitrary number of inputs.

For nodule matching across longitudinal scans, make sure to include `T0`, `T1`, or `T2` in the filename immediately preceding `_0000` (e.g., `nlst_123456T0_0000.nii.gz` for a NIFTI file or `nlst_123456T0_0000` for a directory name).

### Configuration

<pre>
```{
    "device": 0, # [int] the GPU to use for processing
    "detection_model": "biomedparse++", # ["biomedparse" | "biomedparse++" | "total_segmentator"]
    "lung_mask_mode": "mask", # ["mask" | "range" | null]
    "lung_vessel_overlap_threshold": null, # [float | null] in the range of [0, 1]
    "p_f_threshold": 0.2, # [float] in the range of [0, 1]
    "prompt_type": "mask", # ["mask" | "bbox" | "pos_point"]
    "prompt_subset_type": "all" ["all" | "maximum" | "median"]
}```
</pre>

Our pipeline works in two stages.

1. We have an initial detection using the `detection_model` specified. With `lung_mask_mode`, we also optionally incorporate a mask of the lungs to remove any outputs outside of the lungs -- `mask` will remove any outputs outside of the lungs, `range` will remove any outputs beyond the range of z-slices (but not slices that have some part of the lungs on them), and `null` does not apply anything. Similarly, we can specify a threshold value for `lung_vessel_overlap_threshold` that incorporates the lung *vessel* mask in an effort to further reduce false positives -- a threshold value of 0.2, for instance, will remove any outputs of the mask that have an overlap of at least `lung_vessel_overlap_threshold` with the lung vessel mask. For BiomedParse++ only, we require a `p_f_threshold` between 0 and 1, which reduces false positives as we increase the threshold.

2. We use [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive) to smooth the segmentation outputs in 3D. We prompt `nnInteractive` based on the `prompt_type`, and we also specify what subset of prompts to use with `prompt_subset_type`.

### Execution

With the `-t` flag, specify the task to execute (`segment`, `register`, or `full` which performs both in sequence). With the `-d` flag, specify the dataset to process (i.e., the directory name [`my_dataset`, for instance, as above]). `--v` shows verbose outputs.

`cd src && python nodule_volumes.py -t full -d nlst --v`

After running the above script, the pipeline will generate the following directories within the dataset directory:

1. `lung_masks` -- if applicable, the masks of the lungs for the input scans
2. `lung_vessel_masks` -- if applicable, the masks of the lung vessels for the input scans
3. `results` -- timestamped directories containing the initial segmentation masks (`_initial.nii.gz`), the final segmentation masks, the registered masks, and the nodule volume outputs as a JSON.
4. `transforms` -- cached transforms for registration
