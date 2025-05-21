# nodule_volumes

Note that we have a synced fork for the [BiomedParse](https://github.com/evanrubel/BiomedParse) repository at `src/segment/models/BiomedParse`.

See additional documentation [here](https://docs.google.com/document/d/1My76WuBxeqBuQXIBevDKrWPAox0fJdXXWl1wikzfgds/edit?usp=sharing).


## Installation

1. `git clone git@github.com:evanrubel/nodule_volumes.git`
2. `cd src/segment/models && git clone git@github.com:evanrubel/BiomedParse.git`
3. For the BiomedParse weights, download `biomedparse_v1.pt` from [here](https://huggingface.co/microsoft/BiomedParse/tree/main) and put it in `src/segment/checkpoints`.
4. For the nnInteractive weights, download `checkpoint_final.pth` from [here](https://huggingface.co/nnInteractive/nnInteractive/tree/main/nnInteractive_v1.0/fold_0) and put it in `src/segment/checkpoints/nnInteractive/fold_0`.


## Example Usage

`cd src && python nodule_volumes.py -t segment -d toy --v`

## TODOs
[] Figure out the conda environment `nodule_volumes` --> freeze the requirements for reproducibility
