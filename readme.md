# TAP: TARGETED PROMPTING

This is the official repository for our paper *TAP: TARGETED PROMPTING FOR TASK ADAPTIVE GENERATION OF TEXTUAL
TRAINING INSTANCES FOR VISUAL CLASSIFICATION*. We provide the code for reproducing the results 
for all the 8 datasets used in our paper.

## Installation

Our code is built upon the official codebase of the [CoOp](https://github.dev/KaiyangZhou/CoOp) paper and has been 
tested in an environment having `python 3.8.8` and `pytorch 2.0.1` compiled with `CUDA 11.6`. 

As a first step, install `dassl` library (under `TAP/`) in your environment by following the instructions [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation).

To further install all other dependencies, please run the following command, after having your environment activated:

```pip install -r requirements.txt```

## Datasets

Please download and structure your datasets according to the instructions provided in the [CoOp](https://github.dev/KaiyangZhou/CoOp)
official repository. All the `8` datasets should be present in the `data/` directory.

## Descriptions

The generic and dataset specific descriptions for all the 8 datasets are present in the `descriptions/` directory.

## Experiments

### TAP
To reproduce the results for `TAP` all the 8 datasets in Table 1, please run the following command:

```bash scripts/tap.sh <dataset_name>```

where `<dataset_name>` can be one of `dtd` `oxford_flowers` `imagenet_r` `fgvc_aircraft` `food101` `eurosat` `ucf101` `sun397`

### Zero-Shot
Similarly, to obtain zero-shot CLIP results with the single prompt template `a photo of a {category}`. Please run: 

```bash scripts/zeroshot.sh <dataset_name>```

by replacing the `<dataset_name>` with one of the 8 datasets mentioned above.


