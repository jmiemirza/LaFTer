# LaFTer: Label-Free Tuning of Zero-shot Classifier using Language and Unlabeled Image Collections

This is the official repository for our paper [LaFTer](https://arxiv.org/abs/2305.18287), which has been accepted for 
publication at NeurIPS 2023. 

In this paper, we show that for vision-language models (VLMs), we can train a neural network to classify textual
descriptions of visual instances and this network can directly be applied to classify visual data as well. 
This is possible due to the shared image-text embedding space learned by the VLMs during their large scale 
contrastive pre-training. To further enhance the classification performance for the downstream datasets, 
we employ this text-only pre-trained classifier in a pseudo-labeling pipeline to finetune the visual encoder.

We provide the code for reproducing the results 
for all the 12 datasets used in our paper.

## Installation

Our code is built upon the official codebase of the [CoOp](https://github.dev/KaiyangZhou/CoOp) paper and has been 
tested in an environment with `python 3.8.8` and `pytorch 13.1.1` compiled with `CUDA 11.1`. 

As a first step, install `dassl` library (under `LaFTer/`) in your environment by following the instructions [here](https://github.com/KaiyangZhou/Dassl.pytorch#installation).

To further install all other dependencies, please run the following command, after having your environment activated:

```
pip install -r requirements.txt
```

## Datasets

Under `LaFTer/` first make an empty data folder: 

```
mkdir data
```

Then download and structure your datasets according to the instructions provided in the [CoOp](https://github.dev/KaiyangZhou/CoOp)
official repository. All the `12` datasets should be present in the `data/` directory.

## Descriptions

The class-wise descriptions for the `12` datasets are present in `descriptions/generic` directory. 
The code for generating these descriptions is also provided in the `descriptions/generate_descriptions.py` file.

## Experiments

### LaFTer
To run the full `LaFTer` pipeline, please run the following command:

```
bash scripts/LaFTer.sh <dataset_name>
```

where `<dataset_name>` can be `dtd`, `eurosat`, etc.
### Zero-Shot
Similarly, to obtain zero-shot CLIP results with the single prompt template `a photo of a {category}`. Please run: 

```
bash scripts/zeroshot.sh <dataset_name>
```

by replacing the `<dataset_name>` with one of the 12 datasets evaluated in the paper.


#### To cite us: 
```bibtex
@InProceedings{mirza2023lafter,
    author    = {Mirza, M. Jehanzeb and Karlinsky, Leonid and Lin, Wei and Kozinski, Mateusz and 
                 Possegger, Horst and Feris, Rogerio and Bischof, Horst},
    title     = {LaFTer: Label-Free Tuning of Zero-shot Classifier using Language and Unlabeled Image Collections},
    booktitle = {Conference on Neural Information Processing Systems (NeurIPS)},
    year      = {2023}
}
```

If you are also interested in a follow-up work to LaFter, please check out [TAP: Targeted Prompting](https://arxiv.org/abs/2309.06809).
