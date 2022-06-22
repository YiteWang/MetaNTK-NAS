# MetaNTK-NAS: Global Convergence of MAML and Theory-Inspired Neural Architecture Search for Few-Shot Learning [Paper](https://arxiv.org/abs/2203.09137)

Haoxiang Wang*, Yite Wang*, Ruoyu Sun, Bo Li

In CVPR 2022. 

If you find this repo useful for your research, please consider citing our paper
```
@inproceedings{MetaNTK-NAS,
  title={Global Convergence of MAML and Theory-Inspired Neural Architecture Search for Few-Shot Learning},
  author={Wang, Haoxiang and Wang, Yite and Sun, Ruoyu and Li, Bo},
  booktitle={CVPR},
  year={2022}
}
```

## Overview

This is the PyTorch implementation of MetaNTK-NAS, a training-free NAS method for few-shot learning based on Meta Neural Tangent Kernels (MetaNTK).

## Installation

This repository has been tested with RedHat with Pytorch 1.3.1 on NVIDIA V100 GPUs and Ubuntu with Pytorch 1.10 on GTX 3090 and NVIDIA V100 GPUs. For other platforms, configurations may need to be changed.

#### Required packages

- Common packages: numpy, scipy, scikit-learn, easydict, pillow etc.
- Pytorch packages: [Pytorch](https://pytorch.org/), Torchvision, [torch-optimizer](https://github.com/jettify/pytorch-optimizer).
- Packages for efficient gradient computation: [Opacus](https://opacus.ai/).
- Packages for counting operations and parameters for architectures: [ptflops](https://pypi.org/project/ptflops/).

Or you can simply install all dependencies using:

`pip install -r requirements.txt`

## Usage

### 0. Prepare the dataset

* Please download MiniImageNet and TieredImageNet dataset from [RFS](https://github.com/WangYueFt/rfs).
* Please properly set the `data_paths` in the `prune_launch.py`.

### 1. Search

#### [DARTS_fewshot Space](https://arxiv.org/pdf/1911.11090.pdf)

You may want to check the sample scripts in `scripts` folder. It will call `prune_lanch.py` with predefined configurations. Here are multiple arguments you might want to modify to replicate our experiment results.

- `--gpu`: Which GPU to use during search/train/evaluation.
- `--space`: Which search space to use. In our implementation, we use `'darts_fewshot'`. Choose from `['darts', 'darts_fewshot']`.
- `--dataset`: Dataset to use for search/train/evaluation. If you want to search with NTK, choose one of `['MiniImageNet', 'TieredImageNet']`.  On the other hand, if you want to search with MetaNTK, you will need to add 'Meta' to the front. Use `'MetaMiniImageNet'`, for example.
- `--seed`: Manual seed. For random seed, set it to `-1`.
- `--max_nodes`: Number of intermediate nodes in each cell. In our experiments, we use `3` as  default value.
- `--dartsbs`: (Meta) Batch size used for searching, make sure the total number of samples used for NTK and MetaNTK are the same. We use meta batch size of `3` as default value for MetaNTK-NAS. Since we use 5 way 1 query/support sample as one meta batch, 30 should be used for NTK as batch size as a fair comparison.
- `--ntk_type`: Search based on condition numbers of NTK or MetaNTK. Choose from `['NTK', 'MetaNTK_anl']`.
- `--ntk_channels`: Initial number of channels of architecture search/train/evaluation. We set this to `48` for all experiments.
- `--ntk_layers`: Number of cells to stack for the final architecture during search/train/evaluation. In our experiments, we use `5` and `8`.
- `--algorithm`: The algorithm of constructing MetaNTK kernel. Choose from `['MAML','ANIL']`.
- `--inner_lr_time`: The product of inner loop learning rate and training time. The default value is `1000` (we treat inner_lr_time larger than 1000 as infinity).
- `--reg_coef`: the regularization coefficient for the inner loop optimization, we suggest use value larger than `1e-5`. The default value is `1e-3`.
- `--train_after_search`: If automatically train/evaluate the searched architecture. Choose from `['true', 'false']`.

You may also directly call `prune_metantknas.py`, there you will have much more flexibility. Check the file for more details.

### 2. Evaluation

* You can set `train_after_search` to `true`  in script provided so architecture evaluation will automatically run after the search is done.
* You can also use `eval_searched_arch.py` to train/evaluate obtained architectures (by its genotype).

## To-Do
- [ ] Provide searched architectures (in a Dropbox or Google Drive folder)
- [ ] Provide optimal hyperparameters (in `scripts/`) for the fine-tuning part of evaluation, i.e., finetuning a linear classifier on the top of the trained searched architecture for test few-shot tasks. 

## Acknowledgement

* Code base from:
  * [MetaNAS](https://github.com/boschresearch/metanas): We mainly use the model provided by MetaNAS. Check `eval_lib/rfs_models/augment_cnn.py`.
  * [TE-NAS](https://github.com/VITA-Group/TENAS): Our architecture search code is developed based on TENAS.
  * [RFS](https://github.com/WangYueFt/rfs): Our architecture evaluation code is developed based on RFS.
* **This work utilizes resources supported by the National Science Foundation’s Major Research Instrumentation program, grant #1725729, as well as the University of Illinois at Urbana-Champaign.**
