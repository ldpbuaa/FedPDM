# Progressive Distribution Matching for Federated Semi-supervised Learning
This repo provides the guidelines to reproduce the experimental results reported in our paper.

# Python Environment Setup
We use Anaconda for Python package installation and management.
We implement our algorithms with Pytorch 2.1.0.
To install the required packages for running experiments:
``` bash
conda create -n pytoch-2100 python=3.9.13
conda activate pytoch-2100 
conda install -f requirements.txt
```

# Dataset Preparation
We set the default datasets root directory as `~/data/`.
It can also be specified by `--data-dir=path/to/root/` argument.
We evaluate FSSL algorithms on five datasets, among which FashionMNIST, SVHN, CIFAR10 and CIFAR100
will be automatally downloaded at the first run of experiments, and the ISIC2018 dataset
can be downloaded at [ISIC2018 official website](https://challenge.isic-archive.com/data/#2018).
The raw datasets should be palced under `ISIC/raw` directory and organized as follows:
```
raw
├── ISIC2018_Task3_Test_GroundTruth.csv
├── ISIC2018_Task3_Test_Input
├── ISIC2018_Task3_Training_GroundTruth.csv
└── ISIC2018_Task3_Training_Input
```
We patition raw dataset by the category of each image, and then resize images to 240x240. 
The Python script for data pre-processing can be found at `FedPDM/fed/datasets/isic2018.py`

# Running Experiments
Experiments can be excuted by running:
```
./f --action FedPDM  \
--mode < client_training_mode > \
--dataset < dataset_name > \
--model < model_name > \
--data-dir < dataset_root_dir > \
--num-gpus < num_of_gpus > \
--num-processes < num_of_process > \
--num-clients < num_of_clients > \
--split-mode dirichlet \
--split-alpha < dataset_split_alpha > \
--data-transform < data_augmentation > \
--num-labeled < num_of_labeled_data > \
--max-rounds < max_training_rounds > \
--batch-size < training_batch_size > \
--eval-batch-size < evaluation_batch_size > \
--epochs < local_epochs > \
--train-fraction < client_sampling_ratio > \
--learning-rate < client_learning_rate > \
--lr-scheduler < learning_rate_scheduler > \
--lr-decay-rounds < learning_rate_decay_rounds > \
--optimizer-weight-decay < weight_decay > \
--optimizer-momentum < optimizer_momentum > \
--sup-epochs < local_epoch_for_labeled_clients > \
--unsup-epochs < local_epoch_for_unlabeled_clients > \
--num-sup-clients < num_of_labeled_clients > \
--client-eval=True \
--matching=True \
--logit-adjust-tau=< logits_adjustment_tau > \
--seed < random_seed >
```
