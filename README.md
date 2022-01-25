# Translator: A Transfer Learning Approach to Facilitate Single-Cell ATAC-Seq Data Analysis from Reference Dataset

Translator is a deep generative model-based representation learning model to create better cell embeddings for lower-quality samples with the help of a reference dataset. This implementation uses PyTorch as its backend. 

## Setup

To install the software, please clone this repository:

```
git clone https://github.com/aicb-ZhangLabs/Translator.git
```

Then, please install all dependencies by using anaconda:

```
conda env create -f translator.yml
conda activate translator
```

## Model training with reference dataset

To train the model, use the following command:

```
python train.py [OPTIONS]
```

There are certain required parameters:

- Data Type (-d): the dataset type to choose. For training with only the reference dataset, use "SimDataset"; for training with both reference and target datasets, use "SplitSimDataset"
- File name and type name (`--file_name` and `--type_name`): file path (.npz and .csv) of the reference dataset
- File name 2 and type name 2 (`--file_name2` and `--type_name2`): file path (.npz and .csv) of the target dataset (if used in training)

For a list of descriptions of optional parameters, use the `-h` flag. 

## Embedding generation

By default, after training the model, the program will automatically generate embeddings for both the reference dataset and the target dataset (if any). They will all be saved in the provided results folder.

## Save the model

By default, the program will save the trained model as two pytorch dictionary files (.pt) in the results folder. 

## Inference using the model

To create embeddings for target datasets using the model, use the following command:

```
python infer.py [OPTIONS]
```

There are some required parameters:

- File name and type name (`--file_name` and `--type_name`): file path (.npz and .csv) of the target dataset



## Embedding visualization and evaluation
