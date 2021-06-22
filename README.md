# Chest Radiography Classifcation
This repository contains the code for our paper:
Artificial versus Human Intelligence – Superior Diagnostic Accuracy of Deep Learning versus Radiologists in the Assessment of Bedside Chest Radiographs

## Installation
To clone this repository please run

``` bash
git@github.com:FirasGit/chest_radiography_classification.git
```

then navigate into the cloned repository using 

``` bash
cd chest_radiography_classification
``` 

and create a conda environment 

```
conda create -n chest_radiography_classification python=3.8
```

Once activating the environment using

```
conda activate chest_radiography_classification
```

run 
``` bash
pip install -r requirements.txt
```
to install all dependencies.

## Train the model
In order to train the model you need to request access to our internal dataset or use your own dataset.
To train the model, run the following command

``` bash
python training/lightning_trainer.py meta.prefix_name=<your_preferred_run_name> model.name=<model_name> optimizer.learning_rate=<learning_rate> annotations.path_to_train_annotation_csv=<path_to_training_set> annotations.path_to_valid_annotation_csv=<path_to_validation_set> annotations.path_to_test_annotation_csv=<path_to_test_set> optimizer.loss_fnc=<loss_function> meta.batch_size=<batch_size>
```
For a more detailed view over all configuration settings, navigate to configs/train_config/base_cfg.yaml

