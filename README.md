# HW03: Semantic Segmentation and Model Monitoring

## Hungry Hungry Hippos
@Bavhead  
@anakash  
@TanushGo  
@sebseb100  

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
[![PyTorch Lightning](https://img.shields.io/badge/pytorch-lightning-blue.svg?logo=PyTorch%20Lightning)](https://github.com/Lightning-AI/lightning)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Weights and Biases](https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)


## Project Overview
Using the dataset from the [IEEE GRSS 2021 Challenge](https://www.grss-ieee.org/community/technical-committees/2021-ieee-grss-data-fusion-contest-track-dse/), we created multiple models to accurately represent the four classes: 
- regions with settlements without electricity
- regions with settlements and electricty
- regions without settlements and electricty
- regions without settlements, but with electricity
The primary focus was the first option listed: regions with settlements that do not have electricity access or availability.  

Three baseline models are provided:
- Segmentation CNN via `src/models/supervised/segmentation_cnn.py`
- FCN ResNet via `src/models/supervised/resnet_transfer.py`
- U-Net via `src/models/supervised/unet.py`

Additionally, the random forests technique can be applied to to any of these models via `scripts/train_random.py`.

## Setting Up Your Virtual Project Environment
To make sure you download all the packages, we utilize a Python virtual environment which is an isolated environment that allows you to run the code with its own dependencies and libraries independent of other Python projects you may be working on. Here's how to set it up:

1. Navigate to the final-project-hungry-hungry-hippo directory in your terminal.

2. Create a virtual environment:
   
   `python3 -m venv esdenv`
3. Activate the virtual environment:
   * On macOS and Linux:
  
        `source esdenv/bin/activate`
   * On Windows:
  
        `.\esdenv\Scripts\activate`
4. Install the required packages:
    `pip install -r requirements.txt`

To deactivate the virtual environment, type `deactivate`.

## Usage 

### Weights and Biases
We use wandb for our logging needs. After the environment is set up, run `wandb login` which will guide you through the login process to link your account via the API key provided in your account or team.

### Model Training 
To train one of the three main models, run `python -m scripts.train`. Specifics of which model to choose, which parameters to train on, and how to log this data via weights and biases can be modified in `scripts/train.py/. The model type and its hyperparameters can also be modified via the run command mentioned, and the possible arguments + their descriptions are listed at the bottom of the training file.

### Hyperparameter Sweeps
To run sweeps using wandb that can automate the process of finding optimal parameters, run `python scripts/train_sweeps.py --sweep_file=scripts/sweep.yml` where `sweep.yml` is a YAML file. More instructions on the specific format of information and the possible options can be found [here](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration). Configuration for which wandb project to use and how many models to train can be found within `train_sweeps.py'.

### Validation
Validate with the script `scripts/evaluate.py` where you will load the model weights from the relevant (or last) checkpoint in `data/models/<modelname/*.ckpt' and make a forward pass through your model in order to generate prediction masks. This script also has some editable options in the main method at the bottom of the file, mainly in regards to the location of the checkpoint to use, and other data locations to use or output to.

### Images to Inspect and/or Save
- Visualization of restitched ground truth and predictions in `data/predictions/plot/`
- Reconstructed 16x16 predictions for each model architecture in `data/predictions/<Tile*>`
  
