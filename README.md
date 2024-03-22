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

## Getting Started
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


## Images to Save and Inspect
- Visualization of restitched ground truth and predictions from `scripts/evaluate.py`
- Reconstructed 16x16 predictions for each model architecture in `data/predictions/<modelname>`
  
# Modeling


- `__init__`
- `forward`


For more information on how to use PyTorch Lightning with PyTorch as well as helpful tutorials, see:
- [PyTorch Lightning: Basic Skills](https://lightning.ai/docs/pytorch/latest/levels/core_skills.html)
- [A Recipe for Training Neural Networks](http://karpathy.github.io/2019/04/25/recipe/)



![FCNResnet101](assets/fcn.png)

![UNet](assets/unet.png)

## Training
We will train the models using the model architectures defined above in conjunction with the PyTorch Lightning Module for ease of running the training step in `train.py.` To monitor model training make sure to make an account with Weights and Biases for yourself and then create a team. For details on how to get started see [How to Use W&B Teams For Your University Machine Learning Projects for Free](https://wandb.ai/ivangoncharov/wandb-teams-for-students/reports/How-to-Use-W-B-Teams-For-Your-University-Machine-Learning-Projects-For-Free---VmlldzoxMjk1Mjkx).

### `ESDConfig` Python Dataclass
In `train.py` we have created an `ESDConfig` dataclass to store all the paths and parameters for experimenting with the training step. If you notice, in the main function we have provided you with code that utilize the library `argparse` which takes command line arguments using custom flags that allow the user to overwrite the default configurations defined in the dataclass we provided. When running train, for example, if you would like to run training for the architecture `SegmentationCNN` for five epochs you would run:

`python -m scripts.train --model_type=SegmentationCNN --max_epochs=5`


### Hyperparameter Sweeps
We will be using Weights and Biases Sweeps by configuring a yaml file called `sweeps.yml` in order to automate hyperparameter search over metrics such as batch size, epochs, learning rate, and optimizer. You may also experiment with the number of encoders and decoders you would like to add to your model architecture given that you are sensitive to the dimensions of your input image and the dimensions of the output prediction with respect to the ground truth. Some useful articles on how to perform sweeps and use the information to choose the best hyperparameter settings for your model can be found:
- [Tune Hyperparameters](https://docs.wandb.ai/guides/sweeps)
- [Running Hyperparameter Sweeps to Pick the Best Model](https://wandb.ai/wandb_fc/articles/reports/Running-Hyperparameter-Sweeps-to-# U-Net: Segmentation `src/models/unsupervised/unet.py`
Pick-the-Best-Model--Vmlldzo1NDQ0OTIy)

To run training with the hyperparameter sweeps you define in `sweeps.yml` please run `train_sweeps.py --sweep_file=sweeps.yml` provided for you.

## Validation
You will run validation using the script `evaluate.py` where you will load the model weights from the last checkpoint and make a forward pass through your model in order to generate prediction masks. Similar to `ESDConfig` in `train.py`, `EvalConfig` is the dataclass that sets the default configuration for the validation loop when arguments are not passed via command line. Note the use of `argparse` in the main function and its similarities to the `train.py` file.

