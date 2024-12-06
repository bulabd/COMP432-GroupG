# COMP-432 Project

## Project Description
This project focuses on using Convolutional Neural Networks (CNNs), for image classification tasks in the context of medical applications. For part 1 of the project, the goal is to train and evaluate a ResNet-34 model on a dataset consisting of Colorectal Cancer Images (MUS, NORM, STR). For part 2, the goal is to use the encoder obtained from part 1 and a pre-trained ImageNet (ShuffleNet) encoder on new datasets of Prostate Cancer and Animal Faces images for classification. In the end, the performance of these encoders shall be analyzed and compared.

## Requirements
It is strongly recommended to run the code in google colab. Can be run in a python IDE or Jupyter but need to use custom path to the dataset folders.
- The following libraries are required to run the code:
    - os
    - numpy
    - matplotlib
    - sklearn
    - torch
    - torchvision

## Instructions on how to train/validate the model
Use the [model training notebook](/Source%20Code%20ipynb%20files/Part%201/part1_model_training.ipynb). Run each code cell in the top to bottom order.

## Instructions on how to run pre-trained model on sample dataset
Use the [encoder from part 1](/Source%20Code%20ipynb%20files/Part%202/part2_encoder_from_part1.ipynb)
and the [pre-trained encoder](/Source%20Code%20ipynb%20files/Part%202/part2_shufflenet_encoder.ipynb)
Run each code cell in the top to bottom order.

## Dataset links:
The project datasets are downloaded when running the .ipynb notebooks in google collab. 

(Optional) The datasets can be downloaded from the following links and placed inside your project folder. The path in each file should be changed to the location of your datasets.

**Dataset 1: Colorectal Cancer Classification** 
    [Original Dataset](https://zenodo.org/record/1214456)
    | [Project Dataset](https://1drv.ms/u/s!AilzKc-njjP7mN0NOZvxl0TPAUxmig?e=K0TpeX)

**Dataset 2: Prostate Cancer Classification** 
    [Original Dataset](https://zenodo.org/record/4789576)
    | [Project Dataset](https://1drv.ms/u/s!AilzKc-njjP7mN0M_LjB5xeAydDsrA?e=0obzsx)

**Dataset 1: Animal Faces Classification** 
    [Original Dataset](https://www.kaggle.com/datasets/andrewmvd/animal-faces)
    | [Project Dataset](https://1drv.ms/u/s!AilzKc-njjP7mN0LqoRZvUYONY9sbQ?e=wxWbip)