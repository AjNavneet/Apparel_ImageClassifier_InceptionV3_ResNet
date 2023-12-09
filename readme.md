# Image Classification using InceptionV3 and ResNet

## Business Objective

**Image Classification** - Image classification refers to a process in computer vision that can classify an image according to its visual content. Image classification applications include recognizing various objects, such as vehicles, people, moving objects, etc., on the road to enable autonomous driving.

**Transfer Learning** - The reuse of a pre-trained model on a new problem is known as transfer learning in machine learning. A machine uses the knowledge learned from a prior assignment to increase prediction about a new task in transfer learning.

---

## Data Description

We have a dataset of t-shirt images categorized into 2 types, namely plain(solid) and topography (contains text). The training, testing, and validation folders contain two subfolders (plain and topography) each.

- Training - around 600 images
- Testing - around 100 images
- Validation - around 150 images

---

## Aim

A model is to be built that can correctly classify a t-shirt image into plain or topographic using InceptionV3 and Resnet.

---

## Tech Stack

- Language - `Python`
- Libraries - `pandas`, `numpy`, `seaborn`, `matplotlib`, `sklearn`, `tensorflow`

---

## Approach

1. Importing the required libraries.
2. Load and read the data images.
3. Data Visualization.
4. Model Training
   - Create a function for initiating Inception and Resnet models.
   - Create a function that defines the optimizers.
   - Add layers to the pretrained model.
   - Function to compile the model.
   - Fit and train the model.
   - Plot the results.
   - Save the model (.h5 file).
5. Create train and validation image data generators.
6. Model Testing
   - Create testing images data generator.
   - Load the saved model.
   - Perform predictions by plotting the confusion matrix.
   - Analyze the incorrect predictions.

---

## Modular Code Overview

1. **Input Folder** - It contains all the data that we have for analysis. We have three folders: Train, Test, and Valid. These two subfolders are present:
   - Plain
   - Topography

2. **Source Folder** - This is the most important folder of the project. This folder contains all the modularized code for all the above steps in a modularized manner. This folder consists of:
   - `Engine.py`
   - `ML_Pipeline`

   The `ML_Pipeline` is a folder that contains all the functions put into different Python files, which are appropriately named. These Python functions are then called inside the `engine.py` file.

3. **Output Folder** - The output folder contains the fitted model that we trained for this data. This model can be easily loaded and used for future use, and the user need not have to train all the models from the beginning.

4. **Lib** - This folder contains 1 notebooks:
   - `image_classification.ipynb`

---

## Code execution:
1. Before executing the code, change the `home_dir` variable in `src/config.ini` file to this project folder

2. Move to `src` folder:
    ```bash
    cd src
    ```

3. Execute the `engine.py` file 
    ```
    python engine.py
    ```

4. To run hyperparameter tuning, set the `hp_tune` param in `src/config.ini` to `True` and set your choice of hyper parameters in the `param_grid` and run the following command
    ```
    python engine.py
    ```
---

## Refrence Resources:
1. [Transfer learning](https://blogs.nvidia.com/blog/2019/02/07/what-is-transfer-learning/)
2. [Resnet Paper](https://arxiv.org/pdf/1512.03385.pdf)
3. [Inception Paper](https://arxiv.org/pdf/1409.4842.pdf)

---
