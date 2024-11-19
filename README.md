# Satellite Image Classification

This repository contains the implementation of 2 classification models for satellite images:
- K-Nearest Neighbours
- Convolutional Neural Network

## Dataset

The dataset contains 5631 satellite images belonging to 4 classes: cloudy, desert, green area and water. 

## Repository structure

```
├── checkpoints
│   └── best_model.pth
├── data
│   ├── cloudy
│   ├── desert
│   ├── green_area
│   └── water
├── README.md
├── run.ipynb
└── src
    ├── data
    │   └── TransformDataset.py
    └── utils
        ├── general_utils.py
        └── train_utils.py
```
