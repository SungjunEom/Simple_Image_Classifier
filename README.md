# Simple Image Classifier
This repository contains a very simple image classifier and will be used for testing some adversarial attacks and more.

### Installation
```
docker build -t simple_classifier .
docker run --gpus all --rm -it --name container1 simple_classifier
```

### Usage
To train a model with 13 epochs and wandb:
```
python train.py --wandb True --epochs 13
```