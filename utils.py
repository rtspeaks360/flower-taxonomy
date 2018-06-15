# -*- coding: utf-8 -*-
# @Author: rishabh
# @Date:   2018-02-15 20:11:00
# @Last Modified by:   Rishabh Thukral
# @Last Modified time: 2018-06-15 19:49:34

'''
MIT License

Copyright (c) 2018 Rishabh Thukral
'''

### Imports for basic python modules and PyTorch essentials.
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms
### end

# [START load_label_map to load dictionary mapping class indices to class names from a JSON file]
def load_label_map(json_path):
    ''' Load a dictionary mapping class indices to class names from a JSON file
    '''

    import json

    with open(json_path, 'r') as f:
        label_map = json.load(f)
    return label_map
# [END]

# [START process_image function to process the image to make it acceptable by pytorch]
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model '''

    # Resize and crop out the center 224x224 square
    aspect = image.size[0] / image.size[1]
    if aspect > 0:
        image.thumbnail((10000, 256))
    else:
        image.thumbnail((256, 10000))
    left_margin = (image.width - 224) / 2
    top_margin = (image.height - 224) / 2
    image = image.crop((left_margin, top_margin, left_margin + 224, top_margin + 224))

    # Now normalize...
    image = np.array(image) / 255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std

    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose((2, 0, 1))

    return image
# [END]

# [START load_data funcion to handle the process of data augmentation, loading the data and making dataloaders for training]
def load_data(train_dir, valid_dir, test_dir, batch_size_received=32):
        

    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomRotation(45),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])}

    dirs = {'train': train_dir, 'valid': valid_dir, 'test': test_dir}
    image_datasets = {x: datasets.ImageFolder(dirs[x], transform=data_transforms[x])
                      for x in ['train', 'valid', 'test']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size_received, shuffle=True)
                   for x in ['train', 'valid', 'test']}
    return image_datasets, dataloaders\
# [END]