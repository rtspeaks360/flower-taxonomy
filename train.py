# -*- coding: utf-8 -*-
# @Author: rishabh
# @Date:   2018-02-15 04:43:40
# @Last Modified by:   rishabh
# @Last Modified time: 2018-06-14 18:51:47

'''
MIT License

Copyright (c) 2018 Rishabh Thukral
'''

### Basic Python modules and Pytorch imports
import utils
import classifier

import time

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms

import argparse
import os

import numpy as np
### End

### Parser definition
parser = argparse.ArgumentParser(
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data_directory', type=str, default = "data",
                    help="data_directory")
parser.add_argument('--in_file', type=str, default = "label_map.json",
                    help='input json file for label_maps')
parser.add_argument('--save_dir', type=str, default='save',
                    help='directory to store checkpointed models')
parser.add_argument('--batch_size', type=int, default=32,
                    help='minibatch size')
parser.add_argument('--arch', type=str, default = 'vgg16',
                    help='available architectures are vgg 11, 13, 16, 19')
parser.add_argument('--num_epochs', type=int, default=10,
                    help='number of epochs')
parser.add_argument('--print_every', type=int, default=20,
                    help='print frequency')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--dropout_prob', type=float, default=0.5,
                    help='probability of dropping weights')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='run the network on the GPU')
parser.add_argument('--init_from', type=str, default=None,
                    help='initialize network from checkpoint')

args = parser.parse_args()
print(args)
if not os.path.isdir(args.save_dir):
    raise OSError(f'Directory {args.save_dir} does not exist.')
### End


# [START Validation function]
def validation(model, val_data, criterion, cuda=False):
    val_start = time.time()
    running_val_loss = 0
    accuracy = 0
    for inputs, labels in val_data:
        inputs, labels = Variable(inputs), Variable(labels)

        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()

        outputs = model.forward(inputs)
        val_loss = criterion(outputs, labels)

        ps = torch.exp(outputs.data)

        _, predicted = ps.max(dim=1)

        equals = predicted == labels.data
        accuracy += torch.sum(equals) / len(equals)

        running_val_loss += val_loss.data[0]
    val_time = time.time() - val_start
    print("Valid loss: {:.3f}".format(running_val_loss / len(dataloaders['valid'])),
          "Accuracy: {:.3f}".format(accuracy/len(dataloaders['valid'])),
          "Val time: {:.3f} s/batch".format(val_time / len(dataloaders['valid'])))
# [END]

# [START Train function to train the model.]
def train(dataloaders, model, epochs=10, cuda=False):
    print_every_n = args.print_every

    # Define the loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    if cuda:
        model.cuda()
    else:
        model.cpu()

    model.train()
    for e in range(epochs):
        print(f"Epoch {e+1}/{epochs}")
        counter = 0
        running_loss = 0
        for inputs, labels in dataloaders['train']:
            counter += 1

            # Training pass
            inputs, labels = Variable(inputs), Variable(labels)

            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]

            if counter % print_every_n == 0:
                print(f"Step: {counter}")
                print(f"Training loss {running_loss/counter:.3f}")
                model.eval()
                validation(model, dataloaders['valid'], criterion, cuda=cuda)
                model.train()
        else:
            # Validation pass
            train_end = time.time()
            model.eval()
            validation(model, dataloaders['valid'], criterion, cuda=cuda)

    if 'test' in dataloaders:
        if cuda:
            model.cuda()
        model.eval()
        validation(model, dataloaders['test'], criterion, cuda=cuda)
# [END]


# [START Main function where magic happens]
if __name__ == "__main__":
    
    #paths to directories for training data, test data, validation data and the json label map
    data_directory = args.data_directory
    train_dir = data_directory + "/train"
    valid_dir = data_directory + "/valid"
    test_dir = data_directory + "/test"

    json_path = args.in_file

    # Calling the load_data function to load the data
    image_datasets, dataloaders = utils.load_data(train_dir, valid_dir, test_dir, args.batch_size)

    # Calling the load_label_map function to load the label map for the dataset from the given json file
    label_map = utils.load_label_map(json_path)



    # Calling the build model function to build the model
    model_ = classifier.build_model(4096, len(label_map), args.dropout_prob, args.arch)

    # Calling the train function to train the model
    train(dataloaders, model_, epochs = args.num_epochs, cuda = args.gpu)

    # Calling the save model function to save the trained model
    model_.class_to_idx = image_datasets['train'].class_to_idx
    model_.cpu()
    model.save_model(model_, args.arch, args.save_dir)
# [END]