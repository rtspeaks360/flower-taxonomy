# -*- coding: utf-8 -*-
# @Author: rishabh
# @Date:   2018-02-15 20:11:00
# @Last Modified by:   rishabh
# @Last Modified time: 2018-06-14 16:12:54
'''
MIT License

Copyright (c) 2018 Rishabh Thukral
'''

import argparse

import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

from classifier import load_model
from utils import load_label_map, process_image


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('input', type=str, default=None,
                    help='Path to input image')
parser.add_argument('checkpoint', type=str, default=None,
                    help='Load checkpoint for prediction')
parser.add_argument('--gpu', action='store_true', default=False,
                    help='Run the network on a GPU')
parser.add_argument('--top_k', type=int, default=5,
                    help='Predict the top K character probabilities')
parser.add_argument('--category_names', type=str, default=None,
                    help='Path to JSON file mapping categories to names')

args = parser.parse_args()


def predict(image_path, model, top_k=5, cuda=False):
    """ Load an image, run it through a model and predict the class with probabilities

        Arguments
        =========

        image_path : Path to image for class prediction
        model : Model to use as a classifier
        top_k : Return the top K most probable classes
        cuda : Set to True to run the model on GPU

    """
    image = Image.open(image_path)
    image = process_image(image)

    # Convert to PyTorch tensor and do a forward pass
    # Model expects a FloatTensor, from_numpy gives a DoubleTensor
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor)
    inputs = Variable(image_tensor, requires_grad=False)

    # Need to add an extra dimension at the beginning for the batch of 1
    inputs = inputs.unsqueeze(0)

    if cuda:
        inputs = inputs.cuda()
        model.cuda()

    # Model returns log_softmax, exponential to get probabilities
    ps = torch.exp(model.forward(inputs))

    if cuda:
        ps.cuda()

    top_probs, top_labels = ps.topk(top_k)
    top_probs, top_labels = top_probs.data.cpu().numpy().squeeze(), top_labels.data.cpu().numpy().squeeze()

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}

    if top_k == 1:
        top_classes = [idx_to_class[int(top_labels)]]
        top_probs = [float(top_probs)]
    else:
        top_classes = [idx_to_class[each] for each in top_labels]

    return top_probs, top_classes


if __name__ == '__main__':

    image_path = args.input
    model = load_model(args.checkpoint)

    model.eval()
    # if args.gpu:
        # model.cuda()
    top_probs, top_classes = predict(image_path, model, top_k=args.top_k, cuda=args.gpu)

    if args.category_names is not None:
        class_to_name = load_label_map(args.category_names)
        top_classes = [class_to_name[each] for each in top_classes]

    for name, prob in zip(top_classes, top_probs):
        print(f"{name}: {prob:.3f}")

