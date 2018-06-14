# -*- coding: utf-8 -*-
# @Author: rishabh
# @Date:   2018-02-15 20:11:00
# @Last Modified by:   rishabh
# @Last Modified time: 2018-06-14 10:08:08

'''
MIT License

Copyright (c) 2018 Rishabh Thukral
'''
import datetime
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, models, transforms

# [START The Feed Forward Classifier defines the custom classifier that will be trained on top of a pretrained model.]
class FFClassifier(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, drop_prob):
        super().__init__()

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, hidden_features)
        self.fc3 = nn.Linear(hidden_features, out_features)

        self.drop = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)

        x = F.log_softmax(x, dim=1)
        return x
# [END]

# [START build model function is used to load a pretrained model and then override the custom classifier on top of it.]
def build_model(fc_hidden, fc_out, drop_prob, arch='vgg16'):

    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)

    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print (arch + " - model not available")

    for param in model.parameters():
        param.requires_grad = False

    fc_in = model.classifier[0].in_features

    # Create your own classifier
    net = FFClassifier(fc_in, fc_hidden, fc_out, drop_prob)

    # Put your classifier on the pretrained network
    model.classifier = net

    return model
# [END]

# [START save_model function to create a checkpoint and save the current model]
def save_model(model, arch, save_directory):
    date = datetime.datetime.utcnow().strftime("_%Y-%m-%d_%H:%M:%S")

    torch.save({'arch': arch,
            'hidden': 4096,
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx}, 
            save_directory + '/classifier_' + arch + date + '.pt')

    return
# [END]

# [START load_model function to load a pretrained model into memory from a checkpoint.]
def load_model(checkpoint_path, arch='vgg16', drop_prob = 0.5):
    checkpoint = torch.load(checkpoint_path)
    
    if arch == 'vgg11':
        model = models.vgg11(pretrained=True)
        
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        
    elif arch == 'vgg16':
        model = models.vgg16(pretrained=True)

    elif arch == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print (arch + " - model not available")

    for param in model.parameters():
        param.requires_grad = False
    
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Create the classifier
    net = FFClassifier(25088, checkpoint['hidden'], len(model.class_to_idx), drop_prob)

    # Put the classifier on the pretrained network
    model.classifier = net
    
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
# [END]




