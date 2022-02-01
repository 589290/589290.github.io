import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from PIL import Image
import argparse
import time
import random, os, json
random.seed(42)


def main():
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir',          type=str,   default='./flowers/')
    parser.add_argument('--save_checkpoint', type=str,   default='./checkpoint_arch.pth')
    parser.add_argument('--category_names',  type=str,   default='./cat_to_name.json')
    parser.add_argument('--arch',            type=str,   default='vgg13', choices=['vgg13', 'vgg16', 'vgg19'])
    parser.add_argument('--learning_rate',   type=float, default=0.01)
    parser.add_argument('--epochs',          type=int,   default=10)
    parser.add_argument('--hidden_units',    type=int,   default=1024)
    parser.add_argument('--gpu',             type=bool,  default=True)
    parser.add_argument('--test',            type=bool,  default=True)
    args = parser.parse_args()

    # load data
    train_dir = args.data_dir + '/train'
    test_dir  = args.data_dir + '/test'
    valid_dir = args.data_dir + '/valid'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    test_transforms  = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])
    train_data  = datasets.ImageFolder(args.data_dir + '/train', transform=train_transforms)
    test_data   = datasets.ImageFolder(args.data_dir + '/test',  transform=test_transforms)
    valid_data  = datasets.ImageFolder(args.data_dir + '/valid', transform=valid_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader  = torch.utils.data.DataLoader(test_data,  batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

    # load label mapping
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # build network
    model = getattr(models, args.arch)(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False # prevent backdrop
    num_features = model.classifier[0].in_features ## for vgg networks
    # input_size = model.classifier.in_features # for Densenet
    # model.fc.in_features # for Resnet networks
    classifier = nn.Sequential(OrderedDict([
                              ('fc1',    nn.Linear(num_features, args.hidden_units)),
                              ('drop',   nn.Dropout(p=0.5)),
                              ('relu',   nn.ReLU()),
                              ('fc2',    nn.Linear(args.hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    criterion = nn.CrossEntropyLoss() # loss function
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate)

    # cpu/gpu
    cuda = True if (args.gpu and torch.cuda.is_available()) else False
    if cuda: model.cuda()
    else:    model.cpu()
    print('training with {} and {} hidden units using {}\n'.format(args.arch, args.hidden_units, ('gpu' if cuda else 'cpu')))
        
    begin = time.time()
    for epoch in range(args.epochs):
        print('epoch: {}/{} '.format(epoch+1, args.epochs))

        # training
        model.train()
        running_loss = 0
        passes = 0
        for data in trainloader:
            passes += 1
            inputs, labels = data
            if cuda: inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:    inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad() # zero accumulated gradients
            logits = model.forward(inputs)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()                
            running_loss += loss.item()
            prob = torch.exp(logits).data
            equality = (labels.data == prob.max(1)[1])
            loss = running_loss / passes
        print('training   loss: {:.5f}'.format(loss))
    
        # validation
        model.eval()
        running_loss = 0
        passes = 0
        for data in validloader:
            passes += 1
            inputs, labels = data
            if cuda: inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:    inputs, labels = Variable(inputs), Variable(labels)
            optimizer.zero_grad() # zero accumulated gradients
            logits = model.forward(inputs)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            prob = torch.exp(logits).data
            equality = (labels.data == prob.max(1)[1])
            if cuda: accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()
            else:    accuracy = equality.type_as(torch.FloatTensor()).mean()
            loss = running_loss / passes
        print('validation loss: {:.5f} accuracy: {:.2f}'.format(loss, accuracy))
        print('duration: {:.0f}m {:.0f}s \n'.format((time.time()-begin)//60, (time.time()-begin)%60))

    # testing
    if args.test:
        model.eval()
        accuracy = 0
        passes = 0
        for data in testloader:
            passes += 1
            images, labels = data
            if cuda: images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:    images, labels = Variable(images), Variable(labels)
            logits = model.forward(images)
            prob = torch.exp(logits).data
            equality = (labels.data == prob.max(1)[1])
            if cuda: accuracy += equality.type_as(torch.cuda.FloatTensor()).mean()
            else:    accuracy += equality.type_as(torch.FloatTensor()).mean()
        print("test accuracy: {:.5f} %".format(100 * accuracy / passes))
        print('duration: {:.0f}m {:.0f}s \n'.format((time.time()-begin)//60, (time.time()-begin)%60))
        
    # save checkpoint
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'arch'         : args.arch,
                  'learning_rate': args.learning_rate,
                  'epochs'       : args.epochs,
                  'model'        : model,
                  'state_dict'   : model.state_dict(),
                  'class_to_idx' : model.class_to_idx,
                  'classifier'   : classifier,
                  'optimizer'    : optimizer.state_dict()}
    torch.save(checkpoint, args.save_checkpoint.replace('arch', args.arch))
    print('checkpoint saved to: {}'.format(args.save_checkpoint.replace('arch', args.arch)))

    
if __name__ == "__main__":
    main()    