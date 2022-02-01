import numpy as np
import matplotlib.pyplot as plt
import torch, torchvision, argparse, json
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder
from collections import OrderedDict
from PIL import Image

# get args
parser = argparse.ArgumentParser()
parser.add_argument('input',             type=str,  default=None)
parser.add_argument('checkpoint',        type=str,  default=None)
parser.add_argument('--category_names',  type=str,  default='./cat_to_name.json')
parser.add_argument('--arch',            type=str,  default='vgg13', choices=['vgg13', 'vgg16', 'vgg19'])
parser.add_argument('--top_k',           type=int,  default=5)
parser.add_argument('--gpu',             type=bool, default=True)
args = parser.parse_args()
# load checkpoint
checkpoint          = torch.load(args.checkpoint.replace('arch', args.arch))
arch                = checkpoint['arch']
learning_rate       = checkpoint['learning_rate']
epochs              = checkpoint['epochs']
model               = checkpoint['model']
model.load_state_dict(checkpoint['state_dict'])
model.class_to_idx  = checkpoint['class_to_idx']
model.classifier    = checkpoint['classifier']
optimizer           = checkpoint['optimizer']
# predict
with open(args.category_names, 'r') as f: cat_to_name = json.load(f) # load label mapping
pimage = Image.open(args.input) # open image file
cuda = True if (args.gpu and torch.cuda.is_available()) else False # cpu/gpu
if cuda: model.cuda()
else:    model.cpu()
print('predicting {} using {}\n'.format(args.input, ('gpu' if cuda else 'cpu')))
if pimage.size[0] > pimage.size[1]: pimage.thumbnail((9999, 256)) # resize
else:                               pimage.thumbnail((256, 9999))
left   = (pimage.width  - 224) / 2 # crop
bottom = (pimage.height - 224) / 2
right  = left   + 224
top    = bottom + 224
pimage = pimage.crop((left, bottom, right, top)) # normalize
pimage = np.array(pimage) / 255
mean   = np.array([0.485, 0.456, 0.406])
std    = np.array([0.229, 0.224, 0.225])
pimage = (pimage - mean) / std
pimage = pimage.transpose((2, 0, 1)) # transpose color channels to 1-D for torch
if cuda: pimage_tensor = torch.from_numpy(pimage).type(torch.cuda.FloatTensor) # np to tensor
else:    pimage_tensor = torch.from_numpy(pimage).type(torch.FloatTensor)
model_inputs = pimage_tensor.unsqueeze(0) # set batch size to 1
probs = torch.exp(model.forward(model_inputs)) # calc
top_probs, top_labs = probs.topk(args.top_k)
top_probs = top_probs.detach().cpu().numpy().tolist()[0] if cuda else top_probs.detach().numpy().tolist()[0] 
top_labs  =  top_labs.detach().cpu().numpy().tolist()[0] if cuda else  top_labs.detach().numpy().tolist()[0]
idx_to_class = {value: key for key, value in model.class_to_idx.items()} # indices to classes
top_classes  = [idx_to_class[label] for label in top_labs]
top_labels   = [cat_to_name[idx_to_class[label]] for label in top_labs]
for x in list(zip(top_labels, top_probs)):
    print('{:6.2f}% {}'.format((x[1]*100), x[0]))
print()