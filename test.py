import pickle
import sys
import numpy as np
from PIL import Image

import torch 
from torch import nn

from torchnlp.word_to_vector import GloVe

import detectron2
from detectron2.data import MetadataCatalog
from detectron2.layers import ShapeSpec, cat

from defrcn.engine import DefaultTrainer, default_argument_parser, default_setup

from Ecai.DeFRCN_tien.defrcn.modeling.modules import *

def extender_net(input_size, output_size=2048, factor = 2):
    model = nn.Sequential(
        nn.Linear(input_size, input_size * factor),
        nn.ReLU(),
        nn.Linear(input_size * factor, input_size * factor * factor),
        nn.ReLU(),
        nn.Linear(input_size * factor * factor, output_size),
        nn.ReLU(),
    )
    
    return model
    
def get_proposals():
    with open('proposal.pkl', 'rb') as f:
        proposals = pickle.load(f)
    
    return proposals
 
def get_gt_classes():
    with open('proposal.pkl', 'rb') as f:
        proposals = pickle.load(f)
        
    gt_classes = cat([p.gt_classes for p in proposals], dim=0)
    return gt_classes
    
def base_class_mapper(text_dim=300):
    glove_vec = GloVe(name='6B', dim=text_dim)
    voc_map = {'aeroplane': 'aeroplane', 'bicycle': 'bicycle', 'boat': 'boat', 'bottle': 'bottle', 
              'car': 'car', 'cat': 'cat', 'chair': 'chair', 'diningtable': 'dining table', 
              'dog': 'dog', 'horse': 'horse', 'person': 'person', 'pottedplant': 'potted plant', 
              'sheep': 'sheep', 'train': 'train', 'tvmonitor': 'tvmonitor', 'bird': 'bird', 
              'bus': 'bus', 'cow': 'cow', 'motorbike': 'motorbike', 'sofa': 'sofa'}
    text_embed = torch.zeros(len(voc_map), text_dim)
    for idx, extend_class in enumerate(voc_map.values()):
        tokens = extend_class.split(' ')
        for token in tokens:
            text_embed[idx] += glove_vec[token]
        text_embed[idx] /= len(tokens)

    return text_embed

def get_feature_pooled():
    with open('feature_pooled.pkl', 'rb') as f:
        feature_pooled = pickle.load(f)
    
    return feature_pooled

def check_image_shape(path):
    img = Image.open(path)
    print(img)
    
def check_tensor():
    a = torch.ones((3, 1, 2))
    b = torch.ones((3, 2, 3))
    c = [a, b]
    c = c.tensor
    print(c)
    

check_tensor()
