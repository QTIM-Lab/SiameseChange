'''
Siamese ROP evaluation
created 5/21/2019

analysis of 100 excluded samples from the dataset
previously annotated by ophthamologists to provide rank of plus disease severity in 100 samples

show that we can learn the severity of clinical grade to a finer degree of continuous variation than just normal, pre-plus, and plus
Euclidean distance from siamese network model should reflect these distances

'''

# WORKING DIRECTORY (should contain a data/ subdirectories)
working_path = '/SiameseChange/'
os.chdir(working_path)

# PyTorch modules
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms, models
from torch.autograd import Variable

# other modules
import os
from glob import glob
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn import metrics
import statistics
import pickle
import seaborn as sns
from PIL import Image

# custom classes
from siamese_ROP_classes import SiameseNetwork101

# CUDA for PyTorch
os.environ['CUDA_VISIBLE_DEVICES']='0' # pick GPU to use
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# loading files, pick which  model from the main.py output
output_folder_name = 'Res101_inter'
output_dir = working_path + "scripts/histories/" + output_folder_name
net = SiameseNetwork101().cuda()
net.load_state_dict(torch.load(output_dir + "/siamese_ROP_model.pth"))
history = pickle.load(open(output_dir + "/history_training.pckl", "rb"))

# load rank order table (NOTE: rank 100 is plus disease, rank 1 is normal)
rank_table = pd.read_csv(working_path + 'data/rank_data.csv')

image_dir = working_path + 'data/'



