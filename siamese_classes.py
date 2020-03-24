"""
Siamese neural network classes and functions
Evaluating ROP change detection 
created 2019-04-23

The same classes can be modified for any dataset containing images that can be paired with category disease severity change labels.

"""


# PyTorch modules
import torch
from torch import nn 
from torch.utils import data 
import torch.nn.functional as F 
from torchvision import transforms, models

# other modules
import os
import pandas as pd
import numpy as np
from PIL import Image
from glob import glob
import random
import itertools


class ROP_dataset_v5(data.Dataset):
    """ 
    Create dataset representation of ROP data 
    - This class returns image pairs with a change label (i.e. change vs no change in a categorical disease severity label) and other metadata
    - Image pairs are sampled so that there are an equal number of change vs no change labels
    - Epoch size can be set for empirical testing
  
    Concepts adapted from: 
    - https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    - https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """  
    def __init__(self, patient_table, image_dir, epoch_size, transform=None):
        """
        Args:
            patient_table (pd.dataframe): dataframe table containing image names, disease severity category label, and other metadata
            image_dir (string): directory containing all of the image files
            transform (callable, optional): optional transform to be applied on a sample
        """
        self.patient_table = patient_table
        self.image_dir = image_dir 
        self.transform = transform
        self.epoch_size = epoch_size 
        if self.transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
 
    def __len__(self):
        return self.epoch_size
 
    def __getitem__(self, idx):
        
        name_list = list(self.patient_table['imageName'])

        # goal is 50:50 distribution of change vs no change
        change_binary = random.randint(0,1) 

        # keep on looping until no change pair created
        while change_binary == 0:

            # pick random image from folder
            # check to see if the image exists and can be loaded, if not move to another random image
            while True:
                random_image = random.choice(name_list)[:-4] + '.png' # note that processed images are all .png type, while patient_table has different types
                if random_image in os.listdir(self.image_dir):
                    break
                else:
                    print('attempted to get following image, but missing: ' + random_image)

            while True:
                paired_image = random.choice(name_list)[:-4] + '.png' # note that processed images are all .png type, while patient_table has different types
                if paired_image in os.listdir(self.image_dir):
                    break
                else:
                    print('attempted to get following image, but missing: ' + random_image)

            # get labels and meta data
            plus_disease_0 = random_image.split('_')[5][:-4]
            plus_disease_1 = paired_image.split('_')[5][:-4]

            if plus_disease_0 == plus_disease_1:
                plus_disease_binary_change = 0 # 0 for no change
            else:
                plus_disease_binary_change = 1 # 1 for change

            if plus_disease_binary_change == change_binary:
                break
  
        # keep on looping until change pair created
        while change_binary == 1:
        
            # pick random image from folder
            # check to see if the image exists and can be loaded, if not move to another random image
            while True:
                random_image = random.choice(name_list)[:-4] + '.png' # note that processed images are all .png type, while patient_table has different types
                if random_image in os.listdir(self.image_dir):
                    break
                else:
                    print('attempted to get following image, but missing: ' + random_image)

            while True:
                paired_image = random.choice(name_list)[:-4] + '.png' # note that processed images are all .png type, while patient_table has different types
                if paired_image in os.listdir(self.image_dir):
                    break
                else:
                    print('attempted to get following image, but missing: ' + random_image)

            # get labels and meta data
            plus_disease_0 = random_image.split('_')[5][:-4]
            plus_disease_1 = paired_image.split('_')[5][:-4]

            if plus_disease_0 == plus_disease_1:
                plus_disease_binary_change = 0 # 0 for no change
            else:
                plus_disease_binary_change = 1 # 1 for change

            if plus_disease_binary_change == change_binary:
                break

        # convert disease severity class labels to numeric form    
 
        if plus_disease_0 == 'No': pd0 = 0
        if plus_disease_0 == 'Pre-Plus': pd0 = 1
        if plus_disease_0 == 'Plus': pd0 = 2

        if plus_disease_1 == 'No': pd1 = 0
        if plus_disease_1 == 'Pre-Plus': pd1 = 1
        if plus_disease_1 == 'Plus': pd1 = 2

        plus_disease_change = pd1 - pd0 # should range from -2 to +2

        if plus_disease_change == 0:
            plus_disease_binary_change = 0
        else:
            plus_disease_binary_change = 1

        # should be same patient ID and eye for both time points
        subject_id_0 = random_image.split('_')[0]
        subject_id_1 = paired_image.split('_')[0]
        eye_0 = random_image.split('_')[4]
        eye_1 = paired_image.split('_')[4]
        # session indicates the time point -- note cannot compare between different patients
        session_0 = random_image.split('_')[2] 
        session_1 = paired_image.split('_')[2] 
 
        label = plus_disease_binary_change # 0 for no change, 1 for change
 
        meta = {"subject_id_0": subject_id_0,
                "subject_id_1": subject_id_1,
                "eye_0": eye_0, 
                "eye_1": eye_1, 
                "plus_disease_0": pd0,
                "plus_disease_1": pd1,
                "plus_disease_change": plus_disease_change,
                "session_0": session_0,
                "session_1": session_1}
     
        # open images and convert to single channel (greyscale from RGB) - note already 8bit images
        img0 = Image.open(self.image_dir + random_image)
        img1 = Image.open(self.image_dir + paired_image)
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1) 

        return img0, img1, label, meta
       
       
class SiameseNetwork101(nn.Module):
    """
    Siamese neural network
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Siamese ResNet-101 from Pytorch library
    """ 
    def __init__(self):
        super(SiameseNetwork101, self).__init__()
        # note that resnet101 requires 3 input channels, will repeat grayscale image x3
        self.cnn1 = models.resnet101(pretrained = True)
        self.cnn1.fc = nn.Linear(2048, 3) # mapping input image to a 3 node output

    def forward_once(self, x):
        output = self.cnn1(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

       
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    Modified from: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7

    """ 

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
