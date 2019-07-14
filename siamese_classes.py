"""
Siamese neural network classes
created 2019-04-23

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
 
class ROP_dataset(data.Dataset):
    """ 
    Create dataset representation of ROP data -- INTRApatient comparison -- has equal sampling of comparisons (not just same vs not same, but actual classes)
  
    """
    def __init__(self, patient_table, image_dir, epoch_size, transform=None):
        """
        Args:
            patient_table (pd.dataframe): from main.py
            image_dir (string): Directory with all the images
            transform (callable, optional): optional transform to be applied on a sample.
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
        
        no_table = self.patient_table.loc[self.patient_table['Ground truth'] == 'No']
        preplus_table = self.patient_table.loc[self.patient_table['Ground truth'] == 'Pre-Plus']
        plus_table = self.patient_table.loc[self.patient_table['Ground truth'] == 'Plus']

        no_name_list = list(no_table['imageName'])
        preplus_name_list = list(preplus_table['imageName'])
        plus_name_list = list(plus_table['imageName'])

        # goal is equal sampling of each class comparison
        # i.e 6 unique combinations (no-no, pre-plus-pre-plus, plus-plus, no-pre-plus, no-plus, pre-plus-plus) - note: order does not matter
        # outside while loop will keep on drawing random image combinations to ensure this distribution
        change_combo = random.randint(0,5) 
 
        # Create No-No pair
        if change_combo == 0:
            # pick random image from folder in the no_name_list (without replacement)
            # check to see if the image exists and can be loaded, if not move to another random image
            while True:
                random_image = random.choice(no_name_list)[:-4] + '.png' # note that processed images are all .png type, while patient_table has different types
                if random_image in os.listdir(self.image_dir):
                    break
                else:
                    print('attempted to get following image, but missing: ' + random_image)

            # then find INTRA patient comparison (can include comparison with self)
            # filter for the same patient by subjectID
            self.patient_table['subjectID_first9'] = [x[0:9] for x in self.patient_table['subjectID']] # note that some subject IDs have additional characters appended, create new column to only take the unique 9 character ID)
            sub_table = self.patient_table[self.patient_table['subjectID_first9'] == random_image[0:9]]
            # filter for the same eye
            which_eye = random_image.split('_')[4]
            sub_table = sub_table[sub_table['eye'] == which_eye]
            # randomly select a row from sub_table, select the file name, and change file name to end with .png
            paired_image = sub_table.sample(n=1)['imageName'].item()[:-4] + '.png'

            # get labels and meta data
            plus_disease_0 = random_image.split('_')[5][:-4]
            plus_disease_1 = paired_image.split('_')[5][:-4]

        # Create Pre-Plus Pre-Plus pair
        if change_combo == 1:
            # pick random image from folder in the preplus_name_list (without replacement)
            # check to see if the image exists and can be loaded, if not move to another random image
            while True:
                random_image = random.choice(preplus_name_list)[:-4] + '.png' # note that processed images are all .png type, while patient_table has different types
                if random_image in os.listdir(self.image_dir):
                    break
                else:
                    print('attempted to get following image, but missing: ' + random_image)

            # then find INTRA patient comparison (can include comparison with self)
            # filter for the same patient by subjectID
            self.patient_table['subjectID_first9'] = [x[0:9] for x in self.patient_table['subjectID']] # note that some subject IDs have additional characters appended, create new column to only take the unique 9 character ID)
            sub_table = self.patient_table[self.patient_table['subjectID_first9'] == random_image[0:9]]
            # filter for the same eye
            which_eye = random_image.split('_')[4]
            sub_table = sub_table[sub_table['eye'] == which_eye]
            # randomly select a row from sub_table, select the file name, and change file name to end with .png
            paired_image = sub_table.sample(n=1)['imageName'].item()[:-4] + '.png'

            # get labels and meta data
            plus_disease_0 = random_image.split('_')[5][:-4]
            plus_disease_1 = paired_image.split('_')[5][:-4]

        # Create Plus Plus pair
        if change_combo == 2:
            # pick random image from folder in the plus_name_list (without replacement)
            # check to see if the image exists and can be loaded, if not move to another random image
            while True:
                random_image = random.choice(plus_name_list)[:-4] + '.png' # note that processed images are all .png type, while patient_table has different types
                if random_image in os.listdir(self.image_dir):
                    break
                else:
                    print('attempted to get following image, but missing: ' + random_image)

            # then find INTRA patient comparison (can include comparison with self)
            # filter for the same patient by subjectID
            self.patient_table['subjectID_first9'] = [x[0:9] for x in self.patient_table['subjectID']] # note that some subject IDs have additional characters appended, create new column to only take the unique 9 character ID)
            sub_table = self.patient_table[self.patient_table['subjectID_first9'] == random_image[0:9]]
            # filter for the same eye
            which_eye = random_image.split('_')[4]
            sub_table = sub_table[sub_table['eye'] == which_eye]
            # randomly select a row from sub_table, select the file name, and change file name to end with .png
            paired_image = sub_table.sample(n=1)['imageName'].item()[:-4] + '.png'

            # get labels and meta data
            plus_disease_0 = random_image.split('_')[5][:-4]
            plus_disease_1 = paired_image.split('_')[5][:-4]

        # Create No Pre-Plus pair (loop until appropriate pair found)
        while change_combo == 3:          
            # pick random image from folder in the pre-plus list (without replacement)
            # check to see if the image exists and can be loaded, if not move to another random image
            while True:
                random_image = random.choice(preplus_name_list)[:-4] + '.png' # note that processed images are all .png type, while patient_table has different types
                if random_image in os.listdir(self.image_dir):
                    break
                else:
                    print('attempted to get following image, but missing: ' + random_image)

            # then find INTRA patient comparison 
            # filter for the same patient by subjectID
            self.patient_table['subjectID_first9'] = [x[0:9] for x in self.patient_table['subjectID']] # note that some subject IDs have additional characters appended, create new column to only take the unique 9 character ID)
            sub_table = self.patient_table[self.patient_table['subjectID_first9'] == random_image[0:9]]
            # filter for the same eye
            which_eye = random_image.split('_')[4]
            sub_table = sub_table[sub_table['eye'] == which_eye]
            # randomly select a row from sub_table, select the file name, and change file name to end with .png
            paired_image = sub_table.sample(n=1)['imageName'].item()[:-4] + '.png'

            # get labels and meta data
            plus_disease_0 = random_image.split('_')[5][:-4]
            plus_disease_1 = paired_image.split('_')[5][:-4]

            if plus_disease_1 == 'No':
                reverse_direction = random.randint(0,1) 
                if reverse_direction == 1:
                    random_image, paired_image = paired_image, random_image
                    plus_disease_0, plus_disease_1 = plus_disease_1, plus_disease_0
                break

        # Create No Plus pair (loop until appropriate pair found)
        while change_combo == 4:          
            # pick random image from folder in the plus list (without replacement)
            # check to see if the image exists and can be loaded, if not move to another random image
            while True:
                random_image = random.choice(plus_name_list)[:-4] + '.png' # note that processed images are all .png type, while patient_table has different types
                if random_image in os.listdir(self.image_dir):
                    break
                else:
                    print('attempted to get following image, but missing: ' + random_image)

            # then find INTRA patient comparison 
            # filter for the same patient by subjectID
            self.patient_table['subjectID_first9'] = [x[0:9] for x in self.patient_table['subjectID']] # note that some subject IDs have additional characters appended, create new column to only take the unique 9 character ID)
            sub_table = self.patient_table[self.patient_table['subjectID_first9'] == random_image[0:9]]
            # filter for the same eye
            which_eye = random_image.split('_')[4]
            sub_table = sub_table[sub_table['eye'] == which_eye]
            # randomly select a row from sub_table, select the file name, and change file name to end with .png
            paired_image = sub_table.sample(n=1)['imageName'].item()[:-4] + '.png'

            # get labels and meta data
            plus_disease_0 = random_image.split('_')[5][:-4]
            plus_disease_1 = paired_image.split('_')[5][:-4]

            if plus_disease_1 == 'No':
                reverse_direction = random.randint(0,1) 
                if reverse_direction == 1:
                    random_image, paired_image = paired_image, random_image
                    plus_disease_0, plus_disease_1 = plus_disease_1, plus_disease_0
                break
 
        # Create Preplus Plus pair (loop until appropriate pair found)
        while change_combo == 5:          
            # pick random image from folder in the plus list (without replacement)
            # check to see if the image exists and can be loaded, if not move to another random image
            while True:
                random_image = random.choice(plus_name_list)[:-4] + '.png' # note that processed images are all .png type, while patient_table has different types
                if random_image in os.listdir(self.image_dir):
                    break
                else:
                    print('attempted to get following image, but missing: ' + random_image)

            # then find INTRA patient comparison 
            # filter for the same patient by subjectID
            self.patient_table['subjectID_first9'] = [x[0:9] for x in self.patient_table['subjectID']] # note that some subject IDs have additional characters appended, create new column to only take the unique 9 character ID)
            sub_table = self.patient_table[self.patient_table['subjectID_first9'] == random_image[0:9]]
            # filter for the same eye
            which_eye = random_image.split('_')[4]
            sub_table = sub_table[sub_table['eye'] == which_eye]
            # randomly select a row from sub_table, select the file name, and change file name to end with .png
            paired_image = sub_table.sample(n=1)['imageName'].item()[:-4] + '.png'

            # get labels and meta data
            plus_disease_0 = random_image.split('_')[5][:-4]
            plus_disease_1 = paired_image.split('_')[5][:-4]

            if plus_disease_1 == 'Pre-Plus':
                reverse_direction = random.randint(0,1) 
                if reverse_direction == 1:
                    random_image, paired_image = paired_image, random_image
                    plus_disease_0, plus_disease_1 = plus_disease_1, plus_disease_0
                break

        # convert class labels to numeric form    
 
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
        subject_id = random_image.split('_')[0]
        eye = random_image.split('_')[4]
        # session indicates the time point -- cannot compare between different patients
        session_0 = random_image.split('_')[2] 
        session_1 = paired_image.split('_')[2] 
 
        label = plus_disease_binary_change # 0 for no change, 1 for change
 
        meta = {"subject_id": subject_id,
                "eye": eye, 
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
 
 
class SiameseNetwork(nn.Module):
    """
    Siamese neural network
    Based on: https://hackernoon.com/facial-similarity-with-siamese-networks-in-pytorch-9642aa9db2f7
    Siamese ResNet-18 from Pytorch library
    """ 
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # note that resnet18 requires 3 input channels, will repeat grayscale image x3
        self.cnn1 = models.resnet18(pretrained = True)
        self.cnn1.fc = nn.Linear(512, 3) # mapping input image to a 3-dimensional space - three classes for other experiment

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