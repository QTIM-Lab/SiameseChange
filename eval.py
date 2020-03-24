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

'''
Load imaging data with annotations
'''

# annotations - data frame with the image data annotations
testing_table = pd.read_csv(working_path + 'data/testing_table.csv')

# processed image directory
image_dir = working_path + 'data/'

# TESTING DATA - INTRA-patient

testing_transforms = transforms.Compose([
    transforms.CenterCrop(224), # pixel crop 
    transforms.ToTensor()
])
 
testing_siamese_dataset = ROP_dataset(patient_table = testing_table,
                                        image_dir = image_dir, 
                                        epoch_size = 1000, # number of training pairs to test on
                                        transform = testing_transforms)

testing_dataloader = torch.utils.data.DataLoader(testing_siamese_dataset, 
                                                  batch_size=1, 
                                                  shuffle=False, 
                                                  num_workers=0)

"""
Testing the siamese network 
- Test set metrics: accuracy and linear-Kappa
"""


def siamese_testing(testing_dataloader, euclidean_distance_threshold, output_folder_name):
    '''
    Arguments
    - testing_dataloader: pytorch dataloader object
    - euclidean_distance_threshold: maximum distance to be considered "same"
    - output_folder_name: the folder where the training/validation output was saved
    '''
    # load the saved model
    output_dir = working_path + output_folder_name
    net = SiameseNetwork().cuda()
    net.load_state_dict(torch.load(output_dir + "/siamese_model.pth"))
    subnet = SiameseNetwork_umap(net) # to extract avgpool layer for umap
  
    #turn off gradients for testing
    with torch.no_grad(): 
        net.eval() # set evaluation mode
 
        # initialize variables
        testing_accuracy_count = 0
        kappa_label, kappa_test_label = [], []
        label_record, euclidean_distance_record = [], []
        subject_id_record, plus_disease_0_record, plus_disease_1_record, plus_disease_change_record, session_0_record, session_1_record = [], [], [], [], [], []

        print('\nTesting has started...')

        # determine testing accuracy and kappa
        # batch size for the testing_dataloader should = 1 
        for i, data in enumerate(testing_dataloader, 0):
            img0, img1, label, meta = data
            img0 = np.repeat(img0, 3, 1) # repeat grayscale image in 3 channels (ResNet requires 3-channel input) 
            img1 = np.repeat(img1, 3, 1)
            img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
            output1, output2 = net.forward(img0, img1)
            avgpool1, avgpool2 = subnet.forward(img0, img1)

            # get test labels using Euclidean distance threshold
            euclidean_distance = F.pairwise_distance(output1, output2)
            euclidean_distance_record.append(euclidean_distance.item())
            testing_label = euclidean_distance > euclidean_distance_threshold # 0 if same, 1 if not same
            label = label.view(1, len(label))
            label_record.append(label.item())
            
            # count for accuracy calculation
            equals = testing_label == label.type(torch.cuda.ByteTensor) # 1 if true
            testing_accuracy_count += equals.type(torch.FloatTensor)

            # append to lists for kappa calculation
            kappa_label.append(label.item())
            kappa_test_label.append(testing_label.item())

            # record meta data
            subject_id_record.append(meta['subject_id'][0])
            plus_disease_0_record.append(meta['plus_disease_0'][0])
            plus_disease_1_record.append(meta['plus_disease_1'][0])
            plus_disease_change_record.append(meta['plus_disease_change'].item())
            session_0_record.append(meta['session_0'][0]) # first time point 
            session_1_record.append(meta['session_1'][0]) # second time point

    # calculate testing accuracy and kappa (non-weighted, since its binary classification)
    testing_accuracy = testing_accuracy_count/len(testing_dataloader)
    testing_kappa = metrics.cohen_kappa_score(kappa_label, kappa_test_label, weights = None)
    
    print(f'\nThe testing accuracy is {100 * testing_accuracy.item():.2f}%. ')
    print('The Cohen kappa is ' + str(round(testing_kappa, 3)) + '.')
    print(f'These metrics were evaluated on {len(testing_dataloader)} testing examples.')

    testing_results = {'labels':label_record, 'euclidean_distances':euclidean_distance_record,
                       'test_labels':kappa_test_label, 'subject_id':subject_id_record, 'plus_disease_0':plus_disease_0_record, 
                       'plus_disease_1':plus_disease_1_record, 'plus_disease_change':plus_disease_change_record,
                       'session_0':session_0_record, 'session_1':session_1_record
                       }

    return testing_results
 
# euclidean distance threshold was selected as the absolute difference between mean Euclidean distance for validation pairs with no class change and mean Euclidean distance for validation pairs with a class change (can be found in the testing/validation history)
testing_results = siamese_testing(testing_dataloader, 
                                  euclidean_distance_threshold = 0.985,
                                  output_folder_name = "experiment_1")
                      
# save history with pickle
with open(output_dir + "/testing_results.pckl", "wb") as f:
    pickle.dump(testing_results, f)

# accuracy calculation
metrics.accuracy_score(y_true = testing_results_df['labels'], y_pred = testing_results_df['test_labels'])

# Cohen Kappa calculation
metrics.cohen_kappa_score(testing_results_df['labels'], testing_results_df['test_labels'], weights = None)

