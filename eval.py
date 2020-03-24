'''
Siamese ROP evaluation
created 5/21/2019

Part 1: 
- Analysis of 100 excluded samples from the dataset, previously annotated by experts for disease severity ranking
- Show that we can learn the severity of clinical grade to a finer degree of continuous variation than just normal, pre-plus, and plus
- Euclidean distance from siamese network model should reflect these distances

Part 2: 
- Analysis of test set paired image comparison median Euclidean distance relative to a randomly sampled pool of normal images

Part 3: 
- Analysis of longitudinal change in disease severity in the test set, using two methods:
- 1. median Euclidean distance relative to a pool of randomly sampled 'normal' images
- 2. pairwise Euclidean distance between two images for direct comparison

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
from siamese_ROP_classes import SiameseNetwork101, img_processing

# CUDA for PyTorch
os.environ['CUDA_VISIBLE_DEVICES']='0' # pick GPU to use
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# loading siamese neural network model from the main.py output
output_folder_name = 'Res101_inter'
output_dir = working_path + "scripts/histories/" + output_folder_name
net = SiameseNetwork101().cuda()
net.load_state_dict(torch.load(output_dir + "/siamese_ROP_model.pth"))
history = pickle.load(open(output_dir + "/history_training.pckl", "rb"))


'''
Part 1: Evaluation using the 100 expert ranked images for disease severity 
- correlate disease severity ranking with Euclidean distance between image of interest and 5 most "normal" images
'''

# load rank order annotationtable (NOTE: rank 100 is plus disease, rank 1 is normal)
rank_table = pd.read_csv(working_path + 'data/rank_data.csv')

image_dir = working_path + 'data/100rankedimages/'

# anchor image, rank 1-5 -- i.e. 5 most normal studies, compare all to this
img_anchor1 = img_processing(Image.open(image_dir + '1.bmp'))
img_anchor2 = img_processing(Image.open(image_dir + '2.bmp'))
img_anchor3 = img_processing(Image.open(image_dir + '3.bmp'))
img_anchor4 = img_processing(Image.open(image_dir + '4.bmp'))
img_anchor5 = img_processing(Image.open(image_dir + '5.bmp'))

img_anchor = [img_anchor1, img_anchor2, img_anchor3, img_anchor4, img_anchor5]

euclidean_distance_record = []
grade_record = []

net.eval()

for i in range(1,101):
    tmp = rank_table[rank_table['consensus_rank'] == i]
    img_comparison = Image.open(image_dir + tmp['imgName'].item()[:-3] + 'bmp')
    img_comparison = img_processing(img_comparison)

    save_euclidean_distance = []
    for j in range(0,5):
        output0, output1 = net.forward(img_anchor[j], img_comparison)
        euclidean_distance = F.pairwise_distance(output0, output1)
        save_euclidean_distance.append(euclidean_distance.item())

    # take average euclidean distance compared to the 5 lowest rank anchor images as the baseline
    euclidean_distance_record.append(statistics.mean(save_euclidean_distance))

    grade_record.append(tmp['grade'].item()) # ordinal disease severity classification
    
    print(str(i) + ' passed')

output_df = pd.DataFrame({'consensus_rank':list(range(1,101)), 
                          'euclidean_distance':euclidean_distance_record,
                          'grade_record':grade_record,
                         })

# Analysis of Rank Results #

output_df[['consensus_rank', 'euclidean_distance']].corr(method = "spearman")

# correlations within categories
output_df[output_df['grade_record'] == 'No'][['consensus_rank', 'euclidean_distance']].corr(method = "spearman")
output_df[output_df['grade_record'] == 'Pre-Plus'][['consensus_rank', 'euclidean_distance']].corr(method = "spearman")
output_df[output_df['grade_record'] == 'Plus'][['consensus_rank', 'euclidean_distance']].corr(method = "spearman")

plt.figure() 
plt.gcf().subplots_adjust(bottom=0.15)
plt.scatter(output_df[output_df['grade_record'] == 'No']['consensus_rank'], output_df[output_df['grade_record'] == 'No']['euclidean_distance'], color='blue', lw=2) 
plt.scatter(output_df[output_df['grade_record'] == 'Pre-Plus']['consensus_rank'], output_df[output_df['grade_record'] == 'Pre-Plus']['euclidean_distance'], color='magenta', lw=2) 
plt.scatter(output_df[output_df['grade_record'] == 'Plus']['consensus_rank'], output_df[output_df['grade_record'] == 'Plus']['euclidean_distance'], color='red', lw=2) 
plt.legend()
plt.xlabel('Plus Disease Consensus Rank', fontsize = 15) 
plt.ylabel('Median Euclidean Distance', fontsize = 15) 
plt.savefig(output_dir + "/Consensus_Rank_vs_Euclidean_distance_withlegend.png") 
plt.close() 


'''
Part 2: Evaluation of Euclidean distance disease severity of all test set images relative to a pool of normal studies
- Take random sample of 10 "normal" images from the test set (normal image pool)
- Do pairwise comparison of each test set image with the normal image pool image (i.e. 10 euclidean distances)
- Take arithmetic median of the euclidean distances to provide a measure of disease severity
'''

# just the default test_table.csv (annotations for the randomly partitioned test set), where each row is one image with its annotations
testing_table_byimage = pd.read_csv(working_path + 'data/testing_table.csv')

# random sample of 10 "normal" images from the test set
random10 = testing_table_byimage[testing_table_byimage['Ground truth'] == 'No'].sample(10)
random10.to_csv(working_path + 'data/random10.csv')

# anchor images from the random 10 images
random10 = pd.read_csv(working_path + 'data/random10.csv')
img_anchor = []
for a in range(len(random10)):
    image_path = image_dir + random10.iloc[a]['imageName'][:-3] + 'png'
    img_anchor.append(img_processing(Image.open(image_path)))

image_path_record = []
euclidean_distance_record = []
grade_record = []

net.eval()

for i in range(len(testing_table_byimage)):
    tmp = testing_table_byimage.iloc[i]
    image_path = image_dir + tmp['imageName'][:-3] + 'png'

    try:
        img_comparison = img_processing(Image.open(image_path))
        image_path_record.append(image_path)

        save_euclidean_distance = []
        for j in range(len(img_anchor)):
            output1, output2 = net.forward(img_anchor[j], img_comparison)
            euclidean_distance = F.pairwise_distance(output1, output2)
            save_euclidean_distance.append(euclidean_distance.item())

        # take average (or median) euclidean distance compared to the the pool of normals
        # euclidean_distance_record.append(statistics.mean(save_euclidean_distance))
        euclidean_distance_record.append(statistics.median(save_euclidean_distance))

        # true ROP grade record
        grade_record.append(tmp['Ground truth'])

        print(str(i) + 'completed')     
    except:
        print(str(i) + 'image is missing from data set')


pooled_normal_test = pd.DataFrame({'image_path':image_path_record, 
                         'euclidean_distance':euclidean_distance_record,
                         'grade_record':grade_record,
                        })

pooled_normal_test.to_csv(working_path + 'data/pooled_normal_test.csv')

### visualization ###
pooled_normal_test = pd.read_csv(working_path + 'data/pooled_normal_test.csv')

# boxplot
plt.figure()
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
plt.tick_params(axis='both', which='major', labelsize=15)
sns.set(style="white")
ax = sns.boxplot(x="grade_record", y="euclidean_distance", data=pooled_normal_test, color='white', showfliers = False, order = ['No', 'Pre-Plus', 'Plus'])

for i,box in enumerate(ax.artists):
    box.set_edgecolor('black')
    box.set_facecolor('white')
    # iterate over whiskers and median lines
    for j in range(5*i,5*(i+1)):
         ax.lines[j].set_color('black')

ax = sns.swarmplot(x="grade_record", y="euclidean_distance", data=pooled_normal_test, color="grey", size = 3, order = ['No', 'Pre-Plus', 'Plus'])
plt.xlabel('Plus Disease Classification', fontsize = 15)
plt.ylabel('Median Euclidean Distance', fontsize = 15)
plt.savefig(output_dir + "/PlusDisease_vs_EuclideanDist_boxplot_pooled_analysis_median.png")
plt.close()

# Spearman rank correlation 
pooled_normal_test_.loc[pooled_normal_test['grade_record'] == 'No', 'grade_record'] = 0
pooled_normal_test_.loc[pooled_normal_test['grade_record'] == 'Pre-Plus', 'grade_record'] = 1
pooled_normal_test_.loc[pooled_normal_test['grade_record'] == 'Plus', 'grade_record'] = 2

pooled_normal_test_[['euclidean_distance', 'grade_record']].corr(method = "spearman")


'''
Part 3: Analysis of longitudinal change in disease severity in the test set, using two methods:
- 1. median Euclidean distance relative to a pool of randomly sampled 'normal' images
- 2. pairwise Euclidean distance between two images for direct comparison
'''
