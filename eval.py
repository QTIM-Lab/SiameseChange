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

Part 4: 
- Example of inference of image Euclidean distance relative to an anchor (pooled 'normal' images)
- Example of two image pairwise Euclidean distance inference

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
from siamese_ROP_classes import SiameseNetwork101, img_processing, anchor_inference, twoimage_inference

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
- 1. median Euclidean distance difference (relative to a pool of randomly sampled 'normal' images)
- 2. pairwise Euclidean distance between two images for direct comparison
'''

pooled_normal_test = pd.read_csv(working_path + 'data/pooled_normal_test.csv')
testing_table = pd.read_csv(working_path + 'data/testing_table.csv')

# loop through each patient in the test set, find both eyes, get two time points per eye if available
# no same image comparison. time points should have directionality in this case

subjectID_record = []
eye_record = []
PD_A_record = []
PD_B_record = []
PD_change_record = []
PD_change_binary_record = []

euclidean_distance_diff_record = []
pairwise_euclidean_distance_record = []

# represent labels as numerics
testing_table.loc[testing_table['Ground truth'] == 'No', 'Ground truth'] = 0
testing_table.loc[testing_table['Ground truth'] == 'Pre-Plus', 'Ground truth'] = 1
testing_table.loc[testing_table['Ground truth'] == 'Plus', 'Ground truth'] = 2

# reformat the image_paths to match pooled_normal_test format
testing_table['image_path'] = [(image_dir + a[:-3] + 'png') for a in testing_table['imageName']]

net.eval()
for i in list(set(testing_table['subjectID'])):
    # left eye combos
    pick_L = testing_table[testing_table['subjectID'] == i][testing_table['eye'] == 'os']
    # right eye combos
    pick_R = testing_table[testing_table['subjectID'] == i][testing_table['eye'] == 'od']

    # randomly pick an eye combo with change if it exists (since change is underrepresented)
    # if does not exist, pick any random combo

    for p in [pick_L, pick_R]:
        if len(p) > 1: # need at least two images to compare
            possible_labels = list(set(p['Ground truth']))
            if len(possible_labels) > 1: # pick change example if available
                pick2labels = np.random.choice(possible_labels, 2, replace = False)
                pick1 = p[p['Ground truth'] == pick2labels[0]].sample()
                pick2 = p[p['Ground truth'] == pick2labels[1]].sample()
                pick1, pick2 = pick1.squeeze(), pick2.squeeze()
            else: # pick no change example
                no_change_sample = p.sample(2, replace = False)
                pick1 = no_change_sample.iloc[0]
                pick2 = no_change_sample.iloc[1]

            time_diff = pick2['PMARaw'].item() - pick1['PMARaw'].item()
            if time_diff < 1: 
                # if first time point is later than second time point, reverse order
                pick1, pick2 = pick2, pick1

            if pick1['eye'] == 'os': 
                R_eye = 0
            else: 
                R_eye = 1

            PD_A = pick1['Ground truth'].item()
            PD_B = pick2['Ground truth'].item()
            PD_change = PD_B - PD_A
            if PD_change == 0:
                PD_change_binary = 0
            else: 
                PD_change_binary = 1

            pooled_normal_test_A = pooled_normal_test[pooled_normal_test['image_path'] == pick1['image_path']]
            pooled_normal_test_B = pooled_normal_test[pooled_normal_test['image_path'] == pick2['image_path']]
            euclidean_distance_A = pooled_normal_test_A['euclidean_distance'].iloc[0]
            euclidean_distance_B = pooled_normal_test_B['euclidean_distance'].iloc[0]
            euclidean_distance_diff = euclidean_distance_B - euclidean_distance_A

            with torch.no_grad():
                img_comp1 = img_processing(Image.open(pick1['image_path']))
                img_comp2 = img_processing(Image.open(pick2['image_path']))
                output1, output2 = net.forward(img_comp1, img_comp2)
                pairwise_euclidean_distance = F.pairwise_distance(output1, output2).item()

            subjectID_record.append(i)
            eye_record.append(R_eye)
            PD_A_record.append(PD_A)
            PD_B_record.append(PD_B)
            PD_change_record.append(PD_change)
            PD_change_binary_record.append(PD_change_binary)
            euclidean_distance_diff_record.append(euclidean_distance_diff)
            pairwise_euclidean_distance_record.append(pairwise_euclidean_distance)

            print('patient ' + str(i) + ' for eye ' + pick1['eye'])

longitudinal_change_tab = pd.DataFrame({ 'subjectID':subjectID_record,
                                         'R_eye':eye_record,
                                         'PD_A':PD_A_record,
                                         'PD_B':PD_B_record,
                                         'PD_change':PD_change_record,
                                         'PD_change_binary':PD_change_binary_record,
                                         'euclidean_distance_diff':euclidean_distance_diff_record,
                                         'pairwise_euclidean_distance':pairwise_euclidean_distance_record,
                                        })

longitudinal_change_tab.to_csv(working_path + 'data/longitudinal_change_tab.csv')
longitudinal_change_tab = pd.read_csv(working_path + 'data/longitudinal_change_tab.csv')

testing_results_df = longitudinal_change_tab 

# boxplot for euclidean distance diff
plt.figure()
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
plt.tick_params(axis='both', which='major', labelsize=15)
sns.set(style="white")
ax = sns.boxplot(x="PD_change", y="euclidean_distance_diff", data=testing_results_df, color = 'white', showfliers = False)

for i,box in enumerate(ax.artists):
    box.set_edgecolor('black')
    box.set_facecolor('white')
    # iterate over whiskers and median lines
    for j in range(5*i,5*(i+1)):
         ax.lines[j].set_color('black')

ax = sns.swarmplot(x="PD_change", y="euclidean_distance_diff", data=testing_results_df, color="grey")
plt.xlabel('Longitudinal Plus Disease Grade Change', fontsize = 15)
plt.ylabel('Euclidean Distance Difference', fontsize = 15)
plt.savefig(output_dir + "/PDchange_vs_EuclideanDist_boxplot_euclidean_distance_diff.png")
plt.close()

# Spearman rank correlation 
testing_results_df[['euclidean_distance_diff', 'PD_change']].corr(method = "spearman")


# boxplot for euclidean distance pairwise
testing_results_df['PD_change'] = abs(testing_results_df['PD_change'])

plt.figure()
plt.tight_layout()
plt.gcf().subplots_adjust(bottom=0.15, left=0.15)
plt.tick_params(axis='both', which='major', labelsize=15)
sns.set(style="white")
ax = sns.boxplot(x="PD_change", y="pairwise_euclidean_distance", data=testing_results_df, color = 'white', showfliers = False)

for i,box in enumerate(ax.artists):
    box.set_edgecolor('black')
    box.set_facecolor('white')
    # iterate over whiskers and median lines
    for j in range(5*i,5*(i+1)):
         ax.lines[j].set_color('black')

ax = sns.swarmplot(x="PD_change", y="pairwise_euclidean_distance", data=testing_results_df, color="grey")
plt.xlabel('Longitudinal Magnitude of Plus Disease Grade Change', fontsize = 15)
plt.ylabel('Pairwise Euclidean Distance', fontsize = 15)
plt.savefig(output_dir + "/PDchange_vs_EuclideanDist_boxplot_euclidean_distance_pairwise.png")
plt.close()

# Spearman rank correlation 
testing_results_df[['pairwise_euclidean_distance', 'PD_change']].corr(method = "spearman")

'''
Part 4: Example inference
- Example of inference of image median Euclidean distance relative to an anchor (pooled 'normal' images)
- Example of inference of two image pairwise Euclidean distance
'''

# loading siamese neural network model from the main.py output
output_folder_name = 'Res101_inter'
output_dir = working_path + "scripts/histories/" + output_folder_name
net = SiameseNetwork101().cuda()
net.load_state_dict(torch.load(output_dir + "/siamese_ROP_model.pth"))
history = pickle.load(open(output_dir + "/history_training.pckl", "rb"))

### inference relative to pool of normal images 

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

# takes the img_anchor, image of interest, and siamese neural network model as inputs
anchor_inference(img_anchor, image_dir + 'imagename.png', net)

### pairwise inference

# example of determining euclidean distance between two images
# takes the two images of interest and siamese neural network model as inputs
twoimage_inference(image_dir + 'image1.png',
                   image_dir + 'image2.png',
                   net)
