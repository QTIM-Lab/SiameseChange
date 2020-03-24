'''
Siamese neural network implementation for change detection
Originally created 4/23/19

'''

# WORKING DIRECTORY (should contain a data/ subdirectory)
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
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statistics
import pickle

# custom classes
from siamese_classes import ROP_dataset_v5, SiameseNetwork101, ContrastiveLoss

# CUDA for PyTorch
os.environ['CUDA_VISIBLE_DEVICES']='0' # pick GPU to use
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


'''
Load imaging data with annotations (from the output of the partition.py script)
'''
 
# annotations
training_table = pd.read_csv(working_path + 'data/training_table.csv')
validation_table = pd.read_csv(working_path + 'data/validation_table.csv')

# processed image directory
image_dir = working_path + 'data/'

# TRAINING DATA - INTRA-patient

# transforms used for retinopathy of prematurity
training_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5), # aim to train to be invariant to laterality of eye
    transforms.RandomRotation(10), # rotate +/- 5 degrees around center
    transforms.RandomCrop(224), # pixel crop 
    transforms.ColorJitter(brightness = 0.03, contrast = 0.03), # brightness and color variation of +/- 5%
    transforms.ToTensor()
])

training_siamese_dataset = ROP_dataset(patient_table = training_table,
                                        image_dir = image_dir, 
                                        epoch_size = 1000,
                                        transform = training_transforms)
 
training_dataloader = torch.utils.data.DataLoader(training_siamese_dataset, 
                                                  batch_size=32, 
                                                  shuffle=False, 
                                                  num_workers=0)
 
# VALIDATION DATA - INTRA-patient

validation_transforms = transforms.Compose([
    transforms.CenterCrop(224), # pixel crop 
    transforms.ToTensor()
])

validation_siamese_dataset = siamese_dataset(patient_table = validation_table,
                                        image_dir = image_dir, 
                                        epoch_size = 1000,
                                        transform = validation_transforms)

validation_dataloader = torch.utils.data.DataLoader(validation_siamese_dataset, 
                                                  batch_size=32, 
                                                  shuffle=False, 
                                                  num_workers=0)

 
'''
Training the siamese network 

'''

def siamese_training(training_dataloader, validation_dataloader, output_folder_name, learning_rate = 0.00001):
    '''
    - Implements siamese network training/validation with return of network weights and history of losses and accuracies
    - Implementation uses early stopping, saving the model with the best validation loss

    Arguments
    - training_dataloader: pytorch dataloader object
    - validation_dataloader: pytorch dataloader object
    - output_folder_name: name of folder to make in the working directory where the results will be saved
    - learning_rate: for Adam optimizer

    ''' 
    net = SiameseNetwork().cuda()
    criterion = ContrastiveLoss()
    criterion.margin = 2.0  # contrastive loss function margin
    optimizer = optim.Adam(net.parameters(),lr = learning_rate)

    # Initialization
    num_epochs = 1000
    training_losses, validation_losses = [], []
    training_accuracies, validation_accuracies = [], []
    euclidean_distance_threshold = 1

    # Early stopping initialization
    epochs_no_improve = 0
    max_epochs_stop = 5 # "patience" - number of epochs with no improvement in validation loss after which training stops
    validation_loss_min = np.Inf
    validation_max_accuracy = 0
    history = []

    output_dir = working_path + output_folder_name
    os.mkdir(output_dir)
    f = open(output_dir + "/history_training.txt", 'a+')
    f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" + "Training starting now...\n")
    f.close()

    for epoch in range(0, num_epochs):
        print("epoch training started...")

        # keep track of training and validation loss each epoch
        training_loss = 0
        validation_loss = 0

        # keep track of accuracy
        training_accuracy_history = []
        validation_accuracy_history = []

        # keep track of euclidean_distance and label history each epoch
        training_euclidean_distance_history = []
        training_label_history = []
        validation_euclidean_distance_history = []
        validation_label_history = []

        # model set to train
        net.train()
          
        # training loop
        for i, data in enumerate(training_dataloader, 0):
            # train neural network
            img0, img1, label, meta = data
            img0 = np.repeat(img0, 3, 1) # repeat grayscale image in 3 channels (ResNet requires 3-channel input) 
            img1 = np.repeat(img1, 3, 1)
            img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
            optimizer.zero_grad() # clear gradients
            output1, output2 = net.forward(img0, img1)
            loss_contrastive = criterion(output1, output2, label.float())
            loss_contrastive.backward()
            optimizer.step()

            # keep track of training loss
            training_loss += loss_contrastive.item()

            # evaluate training accuracy
            net.eval()
            output1, output2 = net.forward(img0, img1)
            net.train()
            euclidean_distance = F.pairwise_distance(output1, output2)
            training_label = euclidean_distance > euclidean_distance_threshold # 0 if same, 1 if not same
            label = label.view(1, len(label))
            equals = training_label == label.type(torch.cuda.ByteTensor) # 1 if true
            acc_tmp = torch.Tensor.numpy(equals.cpu())[0]
            training_accuracy_history.extend(acc_tmp)

            # save euclidean distance and label history 
            euclid_tmp = torch.Tensor.numpy(euclidean_distance.detach().cpu())
            training_euclidean_distance_history.extend(euclid_tmp)
            label_tmp = torch.Tensor.numpy(label.cpu())[0]
            training_label_history.extend(label_tmp)

        else:
            print("validation started...")
            #turn off gradients for validation
            with torch.no_grad(): 
                net.eval() # set evaluation mode

                # determine validation loss and validation accuracy
                for j, data2 in enumerate(validation_dataloader, 0):
                    img0, img1, label, meta = data2
                    img0 = np.repeat(img0, 3, 1) # repeat grayscale image in 3 channels (ResNet requires 3-channel input) 
                    img1 = np.repeat(img1, 3, 1)
                    img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label).cuda()
                    output1, output2 = net.forward(img0, img1)
                    loss_contrastive = criterion(output1, output2, label.float())
                    validation_loss += loss_contrastive.item()
                     
                    # evaluate validation accuracy using a Euclidean distance threshold
                    euclidean_distance = F.pairwise_distance(output1, output2)
                    validation_label = euclidean_distance > euclidean_distance_threshold # 0 if same, 1 if not same
                    label = label.view(1, len(label))
                    equals = validation_label == label.type(torch.cuda.ByteTensor) # 1 if true
                    acc_tmp = torch.Tensor.numpy(equals.cpu())[0]
                    validation_accuracy_history.extend(acc_tmp)
                    
                    # save euclidean distance and label history 
                    euclid_tmp = torch.Tensor.numpy(euclidean_distance.detach().cpu()) # detach gradient, move to CPU
                    validation_euclidean_distance_history.extend(euclid_tmp)
                    label_tmp = torch.Tensor.numpy(label.cpu())[0]
                    validation_label_history.extend(label_tmp)

            # calculate average training and validation losses (averaged across batches for the epoch)
            training_loss_avg = training_loss/len(training_dataloader)
            validation_loss_avg = validation_loss/len(validation_dataloader)

            # training and validation accuracy calculation
            training_accuracy = statistics.mean(np.array(training_accuracy_history).tolist())
            validation_accuracy = statistics.mean(np.array(validation_accuracy_history).tolist())

            # Save the model if validation loss decreases
            if validation_loss_avg < validation_loss_min:
                # save model
                torch.save(net.state_dict(), output_dir + "/siamese_model.pth")
                # track improvement
                epochs_no_improve = 0
                validation_loss_min = validation_loss_avg
                validation_max_accuracy = validation_accuracy
                best_epoch = epoch
 
            # Otherwise increment count of epochs with no improvement
            else: 
                epochs_no_improve += 1 
                # Trigger EARLY STOPPING
                if epochs_no_improve >= max_epochs_stop:
                    print(f'\nEarly Stopping! Total epochs (starting from 0): {epoch}. Best epoch: {best_epoch} with loss: {validation_loss_min:.2f} and acc: {100 * validation_max_accuracy:.2f}%')
                    # Load the best state dict (at the early stopping point)
                    net.load_state_dict(torch.load(output_dir + "/siamese_model.pth"))
                    # attach the optimizer
                    net.optimizer = optimizer

                    # save history with pickle
                    with open(output_dir + "/history_training.pckl", "wb") as f:
                        pickle.dump(history, f)

                    f = open(output_dir + "/history_training.txt", 'a+')
                    f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" +
                            "Early stopping! Total epochs (starting from 0): {:.0f}\n".format(epoch) +
                            "Best epoch: {:.0f}\n".format(best_epoch) +
                            "Validation loss at best epoch: {:.3f}\n".format(validation_loss_min) +
                            "Validation accuracy at best epoch: {:3f}\n".format(validation_max_accuracy)
                            )
                    f.close()

                    return net, history

        # after each Epoch

        # append to lists for graphing
        training_losses.append(training_loss_avg)
        validation_losses.append(validation_loss_avg)
        training_accuracies.append(training_accuracy)
        validation_accuracies.append(validation_accuracy)

        # training euclidean distance stats
        # extract euclidean distances if label is 0 or 1
        euclid_if_0 = [b for a, b in zip(training_label_history, training_euclidean_distance_history) if a == 0]
        euclid_if_1 = [b for a, b in zip(training_label_history, training_euclidean_distance_history) if a == 1]
        euclid_if_0 = np.array(euclid_if_0).tolist()
        euclid_if_1 = np.array(euclid_if_1).tolist()
        
        # summary statistics for euclidean distances
        mean_euclid_0t = statistics.mean(euclid_if_0) 
        std_euclid_0t = statistics.pstdev(euclid_if_0) # population stdev
        mean_euclid_1t = statistics.mean(euclid_if_1)
        std_euclid_1t = statistics.pstdev(euclid_if_1) # population stdev
        euclid_diff_t = mean_euclid_1t - mean_euclid_0t

        # validation euclidean distance stats
        # extract euclidean distances if label is 0 or 1
        euclid_if_0 = [b for a, b in zip(validation_label_history, validation_euclidean_distance_history) if a == 0]
        euclid_if_1 = [b for a, b in zip(validation_label_history, validation_euclidean_distance_history) if a == 1]
        euclid_if_0 = np.array(euclid_if_0).tolist()
        euclid_if_1 = np.array(euclid_if_1).tolist()
        
        # summary statistics for euclidean distances
        mean_euclid_0v = statistics.mean(euclid_if_0) 
        std_euclid_0v = statistics.pstdev(euclid_if_0) # population stdev
        mean_euclid_1v = statistics.mean(euclid_if_1)
        std_euclid_1v = statistics.pstdev(euclid_if_1) # population stdev
        euclid_diff_v = mean_euclid_1v - mean_euclid_0v

        # after the epoch is completed, adjust the euclidean_distance_threshold based on the validation mean euclidean distances
        euclidean_distance_threshold = (mean_euclid_0v + mean_euclid_1v) / 2

        # store in history list
        history = [training_losses, validation_losses, training_accuracies, validation_accuracies,
                   euclid_diff_t, euclid_diff_v]
 
        # save history with pickle
        with open(output_dir + "/history_training.pckl", "wb") as f:
            pickle.dump(history, f)
 
        print("Epoch number: {:.0f}\n".format(epoch),
            "Training loss: {:.3f}\n".format(training_loss_avg),
            "Training accuracy: {:.3f}\n".format(training_accuracy),
            "Validation loss: {:.3f}\n".format(validation_loss_avg),
            "Validation accuracy: {:.3f}\n".format(validation_accuracy),
            "\nTraining \nLabel0 euclidean distance mean: {:.3f}\n".format(mean_euclid_0t),
            "Label0 euclidean distance std: {:.3f}\n".format(std_euclid_0t),
            "Label1 euclidean distance mean: {:.3f}\n".format(mean_euclid_1t),
            "Label1 euclidean distance std: {:.3f}\n".format(std_euclid_1t),
            "Euclidean distance mean diff: {:.3f}\n".format(euclid_diff_t),
            "\nValidation \nLabel0 euclidean distance mean: {:.3f}\n".format(mean_euclid_0v),
            "Label0 euclidean distance std: {:.3f}\n".format(std_euclid_0v),
            "Label1 euclidean distance mean: {:.3f}\n".format(mean_euclid_1v),
            "Label1 euclidean distance std: {:.3f}\n".format(std_euclid_1v),
            "Euclidean distance mean diff: {:.3f}\n".format(euclid_diff_v),
            "Euclidean distance threshold update: {:.3f}\n".format(euclidean_distance_threshold)
            )

        # write history to file
        f = open(output_dir + "/history_training.txt", 'a+')
        f.write(datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n" +
            "Epoch number: {:.0f}\n".format(epoch) +
            "Training loss: {:.3f}\n".format(training_loss_avg) +
            "Training accuracy: {:.3f}\n".format(training_accuracy) +
            "Validation loss: {:.3f}\n".format(validation_loss_avg) +
            "Validation accuracy: {:.3f}\n".format(validation_accuracy) +
            "\nTraining \nLabel0 euclidean distance mean: {:.3f}\n".format(mean_euclid_0t) +
            "Label0 euclidean distance std: {:.3f}\n".format(std_euclid_0t) +
            "Label1 euclidean distance mean: {:.3f}\n".format(mean_euclid_1t) +
            "Label1 euclidean distance std: {:.3f}\n".format(std_euclid_1t) +
            "Euclidean distance mean diff: {:.3f}\n".format(euclid_diff_t) +
            "\nValidation \nLabel0 euclidean distance mean: {:.3f}\n".format(mean_euclid_0v) + 
            "Label0 euclidean distance std: {:.3f}\n".format(std_euclid_0v) + 
            "Label1 euclidean distance mean: {:.3f}\n".format(mean_euclid_1v) + 
            "Label1 euclidean distance std: {:.3f}\n".format(std_euclid_1v) +
            "Euclidean distance mean diff: {:.3f}\n".format(euclid_diff_v) +
            "Euclidean distance threshold update: {:.3f}\n".format(euclidean_distance_threshold) + "\n"
            )
        f.close()
 
    # Load the best state dict (at the early stopping point)
    net.load_state_dict(torch.load(output_dir + "/siamese_model.pth"))
    # After training through all epochs attach the optimizer
    net.optimizer = optimizer

    # Return the best model and history
    print(f'\nAll Epochs completed! Total epochs (starting from 0): {epoch}. Best epoch: {best_epoch} with validation loss: {validation_loss_min:.2f} and acc: {100 * validation_max_accuracy:.2f}%')
    return net, history

# siamese training 
net, history = siamese_training(training_dataloader = training_dataloader, 
                               validation_dataloader = validation_dataloader, 
                               output_folder_name = 'experiment_1')
   
# Training/validation learning curves
plt.title("Number of Training Epochs vs. Contrastive Loss")
plt.xlabel("Training Epochs")
plt.ylabel("Contrastive Loss")
plt.plot(range(0, len(history[0])), history[0], label = "Training loss")
plt.plot(range(0, len(history[1])), history[1], label = "Validation loss")
plt.legend(frameon=False)
plt.savefig(output_dir + "/Learning_curve.png")
plt.close()
