#!/usr/bin/env python
# coding: utf-8

# # PyTorch / PIL

# In[1]:


import numpy as np
import pandas as pd
import importlib
import pickle
import os

import PIL
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import timeit

from torch.utils.data import Dataset, DataLoader,TensorDataset
from torchvision import transforms


# In[2]:


import re

label_types = ['white_blood_cell','debris','malariae', 'falciparum', 'ovale']

def create_df(dir_name):
    directory = dir_name
    augmented_df = pd.DataFrame(columns=['image','class'])

    idx = 0
    for image_name in os.listdir(directory):
        if 'white_blood_cell' in image_name:
            label_type = 'white_blood_cell'
        else:
            label_type = image_name.split('_')[0]

        class_number = label_types.index(label_type)
        augmented_df.loc[idx] = [image_name,class_number]
        idx += 1
    return augmented_df

train_labels_df = create_df('./split_gamma_0.5/train')
train_labels_df_sobel = create_df('./splitSobel/train')

test_labels_df = create_df('./split/test')


# In[23]:


label_types.index('ovale')


# In[3]:


# train_labels_df = pd.read_csv('original_train.csv')
# test_labels_df = pd.read_csv('original_test.csv')

train_PIL_images = [] 
test_PIL_images = [] 

train_PIL_images_sobel = []
train_labels_sobel = []

train_labels = []
test_labels = []

for filename in train_labels_df['image'].values: ##to keep mapping with classes
    train_PIL_images.append(Image.open('split_gamma_0.5/train/'+filename).copy())
    train_labels.append(train_labels_df.loc[train_labels_df['image'] == filename, 'class'].iloc[0])
    
for filename in train_labels_df_sobel['image'].values: ##to keep mapping with classes
    train_PIL_images_sobel.append(Image.open('splitSobel/train/'+filename).copy())
    train_labels_sobel.append(train_labels_df_sobel.loc[train_labels_df_sobel['image'] == filename, 'class'].iloc[0])
    
for filename in test_labels_df['image'].values: ##to keep mapping with classes
    test_PIL_images.append(Image.open('split/test/'+filename).copy())
    test_labels.append(test_labels_df.loc[test_labels_df['image'] == filename, 'class'].iloc[0])


# In[4]:


X_train = []
X_train.extend(train_PIL_images)
X_train.extend(train_PIL_images_sobel)

Y_train = []
Y_train.extend(train_labels)
Y_train.extend(train_labels_sobel)
len(Y_train)


# In[24]:


test_labels_df['class'].value_counts()


# ## Dataset/DataLoader for PyTorch

# In[6]:


"""Custom Datasets that obtain lists of PIL Images 
and labels as input
"""

class ListsTrainDataset(Dataset):
    def __init__(self, list_of_images, list_of_labels, transform=None):
        
        self.data = list_of_images
        self.labels = np.asarray(list_of_labels).reshape(-1,1)
        self.transform = transform

    def __getitem__(self, index):
        single_image = self.data[index]
        single_image_label = self.labels[index]
        # Transform image to tensor
        if self.transform is not None:
            img_as_tensor = self.transform(single_image)
        # Return image and the label
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return len(self.data)



class ListsTestDataset(Dataset):
    def __init__(self, list_of_images, transform=None):
        """
        Args:
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data = list_of_images
        self.transform = transform

    def __getitem__(self, index):
        single_image = self.data[index]
        if self.transform is not None:
            img_as_tensor = self.transform(single_image)
        # Return image ONLY
        return img_as_tensor

    def __len__(self):
        return len(self.data)


# In[7]:


def create_train_val_datasets(X_train, y_train, X_val = None, y_val = None, norm_params = None):

        print(norm_params)
        val_transforms = transforms. Compose([
            transforms.Resize(size=(64, 64)),
            # transforms.RandomCrop(64),
#             transforms.Grayscale(),
            transforms.ToTensor(),
#             transforms.Normalize(mean=[norm_params['train_norm_mean']],
#                         std =[norm_params['train_norm_std']])
        ])

        train_transforms = transforms. Compose([
            transforms.Resize(size=(64, 64)),
#             transforms.Grayscale(),
            # transforms.resize(image, (64, 64)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=360),
            # transforms.RandomAffine(16),
            transforms.ToTensor(),
#             transforms.Normalize(mean=[norm_params['train_norm_mean']],
#                         std =[norm_params['train_norm_std']])
        ])

        train_dataset = ListsTrainDataset(X_train, y_train, transform = train_transforms)

        if X_val is None and y_val is None:
            return train_dataset

        elif X_val is not None:
            test_dataset = ListsTrainDataset(X_val, y_val, transform = val_transforms)
        else:
            test_dataset = ListsTestDataset(X_val, transform = test_transforms)

        return (train_dataset, test_dataset)


# ## Preprocessing during validation

# In[8]:


import cv2

def apply_sobel(image):
    kernel_vert = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
    kernel_horz = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.filter2D(image, -1, kernel_vert)
    image = cv2.filter2D(image, -1, kernel_horz)
    
    return image

def apply_gamma(image, gamma=0.5):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# In[9]:


import cv2

def apply_sobel(image):
    kernel_vert = np.array([[1,0,-1], [2,0,-2], [1,0,-1]])
    kernel_horz = np.array([[-1,-2,-1], [0,0,0], [1,2,1]])

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.filter2D(image, -1, kernel_vert)
    image = cv2.filter2D(image, -1, kernel_horz)
    
    return image

def apply_gamma(image, gamma=0.5):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# In[10]:


from scipy import ndimage
from skimage.transform import rescale, resize


def augment_and_transform_for_prediction(im):
    
    num_of_rotations = 5
    
    im = transforms.ToPILImage()(im.squeeze())
    im = np.array(im)
    
    augmented_image_list = list()
    augmented_image_list.append(im) ##original
    ##augment
    for i in range(num_of_rotations):
        rotated = ndimage.rotate(im, np.random.randint(0, high = 360))
        augmented_image_list.append(rotated)

    ##Transformed

    re_transform_for_cnn = transforms. Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(64, 64)),
                transforms.ToTensor()
            ])

    sobel_images = []
    for im in augmented_image_list:
        sobel_images.append(apply_gamma(im))
        sobel_images.append(apply_sobel(im))

    augmented_image_list.extend(sobel_images)

    ##back to PIL and Torch Tensor

    tensor_list = []
    for im in augmented_image_list:
        tensor_list.append(re_transform_for_cnn(im).unsqueeze(0))

    final_tensor = torch.Tensor()
    final_tensor.size()
    for i, image in enumerate(tensor_list):
        if i == 0:
            final_tensor = tensor_list[i]
        else:
            final_tensor = torch.cat((final_tensor, image),0)
#     print(final_tensor.size())
    return final_tensor


# ## Training

# In[11]:


def save_model(epoch, model, optimizer, scheduler, name = 'trained_model.pt'):
    train_state = {
    'epoch': epoch,
    # 'model' : model,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict()
    }
    print("Saved model at: "+str(name))
    torch.save(train_state, 'models/'+str(name))


# In[12]:


def train_and_validate(model, train_loader, test_loader,
                       num_epochs, device = torch.device("cuda:0"),
                       learning_rate = 0.001,
                       weight_decay = 0,
                       multiGPU = False,
                       save_name = 'trained_model.pt'):
    batch_size = train_loader.batch_size
    criterion = nn.CrossEntropyLoss();
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay);
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate,
    #                                 weight_decay = weight_decay,
    #                                 momentum = 0.6);

    patience = 15 if weight_decay > 0 else 10
    step_size = 25 if weight_decay > 0 else 15

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'max', factor=0.1, patience=patience, verbose=True)
    #Training
    print("lr:{} wd:{}".format(learning_rate, weight_decay))
    model.train().to(device)
#     if isinstance(model, EnsembleClassifier):
#         if multiGPU == True:
#             print("multiGPU")
#             model.set_devices_multiGPU()

    history = {'batch': [], 'loss': [], 'accuracy': []}
    best_val_accuracy = 0
    for epoch in range(num_epochs):
        # scheduler.step()
        model.train()
        tic=timeit.default_timer()
        losses = [] #losses in epoch per batch
        accuracies_train = [] #accuracies in epoch per batch
        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images).to(device)
            labels = Variable(labels).squeeze(1).long().to(device)#.cpu()
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            _, argmax = torch.max(outputs, 1)
            accuracy_train = (labels == argmax.squeeze()).float().mean()*100
            accuracies_train.append(accuracy_train.cpu())
            # Show progress
            if (i+1) % 32 == 0:
                log = " ".join([
                  "Epoch : %d/%d" % (epoch+1, num_epochs),
                  "Iter : %d/%d" % (i+1, len(train_loader.dataset)//batch_size)])
                print('\r{}'.format(log), end=" ")

        epoch_log = " ".join([
          "Epoch : %d/%d" % (epoch+1, num_epochs),
          "Training Loss: %.4f" % np.mean(losses),
          "Training Accuracy: %.4f" % np.mean(accuracies_train)])
        print('\r{}'.format(epoch_log))

        ##VALIDATION SCORE AFTER EVERY EPOCH
        model.eval()
        correct = 0
        total = 0

        total_labels = torch.Tensor().long()
        total_predicted = torch.Tensor().long()

        for images, labels in test_loader:
            augmented_images = augment_and_transform_for_prediction(images)
            augmented_images = Variable(augmented_images).to(device)
            labels = labels.squeeze(1)
            outputs = model(augmented_images)
            
            probabilities = torch.exp(nn.LogSoftmax()(outputs))
            predicted = torch.argmax(torch.mean(probabilities, 0))

#             print(predicted)
#             print(labels)
            total += labels.size(0)
        #     print(total)
            correct += (predicted.cpu().long() == labels).sum()
            total_labels = torch.cat((total_labels,labels))
            total_predicted = torch.cat((total_predicted, predicted.cpu().long().unsqueeze(dim=0)))
            val_accuracy = 100*correct.item() / total
        print('VALIDATION SET ACCURACY: %.4f %%' % val_accuracy)
        scheduler.step(correct.item() / total)

        ###Results for analysis###
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_model(epoch, model, optimizer, scheduler, name = save_name)
            pickle.dump(total_predicted.cpu().long(), open("test_predicted.pkl", "wb"))
            pickle.dump(total_labels.long(), open("test_labels.pkl", "wb"))

        toc=timeit.default_timer()
        if epoch+1 == 70:
            for group in optimizer.param_groups:
                if 'lr' in group.keys():
                    if group['lr'] == 0.001:
                        group['lr'] == 0.0001
                        scheduler._reset()
                        print("MANUAL CHANGE OF LR")
        print(toc-tic)
    return model


# In[13]:


#Device selection and CNN
from collections import OrderedDict
import math

if torch.cuda.device_count()>1:
    device = torch.device("cuda:1") #Multi-GPU
elif torch.cuda.device_count()>0:
    device = torch.device("cuda:0") #Single-GPU
else:
    device = torch.device("cpu") #No-GPU


# In[14]:


##=============TRAIN==============##
from torchvision.models.resnet import Bottleneck
import NNs
from NNs import *
importlib.reload(NNs)

#datasets/dataloaders
train_dataset, test_dataset = create_train_val_datasets(X_train, Y_train,
                                                       test_PIL_images, test_labels)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32,
    shuffle = True, num_workers=2)

test_loader = torch.utils.data.DataLoader(test_dataset,
                            batch_size = 1, shuffle = False)

#CNN initialization and training


cnn = ResNetDynamic(Bottleneck, [3, 4, 6, 3],
            num_layers = 2, pretrained_nn = None)

trained_model = train_and_validate(cnn, train_loader, test_loader,
                                   num_epochs=30,
                                   learning_rate = 0.0001,
                                   weight_decay = 0,
                                   device = device,
                                   save_name = 'trained_model.pt')


# In[ ]:


##=============TRAIN==============##
from torchvision.models.resnet import Bottleneck
import NNs
from NNs import *
importlib.reload(NNs)

#datasets/dataloaders
train_dataset, test_dataset = create_train_val_datasets(X_train, Y_train,
                                                       test_PIL_images, test_labels)

test_loader = torch.utils.data.DataLoader(test_dataset,
                            batch_size = 1, shuffle = False)

for images, labels in test_loader:
    break


# In[ ]:


# best_model = cnn
# best_model.load_state_dict(torch.load('models/trained_model.pt')['state_dict'])

# correct = 0
# total = 0

# total_labels = torch.Tensor().long()
# total_predicted = torch.Tensor().long()

# for idx, (images, labels) in enumerate(test_loader):
#     augmented_images = augment_and_transform_for_prediction(images)
#     augmented_images = Variable(augmented_images).to(device)
#     labels = labels.squeeze(1)
#     outputs = best_model(augmented_images)
#     predicted = torch.argmax(torch.mean(outputs, 0))
#     total += labels.size(0)
# #     print(total)
#     correct += (predicted.cpu().long() == labels).sum()
# #     print(correct)
    
#     total_labels = torch.cat((total_labels,labels))
#     total_predicted = torch.cat((total_predicted, predicted.cpu().long().unsqueeze(dim = 0)))

#     val_accuracy = 100*correct.item() / total


# In[ ]:


total_predicted


# In[ ]:


# best_model = cnn
# best_model.load_state_dict(torch.load('models/trained_model.pt')['state_dict'])

# trained_model = train_and_validate(best_model, train_loader, test_loader,
#                                    num_epochs=30,
#                                    learning_rate = 0.0001,
#                                    weight_decay = 0,
#                                    device = device,
#                                    save_name = 'trained_model.pt')


# In[15]:


predicted = pickle.load(open("test_predicted.pkl", "rb"))
test_labels_real = pickle.load(open("test_labels.pkl", "rb"))


# In[17]:


results_df = pd.DataFrame.from_dict({'label' : test_labels, 'prediction' :predicted })
results_df


# In[18]:


## per class accuracy
class_df = pd.DataFrame()
for cl in results_df['label'].unique():
    unique_df = results_df.loc[results_df['label'] == cl]
    correct = len(unique_df.loc[unique_df['prediction'] == cl])
    total = len(unique_df)
    acc = correct/total
    results_dict = {'Class' : cl , 'Total' : total, 'Correct' : correct, 'Accuracy' : acc}
    print(results_dict)


# In[63]:


def create_diagnosis(x):
    if x == 0 or x == 1: ##white_blood or debris
        return 0
    else:
        return 1
        
actual_infection = results_df.label.apply(lambda s: pd.Series({'infection': create_diagnosis(s)}))
predicted_infection = results_df.prediction.apply(lambda s: pd.Series({'pred_infection': create_diagnosis(s)}))



final = pd.concat([results_df, actual_infection, predicted_infection], axis =1)


# In[79]:


final.pred_infection.value_counts()


# In[80]:


print((TN, FP, FN, TP))


# In[67]:


from sklearn.metrics import confusion_matrix, classification_report

cf = confusion_matrix(final.infection, final.pred_infection)
print(cf)


# In[78]:


TN, FP, FN, TP = cf.ravel()

sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)
precision = TP/(TP+FP)
print('precision: {}, sensitivity: {}, specificity: {}'.format(precision, sensitivity, specificity))


# In[99]:


# im = Image.open('split/train/ovale_3243.png')

# import matplotlib.pyplot as plt
# %matplotlib inline
# import scipy.misc

# new = apply_sobel(np.array(im))
# new = apply_gamma(new)

# # scipy.misc.imsave('im.png', new)
# plt.imshow(new)

