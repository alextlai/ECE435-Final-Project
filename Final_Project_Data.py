# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 19:29:40 2020

@author: Alexander Lai
"""
from scipy.io import loadmat
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.utils import shuffle
from PIL import Image
import tensorflow as tf
from tensorflow import keras

#%% Function to create new dataset for photos with no digit
def get_noclass_data(purpose):
    image_data = []
    
    #Iterate through folder to grab images
    for filename in glob.glob('train2/*.jpg'):
        whole_image = Image.open(filename)
        data = np.asarray(whole_image)
        height,width,channel = np.shape(data)
        
        #Iterate through images to segregate into 32x32 images
        for i in range(int(height/32)):
            for j in range(int(width/32)):
                y_pixel= i*32
                x_pixel= j*32
                image_cropped=whole_image.crop((x_pixel,y_pixel,x_pixel+32,y_pixel+32))
                image_crp_data = np.asarray(image_cropped)
                image_data.append(image_crp_data)
    
    #Shuffle Images
    image_data = shuffle(image_data)

    #Store Values from images in tuple 
    number_test = 20000
    xtrain = image_data[0:number_test]
    ytrain = [[10] for i in range(len(xtrain))]
    train2 = [xtrain,ytrain]
    
    xtest = image_data[(number_test+1):len(image_data)]
    ytest = [[10] for i in range(len(xtest))]
    test2 = [xtest,ytest]
    
    #Display images
    if purpose == 'train':
        fig1 = plt.figure(figsize=(7,3))
        gs = fig1.add_gridspec(4,8)
        ax1 = fig1.add_subplot(gs[:,4:8])
        ax1.imshow(data)
        plt.axis('off')
        
        for i in range(4):
            for j in range(4):
                ax2 = fig1.add_subplot(gs[i,j])
                ax2.imshow(image_data[0+i*4+j])
                plt.axis('off')
        return train2
    if purpose == 'test':
        return test2
    
#%% Function to Import SVHN data for Model
def get_process_data(filename1,purpose):
    #Function to get data from .mat svhn files. Takes the file name as and argument and returns
    #processed x and y data in the form of normalized images and values 0-9 
    #respectively
    data = loadmat(filename1)
    data_X, data_Y = data['X'],data['y'] 
    data_noclass = get_noclass_data(purpose)
    data_X2,data_Y2 = data_noclass[0],data_noclass[1]

    #Transpose data_X so index of whole image is first and combine with data_X2
    data_X = np.transpose(data_X, (3,0,1,2))
    data_X = np.concatenate((data_X,data_X2),axis=0)
    
    #Change the values of 10 to 0 in data_Y and combine with data_y2
    for i in range(len(data_Y[:,0])):
        if data_Y[i,0] == 10:
            data_Y[i,0] = 0
    data_Y = np.concatenate((data_Y,data_Y2),axis=0)
    
    #Convert data_Y to categorical data
    data_Y = keras.utils.to_categorical(data_Y,11)

    #Shuffle Data
    data_X, data_Y = shuffle(data_X, data_Y)
    
    #Convert image to greyscale (NOT USED)
    #data_X = np.dot(data_X,[0.2999,0.587,0.1440])
    #data_X = np.expand_dims(data_X, axis=3) 
    #print(np.shape(data_X))      
    #plt.imshow(greyscale_x[0], cmap=plt.cm.binary)
    
    #Standardize image across rgb channels (NOT USED)
    #mean_X = np.mean(data_X,axis=tuple([0,1,2]))
    #std_X = np.std(data_X,axis=tuple([0,1,2]))
    #data_X = (data_X - mean_X)/std_X
    
    #Normalize image to values between 0 and 1
    data_X = data_X/255
    
    #Crop data set to be 18x32
    # data_X = data_X[:,:,7:25,:]
    
    #Display Data
    if purpose == 'train':
        fig2 = plt.figure(figsize=(7,2),constrained_layout=True)
        gs2 = fig2.add_gridspec(2,6)
        for i in range(2):
            for j in range(6):
                ax3 = fig2.add_subplot(gs2[i,j])
                ax3.imshow(data_X[0+i*6+j])
                plt.axis('off')
                ax3.set_title('%d' %np.argmax(data_Y[0+i*6+j]))
            
    return data_X, data_Y
