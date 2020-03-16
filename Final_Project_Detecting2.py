# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:56:47 2020

@author: Alexander Lai
"""
#%% Prepare data from dataset
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from Final_Project_Supression import non_max_supression

import tensorflow as tf
from tensorflow import keras

#%% Sliding Window Operation
#import images
#image_numbers = [1,9,11,66,540,383,596,2018(GOOD),2115,3920(GOOD),3957(OKAY),3893,3916,3921,2794,2501,1431,
# 3691(GOOD),2329]
#Good_bad_ex = [95,338,760,1221,305,1401,2250,115,3329,124,171(increase probability thresh),
# 74]

#choices = [3957,2018,3920,3921]
def sliding_window(image_file,model,prob_thresh):
    #Function performs slidding window task, inputs image_num,model file, and threshold for
    #probability. It ouputs box dimensions that have probability above threshold,
    #the associated class/probabilities, and the array of the 200 pixel high scaled image
    image = Image.open(image_file)
    image_data = np.asarray(image)
    
    #reset image height to be 200
    new_height = 200 
    image_shape = np.shape(image_data)
    org_scale = new_height/image_shape[0]
    new_width = int(image_shape[1]*org_scale)
    image_resized = image.resize((new_width,new_height))
    image_firststep = image_resized
    
    #sliding window settings
    pyramid_height = 3  #Specify how many times photo is reduced
    # win_size = [32,32]
    win_size = [18,32]  #Size of sliding window
    input_size = [32,32]    #Network input size
    win_stride = 10     #Stride of sliding window
    scale = 1   #Intialize first layer to have scale 1
    reduce = 2      #Scaling factor (2 indicates image is scaled by half each layer)
    color = ['r','g','b']   #Colors of boxes for pyramid height
    
    #plot initial images
    fig,ax = plt.subplots(1)
    ax.imshow(image_resized)
    
    #Set initial array for boundary box locations and final class
    boxes,classes,predicts = [],[],[]
    
    #Import model for prediction
    model = keras.models.load_model(model)
    #model = keras.models.load_model('finalproject_model.h5')
    # model = keras.models.load_model('finalproject_reduce_model.h5')
    #model = keras.models.load_model('finalproject_reduce_crop_model.h5')
    #model = keras.models.load_model('finalproject_reduce_sgd.h5') #BEST SO FAR
    #model = keras.models.load_model('final_project_test_crop.h5') 
    #model = keras.models.load_model('finalproject_greyscale_model.h5')
    
    #Perform sliding window task
    for i in range(pyramid_height):
        x_coord = 0     #Initialize x and y coordinates
        y_coord = 0
        
        #Loop through image to slide window
        while (x_coord + win_size[0]) <= new_width:
            while (y_coord + win_size[1]) <= new_height:
                image_cropped=image_resized.crop((x_coord,y_coord,x_coord+win_size[0],
                                                  y_coord+win_size[1])) #crop image according to window size
                #make image 32x32
                input_feed= Image.new('RGB',(input_size[0],input_size[1]),(0,0,0)) 
                input_feed.paste(image_cropped, (int((input_size[0]-win_size[0])/2),
                                                              int((input_size[1]-win_size[1])/2))) 
                image_crp_data = np.asarray(input_feed) #convert image to array
                # image_crp_data = np.asarray(image_cropped)
                image_crp_data = image_crp_data/255     #normalize image
                image_crp_data = np.expand_dims(image_crp_data, axis=0)     
                #greyscale_crop= np.dot(image_crp_data,[0.2999,0.587,0.1440])
                #greyscale_crop = np.expand_dims(greyscale_crop, axis=3)
                predict= model.predict(image_crp_data)      #feed cropped image into CNN
                #predict = model.predict(greyscale_crop)    #model for greyscale
                rect_coord = [int(x_coord*(1/scale)),int(y_coord*(1/scale))]    #store coordinate of window
                
                if np.max(predict[0,0:9]) > prob_thresh: #plot and store box if probability is high enough
                    box_coord =np.array([int(x_coord*(1/scale)),int(y_coord*(1/scale)),
                                 int((x_coord+win_size[0])*(1/scale)), 
                                 int((y_coord+win_size[1])*(1/scale))])
                    boxes.append(box_coord) #Store box coordinates
                    predicts.append(np.max(predict))    #Store probabilities
                    classes.append(np.argmax(predict[0,0:9]))   #store class labels
                    
                    #plot boxes above probability threshold
                    rect = patches.Rectangle((rect_coord[0],rect_coord[1]),
                                             int(win_size[0]*(1/scale)),
                                             int(win_size[1]*(1/scale)),linewidth=1,
                                             edgecolor=color[i],facecolor='none')
                    ax.text(box_coord[0]+(box_coord[2]-box_coord[0])/2,
                             box_coord[1]-3,"%.001f" % np.max(predict), size=12, 
                             color = color[i])
                    ax.add_patch(rect)
                y_coord += win_stride   
            y_coord = 0
            x_coord += win_stride
        scale = scale/2 #update scale
        #Scale image for pyramid
        new_width = int(new_width/reduce)   
        new_height = int(new_height/reduce)
        image_resized = image.resize((new_width,new_height))
    
    return image_firststep,boxes,classes,predicts
image_firststep,boxes,classes,predicts = sliding_window('test/3921.png','finalproject_model.h5',.7)
image_firststep1,boxes1,classes1,predicts1 = sliding_window('test/124.png','finalproject_model.h5',.7)
image_firststep2,boxes2,classes2,predicts2 = sliding_window('test/74.png','finalproject_model.h5',.7)
 #%% Non-maximum supression
non_max_supression(image_firststep,boxes,classes,predicts,threshold = 0.01)
non_max_supression(image_firststep1,boxes1,classes1,predicts1,threshold = 0.01)
non_max_supression(image_firststep2,boxes2,classes2,predicts2,threshold = 0.01)

