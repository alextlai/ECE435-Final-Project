# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 14:57:19 2020

@author: Alexander Lai
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
#%% Function to perform non-maximum supression
def non_max_supression(first_image,boxes,classes,predicts,threshold):
    #This function takes in first_image (in order to plot the optimal box on the original image)
    #The boxes found by the moving window, and the classes/probabilities associated with each those boxes.
    #Threshold is the overlap threshold for supression.
    #The function then performs non-maximumm supression, plots the optimal boxes
    #and outputs the coordinates of the optimal box
    
    #Plot the original image
    fig2,ax2 = plt.subplots(1,figsize = (6,2))
    ax2.imshow(first_image)
    
    #Make boxes and classes into a numpy array
    boxes = np.array(boxes)
    classes = np.array(classes)
    
    opt_box = [] #initalize matrix for the optimal boxes
    area = (boxes[:,2]-boxes[:,0]+1)*(boxes[:,3]-boxes[:,1]+1) #find areas 
    indexes = np.argsort(predicts) #Sort boxes according to lowest to highest confidence
    
    #Non maximum supression by eliminating boxes by IOU
    while len(indexes) > 0:
        final_index = len(indexes)-1 #grab last index
        i = indexes[final_index] #grab value of the last index
        opt_box.append(i) #save box associated with highest confidence 
        suppress = [final_index] #store the index associated with this box for supression
        
        #Iterate through all the boxes to calculate IOU
        for y2_pos in range(final_index):
            j = indexes[y2_pos] 
            #Calculate dimensions of overlapping boxes
            inner_x1,inner_y1 = max(boxes[i,0], boxes[j,0]), max(boxes[i,1], boxes[j,1])
            inner_x2,inner_y2 = min(boxes[i,2], boxes[j,2]), min(boxes[i,3], boxes[j,3])
            width = max(0, inner_x2 - inner_x1 + 1)
            height = max(0, inner_y2 - inner_y1 + 1)
            #Calculate IOU and see if it goes surpasses threshold
            overlap = float(width * height) / (area[j]+area[i]-(width*height))
            if overlap > threshold:
                suppress.append(y2_pos) #if surpassed store for removal
        indexes = np.delete(indexes, suppress) #remove indexes 
        
    # Draw Optimal Boxes
    for m in range(len(opt_box)):
        rect2 = patches.Rectangle((boxes[opt_box[m],0],boxes[opt_box[m],1]),
                                 (boxes[opt_box[m],2]-boxes[opt_box[m],0]),
                                 (boxes[opt_box[m],3]-boxes[opt_box[m],1]),
                                 edgecolor='r',facecolor='none')
        ax2.text(boxes[opt_box[m],0]+(boxes[opt_box[m],2]-boxes[opt_box[m],0])/2,
                 boxes[opt_box[m],1]-3,"%d" % classes[opt_box[m]], size=12, 
                 color = (1,0,0))
        ax2.add_patch(rect2)  
    
    return opt_box 
