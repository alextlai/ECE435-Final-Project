# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 20:19:48 2020

@author: Alexander Lai
"""

#%% Import Libraries
from Final_Project_Data import get_process_data

import tensorflow as tf
from tensorflow import keras

#%% Import and process Data
train_file = 'train_32x32.mat'
test_file = 'test_32x32.mat'
train_X,train_Y = get_process_data(train_file,purpose='train')
test_X,test_Y = get_process_data(test_file,purpose='test')

#%% Build CNN model
model = keras.models.Sequential([
    #Original Input Shape
    #keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,3)),
    
    #Cropped Input Shape
    keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,18,3)),

    #Greyscale Shape
    #keras.layers.Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(32,32,1)),
    
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32,(3,3),activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(.3),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64,(3,3),activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Dropout(.3),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(.5),
    keras.layers.Dense(11,activation='softmax')
])
print(model.summary())

#%% Run CNN model
# sgd = keras.optimizers.SGD(learning_rate = .01, momentum = 0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(train_X,train_Y,epochs=5)
loss,acc = model.evaluate(test_X,test_Y)
#model.save('finalproject_reduce_crop_model.h5')
# model.save('final_project_test_crop.h5')
#model.save('finalproject_reduce_model.h5')s
#model.save('finalproject_model.h5')
#model.save('finalproject_greyscale_model.h5')