# ECE435-Final-Alexander Lai
This repository contains the files I developed for the ECE435 Final Project. 
For the final project I developed an digit detection/classification algorithm for the SVHN dataset. This repository is a supplement for the jupyter notebook submission. For the juptyer notebook to work the following files in this repository are needed:
1. '__Final_Project_Data.py__' - python file containing functions to process data in test 
2. '__Final_Project_Supression.py__' - python file containing functions for non maximum supression
3. '__finalproject_model.h5__' - the model I trained 
4. '__Test2__' - folder containing google street view images that do not contain digits (for training)

The following two folders must be obtained from the SVHN website because they are too big. The website is as follows:
http://ufldl.stanford.edu/housenumbers/
5. '__train_32x32.mat__' and '__test_32x32.mat__' - SVHN 32x32 images for training and testing. The SVHN file 'extra_32x32.mat' was not used in this code
6. '__Test__' - folder containing SVHN full images for digit detection (not training). Download 'test.tar.gz' from the SVHN website and place images in this folder.

Beyond these files, I have also included two files: '__Final_Project_Training2.py__' and '__Final_Project_Detection2.py__' which are pretty much the main files that do everything the jupyter notebook does. They also require all of the files above to work.
The youtube link describing this project can be found here: 
