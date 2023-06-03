# KISD_MachineLearning_Project
CerberCNN for semantic segmentation


Three interconnected "necks" joining the body. 
A lot of connections allowing fast learning, guaranting small model size.

Cerber is learning fast. But... Cerber likes eating RAM. 

For now, with such generated training dataset, it acts like filter for inhomogeneities other than 2D gaussian "dots/ellipses".

Worse part now...

1.	Introduction

This convolutional neural network is called by me „Cerber” because of its architecture – input data feeds three partially independent (but interconnected) computational blocks, which at some point merge into common core.

Main purpose of this network was semantic segmentation of microscopic images for nuclear particle tracks detection. However, only training dataset limits its possible uses.
The output should be a probability matrix – if a pixel belongs to class „particle track” or not. In fact, with my dataset, this network acts like a filter – making „background” much more flat and uniform, with only small, circular or oval (tracks), inhomogeneities enhancing (2D gaussian shapes with SDx and SDy <5 pixels).

Data preparation

Everything is in provided code – User needs four folders with images. Two folders for training (input image, expected output image) and two for validation. Remember to edit paths – not only data paths, but also model plot path and model save path.

Images should be (as for today) equal in size and format.

First run

It is advised to run the code in debug mode until all problems are solved. User do not need any special software to run this code – ordinary Python console is enough.

2.	Input shapes

User can change input shapes in the tensorflow.keras.layers.Input layer. As for today, author was unable to attempt input resizing without crashing the code – therefore shapes are fixed. Still working on this issue.
Author advises small batch size, because of limited RAM resources.

3.	Hardware limitations

In general, this line in a code is crucial:

395   model = cerber(1,4)

„Cerber” function has two arguments: batch size and basic number of trainable parameters. The example above results in consuming 12-16 GB of RAM (90 000 trainable parameters). Also, training/validation dataset was >300 1608x1608 bitmaps.

It is advised to start with model = cerber(1,1). 

4.	Smaller codes in project

There are only two codes (not crucial to run main network):

1.	training_data_generator.py
2.	track_analyzer.py

training_data_generator provides artificially generated bitmaps. This code is using multiprocessing in the function called grinder. Please edit this function to fit Your device before running the program.
Track_analyzer is a sample code allowing use of trained and saved models. In general this program was created to make fixed executables with trained models. However, the code still needs refinement.

Please look at the paths in all provided codes – there is high risk of overlooking them.
