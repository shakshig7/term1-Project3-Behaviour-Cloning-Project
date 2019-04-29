
# Behaviorial Cloning Project



[cropped]: ./preprocessed/cropped.png "Cropped image"
[resize]: ./preprocessed/resize.png "Resize image"
[before_flip]: ./preprocessed/before_flip.png "Original image"
[after_flip]: ./preprocessed/after_flip.png "Flipped image"
[left]: ./preprocessed/left.jpg "Left image"
[center]: ./preprocessed/center.jpg "Center image"
[right]: ./preprocessed/right.jpg "Right image"



### The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Files Submitted & Code Quality
**1. Submission includes all required files and can be used to run the simulator in autonomous mode**

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

**2. Submission includes functional code**

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

`python drive.py model.h5`

**3. Submission code is usable and readable**

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy
**1. An appropriate model architecture has been employed**
My model consists of a convolution neural network with 5x5 filter sizes and depths between 6 and 12 (model.py lines 94-100)

The model includes RELU layers to introduce nonlinearity (code line 95 and 99), and the data is normalized in the model using a Keras lambda layer (code line 92).
A model summary is as follows:


```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 80, 320, 3)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 32, 32, 3)     0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 28, 28, 6)     456         lambda_1[0][0]                   
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 14, 14, 6)     0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 10, 10, 12)    1812        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 5, 5, 12)      0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 300)           0           maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           30100       flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 100)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        activation_1[0][0]               
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 50)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             51          activation_2[0][0]               
====================================================================================================
Total params: 37,469
Trainable params: 37,469
Non-trainable params: 0
____________________________________________________________________________________________________`


```

**2. Attempts to reduce overfitting in the model**

I have used **MAX POOLING** as the regularization technique (model.py lines 96 and 100).Also, I have kept the training epochs low: only five epochs. In addition to that, I split my sample data into training and validation data. Using 80% as training and 20% as validation (model.py line 33).

**3. Model parameter tuning**

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 117).

**4. Appropriate training data**
Training data was chosen to keep the vehicle driving on the road. I have used the data provided by Udacity. The simulator provides three different images: center, left and right cameras. Each image was used to train the model. Also, I have added some recovery data. This means that data should be captured starting from the point of approaching the edge of the track (perhaps nearly missing a turn and almost driving off the track) and recording the process of steering the car back toward the center of the track to give the model a chance to learn recovery behavior.

### Model Architecture and Training Strategy
**1. Solution Design Approach**

My first step was to use a convolution neural network model with 2 convolution layer and 3 fully connected layers.
After training suing the images from udacity dataset, in simulator car went straight into the lake.
I added a cropping layer in the start of the model so that only the section of the image containing road is used.
Image after cropping -
![alt text][cropped]

I needed to do some data preprocessing techniques.So , I added a Lambda layer to call the prerocesssing function (model.py line 92).The preprocessing function has code to resize and normalize the input images (model.py lines 26-30). 
![alt text][resize]
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.  
To avoid the model from overfitting, I added max pooling layer after each convolutional layer.
This time, the car was moving out of the road after crossing the layer.So, I augmented the data by adding the same image flipped (lines 52 - 64). In addition to that, the left and right camera images where introduced with a correction factor on the angle to help the car go back to the lane(lines 50 - 63). 

The final step was to run the simulator to see how well the car was driving around track one.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

**2. Final Model Architecture**

The final model architecture (model.py lines 86-114) consisted of a convolution neural network with the following layers and layer sizes.

The input is 66x200xC with C = 3 RGB color channels.
###Architecture 

Layer 1: **Cropping** : Cropping image by dimension `([60,20],[0,0])`.

Layer 2: **Lambda** : Resizing image to 32x32 and normalization to range `(-0.5, 0.5)`.

Layer 3: **Convolutional** : Kernel 5x5 with filter depth 6 and output shape is 28, 28, 6 with relu activation function.

Layer 4: **Max pooling** : Output shape is 14,14,6.

Layer 5: **Convolutional** : Kernel 5x5 with filter depth 12 and output shape is 10, 10, 12 with relu activation function.

Layer 6: **Max pooling** : Output shape is 5, 5, 12.

**Flatten** with 300 output

Layer 7 : **Fully connected** : Output is 100 with relu activation function.

Layer 8: **Fully connected** : Output is 50 with relu activation function.

Final Layer : **Fully connected** : 1 output for the steering angle.

**3. Creation of the Training Set & Training Process**

I have used the Udacity Training Data and added some recovery data from the first track.I have used left and right camera images as well.

Left Image |  Right Image
:---------   |:--------
![alt text][left] | ![alt text][right]


In addition to that,I have flipped the images to get additional images.

Original Image |  Flipped Image
:---------   |:--------
![alt text][before_flip]   |  ![alt text][after_flip]


I finally randomly shuffled the data set and put 0.2% of the data into a validation set.
The validation set helped determine if the model was over or under fitting.I have used 5 epochs and an adam optimizer so that manually training the learning rate wasn't necessary.
After this training, the car was driving down the road all the time on the first track.
