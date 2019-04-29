#BEHAVIOUR CLONING PROJECT

import keras
import csv
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,Model
from keras.layers import Flatten,Dense,Lambda,Cropping2D,Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2


#Loading Data using csv
samples = []

with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)


#Data preprocessing function - RESIZING to 32x32 AND NORMALIZING 
def data_preprocess(images):
    import tensorflow as tf
    images = tf.image.resize_images(images,(32,32))
    image_normalize = images /255.0 - 0.5
    return image_normalize

#Splitting data for training and validation images
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

#Generator Function
def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            steering = []
            for batch_sample in batch_samples:
                # ADDING CENTER CAMERA IMAGES TO TRAINING DATA
                image_path = batch_sample[0].strip()
                center_image = plt.imread(image_path)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                steering.append(center_angle)

                #ADDING LEFT CAMERA IMAGES TO TRAINING DATA
                left_image_path = batch_sample[1].strip()
                left_image = plt.imread(left_image_path)
                left_angle =center_angle + 0.4
                images.append(left_image)
                steering.append(left_angle)

                #ADDING RIGHT CAMERA IMAGES TO TRAINING DATA
                right_image_path = batch_sample[1].strip()
                right_image = plt.imread(right_image_path)
                right_angle =center_angle - 0.2
                images.append(right_image)
                steering.append(right_angle)

                #FLIPPING ALL THE IMAGES TO GET MORE DATA AND ADDING TO TRAINING DATA
                img_list = [center_image, left_image, right_image]
                str_list = [center_angle, left_angle, right_angle]
                for i in range(3):
                    flipped_image = cv2.flip(img_list[i],1)
                    flipped_angle = -1*str_list[i]
                    images.append(flipped_image)
                    steering.append(flipped_angle)


            
            X_train = np.array(images)
            y_train = np.array(steering)
            yield (shuffle(X_train, y_train))

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=64)

#Model ARCHITECTURE OF 2 CONVOLUTIONAL LAYER AND 3 DENSE LAYER
model = Sequential()

#Cropping layer to crop image to see only the road section
model.add(Cropping2D(cropping=((60,20),(0,0)),input_shape=(160,320,3)))

#Lambda function for data preprocessing
model.add(Lambda(data_preprocess))

#1st convolutional layer using 5x5 filter and relu activation function
model.add(Convolution2D(6,5,5,activation='relu'))
model.add(MaxPooling2D())

#2nd convolutional layer using 5x5 filter and relu acrivation function
model.add(Convolution2D(12,5,5,activation='relu'))
model.add(MaxPooling2D())

#Flatten layer to get single output
model.add(Flatten())

#1st dense layer to get output as 100
model.add(Dense(100))
model.add(Activation('relu'))

#2nd dense layer to get output as 50
model.add(Dense(50))
model.add(Activation('relu'))

#3rd dense layer to get single output of steering angle
model.add(Dense(1))

#compiling the model
model.compile(loss='mse',optimizer='adam')
history_object = model.fit_generator(train_generator, samples_per_epoch= 6*len(train_samples), validation_data=validation_generator,nb_val_samples=len(validation_samples), nb_epoch=5,verbose =1)
model.save('model.h5')








    



