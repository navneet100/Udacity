

import os
import cv2

import numpy as np
import pandas as pd

import matplotlib.image as mimg
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Lambda, Dropout, Activation
from keras.optimizers import Adam

#import tensorflow as tf
#tf.python.control_flow_ops = tf

drivingData = pd.read_csv('./data/driving_log.csv')

image_path = './data/'
np.random.seed(777)


#define final image sizes before feeding for training
ch, img_rows, img_cols = 3, 16, 32
#ch, img_rows, img_cols = 3, 80, 160
#ch, img_rows, img_cols = 3, 64,64
shift = 0.17 # adding shift to left and right camera imagez


# adding brightness to cover dark areas
def augment_brightness(image):
    brightness = 0.25 + np.random.uniform()
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:,:,2] = image[:,:,2]*brightness 
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)


# ## selecting ROI
def roi(image):
    image = image[60:140, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return cv2.resize(image, (img_cols,img_rows),interpolation=cv2.INTER_AREA)

## Preprocess training data to generate more images on the fly
def preprocess_training_Data(idx):
    steering_angle = drivingData.iloc[idx].steering
    
    j = np.random.randint(3) ##Choose a ranodm number 
    
    if j == 0:
        image = mimg.imread(os.path.join(image_path, drivingData.iloc[idx].left.strip())) ##Left image
        steering_angle += shift
    elif j == 1:    
        image = mimg.imread(os.path.join(image_path, drivingData.iloc[idx].center.strip())) ##Center image
    else:
        image = mimg.imread(os.path.join(image_path, drivingData.iloc[idx].right.strip()))  ##Right image
        steering_angle -= shift
        
    
    ##convert from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    ##augment the brightness
    image = augment_brightness(image)
    
    ##Apply roi
    image = roi(image)
   
    #flip images
    flip = np.random.randint(2)
    if flip == 0:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    
    image = np.array(image)    
    
    return image, steering_angle    


##Preprocess validation data
def preprocess_validation_data(row_data):
    image = mimg.imread(os.path.join(image_path, row_data['center'].strip()))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = np.array(roi(image))
    #image = return_image(image)
    return image   


## Generator for training data
def training_data_generator(data, batch_size=64):
    batch_images = np.zeros((batch_size,img_rows, img_cols,ch), dtype = 'float32')
    batch_steering_angles = np.zeros((batch_size,1), dtype = 'float32')
    
    while 1:
        for i in range(batch_size):
            idx = np.random.randint(data.shape[0])
            img, str_angle = preprocess_training_Data(idx)  
            batch_images[i] = img
            batch_steering_angles[i] = str_angle
        
        yield batch_images, batch_steering_angles


##Validation data generator
def validation_data_generator(data):
    while 1:
        for i in range(len(data)):
            row_data = data.iloc[i]
            img = preprocess_validation_data(row_data)
            img = np.array(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]), dtype='float32')
            steer = row_data['steering']
            steer = np.array([[steer]],dtype='float32')
            yield img, steer

# randomly divide data into training and validation datasets
drivingData = drivingData.sample(frac=1).reset_index(drop=True)
train_data = training_data_generator(data = drivingData.sample(frac=0.9),batch_size=64)
val_data = validation_data_generator(data=drivingData.sample(frac=0.1))


model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(img_rows, img_cols,ch)))
#model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(img_rows, img_cols,ch)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="same", init="he_normal"))
model.add(Activation('relu'))
#model.add(Dropout(0.5)) 
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="same", init="he_normal"))
model.add(Activation('relu'))
#model.add(Dropout(0.5)) 
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="same", init="he_normal"))
model.add(Activation('relu'))
#model.add(Dropout(0.5)) 
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same", init="he_normal"))
model.add(Activation('relu'))
model.add(Dropout(0.5)) 
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="same", init="he_normal"))

model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
#model.add(Dropout(0.5)) 
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

opt = Adam(lr=0.0001)

#model.compile(loss="mse", optimizer='adam')
model.compile(loss="mse", optimizer=opt)

#model.summary()
#model.fit_generator(train_data, samples_per_epoch = 23296, validation_data= val_data, nb_val_samples= 1000,nb_epoch=5)
model.fit_generator(train_data, samples_per_epoch = 20096, validation_data= val_data, nb_val_samples= 1000,nb_epoch=6)


#save model to use with live driving data
model_json = model.to_json()
model_name = 'model_017'
with open(model_name+'.json', "w") as json_file:
    json_file.write(model_json)

model.save_weights(model_name+'.h5')
