import os
import cv2
#import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import matplotlib.image as mimg
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Flatten, Lambda, Dropout, Activation
from keras.optimizers import Adam

import tensorflow as tf
tf.python.control_flow_ops = tf

ch, final_img_rows, final_img_cols = 3, 16 , 32
image_path = './data/'    
    
def add_random_shadow(image):
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160 
    bot_y = 320 * np.random.uniform()
    image_hls = cv2.cvtColor(image,cv2.COLOR_RGB2HLS)
    shadow_mask = 0*image_hls[:,:,1]
    X_m = np.mgrid[0:image.shape[0],0:image.shape[1]][0]
    Y_m = np.mgrid[0:image.shape[0],0:image.shape[1]][1]
    shadow_mask[((X_m-top_x)*(bot_y-top_y) -(bot_x - top_x)*(Y_m-top_y) >=0)]=1

    if np.random.randint(2)==1:
        random_bright = .5
        cond1 = shadow_mask==1
        cond0 = shadow_mask==0
        if np.random.randint(2)==1:
            image_hls[:,:,1][cond1] = image_hls[:,:,1][cond1]*random_bright
        else:
            image_hls[:,:,1][cond0] = image_hls[:,:,1][cond0]*random_bright    
    image = cv2.cvtColor(image_hls,cv2.COLOR_HLS2RGB)
    return image  
    
def augment_brightness_camera_images(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = 0.25 + np.random.uniform()
    image[:, :, 2] = image[:, :, 2] * random_bright
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image
    
def return_image(img, color_change=True):
    # Take out the dash and horizon
    #img_shape = img.shape
    #img = img[60:img_shape[0] - 25, 0:img_shape[1]]
    img = img[60:140, :]
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = (cv2.resize(img, (final_img_cols,final_img_rows), interpolation=cv2.INTER_AREA))
    return np.float32(img)
    
def trans_image(image, steer, trans_range):
    # Translation
    rows, cols, channels = image.shape
    tr_x = trans_range * np.random.uniform() - trans_range / 2
    steer_ang = steer + tr_x / trans_range * 2 * .2
    tr_y = 10 * np.random.uniform() - 10 / 2
    # tr_y = 0
    Trans_M = np.float32([[1, 0, tr_x], [0, 1, tr_y]])
    image_tr = cv2.warpAffine(image, Trans_M, (cols, rows))
    return image_tr, steer_ang
    
    
def augment_images(x, y):
    steering = y
    #image = load_img("{0}".format(x))
    #image = img_to_array(image)
    # Augment brightness so it can handle both night and day
    image = x
    image = add_random_shadow(image)
    image = augment_brightness_camera_images(image)
    trans = np.random.random()
    if trans < .2:
        # 20% of the time, return the original image
        return return_image(image), steering
    trans = np.random.random()
    if trans > .3:
        # Flip the image around center 70% of the time
        steering *= -1
        image = cv2.flip(image, 1)
    trans = np.random.random()
    if trans > .5:
        # Translate 50% of the images
        image, steering = trans_image(image, steering, 150)
    trans = np.random.random()
    if trans > .8:
        # 20% of the time, add a little jitter to the steering to help with 0 steering angles
        steering += np.random.uniform(-1, 1) / 60
    image = return_image(image)
    return image, steering

## Define the steps here for how the pipeline is going to be exceuted
def preprocess_trainData(X_train_data, y_train_data):
    
    shift = 0.17
    
    idx = np.random.randint(X_train_data.shape[0])
        
    steering_angle = y_train_data.iloc[idx]
    
    j = np.random.randint(3) ##Choose a ranodm number 
    


    if j == 0:
        imgPath = X_train_data['left'].iloc[idx]
        imgPath = imgPath.strip()
        imgPath = os.path.join(image_path, imgPath)
        image = mimg.imread(imgPath)  ##Left image
        steering_angle += shift
    elif j == 1:    
        imgPath = X_train_data['center'].iloc[idx]
        imgPath = imgPath.strip()
        imgPath = os.path.join(image_path, imgPath)
        image = mimg.imread(imgPath)  ####Center image
    else:
        #image = mimg.imread((os.path.join(image_path, (X_train_data['right'][idx])).strip()).replace(" ",""))  ##Right image
        imgPath = X_train_data['right'].iloc[idx]
        #print(imgPath)
        imgPath = imgPath.strip()
        #print(imgPath)
        imgPath = os.path.join(image_path, imgPath)
        #print(imgPath)
        image = mimg.imread(imgPath)  ##Right image
        steering_angle -= shift
        
    
    ##Chaneg from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = np.array(image)    
    image, steering_angle = augment_images(image, steering_angle)
    
    
    return image, steering_angle    


##Preprocessing for validation data
def preprocess_valid_data(row_data):
    image = mimg.imread(os.path.join(image_path, row_data['center'].strip()))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = return_image(image)
    return image   


## Define custom trainImageDataGenerator
def training_data_generator(X_train_data,y_train_data, batch_size=64):
    batch_images = np.zeros((batch_size,final_img_rows, final_img_cols,ch), dtype = 'float32')
    batch_steering_angles = np.zeros((batch_size,1), dtype = 'float32')
    
    while 1:
        for i in range(batch_size):            
            img, str_angle = preprocess_trainData(X_train_data,y_train_data)  
            batch_images[i] = img
            batch_steering_angles[i] = str_angle
        
        yield batch_images, batch_steering_angles


##Define custom validImageDataGenerator
def validation_data_generator(X_val_data,y_val_data):
    while 1:
        for i in range(len(X_val_data)):
            row_data = X_val_data.iloc[i]
            img = preprocess_valid_data(row_data)
            img = np.array(img.reshape(1, img.shape[0], img.shape[1], img.shape[2]), dtype='float32')
            steer = y_val_data.iloc[i]
            steer = np.array([[steer]],dtype='float32')
            yield img, steer



def readData(filename):
    driving_log = pd.read_csv(filename) 

    ##Variables to be used in processing pipeline    
    X_data = driving_log[['center','left','right']]
    y_data = driving_log['steering']
    return X_data, y_data

def splitAndGenerateData(X_data, y_data):
    X_train_data, X_val_data,y_train_data, y_val_data = train_test_split(X_data, y_data, test_size=0.1, random_state=123)
    
    train_data = training_data_generator(X_train_data,y_train_data, batch_size=128)
    
    val_data = validation_data_generator(X_val_data,y_val_data)
    return train_data, val_data


def buildModel(train_data, val_data):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(final_img_rows, final_img_cols,ch)))
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
    
    #default lr=0.001
    opt = Adam(lr=0.0001)
    model.compile(loss="mse", optimizer=opt)
    
    
    model.fit_generator(train_data, samples_per_epoch = 20096, validation_data= val_data, nb_val_samples= 1000,nb_epoch=5)
    #model.fit_generator(train_data, samples_per_epoch = 20224, validation_data= val_data, nb_val_samples= 1000,nb_epoch=5)
    return model

def saveModelAndWeights(model):
    model_json = model.to_json()
    model_name = 'model_029a'
    with open(model_name+'.json', "w") as json_file:
        json_file.write(model_json)
    
    model.save_weights(model_name+'.h5')


if __name__ == '__main__':
    np.random.seed(123)
    
    filename = './data/driving_log.csv'
    
    X_data, y_data = readData(filename)
    
    train_data, val_data = splitAndGenerateData(X_data, y_data)
    
    model = buildModel(train_data, val_data)
    saveModelAndWeights(model)