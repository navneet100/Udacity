import cv2
import h5py
import rospy
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from styx_msgs.msg import TrafficLight

import keras
from keras.models import Model
from keras.preprocessing.image import img_to_array
from keras.layers import Input, Activation, Concatenate
from keras.layers import Flatten, Dropout
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers import AveragePooling2D

from keras.models import model_from_json
from keras.optimizers import Adam,Adagrad, Nadam

import tensorflow as tf
import json

#import keras.backend as K

K.set_image_dim_ordering('tf')



class TLClassifier(object):
    def __init__(self):
        #self.model = SqueezeNet()
        #self.model.load_weights("light_classification/weights/traffic_model-255-val_acc-0.67.hdf5")
	self.i = 0
	model_path = 'light_classification/weights/latest_simulator_data_model_2.json'
	weights_path = 'light_classification/weights/model_2_weights.hdf5'


	with open(model_path, 'r') as f:
		model_json = f.read()

        adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        #adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
        #nadam = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004)


        self.model = model_from_json(model_json)
        self.model.load_weights(weights_path)

	self.graph = tf.get_default_graph()
        rospy.logdebug("Model loaded.")

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #d = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	#d = image

        image = img_to_array(image)
        image /= 255.0
        image = np.expand_dims(image, axis=0)

	with self.graph.as_default():
		preds = self.model.predict(image)[0]
	#print(preds)
	#max_index = np.argmax(preds)

	state = TrafficLight.UNKNOWN

	if preds[0] > 0.7:
		print("Traffic Light - Green")
		state = TrafficLight.GREEN
	elif preds[1] > 0.7:
		print("Traffic Light - Yellow")
		state = TrafficLight.YELLOW
	elif preds[2] > 0.7:
		print("Traffic Light - Red")
		state = TrafficLight.RED
	else:
		print("Traffic Light - None")
		state = TrafficLight.UNKNOWN

        #cv2.putText(d, strn, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        #cv2.imshow("Camera stream", d)
	#cv2.waitKey(1)
	#self.i += 1
	#o_img_path = 'light_classification/o_images/my_images_' + str(self.i) + '.jpg'
	#cv2.imwrite(o_img_path,d)
        #cv2.waitKey(1)


	#print(strn + str(self.i))

	#cv2.imshow("Camera stream", d)


        return state
