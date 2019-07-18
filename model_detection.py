#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Reshape
from keras.layers import AveragePooling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers import *
import keras
import numpy as np
import cv2
from matplotlib import pyplot as plt
from generator_detection import *
import tensorflow as tf

def return_grid(y_pred,grid_shape):
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])      #grid shape=(13, 13, 1, 2)
    grid = K.cast(grid, K.dtype(y_pred))
    return grid
def return_grid_np(grid_shape=(13,13)):
    grid_y = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
        [1, grid_shape[1], 1, 1])
    grid_x = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
        [grid_shape[0], 1, 1, 1])
    grid = np.concatenate([grid_x, grid_y],axis=-1)      #grid shape=(13, 13, 1, 2)
    grid=grid.astype("float32")
    return grid


def loop_body(b, ignore_mask):
    true_box = tf.boolean_mask(y_true[b,...,0:4], object_mask_bool[b,...,0])
    iou = box_iou(pred_box[b], true_box)
    best_iou = K.max(iou, axis=-1)
    ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
    return b+1, ignore_mask


"""
ignore_mask = tf.TensorArray(K.dtype(y_true), size=1, dynamic_size=True)
object_mask_bool = K.cast(object_mask, 'bool')
_, ignore_mask = K.control_flow_ops.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])
ignore_mask = ignore_mask.stack()
ignore_mask = K.expand_dims(ignore_mask, -1)
"""






def custom_loss(Layer):

    # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
    def loss(y_true,y_pred):
	y_true = K.reshape(y_true,(-1,13,13,3,6))
	y_pred = K.reshape(y_pred,(-1,13,13,3,6))
	anchors=np.array([[10,13],[16,30],[33,23]])  
	anchors=np.reshape(anchors,(1,1,1,3,2))
	anchors=K.constant(anchors,dtype="float32")


        object_mask = y_true[:,:,:,:,4:5]
	face_mask = y_true[:,:,:,:,5:]
		


	grid_shape = K.shape(y_pred)[1:3] # height, width   (13,13)
	grid=return_grid(y_pred,grid_shape)
	#box_xy = (K.sigmoid(y_pred[..., :2]) + grid)/ K.cast(grid_shape[::-1],K.dtype(y_pred))

        raw_true_xy = y_true[..., :2]*[13.0,13.0] - grid
        raw_true_wh = K.log((y_true[..., 2:4]+K.epsilon()) * [416,416]/anchors)
        raw_true_wh = K.switch(object_mask, raw_true_wh, K.zeros_like(raw_true_wh)) # avoid log(0)=-inf


        #raw_true_wh = y_true[..., 2:4]
        box_loss_scale = 2 - y_true[...,2:3]*y_true[...,3:4]

 	xy_loss = object_mask * box_loss_scale *K.square(raw_true_xy- y_pred[...,0:2])
        wh_loss = object_mask * box_loss_scale * 0.5 * K.square(raw_true_wh-y_pred[...,2:4])


	class_loss=object_mask * K.binary_crossentropy(object_mask, (K.sigmoid(y_pred[...,4:5])))+ \
			  (1-object_mask) * K.binary_crossentropy(object_mask, (K.sigmoid(y_pred[...,4:5])))
 					
	face_loss=object_mask * K.binary_crossentropy(object_mask, K.sigmoid(y_pred[...,5:]))+ \
			  (1-object_mask) * K.binary_crossentropy(object_mask,K.sigmoid(y_pred[...,5:]))


	#loss1=5*K.mean(K.square(y_true[...,2:]-y_pred[...,2:]))
		
	#loss1=-(object_mask*K.log(class_prob)+(1-object_mask)*K.log(1-class_prob))
	#loss2=-(face_mask*K.log(face_prob)+(1-face_mask)*K.log(1-face_prob))

        s=K.sum(object_mask)
        #loss=(K.sum(xy_loss)/s)+(K.sum(wh_loss)/s)+ K.sum(class_loss)+K.sum(face_loss)
	#loss=K.mean(K.square(y_true-y_pred)) +xy_loss
	loss=10*xy_loss+wh_loss+class_loss+face_loss
	#loss=xy_loss+wh_loss+class_loss+face_loss
        #loss = tf.Print(loss, [loss, K.sum(xy_loss)/s, K.sum(wh_loss)/s, K.sum(class_loss)/s], message='\n loss: ')
        #loss=tf.Print(loss, [K.sum(xy_loss)/s, K.sum(wh_loss)/s, K.sum(class_loss)/s,K.sum(face_loss)/s], message='\n loss: ')

	return loss
    # Return a function
    return loss


def mul_layers(I,n_feature=8,kernel_size=(4,4),num_layers=2,act="tanh"):

    x=Conv2D(n_feature,kernel_size,activation=act,padding="same")(I)
    x=MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    for i in range(num_layers-1):
	x=Conv2D(n_feature,kernel_size,activation=act,padding="same")(x)
	x=MaxPooling2D(pool_size=(2,2), strides=2)(x)
   
    return x 



def intern_model():
    I=Input(shape=(416, 416, 3))
    x=mul_layers(I,n_feature=32,kernel_size=(4,4),num_layers=2,act="relu")
    x=mul_layers(x,n_feature=64,kernel_size=(6,6),num_layers=2,act="relu")
    x=mul_layers(x,n_feature=32,kernel_size=(6,6),num_layers=1,act="relu")
    x=Conv2D(28,(2,2),activation="relu",padding="same")(x)
    x=Conv2D(28,(2,2),activation="relu",padding="same")(x)
    x=Conv2D(18,(2,2),activation="relu",padding="same")(x)
    x=Conv2D(18,(2,2),activation="linear",padding="same")(x)


    model=Model(inputs=[I], outputs=[x])
    model.compile(loss=custom_loss(Layer),optimizer="rmsprop" 	,metrics=['accuracy'])
    #model.compile(loss='mse',optimizer="rmsprop",metrics=['accuracy'])
    model.summary()
    return model
    
#model=intern_model() 


