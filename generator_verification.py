from keras.utils import Sequence
import numpy as np
import glob
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt

DATA_PATH="/home/ecsuiplab/FACE_VERIFICATION/DATA/VERIFICATION/olivetti_faces/"
#DATA_PATH="/home/dhriti/Documents/ISI/project/src_veri/olivettifaces/"
def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

class ImageSequence(Sequence):
    def __init__(self,  batch_size=4, input_size=(256,256)):
        self.image_seq_path=DATA_PATH
        self.input_shape=input_size
        self.batch_size = batch_size
        self.epoch = 0
        self.SHAPE_Y=self.input_shape[0]
        self.SHAPE_X=self.input_shape[1]
        self.dirs=os.listdir(self.image_seq_path)
	self.num_dir=len(self.dirs)


    def __len__(self):
        return (100)


    def read_image(self,file_path):
        Img=cv2.imread(file_path)
	Img=rgb2gray(Img)
        Img=cv2.resize(Img,(256,256))
        Img=Img[:,:,np.newaxis]/255.0
        return Img

    def __getitem__(self, idx):
        x_batch = []
        c=0
	X=[]
	X1=[]
	X2=[]
	Y=[]
        while(c<self.batch_size):
	    id1=np.random.randint(0,self.num_dir) 
	    flag=np.random.randint(0,2) 
	    if(flag==0): 
	        id2=np.random.randint(0,self.num_dir) 
		if(id1==id2):
		    continue
	        dir1=self.image_seq_path+self.dirs[id1]+"/"
	        dir2=self.image_seq_path+self.dirs[id2]+"/"
		files1=os.listdir(dir1)
		files2=os.listdir(dir2)
		file_path1=dir1+files1[np.random.randint(len(files1))]
		file_path2=dir2+files2[np.random.randint(len(files2))]
		#Y.append(np.zeros((128,)))
		Y.append(0)

	    else:
	        dir1=self.image_seq_path+self.dirs[id1]+"/"
		files1=os.listdir(dir1)
		file_path1=dir1+files1[np.random.randint(len(files1))]
		file_path2=dir1+files1[np.random.randint(len(files1))]
		Y.append(1)	
		#Y.append(np.ones((128,)))
	

            I1=self.read_image(file_path1)
            I2=self.read_image(file_path2)
	    #X.append([I1,I2])
	    X1.append(I1)
	    X2.append(I2)
	    #print file_path1,file_path2,Y[-1]	
	    c=c+1
	
        #X=np.array(X)
        X1=np.array(X1)
        X2=np.array(X2)
	Y=np.array(Y)
	 
        #return (X,Y)
        return ([X1,X2],Y)

    def on_epoch_end(self):
        self.epoch += 1

