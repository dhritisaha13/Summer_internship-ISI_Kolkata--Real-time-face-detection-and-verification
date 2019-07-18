#!/usr/bin/env python
# coding: utf-8

# In[1]:
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="1"


from keras import backend as K
from keras.layers import Input, Dense, Conv2D, Reshape
from keras.layers import AveragePooling2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.layers import Layer
import keras
import numpy as np
import cv2
from matplotlib import pyplot as plt


Anchors=[(10,13),(16,30),(33,23)]


class DataGenerator(keras.utils.Sequence):
    def __init__(self,batch_size=10, dim=(416,416)):
        self.batch_size = batch_size
    def read_annotation(self,annotation_path ='/home/ecsuiplab/FACE_VERIFICATION/DATA/DETECTION/image_annot.txt'):
        with open(annotation_path) as f:
            annotation = f.readlines()
        return annotation

    def func_find_boxes_and_resize_single_image(self,annotation_line,input_shape,max_boxes=15):
        line = annotation_line.split()
        image = cv2.imread(line[0])
        ih,iw,c = image.shape
        h,w = input_shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        scale = min(float(w)/iw, float(h)/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)
        dy = (w-nw)//2
        dx = (h-nh)//2
     
        image = cv2.resize(image,dsize=(nw,nh),interpolation = cv2.INTER_CUBIC)
        new_image = np.zeros((h,w,3),dtype ="float32")
        new_image[dx:dx+image.shape[0],dy:dy+image.shape[1],:] = image[:,:,:]
        image_data = np.array(new_image)/255.
        
     
        box[:, [0,2]] = box[:, [0,2]]*scale + dy
        box[:, [1,3]] = box[:, [1,3]]*scale + dx
        box_data = np.zeros((max_boxes,5),dtype = 'float32')
        box_data[:len(box)] = box[:max_boxes]
        return image_data, box_data

    def func_eliminate_small_boxes(self,box,threshold=900):
	counter=0
	selected_box=[]
	for i in range(box.shape[0]):	
	    b=box[i]
	    area = (b[2]-b[0])*(b[3]-b[1])
	    if(area<threshold):
		box[i]=0
 
        count=np.sum(box[:,:4])
	if(count==0):
	    return False,box
	else:
	    return True,box
       
       		

    def func_generate_random_data(self,annotation_lines, batch_size, input_shape, max_boxes):
        n = len(annotation_lines)
        i = 0
        image_data = []
        box_data = []
        for b in range(batch_size):
	    flag=False
	    while(flag==False):
                np.random.shuffle(annotation_lines)
                image, box = self.func_find_boxes_and_resize_single_image(annotation_lines[i], input_shape,max_boxes)
		flag,box = self.func_eliminate_small_boxes(box)
            image_data.append(image)
            box_data.append(box)
            i = (i+1) % n
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        return image_data, box_data
    def func_produce_output_boxes(self,image_data,boxes,input_shape=(416,416)):
    

        boxes_centre = (boxes[..., 0:2] + boxes[..., 2:4]) // 2
        boxes_width_height = boxes[..., 2:4] - boxes[..., 0:2]
        output_boxes=np.zeros((boxes.shape[0],boxes.shape[1],6))
        output_boxes[:,:,:2]=boxes_centre
        output_boxes[:,:,2:4]=boxes_width_height
        
        for b in range(boxes.shape[0]):
            for i in range(boxes.shape[1]):
                if(output_boxes[b,i,2]>0):
                    output_boxes[b,i,4:6]=1
            
        return image_data, output_boxes

    def box_iou(self,b1, b2):
        b1_xmin,b1_ymin,b1_xmax,b1_ymax = b1
        b2_xmin,b2_ymin,b2_xmax,b2_ymax = b2
        xi1 = max(b1_xmin,b2_xmin)
        yi1 = max(b1_ymin,b2_ymin)
        xi2 = min(b1_xmax,b2_xmax)
        yi2 = min(b1_ymax,b2_ymax)
        inter_area = (yi2-yi1)*(xi2-xi1)
        box_area = (b1_xmax-b1_xmin)*(b1_ymax-b1_ymin)
        anchor_area = (b2_xmax-b2_xmin)*(b2_ymax-b2_ymin)
        union_area = box_area+anchor_area-inter_area
        iou = inter_area/union_area
        return iou
	
    #Input : image_data(Batch_Size x 416 x 416 x 3), true_boxes 
    
    def find_the_best_anchor(self,image_data,true_boxes,input_shape = (416,416),anchors = Anchors):
        num_of_img = true_boxes.shape[0] #Batch Size
        num_of_max_boxes = true_boxes.shape[1] #Maximum number of boxes
        output = np.zeros((num_of_img,13,13,3,6)) #Output Matrix
        for i in range(num_of_img):#For for number of images in batch
            for j in range(true_boxes.shape[1]): #For maximum number of boxes
                    
                cx,cy,ih,iw,cl,p = true_boxes[i,j,:] #Extracting the tuple
                    
                    #Calutaing the xmin,ymin xmax,ymax coordinates of the bounding box.
                b_xmin = cx - int(iw/2)
                b_ymin = cy - int(ih/2)
                b_xmax = b_xmin + iw
                b_ymax = b_ymin + ih
                    
                b1 = [b_xmin,b_ymin,b_xmax,b_ymin]
                    
                    
                iou_list = [] #iou list for storing the iou calculated for each anchors
                for a in anchors: #Iterating over all anchors
            
                    #Calutaing the xmin,ymin xmax,ymax coordinates of the anchor box.
                    a_xmin = cx - int(a[1]/2) 
                    a_ymin = cy - int(a[0]/2)
                    a_xmax = a_xmin + a[1]
                    a_ymax = a_ymin + a[0]

                    b2 = [a_xmin,a_ymin,a_xmax,a_ymax]

                    iou = self.box_iou(b1,b2)
                    iou_list.append(iou)
                
                idx=np.clip(int(round(cx/416.0 *13)),0,12)
                idy=np.clip(int(round(cy/416.0 * 13)),0,12)
                output[i,idx,idy,np.argmax(iou_list),:] = (cx/416.0,cy/416.0,ih/416.0,iw/416.0,cl,p)
        
        
        return image_data,output
    def __getitem__(self, index):
        annotation_lines = self.read_annotation()
        image_data,box_data = self.func_generate_random_data(annotation_lines, batch_size=self.batch_size, input_shape=(416,416), max_boxes=15)
        image_data,output_box = self.func_produce_output_boxes(image_data,box_data)
        X,Y = self.find_the_best_anchor(image_data,output_box)
        Y = np.reshape(Y,(Y.shape[0],Y.shape[1],Y.shape[2],18))
        return X,Y
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return 50
    
    
def sigmoid(x, derivative=False):
    sigm = 1. / (1. + np.exp(-x))

    return sigm   



def reverse_find_the_best_anchor(image_data,true_boxes,input_shape=(13,13),anchors=Anchors):
    
    #plt.imshow(image_data)
    #plt.show()
    grid_size=true_boxes.shape[0:2]
    number_of_anchors=true_boxes.shape[2]
    num_of_max_boxes =15 #Maximum number of boxes
    
    output = np.zeros((num_of_max_boxes,6)) #Output Matrix
    print(true_boxes.shape)
    
    list_prob=[]
    
    for j in range(grid_size[0]):
        for k in range(grid_size[1]):
            for l in range(number_of_anchors):
                cx,cy,iw,ih,cl,p = true_boxes[j,k,l,:]
                list_prob.append(true_boxes[j,k,l,:])
    
    list_prob = np.array(list_prob)
    list_prob = list_prob[np.argsort(list_prob[:, 5])]
    list_prob = list_prob[::-1]
    list_prob = list_prob[:15]
    
    for i in range(len(list_prob)):
        cx,cy,iw,ih,cl,p = list_prob[i]
        cx = int(round(cx * 416))
        cy = int(round(cy * 416))
        iw = iw * 416
        ih = ih * 416
        list_prob[i] = cx,cy,iw,ih,cl,p
    return image_data,list_prob

                



def super_impose(image1,box,prob=0.9):
    image=image1.copy()
    for b in  box:
        cx,cy,iw,ih,cl,p = b
        cx=cx
        cy=cy
        ih=ih
        iw=iw
        #cl = sigmoid(cl)
        #p = sigmoid(p)
        if(p>=prob):
            cv2.rectangle(image,(int(cx-iw/2),int(cy-ih/2)),(int(cx+iw/2),int(cy+ih/2)),(0,255,0),3)

    return image
    #plt.imshow(image)
    #plt.show()
    #plt.clf()
    #plt.close()



def show_images(f,image):
    plt.imshow(image)
    #plt.show()
    plt.pause(0.02)
    plt.clf()
    #plt.close()
    






"""
imgen= DataGenerator()
x,y = imgen.__getitem__(0)

y_gt=np.reshape(y,y.shape[:3]+(3,6))
im_data,list_prob_data = reverse_find_the_best_anchor(x[0],y_gt[0])

super_impose(im_data,list_prob_data)
   
"""

  
