import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import numpy as np
import cv2
import sys
sys.path.insert(0,"/home/ecsuiplab/FACE_VERIFICATION/SRC/")
from DETECTION.model_detection import *

from matplotlib import pyplot as plt


def frame_capture():

    model=intern_model()
    model.load_weights("/home/ecsuiplab/FACE_VERIFICATION/SRC/DETECTION/MODELS/bounding_weights2.h5")
    cap = cv2.VideoCapture(0)
    f=plt.figure("dis")
    while(True):

        ret, frame = cap.read()
        #frame = cv2.imread("/media/newhd/data/face_detection/WIDER_dataset/WIDER_train/images/0--Parade/0_Parade_marchingband_1_116.jpg")
        frame = func_resize(frame)
        y_out=model.predict(frame)
        y_out = post_process(y_out)
        im_data,list_prob_data = reverse_find_the_best_anchor(frame[0],y_out[0])
        frame=super_impose(im_data,list_prob_data,prob = 0.2)
	f=plt.figure("dis")
	show_images(f,frame)

    cap.release()
    cv2.destroyAllWindows()

def func_resize(image,input_shape=(416,416)):

    ih,iw,c = image.shape
    h,w = input_shape
    scale = min(float(w)/iw, float(h)/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)
    dy = (w-nw)//2
    dx = (h-nh)//2
    image = cv2.resize(image,dsize=(nw,nh),interpolation = cv2.INTER_CUBIC)
    new_image = np.zeros((h,w,3),dtype ="float32")
    new_image[dx:dx+image.shape[0],dy:dy+image.shape[1],:] = image[:,:,:]
    image_data = np.array(new_image)/255.
    image_data=image_data[np.newaxis,:,:,:]
    return image_data

def post_process(predicted_y):

   predicted_y=np.reshape(predicted_y,predicted_y.shape[:3]+(3,6))
   grid=return_grid_np()
   predicted_y[:,:,:,:,:2]=(predicted_y[:,:,:,:,:2]+grid)/[13,13]
   predicted_y[:,:,:,:,4:]=sigmoid(predicted_y[:,:,:,:,4:])
   anchors=np.array([[10,13],[16,30],[33,23]])
   anchors=np.reshape(anchors,(1,1,1,3,2))
   predicted_y[:,:,:,:,2:4]=np.exp(predicted_y[:,:,:,:,2:4])*anchors/[416.0,416.0]

   return predicted_y
    

frame_capture()
