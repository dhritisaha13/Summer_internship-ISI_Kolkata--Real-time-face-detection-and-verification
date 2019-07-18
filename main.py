import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from models import *
from generator_28 import *
from database import *

ob=DBMS()
img_gen = ImageSequence()
mycol = ob.Create_Database("Face_Data10","face8")


saimese_model,base_model=create_siamese_network(input_shape=(256,256,1))
saimese_model.load_weights("models/saimese_weights.h5")
base_model.load_weights("models/base_weights.h5")


for i in range(1000):

    I,P =img_gen.__getitem__(3)
    V = ob.get_vector(I,base_model)
    path,dist=ob.Insert_Data(mycol,P,V)
    #print path,dist
    #print "**********",P,path[0]

   

