import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from models_verification import *
from generator_verification import *




saimese_model,base_model=create_siamese_network(input_shape=(256,256,1))
saimese_model.compile(loss="binary_crossentropy",optimizer="adam")
#saimense_model.compile(loss="mse",optimizer="adam")

img_gen=ImageSequence()
saimese_model.fit_generator(img_gen,epochs=1000)

saimese_model.save_weights("MODELS/saimese_weights.h5")
base_model.save_weights("MODELS/base_weights.h5")

saimese_model.load_weights("MODELS/saimese_weights.h5")
base_model.load_weights("MODELS/base_weights.h5")

[X1,X2],Y =img_gen.__getitem__(3)


#-----------------------Testing-----------------------------------
saimese_model,base_model=create_siamese_network(input_shape=(256,256,1))
saimese_model.load_weights("MODELS/saimese_weights.h5")
base_model.load_weights("MODELS/base_weights.h5")

[X1,X2],Y=img_gen.__getitem__(3)

vec1=base_model.predict(X1)
vec2=base_model.predict(X2)

sum_vec=np.mean(np.abs(vec1-vec2),axis=-1)

   

