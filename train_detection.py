import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from generator_detection import *
from model_detection import *
from model_new import *




imgen= DataGenerator(batch_size=10)
x,y = imgen.__getitem__(0)




model=intern_model()

I=Input(shape=(416,416,3))
#model=yolo_body(I,3,1)
#model.compile(loss=custom_loss(Layer),optimizer="adam"      ,metrics=['accuracy'])

#model.load_weights("models/bounding_weights2.h5")
model.fit_generator(imgen,epochs=200000)
#model.fit(x,y,epochs=1000)
model.save_weights("MODELS/bounding_weights3.h5")
model.load_weights("MODELS/bounding_weights3.h5")





y_gt=np.reshape(y,y.shape[:3]+(3,6))
y_out=model.predict(x)
y_out=np.reshape(y_out,y_out.shape[:3]+(3,6))
grid=return_grid_np()
y_out[:,:,:,:,:2]=(y_out[:,:,:,:,:2]+grid)/[13,13]
y_out[:,:,:,:,4:]=sigmoid(y_out[:,:,:,:,4:])
anchors=np.array([[10,13],[16,30],[33,23]])
anchors=np.reshape(anchors,(1,1,1,3,2))
y_out[:,:,:,:,2:4]=np.exp(y_out[:,:,:,:,2:4])*anchors/[416.0,416.0]




#idx=np.random.randint(x.shape[0])
#im_data,list_prob_data = reverse_find_the_best_anchor(x[idx],y_gt[idx])
im_data,list_prob_data = reverse_find_the_best_anchor(x[1],y_out[1])
super_impose(im_data,list_prob_data,prob = 0.41)
