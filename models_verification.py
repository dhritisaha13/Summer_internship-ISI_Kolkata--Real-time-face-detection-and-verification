from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, Add, Dropout, concatenate
from keras.layers import *

def unet_down_1(filter_count, inputs, activation='relu', pool=(2, 2), n_layers=3):
    down = inputs
    for i in range(n_layers):
        down = Conv2D(filter_count, (3, 3), padding='same', activation=activation)(down)
        down = BatchNormalization()(down)

    if pool is not None:
        x = MaxPooling2D(pool, strides=pool)(down)
    else:
        x = down
    return (x, down)

def create_base_model(input_shape=(256,256,1)):

    n_layers_down = [2, 2, 2, 2, 2, 2]
    n_filters_down = [16,32,64, 96, 144, 192]
    n_filters_center=256
    n_layers_center=4
    
    activation='relu'
    inputs = Input(shape=input_shape)
    x = inputs
    x = BatchNormalization()(x)
    xbn = x
    depth = 0
    back_links = []
    for n_filters in n_filters_down:
        n_layers = n_layers_down[depth]
        x, down_link = unet_down_1(n_filters, x, activation=activation, n_layers=n_layers)
        back_links.append(down_link)
        depth += 1

    center, _ = unet_down_1(n_filters_center, x, activation=activation, pool=None, n_layers=n_layers_center)


    # center
    x1 = center
    x1=Flatten()(x1)
    out=Dense(4096,activation="sigmoid")(x1)
    model = Model(inputs=inputs, outputs=out)
    return model

def create_siamese_network(input_shape=(256,256,1)):
    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)
   
    base_model=create_base_model(input_shape=input_shape) 
 
    vec1=base_model(input1)
    vec2=base_model(input2)
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    vec = L1_layer([vec1,vec2]) 
    
    out=Dense(1,activation="sigmoid")(vec)
    siamese_model = Model(inputs=[input1,input2], outputs=out)
  
    return siamese_model,base_model


