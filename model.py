import tensorflow as tf
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import backend
from tensorflow.compat.v1 import keras
import numpy as np

print("hi")

class net:
    def __init__(self):
        self.input_size=512
        input = keras.Input(shape=(512,512,3), dtype=tf.float32)
        start = layers.Conv2D(filters = 3,kernel_size=1, strides=1)(input)
        start = layers.MaxPooling2D()(start)
        hg1 = self.hg_module(start,fil=3)
        hg2 = self.hg_module(hg1,fil=3)
        depth=tf.compat.v1.space_to_depth(input=hg2,block_size=4)
        hg3=self.hg_module(depth,fil=48)
        hg4=self.hg_module(hg3,fil=48)
        heat=self.reduce_dim(hg4,48)
        norm_heat=tf.compat.v1.math.l2_normalize(heat,axis=3)

        self.model=keras.Model(inputs=input,outputs=norm_heat)

    def forward(self,input):
        '''
        input needs to be size of (b ,512, 512, 3)
        '''
        return self.model(input)

    def reduce_dim(self, input,num):
        splits=tf.compat.v1.split(value=input,num_or_size_splits=num,axis=3)
        _x=splits[0]
        for i in range(1,len(splits)):
            _x=layers.Add()([_x,splits[i]])


        return _x

    def hg_module(self,input,fil=3):
        
        #input =  np.random.rand(b,256,256,3)
        
        ##down
        skip256 = input
        _x = layers.Conv2D(filters = fil,kernel_size=1, strides=1)(input)
        _x = layers.MaxPooling2D()(_x)
        skip128 = _x
        _x = layers.Conv2D(filters = fil,kernel_size=1, strides=1)(_x)
        _x = layers.MaxPooling2D()(_x)
        skip64 = _x
        _x = layers.Conv2D(filters = fil,kernel_size=1, strides=1)(_x)
        _x = layers.MaxPooling2D()(_x)
        skip32 = _x
        _x = layers.Conv2D(filters = fil,kernel_size=1, strides=1)(_x)
        _x = layers.MaxPooling2D()(_x)
        skip16 = _x
        _x = layers.Conv2D(filters = fil,kernel_size=1, strides=1)(_x)
        _x = layers.MaxPooling2D()(_x)
        skip8 = _x
        _x = layers.Conv2D(filters = fil,kernel_size=1, strides=1)(_x)
        _x = layers.MaxPooling2D()(_x)
        
        ##up
        _x = backend.resize_images(_x,2,2,data_format="channels_last", interpolation = "nearest")
        _x = layers.Add()([_x,skip8])
        _x = backend.resize_images(_x,2,2,data_format="channels_last", interpolation = "nearest")
        _x = layers.Add()([_x,skip16])
        _x = backend.resize_images(_x,2,2,data_format="channels_last", interpolation = "nearest")
        _x = layers.Add()([_x,skip32])
        _x = backend.resize_images(_x,2,2,data_format="channels_last", interpolation = "nearest")
        _x = layers.Add()([_x,skip64])
        _x = backend.resize_images(_x,2,2,data_format="channels_last", interpolation = "nearest")
        _x = layers.Add()([_x,skip128])
        
        _x = backend.resize_images(_x,2,2,data_format="channels_last", interpolation = "nearest")
        
        _x = layers.Add()([_x,skip256])
        
        return _x

    def loss(self,inputs,gt):
        #TODO: maybe add a custom loss function, first try with something from keras
        #both needs to be shape [NONE, 64,64,1]
        _y=self.forward(inputs)
        return tf.keras.losses.Huber(gt,_y)

    def grad(self,inputs,targets):
        with tf.GradientTape() as tape:
            loss_value= loss(inputs,targets)
        return loss_value, tape.gradient(loss_value,self.model.trainable_variables)




if __name__ == "__main__":
    model = net()
    data = np.random.rand(5,512,512,3).astype(np.float32)
    #import ipdb; ipdb.set_trace()
    pred = model.forward(data)
    import ipdb; ipdb.set_trace()
    print(pred.shape)
    

