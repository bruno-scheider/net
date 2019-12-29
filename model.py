import tensorflow as tf
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import backend
import numpy as np

print("hi")

class net:
    def __init__(self):
        self.input_size=512

    def forward(self,input):
        '''
        input needs to be size of (b ,512, 512, 3)
        '''
        start = layers.Conv2D(filters = 3,kernel_size=1, strides=1)(input)
        start = layers.MaxPooling2D()(start)
        hg1 = self.hg_module(start)
        hg2 = self.hg_module(hg1)

        return hg2


    def hg_module(self,input):#, input):
        
        #input =  np.random.rand(b,256,256,3)
        
        ##down
        skip256 = input
        _x = layers.Conv2D(filters = 3,kernel_size=1, strides=1)(input)
        _x = layers.MaxPooling2D()(_x)
        skip128 = _x
        _x = layers.Conv2D(filters = 3,kernel_size=1, strides=1)(_x)
        _x = layers.MaxPooling2D()(_x)
        skip64 = _x
        _x = layers.Conv2D(filters = 3,kernel_size=1, strides=1)(_x)
        _x = layers.MaxPooling2D()(_x)
        skip32 = _x
        _x = layers.Conv2D(filters = 3,kernel_size=1, strides=1)(_x)
        _x = layers.MaxPooling2D()(_x)
        skip16 = _x
        _x = layers.Conv2D(filters = 3,kernel_size=1, strides=1)(_x)
        _x = layers.MaxPooling2D()(_x)
    
        
        ##up
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





if __name__ == "__main__":
    model = net()
    data = np.random.rand(5,512,512,3)
    pred = model.forward(data)
    print(pred.shape)

