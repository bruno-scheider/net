import tensorflow as tf
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import backend
from tensorflow.compat.v1.keras import backend as K
from tensorflow.compat.v1 import keras
import numpy as np

print("hi")

class net:
    def __init__(self):
        sess=tf.Session()
        backend.set_session(sess)
        self.input_size=512

        self.model=self.load_model()

    def forward(self,input):
        '''
        input needs to be size of (b ,512, 512, 3)
        '''
        return self.model(input)

    def load_model(self):
        #TODO: normalization
        input = keras.Input(shape=(512,512,3), dtype=tf.float32)
        _x = layers.Conv2D(filters = 1,kernel_size=1,strides=1)(input)
        _x = layers.MaxPooling2D()(_x)

        ##down
        _x = tf.compat.v1.space_to_depth(input=_x,block_size=2)#shape(NONE, 128,128,4)
        _x = self.conv_norm(128,4,_x)
        skip_128 = _x
        _x = tf.compat.v1.space_to_depth(input=_x,block_size=2)#shape(NONE, 64,64,16)
        _x = self.conv_norm(64,16,_x)
        skip_64 = _x
        _x = tf.compat.v1.space_to_depth(input=_x,block_size=2)#shape(NONE, 32,32,64)
        _x = self.conv_norm(32,64,_x)
        skip_32 = _x
        _x = tf.compat.v1.space_to_depth(input=_x,block_size=2)#shape(NONE, 16,16,256)
        _x = self.conv_norm(16,256,_x)
        skip_16 = _x
        _x = tf.compat.v1.space_to_depth(input=_x,block_size=2)#shape(NONE, 8,8,1024)
        _x = self.conv_norm(8,1024,_x)
        skip_8=_x
        _x = layers.Conv2D(filters=64,kernel_size=1, strides =1)(_x)
        _x = tf.compat.v1.space_to_depth(input=_x,block_size=2)#shape(NONE, 4,4,256)
        _x = self.conv_norm(4,256,_x)


        ##up
        _x = backend.resize_images(_x,2,2,data_format="channels_last", interpolation = "nearest")
        _x = layers.Concatenate(axis=-1)([_x,skip_8])
        _x = layers.Conv2D(filters=1024,kernel_size=1, strides =1)(_x)

        _x = backend.resize_images(_x,2,2,data_format="channels_last", interpolation = "nearest")
        _x = layers.Concatenate(axis=-1)([_x,skip_16])
        _x = layers.Conv2D(filters=256,kernel_size=1, strides =1)(_x)

        _x = backend.resize_images(_x,2,2,data_format="channels_last", interpolation = "nearest")
        _x = layers.Concatenate(axis=-1)([_x,skip_32])
        _x = layers.Conv2D(filters=64,kernel_size=1, strides =1)(_x)

        _x = backend.resize_images(_x,2,2,data_format="channels_last", interpolation = "nearest")
        _x = layers.Concatenate(axis=-1)([_x,skip_64])
        _x = layers.Conv2D(filters=16,kernel_size=1, strides =1)(_x)

        _x = backend.resize_images(_x,2,2,data_format="channels_last", interpolation = "nearest")
        _x = layers.Concatenate(axis=-1)([_x,skip_128])
        output = layers.Conv2D(filters=1,kernel_size=1, strides =1)(_x)
        return keras.Model(inputs=input,outputs=output)


    def conv_norm(self,input_size,fil,input):
        _x = layers.Conv2D(filters = fil,kernel_size=1, strides=1)(input)
        _x = layers.Conv2D(filters = fil,kernel_size=1, strides=1)(_x)
        _x = layers.Conv2D(filters = fil,kernel_size=1, strides=1)(_x)
        return _x

    def loss(self,inputs,gt):
        #TODO: maybe add a custom loss function, first try with something from keras
        #both needs to be shape [NONE, 128,128,1]
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

    #model.model.save('one hour.h5')

    #input_1:0
    #conv2d_24/BiasAdd:0
    import ipdb; ipdb.set_trace()
    print(pred.shape)
