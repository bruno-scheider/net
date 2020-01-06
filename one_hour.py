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
        #_x = backend.resize_images(_x,2,2,data_format="channels_last", interpolation = "nearest")
        
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
        import ipdb; ipdb.set_trace()
        _y=self.forward(inputs)
        return tf.keras.losses.Huber(gt,_y)

    def grad(self,inputs,targets):
        with tf.GradientTape() as tape:
            #added forward pass in grad
            output=self.forward(inputs)
            loss_value= self._loss(output,targets)
        return loss_value, tape.gradient(loss_value,self.model.trainable_variables)

    def _loss(self,pred, target):

        assert not tf.math.reduce_any(tf.math.is_nan(pred))
        assert not tf.math.reduce_any(tf.math.is_nan(target))

        x= tf.reduce_mean(tf.square(pred - target))
        
        return x 

    def focal_loss(self,preds,gt):
        #from makalow 
        
        assert not tf.math.any(tf.math.is_nan(targets))
        assert not tf.math.any(tf.math.is_nan(inputs))
        epsilon=0.0001
        loss = 0

        #print(gt.get_shape().as_list())
        zeros=tf.zeros_like(gt)
        ones=tf.ones_like(gt)
        num_pos=tf.reduce_sum(tf.where(tf.equal(gt,1),ones,zeros))
        loss=0
        #loss=tf.reduce_mean(tf.log(preds))
        
        pos_weight=tf.where(tf.equal(gt,1),ones-preds,zeros)
        neg_weight=tf.where(tf.less(gt,1),preds,zeros)
        #added small epsilon to log
        pos_loss=tf.reduce_sum(tf.log(preds+epsilon) * tf.pow(pos_weight,2))
        neg_loss=tf.reduce_sum(tf.pow((1-gt),4)*tf.pow(neg_weight,2)*tf.log((1-preds+epsilon)))
        loss=loss-(pos_loss+neg_loss)/(num_pos+tf.convert_to_tensor(1e-4))
    
        return loss

    def my_focal_loss(self, preds, gt):
        return tfa.losses.SigmoidFocalCrossEntropy(gt,preds)
        '''pos = tf.math.equals(gt,1)
        pos_inds= tf.where(pos, gt)

        neg = tf.math.not_equals(gt,1)
        neg_inds = tf.where(neg,gt)

        tf.math.pow(1 - gt[neg_inds],4)
        
        loss = 0

        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = tf.math.log(pos_pred) * tf.math.pow(1-pos_pred, 2)
        neg_loss = tf.math.log(1 - neg_pred) * tf.math.pow(neg_pred,2) * neg_weights

        num_pos = tf.reduce_sum(pos_inds)
        pos_loss = tf.reduce_sum(pos_loss)
        neg_loss = tf.reduce_sum(neg_loss)

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss)/ num_pos
        
        '''
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
