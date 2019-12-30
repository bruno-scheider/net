import tensorflow as tf
from tensorflow import keras
from one_hour import *
from get_data import _decode
tf.enable_eager_execution()
net=net()
optimizer=keras.optimizers.SGD(learning_rate=0.01)
num_epoch=10
batch_size= 8




def run_epoch(train_dataset):
    epoch_loss_avg=keras.metrics.Mean()
    #TODO: calculate accuracy

    for (input,gt_heat) in train_dataset:
        
        loss, grads = grad(net.model,input,gt_heat)
        
        #g = tf.get_default_graph()
        #tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=20000)

        optimizer.apply_gradients(zip(grads,net.model.trainable_variables))

        epoch_loss_avg(loss)
    return format(epoch_loss_avg.result())



ids_dataset = tf.data.Dataset.list_files('/home/local/stud/mbzirc/net/data/copter_images/*.jpeg')
dataset = ids_dataset.map(_decode)
batched_data = dataset.batch(batch_size)


for i in range(num_epoch):
    loss=run_epoch(batched_data)
    print(loss)
    #TODO:
    #evaluation run