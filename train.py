import tensorflow as tf
from tensorflow import keras
from model import *


net=net()
optimizer=keras.optimizer.SGD(learning_rate=0.01)
num_epoch=10

def run_epoch(train_dataset):
    epoch_loss_avg=keras.metrics.Mean()
    #TODO: calculate accuracy

    for input,gt_heat in train_dataset:
        
        loss, grads = grad(net.model,input,gt_heat)
        optimizer.apply_gradients(zip(grads,net.model.trainable_variables))

        epoch_loss_avg(loss)
    return format(epoch_loss_avg.result())

for i in range(num_epoch):
    loss=run_epoch(train_data)
    print(loss)
    #TODO:
    #evaluation run