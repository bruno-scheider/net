import tensorflow as tf
from tensorflow import keras
from one_hour import *
from get_data import _decode, _decode_gt
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
tf.enable_eager_execution()
net=net()
#changed loss from 0.01 to 0.001 otherwise the net diverges
optimizer=keras.optimizers.SGD(learning_rate=0.0005)
num_epoch=10
batch_size= 8




def run_epoch(train_dataset):
    epoch_loss_avg=keras.metrics.Mean()
    #TODO: calculate accuracy
    i=0

    for input,gt_heat in train_dataset:
        i+=1
        loss=0
        #import ipdb; ipdb.set_trace()
        #pred = net.forward(input)
        loss, grads = net.grad(input,gt_heat)
        #print("Batch: "+str(i)+ " with loss: " + str(loss))
        #g = tf.get_default_graph()
        #tf.contrib.quantize.create_training_graph(input_graph=g, quant_delay=20000)
        #import ipdb; ipdb.set_trace()
        optimizer.apply_gradients(zip(grads,net.model.trainable_variables))

        epoch_loss_avg(loss)
    return format(epoch_loss_avg.result())


def evaluate_on_image(net):
    path_img= '/home/local/stud/mbzirc/net/data/copter_images/89350.jpeg'
    gt_img= '/home/local/stud/mbzirc/net/data/copter_gt/89350.jpeg'
    #read image and gt
    image = tf.io.read_file(path_img)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [128,128])

    gt = tf.io.read_file(gt_img)
    gt = tf.image.decode_jpeg(gt)
    gt = tf.image.convert_image_dtype(gt, tf.float32)
    #gt = tf.image.grayscale_to_rgb(gt)
    #gt = tf.image.resize(gt, [512,512])
    gt = tf.squeeze(gt)
    
    gt = np.array(gt).astype(np.uint8)*255
    #add dim to pass to net
    image_ = tf.expand_dims(image,0)

    #predict
    image_ = net.forward(image_)
    image_ = tf.squeeze(image)
    
    #convert prediction to numpy
    img = np.array(image_)
    img = np.resize(img,(512,512))*255
    img = img.astype(np.int8)

    #convert image to numpy and resize
    image = np.array(image)
    image = np.resize(image,(512,512))
    image = image.astype(np.int8)

    #stack them horizontaly
    #img=np.hstack((gt,img))
    #import ipdb; ipdb.set_trace()
    Image.fromarray(image).show()
    #plt.imshow(img, cmap='gray', vmin=0, vmax=255)
    #plt.imshow(image, cmap='gray', vmin=0, vmax=255)
    plt.show()
   
    
  



ids_dataset = tf.data.Dataset.list_files('/home/local/stud/mbzirc/net/data/copter_images/*.jpeg')
dataset = ids_dataset.map(_decode)

ids_dataset_gt = tf.data.Dataset.list_files('/home/local/stud/mbzirc/net/data/copter_gt/*.jpeg')
dataset_gt = ids_dataset_gt.map(_decode_gt)
dataset = tf.data.Dataset.zip((dataset, dataset_gt))
#dataset.shuffle(1900, reshuffle_each_iteration=True)
batched_data = dataset.batch(batch_size)

#evaluate_on_image(net)

for i in tqdm(range(num_epoch)):
    #evaluate_on_image(net)
    loss=run_epoch(batched_data)
    print(loss)
    if(i==5):
        #evaluate_on_image(net)
        net.model.save_weights('cpts/weights'+str(i))
    #TODO:
    #evaluation run