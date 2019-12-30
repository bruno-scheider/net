'''FIXME: Detections stimmen.
            Überprüfe width_height_ratio
            Überprüfe draw gaussian'''
import sys
sys.path.append('./utils')
import cv2
import math
import numpy as np
import random
import string
import tensorflow as tf
from config import cfg
from transform import draw_gaussian, gaussian_radius,resize_image,clip_detections
from init_data import MSCOCO
from matplotlib import pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

class Image_data():
    def __init__(self,split):
        print("hinitialized")

    def read_from_disk(self,queue):
        max_tag_len=1
        image       = np.zeros((self.input_size[0], self.input_size[1],3), dtype=np.float32)
        heatmap = np.zeros((self.output_size[0], self.output_size[1]), dtype=np.float32)
        boxes       = np.zeros((max_tag_len,4), dtype=np.int64)
        ratio       = np.ones((max_tag_len,2), dtype=np.float32)
        
        
        # reading image
        image=self.coco.read_img(queue[0]).transpose(1,0,2)

        # reading detections
        detections = self.coco.detections(queue[0])
        # print("__________________________")
        # print(image.shape, detections)
        # print("__________________________")
        
        image, detections = resize_image(image, detections, self.input_size)
        detections = clip_detections(image, detections)

        width_ratio  = self.output_size[1] / self.input_size[1]
        height_ratio = self.output_size[0] / self.input_size[0]

        image = image.astype(np.float32) / 255.
       

        for ind, detection in enumerate(detections):
            
            x_ori, y_ori = (detection[0]+detection[2])//2, (detection[1]+detection[3])//2

            fx = (x_ori * width_ratio)
            fy = (y_ori * height_ratio)
            

            x = int(fx)
            y = int(fy)
            print("__________________________")
            print(image.shape, width_ratio, height_ratio)
            print("__________________________")
            
            if self.gaussian_bump:
                width  = detection[2] - detection[0]
                height = detection[3] - detection[1]

                width  = math.ceil(width * width_ratio)
                height = math.ceil(height * height_ratio)
                print(width, height, width_ratio, height_ratio)
                if self.gaussian_rad == -1:
                    radius = gaussian_radius((height, width), self.gaussian_iou)
                    radius = max(0, int(radius))
                else:
                    radius = self.gaussian_rad

                draw_gaussian(heatmaps, x,y, radius)#statt 0 category
                
            else:
                heatmap[ytl, xtl] = 1
                

            
            center = [x_ori,y_ori]
            
            
        return image, heatmap,center


def get_gt(img_id): 
    annotations = '/home/local/stud/mbzirc/CornerNet_tf/brutrick/EfficientDet/datasets/coco/annotations/instances_train2017.json'
    coco = COCO(annotation_file=annotations)
    bbox = coco.loadAnns(img_id)[0]['bbox']
    return bbox

'''Nur sinnvoll wenn man keine gt gegeben hat'''
def create_gt(bbox):
    heatmap = np.zeros((128, 128), dtype=np.float32)
    
    w,h = image.shape[0:2]
    ratio_w = 128/w
    ratio_h= 128/h

    x_ori, y_ori = (detection[0]+detection[2])//2, (detection[1]+detection[3])//2
    fx = (x_ori * ratio_w)
    fy = (y_ori * ratio_h)
    x = int(fx)
    y = int(fy)

def _decode(filename):
    
    # print(tf.strings.bytes_splits(filename))
    gt_path = '/home/local/stud/mbzirc/net/data/copter_gt'

    path = tf.strings.split([filename], '/')
    path_split = tf.sparse.to_dense(path)

    img_id_with_jpg = tf.compat.as_str_any(tf.convert_to_tensor(path_split)[0,-1])
    img_id_split = tf.strings.split([img_id_with_jpg], '.')
    id_dense = tf.sparse.to_dense(img_id_split)
    import ipdb; ipdb.set_trace()
    img_id = tf.compat.as_str_any(tf.convert_to_tensor(id_dense)[0,0])
    
    
    image = tf.io.read_file(filename)
    
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.uint8)
    
    image = tf.image.resize(image, [128,128])

    gt_image = tf.io.read_file(gt_path+'/'+img_id_with_jpg)
    gt_image = tf.image.decode_jpeg(gt_image)
    gt_image = tf.image.convert_image_dtype(gt_image, tf.uint8)
    gt_image = tf.image.resize(gt_image, [512,512])
    #gt_image=np.convert_to_tensor(gt_image)

    # plt.imshow(img[:,:,0], cmap='gray', vmin=0, vmax=255)
    # plt.show()
   
    
    return image, gt_image

if __name__=='__main__':
    #tf.enable_eager_execution()
    data=Image_data('trainval')
 
    #import ipdb; ipdb.set_trace()
    #img_ids =np.asarray(data.coco.get_all_img())
    ids_dataset = tf.data.Dataset.list_files('/home/local/stud/mbzirc/net/data/copter_images/*.jpeg')
    dataset = ids_dataset.map(_decode)
    import ipdb; ipdb.set_trace()
    #image = _decode(file_path)


    # tf.compat.v1.enable_eager_execution()
    # data=Image_data('trainval')
    # data.inupt_producer()
    
    # images, heatmaps, centers =data.get_batch_data(5)#2
    # sess=tf.InteractiveSession()
    # coord = tf.train.Coordinator()
    # threads=tf.train.start_queue_runners(sess=sess,coord=coord)
    # for i in range(12):
    #     images_,heatmaps_, centers_= sess.run([images, heatmaps, centers])
    #     for j in range(2):
        
    #         img=(images_[j]*255).astype(np.uint8)
    #         heat_1=np.max(heatmaps_[j],axis=-1)
    #         heat_cat=np.zeros_like(heat_1)
    #         heat_cat[np.where(heat_1==1)]=1
    #         heat_arg=np.argmax(heat_cat,-1)
    #         print(heat_arg)
    #         # print('kkkkkkk')
    #         # print(np.where(heat_1==1))
    #         # print(np.where(heat_1>1))
    #         # print('ppppp')
    #         heat_1=heat_1*255
    #         heat_1=heat_1.astype(np.uint8)
    #         heat_1=np.stack([heat_1,heat_1,heat_1],-1)

    #         heat_2=np.max(heatmaps_br_[j],axis=-1)*255
    #         heat_2=heat_2.astype(np.uint8)
    #         heat_2=np.stack([heat_2,heat_2,heat_2],-1)

    #         heat=heat_1+heat_2
    #         heat=cv2.resize(heat,(511,511))
    #         norm=cv2.addWeighted(img,0.5,heat,0.5,0)
    #         box=boxes_[j]
    #         for b in box:
    #             cv2.rectangle(norm ,(b[0],b[1]),(b[2],b[3]),(225,225,0),1)
    #         cv2.imshow('img',norm)
    #         cv2.waitKey(0)
    # coord.request_stop()
    # coord.join(threads)
    # sess.close()