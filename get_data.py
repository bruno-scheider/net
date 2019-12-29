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

class Image_data():
    def __init__(self,split):
        self.coco=MSCOCO(split)
        self.data_rng   = cfg.data_rng
        self.num_image  = len(self.coco.get_all_img())
        self.categories   = cfg.categories
        self.input_size   = cfg.input_size
        self.output_size  = cfg.output_sizes[0]

        self.border        = cfg.border
        #self.lighting      = cfg.lighting
        self.rand_crop     = cfg.rand_crop
        print(self.rand_crop)
        self.rand_color    = cfg.rand_color
        self.rand_scales   = cfg.rand_scales
        self.gaussian_bump = cfg.gaussian_bump
        self.gaussian_iou  = cfg.gaussian_iou
        self.gaussian_rad  = cfg.gaussian_radius

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

    def get_single_data(self,queue):
        images, heatmap, center = tf.py_func(self.read_from_disk,[queue],
            [tf.float32,tf.float32,tf.int64])
        return images, heatmap, center

        
    def inupt_producer(self):
        #DEBUG:
        ## quene_train is empty  
        ## self.coco.get_all_img() give list of strings(name of pic)
       
        quene_train=tf.train.slice_input_producer([self.coco.get_all_img()],shuffle=False)
        self.images, self.heatmap, self.center =self.get_single_data(quene_train)
        
    def get_batch_data(self,batch_size):
        

        #DEBUG:
        ## tf.train.shuffle_batch : List of tensors is NONE, batch_size = 10, 
        ## shapes = [(511, 511, 3),   128,    128,   (128, 128, 1), (128, 128, 1),   128,    (128, 2),   (128, 2),   (128, 4), (128, 2)]
        ##                 |           |       |           |             |            |         |            |          |          |
        ##              images,     tags_tl, tags_br,  heatmaps_tl,   heatmaps_br,  tags_mask, offset_tl, offset_br, boxes_ratio, ratio
        images, heatmaps, centers =tf.train.shuffle_batch([self.images,self.heatmap, self.center],
            batch_size=batch_size,
            shapes=[(self.input_size[0], self.input_size[1],3),(self.output_size[0], self.output_size[1]),
            (2)],
            capacity=100,min_after_dequeue=batch_size,num_threads=1)#num_threads =16
        
        return images,heatmaps, centers


if __name__=='__main__':
    data=Image_data('trainval')
    print(tf.data.Dataset.from_tensor_slices(data.coco.get_all_img(),1))


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