import sys
sys.path.append("./data/cocoapi/PythonAPI/")
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from config import cfg
from tqdm import tqdm
import cv2
import pickle
import numpy as np
import json
import os
from PIL import Image
from get_data import _decode
# sys.path.append('../')


class MSCOCO():
    def __init__(self, split):
        data_dir = '/home/local/stud/mbzirc/CornerNet_tf/brutrick/EfficientDet/datasets'#cfg.data_dir
        result_dir = '/home/local/stud/mbzirc/CornerNet_tf/brutrick/EfficientDet/datasets'
        cache_dir = cfg.cache_dir

        self._split = split
        #verschiedene Datasets
        self._dataset = {
            "trainval": "train2017",
            "minival": "minival2014",
            "testdev": "testdev2017"
        }[self._split]
        #weg zum datenset
        self._coco_dir = os.path.join(data_dir, "coco")
        #weg zur json
        self._label_dir = os.path.join(self._coco_dir, "annotations")
        self._label_file = os.path.join(self._label_dir, "instances_{}.json")
        self._label_file = self._label_file.format(self._dataset)

        #alle Bilder im Ordner
        self._image_dir = os.path.join(self._coco_dir, "images") #, self._dataset
        self._image_file = os.path.join(self._image_dir, "{}")
        
        self._data = "coco"
        #FIXME: wir haben nur eine ID nämlich drohne aber in train.py wird für alle Kategorien eine prediction erwartet
        #SOLUTION: einfach in train.py 80 zu 1 ändern
        #anderer Fehler ist noch irgendwas mit load data
        

        self._cache_file = os.path.join(
           cache_dir, "coco_{}.pkl".format(self._dataset))
        self._load_data()
        self._db_inds = np.arange(len(self._image_ids))

        self._load_coco_data()


    def _load_data(self):
        print("loading from cache file: {}".format(self._cache_file))
        if not os.path.exists(self._cache_file):
            print("No cache file found...")
            self._extract_data()
            with open(self._cache_file, "wb") as f:
                # self._image_ids is img's name.self._detections is {name:box and cat([N,5])}
                pickle.dump([self._detections, self._image_ids], f)
        else:
            with open(self._cache_file, "rb") as f:
                self._detections, self._image_ids = pickle.load(f)
        # count=0
        # for i in self._image_ids:
        #     if 'val' in i:
        #         print(i)
        #         im=cv2.imread('/home/makalo/workspace/code/corner_net/data/coco/images/minival2014/'+i)
        #         cv2.imwrite('/home/makalo/workspace/code/corner_net/data/coco/images/trainval2014/'+i,im)

    def _load_coco_data(self):
        self._coco = COCO(self._label_file)
        with open(self._label_file, "r") as f:
            data = json.load(f)

        coco_ids = self._coco.getImgIds()
        eval_ids = {
            self._coco.loadImgs(coco_id)[0]["file_name"]: coco_id
            for coco_id in coco_ids
        }

        self._coco_categories = data["categories"]
        self._coco_eval_ids = eval_ids

    def class_name(self, cid):
        cat_id = self._classes[cid]
        cat = self._coco.loadCats([cat_id])[0]
        return cat["name"]

    def _extract_data(self):
        self._coco = COCO(self._label_file)
        self._cat_ids = self._coco.getCatIds()

        coco_image_ids = self._coco.getImgIds()

        self._image_ids = [
            self._coco.loadImgs(img_id)[0]["file_name"]
            for img_id in coco_image_ids
        ]
        self._detections = {}
        for ind, (coco_image_id, image_id) in enumerate(tqdm(zip(coco_image_ids, self._image_ids))):
            image = self._coco.loadImgs(coco_image_id)[0]
            bboxes = []
       
            annotation_ids = self._coco.getAnnIds(imgIds=image["id"])
            annotations = self._coco.loadAnns(annotation_ids)
            if(not (annotation_ids==[])):
                for annotation in annotations:
                    bbox = np.array(annotation["bbox"])
                    bboxes.append(bbox)
            else:
                bboxes.append([0,0,0,0])

            bboxes = np.array(bboxes, dtype=float)
            
            if bboxes.size == 0:
                self._detections[image_id] = np.zeros((0, 5), dtype=np.float32)
            else:
                # each image's all boxes and box's cat [N,4]
                self._detections[image_id] = bboxes
        #import ipdb; ipdb.set_trace()

    def get_all_img(self):
        return self._image_ids

    def read_img(self, img_name):
        
        img_path = self._image_file.format(bytes.decode(img_name))
        
        img = np.asarray(Image.open(img_path).convert('RGB'))#cv2.imread(img_path)
        # Image.open(img_path).show()
        if(img.any()==None): 
            print("failed to load img[--]")
            exit()
        
        return img[:,:,::-1].copy()#img.astype(np.float32)

    def detections(self, img_name):
        detections = self._detections[bytes.decode(img_name)]

        return detections.astype(float).copy()

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    

    def evaluate(self, result_json, cls_ids, image_ids, gt_json=None):
        if self._split == "testdev":
            return None

        coco = self._coco if gt_json is None else COCO(gt_json)

        eval_ids = [self._coco_eval_ids[image_id] for image_id in image_ids]
        cat_ids = [self._classes[cls_id] for cls_id in cls_ids]

        coco_dets = coco.loadRes(result_json)
        coco_eval = COCOeval(coco, coco_dets, "bbox")
        coco_eval.params.imgIds = eval_ids
        coco_eval.params.catIds = cat_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        return coco_eval.stats[0], coco_eval.stats[12:]

