from generators.common import Generator
import os
import numpy as np
from pycocotools.coco import COCO
import cv2
from PIL import Image

class CocoQueue(Dataset):
      def __init__(self, data_dir, set_name, **kwargs):
        """
        Initialize a COCO data generator.
        Args
            data_dir: Path to where the COCO dataset is stored.
            set_name: Name of the set to parse.
        """
        self.data_dir = data_dir
        self.set_name = set_name
        self.coco = COCO(os.path.join(data_dir, 'annotations', 'instances_' + set_name + '.json'))
        self.image_ids = self.coco.getImgIds()


        super(CocoGenerator, self).__init__(**kwargs)


    def size(self):
        """ Size of the COCO dataset.
        """
        return len(self.image_ids)

    def num_classes(self):
        """ Number of classes in the dataset. For COCO this is 80.
        """
        return len(self.classes)

    def has_label(self, label):
        """ Return True if label is a known label.
        """
        return label in self.labels

    def has_name(self, name):
        """ Returns True if name is a known class.
        """
        return name in self.classes

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def coco_label_to_label(self, coco_label):
        """ Map COCO label to the label as used in the network.
        COCO has some gaps in the order of labels. The highest label is 90, but there are 80 classes.
        """
        return self.coco_labels_inverse[coco_label]

    def coco_label_to_name(self, coco_label):
        """ Map COCO label to name.
        """
        return self.label_to_name(self.coco_label_to_label(coco_label))

    def label_to_coco_label(self, label):
        """ Map label as used by the network to labels as used by COCO.
        """
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def load_image(self, image_index):
        """
        Load an image at the image_index.
        """
        # {'license': 2, 'file_name': '000000259765.jpg', 'coco_url': 'http://images.cocodataset.org/test2017/000000259765.jpg', 'height': 480, 'width': 640, 'date_captured': '2013-11-21 04:02:31', 'id': 259765}
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.data_dir, 'images', self.set_name, image_info['file_name'])
        image = np.asarray(Image.open(path))
       
        return image

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = {'labels': np.empty((0,), dtype=np.float32), 'bboxes': np.empty((0, 4), dtype=np.float32)}

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):
            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotations['labels'] = np.concatenate(
                [annotations['labels'], [self.coco_label_to_label(a['category_id'])]], axis=0)
            annotations['bboxes'] = np.concatenate([annotations['bboxes'], [[
                a['bbox'][0],
                a['bbox'][1],
                a['bbox'][2],
                a['bbox'][3],
            ]]], axis=0)

        return annotations