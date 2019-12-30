import tensorflow as tf
import numpy as np
import cv2
import os
from model import net
from tqdm import tqdm

from utils.transform import crop_image
from module.forward_module import rescale_dets,soft_nms_merge

from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python.tools import freeze_graph


class Export():
    def __init__(self):
        #self.coco=MSCOCO('minival')
        self.net=net()
       

        self.snapshot_dir = 'path to snapshot_file'
        self.snapshot_file = 'name of snashpot_file'

    def load(self, saver, sess, ckpt_path):
        '''Load trained weights.
        Args:
          saver: TensorFlow saver object.
          sess: TensorFlow session.
          ckpt_path: path to checkpoint file with parameters.
        '''
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, os.path.join(ckpt_path, ckpt_name))
            print("Restored model parameters from {}".format(ckpt_name))
            return True
        else:
            return False

    def export(self):

                #is_training=tf.convert_to_tensor(False)

                images = tf.compat.v1.placeholder(tf.float32, shape=(1, 512, 512, 3), name = 'input_image')

                test_outs=self.net.forward(images)
                #dets_tensor=self.net.decode(*test_outs)

                output_names = ''
                for tensor in test_outs:
                    output_names+=tensor.name[:-2] + ','
                output_names = output_names[:-1]
                print(output_names)

                
                g = tf.get_default_graph()
                #tf.contrib.quantize.create_eval_graph(input_graph=g)

                loader = tf.train.Saver()
                input_saver_def = loader.as_saver_def()

                frozen_graph_def = freeze_graph.freeze_graph_with_def_protos(
                    input_graph_def=tf.get_default_graph().as_graph_def(),
                    input_saver_def=input_saver_def,
                    input_checkpoint=self.snapshot_dir + '/corner_net.ckpt-0',
                    output_node_names=output_names,
                    restore_op_name='',
                    filename_tensor_name='',
                    clear_devices=True,
                    output_graph='',
                    initializer_nodes='')

                #input_names = ['image']
                #transforms = ['strip_unused_nodes']
                #output_names_list = ['heatmaps_up', 'pafs_up', 'heatmaps_smoothed', 'heatmaps_max']
                #transformed_graph_def = TransformGraph(frozen_graph_def, input_names,
                #                                     output_names_list, transforms)
                transformed_graph_def = frozen_graph_def

                with tf.gfile.GFile('./tmp/frozen_graph_without_cornerPooling.pb', 'wb') as f:
                        f.write(transformed_graph_def.SerializeToString())

                with tf.gfile.GFile('./tmp/frozen_graph_without_cornerPooling.pbtxt', 'w') as f:
                        f.write(str(transformed_graph_def))                

                #tflite_convert --output_file="tflite_model.tflite" --graph_def_file="frozen_graph.pb" --input_arrays="input_image" --output_arrays="" --input_shapes=1,512,512,3 --enable_select_tf_ops --allow_custom_ops

                #Quantized:
                # echo "CONVERTING frozen graph to TF Lite file..."
                # tflite_convert \
                # --output_file="${OUTPUT_DIR}/output_tflite_graph_480p_score_thresh_1e-10_test_${ckpt_number}.tflite" \
                # --graph_def_file="${OUTPUT_DIR}/tflite_graph.pb" \
                # --inference_type=QUANTIZED_UINT8 \
                # --input_arrays="${INPUT_TENSORS}" \
                # --output_arrays="${OUTPUT_TENSORS}" \
                # --mean_values=128 \
                # --std_dev_values=128 \
                # --input_shapes=1,480,848,3 \
                # --change_concat_input_ranges=false \
                # --allow_nudging_weights_to_use_fast_gemm_kernel=true \
                # --allow_custom_ops

if __name__ == "__main__":

    e = Export()
    e.export()