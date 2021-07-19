import tensorflow as tf
import numpy as np
import os

import sys


from collections import defaultdict

from matplotlib import pyplot as plt
from PIL import Image
import cv2
import pathlib 


from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

root_dir = os.getcwd()
# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'workspace/annotations/label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
print(category_index)

fine_tuned_model = "workspace/exported-models/PPE_model"

model_dir = pathlib.Path(fine_tuned_model)/"saved_model"
print(model_dir)
model = tf.compat.v1.saved_model.load_v2(str(model_dir))

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  input_tensor = tf.convert_to_tensor(image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
  
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
              image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                      tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("safety_helmet.mp4")
# cap.open("http://192.168.1.4:8080/video")

while(cap.isOpened()):
  try:
# while(True):
    # Capture frame-by-frame
      ret,frame = cap.read()
      image_np = np.array(frame)
      output_dict = run_inference_for_single_image(model,image_np)
      vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks_reframed', None),
        use_normalized_coordinates=True,
        line_thickness=8)
      cv2.namedWindow("output", cv2.WINDOW_NORMAL)
      cv2.imshow('output',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
  except Exception as e:
    pass
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
    
