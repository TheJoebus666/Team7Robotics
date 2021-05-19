import numpy as np
import cv2
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.builders import model_builder

def drawcai(img,boxes,classes,scores,category_index):
    height=img.shape[0]
    width=img.shape[1]

    detectn=np.sum(scores>0.2)
    
    for i in range(detectn): 
        ymin=int(boxes[i][0]*height)
        xmin=int(boxes[i][1]*width)
        ymax=int(boxes[i][2]*height)
        xmax=int(boxes[i][3]*width)        
    
        box_color = (255, 128, 0)  # box color
        box_thickness = 2
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), box_color, box_thickness)
        label_text = category_index[classes[i]+1]["name"] + " (" + str(int(scores[i]*100)) + "%)"
        label_background_color = (125, 175, 75)
        label_text_color = (255, 255, 255) 
        label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        label_xmin = int(xmin)
        label_ymin = int(ymin) - label_size[1]
        if (label_ymin < 1):
            label_ymin = 1
        label_xmax = label_xmin + label_size[0]
        label_ymax = label_ymin + label_size[1]
        cv2.rectangle(img, (label_xmin - 1, label_ymin - 1),(label_xmax + 1, label_ymax + 1),label_background_color, -1)
        cv2.putText(img, label_text, (label_xmin, label_ymax), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_text_color, 1)

def get_model_detection_function(model):
  """Get a tf.function for detection."""

  @tf.function
  def detect_fn(image):
    """Detect objects in image."""
    image, shapes = model.preprocess(image)
    prediction_dict = model.predict(image, shapes)
    detections = model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict, tf.reshape(shapes, [-1])

  return detect_fn



pipeline_config = './train_cabinet_gpu.config'
model_path = './cabinet_model/ckpt-3'
test_folder_path = './dataset/cabinet/testimage/'
label_map_path = './dataset/cabinet/label_map.pbtxt'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(pipeline_config)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=True)
# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(model_path).expect_partial()

detect_fn = get_model_detection_function(detection_model)


label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map,max_num_classes=label_map_util.get_max_label_map_index(label_map),use_display_name=True)
category_index = label_map_util.create_category_index(categories)

import glob
import os
imglist=glob.glob(os.path.join(test_folder_path, "*.jpg"))

for img_path in imglist:

  image_np = cv2.imread(img_path)
  image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

  input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, axis=0), dtype=tf.float32)
  detections, predictions_dict, shapes = detect_fn(input_tensor)

  cvimage = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
  #print(detections['detection_scores'][0].numpy())
  drawcai(cvimage,detections['detection_boxes'][0].numpy(),detections['detection_classes'][0].numpy().astype(np.uint32),detections['detection_scores'][0].numpy(),category_index)

  cv2.imshow('test_img',cvimage)
  cv2.waitKey(0) 

