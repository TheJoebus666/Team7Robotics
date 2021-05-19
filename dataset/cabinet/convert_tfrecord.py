'''
Modified from PASCAL dataset to TFRecord for object_detection.
'''
import hashlib
import io
import os
import glob
from lxml import etree
import PIL.Image
import tensorflow.compat.v1 as tf1

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

def dict_to_tf_example(data,full_path,label_map_dict,ignore_difficult_instances=False,image_subdirectory='JPEGImages'):

  with tf1.gfile.GFile(full_path, 'rb') as fid:
    encoded_jpg = fid.read()
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = PIL.Image.open(encoded_jpg_io)
  if image.format != 'JPEG':
    raise ValueError('Image format not JPEG')
  key = hashlib.sha256(encoded_jpg).hexdigest()

  width = int(data['size']['width'])
  height = int(data['size']['height'])

  xmin = []
  ymin = []
  xmax = []
  ymax = []
  classes = []
  classes_text = []
  truncated = []
  difficult_obj = []
  for obj in data['object']:
    difficult = bool(int(obj['difficult']))
    if ignore_difficult_instances and difficult:
      continue
    difficult_obj.append(int(difficult))

    xmin.append(float(obj['bndbox']['xmin']) / width)
    ymin.append(float(obj['bndbox']['ymin']) / height)
    xmax.append(float(obj['bndbox']['xmax']) / width)
    ymax.append(float(obj['bndbox']['ymax']) / height)
    classes_text.append(obj['name'].encode('utf8'))
    classes.append(label_map_dict[obj['name']])
    truncated.append(int(obj['truncated']))
    
  example = tf1.train.Example(features=tf1.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(data['filename'].encode('utf8')),
      'image/source_id': dataset_util.bytes_feature('0'.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
      'image/object/difficult': dataset_util.int64_list_feature(difficult_obj),
      'image/object/truncated': dataset_util.int64_list_feature(truncated),
  }))
  return example


def main(_):

  datapath = './images/'
  label_map_path = './label_map.pbtxt'
  writer_path = './train.record'

  label_map_dict = label_map_util.get_label_map_dict(label_map_path)
  writer = tf1.python_io.TFRecordWriter(writer_path)
  imglist=glob.glob(os.path.join(datapath, "*.jpg"))

  i = 0
  for filename in imglist:

    path = filename.replace('.jpg','.xml')
    with tf1.gfile.GFile(path, 'r') as fid:
      xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']

    tf_example = dict_to_tf_example(data, filename, label_map_dict,False)
    writer.write(tf_example.SerializeToString())

    if i % 10 == 0:
      print(i, ' of ', len(imglist))
    i=i+1

  writer.close()


if __name__ == '__main__':
  tf1.app.run()

