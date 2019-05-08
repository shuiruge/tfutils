"""Utility functions for creating TFRecord data sets.

Forked from: https://github.com/tensorflow/models/blob/master/research/object_detection/utils/dataset_util.py  # noqa:E501
"""

import tensorflow as tf
from typing import Dict


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# TODO: Add docstring.
def get_input_nodes(dataset: tf.data.Dataset) -> Dict[str, str]:
  """
  Args:
    dataset: A `tf.data.Dataset` instance.

  Returns:
    Dictionary from input feature-name to the corresponding node in the graph.
  """
  dataset_iter = dataset.make_initializable_iterator()
  features = dataset_iter.get_next()

  if isinstance(features, tuple):  # thus including input, target, etc.
    features = features[0]  # input only, assuming that input is the first.

  if isinstance(features, dict):
    raise TypeError

  input_nodes = {}
  for input_name, tensor in features.items():
    input_nodes[input_name] = tensor.name
  return input_nodes
