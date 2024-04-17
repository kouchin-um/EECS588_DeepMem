#!/usr/bin/python3

import sys
import re
import argparse
import random
import os
import tensorflow as tf
import tensorflow.compat.v1 as tf1

MODEL_PATH = 'build_model/model/_ETHREAD/20240413_144602/struct2vec_edge_type.HOP3._ETHREAD-4'  # TF1 model path
TF2_MODEL_PATH = 'build_model/model/_ETHREAD/20240413_144602/tf2_struct2vec_edge_type.HOP3._ETHREAD-4'  # TF2 model path

vars = {}
reader = tf.train.load_checkpoint(MODEL_PATH)
dtypes = reader.get_variable_to_dtype_map()
for key in dtypes.keys():
  vars[key] = tf.Variable(reader.get_tensor(key))
tf.train.Checkpoint(vars=vars).save(TF2_MODEL_PATH)