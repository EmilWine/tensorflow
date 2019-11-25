#!/usr/bin/env python3
import os
os.environ["OMP_NUM_THREADS"] = "1"
#from ipdb import set_trace as dbg
import tensorflow as tf
from tensorflow.python.framework import config
mod = tf.load_op_library("../bazel-bin/tensorflow/core/user_ops/insoundz_ops/tf_insoundz/rankoneupdate/python/ops/_rankoneupdate_ops.so")
config.set_inter_op_parallelism_threads(1)
config.set_intra_op_parallelism_threads(1)

N = 2
alpha = tf.constant(0.9,'complex64')
a = tf.cast(tf.ones(shape=(N,N)),dtype='complex64')
x = tf.cast(3*tf.ones(shape=(N,1)),dtype='complex64')
z = mod.rank_one_update(a,x,alpha)
print(z)

