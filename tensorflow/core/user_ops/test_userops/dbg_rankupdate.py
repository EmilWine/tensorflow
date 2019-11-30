#!/usr/bin/env python3
import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
from datetime import datetime
import timeit
from tensorflow.python.framework import config
import tensorflow as tf
import numpy as np
import tf_insoundz.rankoneupdate as tf_rankoneupdate
import tf_insoundz.hermitian_matmul as tf_hermiaitan_matmul
from ipdb import set_trace
config.set_inter_op_parallelism_threads(1)
config.set_intra_op_parallelism_threads(1)


alpha = tf.constant(0.9,'complex64')
a = tf.cast(tf.zeros(shape=(10,10)),dtype='complex64')
x = tf.cast(2*tf.ones(shape=(10,1)),dtype='complex64')
z = tf_rankoneupdate.rankoneupdate_ops.rank_one_update(a,x,alpha)
print(z)
