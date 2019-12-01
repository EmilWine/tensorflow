#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import tf_insoundz.matrix_scale
from ipdb import set_trace

class C: pass
obj = C(); obj.v = None
alpha = tf.constant(0.1,dtype='float32')
a = tf.Variable([[23,0],[1.5,15]],dtype='float32')

print(a)
z = tf_insoundz.matrix_scale.matrix_scale_ops.matrix_scale(a,alpha)
print(z)
print("a",a)
