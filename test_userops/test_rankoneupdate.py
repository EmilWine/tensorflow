#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
mod = tf.load_op_library("../bazel-bin/tensorflow/core/user_ops/rankoneupdate/rankoneupdate_op.so")
alpha = tf.constant(0.1,dtype='float32')

class C: pass
obj = C(); obj.v = None

@tf.function
def func():
    if obj.v is None:
        obj.v = tf.Variable([[23,0],[1.5,15]],dtype='float32')
    x = tf.transpose(tf.constant([[10,1]],dtype='float32'))
    out =mod.rank_one_update(obj.v,x,alpha) 
    obj.v.assign(out)
    return (out)

print(func())
print(func())
