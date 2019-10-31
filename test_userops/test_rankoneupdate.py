#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import tf_insoundz.rankoneupdate
from ipdb import set_trace

class C: pass
obj = C(); obj.v = None

alpha = tf.constant(0.1,dtype='float32')


@tf.function
def func():
    if obj.v is None:
        obj.v = tf.Variable([[23,0],[1.5,15]],dtype='float32')
    x = tf.transpose(tf.constant([[10,1]],dtype='float32'))
    out = tf_insoundz.rankoneupdate.rankoneupdate_ops.rank_one_update(obj.v,x,alpha)
    obj.v.assign(out)
    return (out)

print(func())
print(func())
