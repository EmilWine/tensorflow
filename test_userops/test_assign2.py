#!/usr/bin/env python3
#!/usr/bin/env python3
from ipdb import set_trace as dbg
import tensorflow as tf
mod = tf.load_op_library("../bazel-bin/tensorflow/core/user_ops/assign_variable_test2_op.so")
#mod = tf.load_op_library("../bazel-bin/tensorflow/core/user_ops/assign_variable_test2_op.so")
class C: pass
obj = C(); obj.v = None

v = tf.Variable([[21,33],[5,10]],dtype='float32')
x = tf.constant([[1,2],[3,4]],dtype='float32')

#v.assign(x)

mod.assign_variable_test2(v, x,'float32')
