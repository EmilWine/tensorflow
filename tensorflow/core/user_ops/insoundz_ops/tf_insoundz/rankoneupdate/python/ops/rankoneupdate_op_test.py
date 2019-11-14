"""Tests for rankoneupdate ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.python 


from tensorflow.python.platform import test
import tf_insoundz.rankoneupdate 

class RankOneeUpdateTest(test.TestCase):
    def testRankOneUpdate(self):
        alpha = tf.constant(0.1,dtype='float32')
        v = tf.Variable([[23,0],[1.5,15]],dtype='float32') 
        x = tf.transpose(tf.constant([[10,1]],dtype='float32'))
        out = tf_insoundz.rankoneupdate.rankoneupdate_ops.rank_one_update(v,x,alpha)

        #self.assertAllClose(out,np.array([[3232, 0], [2.5,15.1]]))
        self.assertAllClose(out,np.array([[33, 0], [2.5,15.1]]))

if __name__ == '__main__':
  test.main()



