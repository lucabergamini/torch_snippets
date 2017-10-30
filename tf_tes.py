import tensorflow
import numpy
from tensorflow import placeholder


m1 = placeholder(dtype="float32",shape=(4,4))
m2 = placeholder(dtype="float32",shape=(4,4))

result = tensorflow.pow(m1-m2,2)

with tensorflow.Session() as sess:
    print "matrix_pow:{}".format(sess.run(result,feed_dict={m1: numpy.arange(16).astype("float32").reshape(4,4),m2:numpy.random.rand(4,4)}))
