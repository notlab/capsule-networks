import tensorflow as tf
import numpy as np

import capsnet.model as model

def run_once():
    inputs = tf.placeholder(shape=(1, 28, 28, 1), dtype=tf.float32)
    caps = model.capsnet(inputs)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        noise = np.random.rand(1, 28, 28, 1)
        caps_out = sess.run(caps, feed_dict={inputs: noise})
        print(caps_out.shape)

run_once()
