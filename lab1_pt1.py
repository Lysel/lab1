import sys
import tensorflow as tf
import numpy as np

w = tf.get_variable(name="w", dtype=tf.float32, initializer=tf.constant([0.0, 0.0]))
a = tf.placeholder(dtype=tf.float32, name="a")
b = tf.placeholder(dtype=tf.float32, name="b")
lr = tf.placeholder(dtype=tf.float32, name="lr")

feed_params = {a: sys.argv[1], b: sys.argv[2], lr: sys.argv[3]}

obj = tf.add(tf.square(a-w[0]), b*tf.square(w[1]-tf.square(w[0])))

def select_optimizer(opt_arg):
    if opt_arg == 'gd':
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(obj, name="gd")
    elif opt_arg == 'gdm':
        train_step = tf.train.MomentumOptimizer(lr, momentum=0.9).minimize(obj, name="gdm")
    elif opt_arg == 'adam':
        train_step = tf.train.AdamOptimizer(lr).minimize(obj, name="adam")
    return train_step

train_step = select_optimizer(sys.argv[4])

init = tf.global_variables_initializer()

# file_writer = tf.summary.FileWriter("c:/dlclass/lab1/exercise", tf.get_default_graph())

sess = tf.Session()
sess.run(fetches=[init])

# loop preset conditions
i=0
min_check = True
inf_check = True
nan_check = True

while all([min_check, inf_check, nan_check]):
    _, get_w, obj_val = sess.run(fetches=[train_step, w, obj], feed_dict=feed_params)
    print(str(i) + ": " + str(obj_val))
    min_check = obj_val > 0.0001
    inf_check = not(np.isinf(obj_val))
    nan_check = not(np.isnan(obj_val))
    i = i+1