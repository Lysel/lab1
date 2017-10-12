import csv
import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# input/output files
in_file = "c:/dlclass/lab1/exercise/part2/lab1_pt2_data.txt"
out_file = "c:/dlclass/lab1/exercise/part2/lab1_pt2_output.txt"

# input polynomial factor
k = 4

a = tf.placeholder(dtype=tf.float32, name="a")
b = tf.placeholder(dtype=tf.float32, name="poly")
sol = tf.pow(a, b, name="solution")

file_writer = tf.summary.FileWriter("c:/dlclass/lab1/exercise/part2", tf.get_default_graph())

sess = tf.Session()

# perform transformation
with open(in_file) as d:
    reader = csv.reader(d)
    k_list = []
    val_list = []
    for row in reader:
        val_list.append(row)
    for i in range(1, k+1):
        k_list.append(i)
    _a, _b, out_sol = sess.run(fetches=[a, b, sol], feed_dict={a: val_list, b:k_list})

# format output to floats rounded 5 decimal places
out_list = out_sol.tolist()
new_out = []
for l in out_list:
    l_str = []
    for v in l:
        l_str.append("%.5f" %v)
    new_out.append(l_str)

print (new_out)

# save output to text file
with open(out_file, 'w') as out:
    for j in new_out:
        for k in j:
            out.write(k + " ")
        out.write("\n")




