# less efficient implementation of block using double for loops
# inline for loops helps, but in next version will use something I found online
# of numpy's vectorized operations 
import numpy as np
_temp1 = np.array([[kernel2d(x_all[i,:], x_all[j,:]) for i in range(num_all)] for j in range(num_all)])
K11 = tf.constant(_temp1, dtype=DTYPE)
print("K11 is assembled")
_temp2 = np.array([[kernel2d_bilapl(x_train[i,:], x_all[j,:]) for j in range(num_all)] for i in range(num_interior)])
K12 = tf.constant(_temp2, dtype=DTYPE)
print("K12 is assembled")
_temp3 = np.array([[kernel2d_double_bilapl(x_train[i,:], x_train[j,:]) for j in range(num_interior)] for i in range(num_interior)])
K22 = tf.constant(_temp3, dtype=DTYPE)
print("K22 is assembled")