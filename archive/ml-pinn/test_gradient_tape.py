import tensorflow as tf

x = tf.Variable(tf.constant([[1.0], [2.0]]), trainable=True)
y = tf.Variable(tf.constant([[1.0], [2.0]]), trainable=True)

with tf.GradientTape(persistent=True) as tape3:
    with tf.GradientTape(persistent=True) as tape2:
        with tf.GradientTape(persistent=True) as tape1:
            w = tf.reduce_sum(x ** 2 + y ** 2)
            dx = tape1.gradient(w, x)
            dy = tape1.gradient(w, y)
        d2x = tape2.gradient(dx, x)
        d2y = tape2.gradient(dy, y)
    d3x = tape3.gradient(d2x, x)
    d3y = tape3.gradient(d2y, y)
d4x = tape3.gradient(d3x, x)
d4y = tape3.gradient(d3y, y)
print(f"{dx=}, {d2x=}, {d3x=}, {d4x=}")