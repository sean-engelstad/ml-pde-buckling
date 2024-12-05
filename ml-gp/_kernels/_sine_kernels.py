import numpy as np
import tensorflow as tf

pi = np.pi

def d2_fact(xbar, p, L):
    # def : xbar = tf.pi * (x - xp) / p
    S = tf.sin(xbar)
    C = tf.cos(xbar)
    return 1.0*pi**2*(S**2 - C**2 + S**2*C**2/L**2)/(L**2*p**2)

def d4_fact(xbar,p, L):
    # def : xbar = tf.pi * (x - xp) / p
    S = tf.sin(xbar)
    C = tf.cos(xbar)
    return pi**4*(-4.0*S**2 + 4.0*C**2 + 3.0*S**4/L**2 - 22.0*S**2*C**2/L**2 + 3.0*C**4/L**2 + 6.0*S**4*C**2/L**4 - 6.0*S**2*C**4/L**4 + 1.0*S**4*C**4/L**6)/(L**2*p**4)

def d6_fact(xbar, p, L):
    # def : xbar = tf.pi * (x - xp) / p
    S = tf.sin(xbar)
    C = tf.cos(xbar)
    return pi**6*(16.0*S**2 - 16.0*C**2 - 60.0*S**4/L**2 + 376.0*S**2*C**2/L**2 - 60.0*C**4/L**2 + 15.0*S**6/L**4 - 345.0*S**4*C**2/L**4 + 345.0*S**2*C**4/L**4 - 15.0*C**6/L**4 + 45.0*S**6*C**2/L**6 - 170.0*S**4*C**4/L**6 + 45.0*S**2*C**6/L**6 + 15.0*S**6*C**4/L**8 - 15.0*S**4*C**6/L**8 + 1.0*S**6*C**6/L**10)/(L**2*p**6)

def d8_fact(xbar, p, L):
    # def : xbar = tf.pi * (x - xp) / p
    S = tf.sin(xbar)
    C = tf.cos(xbar)
    return pi**8*(-64.0*S**2 + 64.0*C**2 + 1008.0*S**4/L**2 - 6112.0*S**2*C**2/L**2 + 1008.0*C**4/L**2 - 840.0*S**6/L**4 + 14616.0*S**4*C**2/L**4 - 14616.0*S**2*C**4/L**4 + 840.0*C**6/L**4 + 105.0*S**8/L**6 - 5460.0*S**6*C**2/L**6 + 16086.0*S**4*C**4/L**6 - 5460.0*S**2*C**6/L**6 + 105.0*C**8/L**6 + 420.0*S**8*C**2/L**8 - 3780.0*S**6*C**4/L**8 + 3780.0*S**4*C**6/L**8 - 420.0*S**2*C**8/L**8 + 210.0*S**8*C**4/L**10 - 644.0*S**6*C**6/L**10 + 210.0*S**4*C**8/L**10 + 28.0*S**8*C**6/L**12 - 28.0*S**6*C**8/L**12 + 1.0*S**8*C**8/L**14)/(L**2*p**8)