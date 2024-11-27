__all__ = ["DTYPE"]

import tensorflow as tf

DTYPE='float32' # default
# DTYPE='float64' # for higher-order derivatives
tf.keras.backend.set_floatx(DTYPE)