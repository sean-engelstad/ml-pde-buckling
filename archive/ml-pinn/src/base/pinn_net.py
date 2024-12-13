__all__ = ["PINN_NeuralNet"]
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, activations

# Define model architecture
class PINN_NeuralNet(tf.keras.Model):
    """ Set basic architecture of the PINN model."""

    def __init__(self, lb, ub, 
            output_dim=1,
            num_hidden_layers=8, 
            num_neurons_per_layer=20,
            activation='tanh',
            kernel_initializer='glorot_normal',
            **kwargs):
        super().__init__(**kwargs)

        self.num_hidden_layers = num_hidden_layers
        self.output_dim = output_dim
        self.lb = lb
        self.ub = ub
        
        # Define NN architecture
        self.scale = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)
        self.hidden = [tf.keras.layers.Dense(num_neurons_per_layer,
                             activation=tf.keras.activations.get(activation),
                             kernel_initializer=kernel_initializer)
                           for _ in range(self.num_hidden_layers)]
        self.out = tf.keras.layers.Dense(output_dim)
        
    def call(self, X):
        """Forward-pass through neural network."""
        Z = self.scale(X)
        for i in range(self.num_hidden_layers):
            Z = self.hidden[i](Z)
        return self.out(Z)

# new version..
# class PINN_NeuralNet(tf.keras.Model):
#     """ Set basic architecture of the PINN model."""

#     def __init__(self, lb, ub, 
#             output_dim=1,
#             num_hidden_layers=8, 
#             num_neurons_per_layer=20,
#             activation='tanh',
#             kernel_initializer='glorot_normal',
#             **kwargs):
#         super().__init__(**kwargs)

#         self.num_hidden_layers = num_hidden_layers
#         self.output_dim = output_dim
#         self.lb = lb
#         self.ub = ub
        
#         # Define NN architecture
#         _layers = [layers.Lambda(
#             lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)]
#         _layers += [
#             layers.Dense(num_neurons_per_layer,
#                                 activation=activations.get(activation),
#                                 kernel_initializer=kernel_initializer)
#             for _ in range(self.num_hidden_layers)
#         ]
#         _layers += [layers.Dense(output_dim)]
#         self.model = Sequential(_layers)
        
#     def call(self, X):
#         return self.model(X)
    
#     # def build(self, input_shape):
#     #     self.model.build(input_shape)
#     #     self.built = True
        
