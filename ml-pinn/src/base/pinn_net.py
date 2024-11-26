__all__ = ["PINN_NeuralNet"]
import tensorflow as tf

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
        layers = [tf.keras.layers.Lambda(
            lambda x: 2.0*(x - lb)/(ub - lb) - 1.0)]
        layers += [
            tf.keras.layers.Dense(num_neurons_per_layer,
                                activation=tf.keras.activations.get(activation),
                                kernel_initializer=kernel_initializer)
            for _ in range(self.num_hidden_layers)
        ]
        layers += [tf.keras.layers.Dense(output_dim)]
        self.model = tf.keras.Sequential(layers)
        
    def call(self, X):
        return self.model(X)
    
    # def build(self, input_shape):
    #     self.model.build(input_shape)
    #     self.built = True
        
