
__all__ = ["PINN_EigenNet", "StrongFormBucklingPINN"]
import tensorflow as tf
from .base.dtype import DTYPE
from .base import PINN_NeuralNet, PINNSolver

class PINN_EigenNet(PINN_NeuralNet):
    """
    A PINN neural network model, except it also has a trainable variable for the eigenvalue.
    This allows learning an eigenvalue PDE not just a regular PDE.
    """
    def __init__(self, *args, **kwargs):
        
        # Call init of base class
        super().__init__(*args,**kwargs)
        
        # Initialize variable for lambda
        self.lambd = tf.Variable(1.0, trainable=True, dtype=DTYPE)
        self.lambd_list = []

class StrongFormBucklingPINN(PINNSolver):
    def __init__(self, D:float, Nxx:float, Nxy:float, Nyy:float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.D = tf.constant(D, dtype=DTYPE)
        self.Nxx = tf.constant(Nxx, dtype=DTYPE)
        self.Nxy = tf.constant(Nxy, dtype=DTYPE)
        self.Nyy = tf.constant(Nyy, dtype=DTYPE)
    
    def fun_r(self, x, y, d4x, d2x2y, d4y, d2x, dxy, d2y):
        """Residual of the PDE"""
        return self.D * (d4x + 2. * d2x2y + d4y) - self.model.lambd * \
            (self.Nxx * d2x + 2. * self.Nxy * dxy + self.Nyy * d2y)
    
    @tf.function # forces tensorflow to compile the graph statically
    def get_r(self):
        """For buckling PINN strong form"""

        # with tf.GradientTape(persistent=True) as tape4:
        #     tape4.watch([self.x, self.y])  # Watch both variables
        with tf.GradientTape(persistent=True) as tape3:
            tape3.watch([self.x, self.y])
            with tf.GradientTape(persistent=True) as tape2:
                tape2.watch([self.x, self.y])
                with tf.GradientTape(persistent=True) as tape1:
                    tape1.watch([self.x, self.y])
                    w = self.model(tf.stack([self.x[:, 0], self.y[:, 0]], axis=1))

        dx = tape1.gradient(w, self.x)
        dy = tape1.gradient(w, self.y)
        d2x = tape2.gradient(dx, self.x)
        d2y = tape2.gradient(dy, self.y)
        dxdy = tape2.gradient(dx, self.y)

        

        d3x = tape3.gradient(d2x, self.x)
        d2xdy = tape3.gradient(d2x, self.y)
        d3y = tape3.gradient(d2y, self.y)

        print(f"{dx=} {dy=} {d2x=} {d3x=}")

        # Compute d4x, d4y, and d2xd2y AFTER exiting all GradientTape contexts
        del tape1, tape2
        d4x = tape3.gradient(d3x, self.x)
        d4y = tape3.gradient(d3y, self.y)
        d2xd2y = tape3.gradient(d2xdy, self.y)
        del tape3
        print(f"{d4x=}, {d4y=}, {d2xd2y=}")
        

        # Cleanup resources
        del tape1, tape2, tape3 # , tape4        

        return self.fun_r(self.x, self.y, d4x, d2xdy, d4y, d2x, dxdy, d2y)
    
    def callback(self, xr=None):
        lambd = self.model.lambd.numpy()
        self.model.lambd_list.append(lambd)
        
        if self.iter % 50 == 0:
            print('It {:05d}: loss = {:10.8e} lambda = {:10.8e}'.format(self.iter, self.current_loss, lambd))
        
        self.hist.append(self.current_loss)
        self.iter += 1
        
    def plot_loss_and_param(self, axs=None):
        if axs:
            ax1, ax2 = axs
            self.plot_loss_history(ax1)
        else:
            ax1 = self.plot_loss_history()
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.plot(range(len(self.hist)), self.model.lambd_list,'-',color=color)
        ax2.set_ylabel('$\\lambda^{n_{epoch}}$', color=color)
        return (ax1,ax2)