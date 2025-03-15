import tensorflow as tf

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.mcmc.internal import leapfrog_integrator as leapfrog_impl

__all__ = [
    'HNNLeapfrogIntegrator',
]


class HNNLeapfrogIntegrator(leapfrog_impl.LeapfrogIntegrator):
    """
    Custom integrator for performing Leapfrog integration using the HNN model.
    
    Replace the traditional target_log_prob_fn, use HNN to calculate the Hamiltonian gradient.
    The Leapfrog steps are implemented based on the logic in run_hnn_hmc.py.
    """
    
    def __init__(self, hnn_model, step_size, num_steps, rho_size=10.0):
        """Initialize the integrator.
        
        Args:
            hnn_model: HNN model instance, used to calculate the Hamiltonian and gradient
            step_size: float scalar or Tensor, representing the leapfrog step size
            num_steps: integer scalar or Tensor, representing the number of leapfrog steps
            rho_size: float scalar, representing the variance scaling factor for momentum sampling
        """
        self._hnn_model = hnn_model
        self._step_size = step_size
        self._num_steps = num_steps
        self._rho_size = rho_size
    
    @property
    def hnn_model(self):
        """Return the HNN model"""
        return self._hnn_model
    
    @property
    def step_size(self):
        """Return the step size parameter"""
        return self._step_size
    
    @property
    def num_steps(self):
        """Return the number of steps parameter"""
        return self._num_steps
    
    @property
    def rho_size(self):
        """Return the rho_size parameter"""
        return self._rho_size
    
    def __call__(self,
                momentum_parts,
                state_parts,
                target=None,
                target_grad_parts=None,
                kinetic_energy_fn=None,
                name=None):
        """Execute the leapfrog integration for num_steps steps.
        
        Args:
            momentum_parts: list of momentum, shape is [13]
            state_parts: list of state, shape is [13]
            target: not used (keep as None)
            target_grad_parts: not used (keep as None)
            kinetic_energy_fn: not used (keep as None)
            name: prefix of the operation name
        
        Returns:
            next_momentum_parts: list of updated momentum
            next_state_parts: list of updated state
            next_target: Hamiltonian of the updated state (scalar)
            next_target_grad_parts: gradient of the updated state
        """
        with tf.name_scope(name or 'hnn_leapfrog_integrate'):
            # extract the state and momentum
            theta = state_parts[0]  # [13]
            rho = momentum_parts[0]  # [13]
            
            # add batch dimension, because HNN expects input with batch dimension
            current_theta = tf.expand_dims(theta, 0)  # [1, 13]
            current_rho = tf.expand_dims(rho, 0)  # [1, 13]
            
            # initialize i
            i = tf.constant(0, dtype=tf.int32)
            
            # define the leapfrog step function
            def body(i, theta, rho):
                # half step update theta
                #_, _, grad_rho = self.hnn_model.compute_gradients(theta, rho)
                #theta = theta + (self.step_size / 2) * grad_rho
                theta = theta + (self.step_size / 2) * rho / self._rho_size
                # full step update rho
                _, grad_theta, _ = self.hnn_model.compute_gradients(theta, rho)
                rho = rho - self.step_size * grad_theta
                
                # half step update theta
                #_, _, grad_rho = self.hnn_model.compute_gradients(theta, rho)
                #theta = theta + (self.step_size / 2) * grad_rho
                theta = theta + (self.step_size / 2) * rho / self._rho_size
                
                return i + 1, theta, rho
            
            # execute num_steps steps of leapfrog integration
            _, next_theta, next_rho = tf.while_loop(
                cond=lambda i, *_: i < self.num_steps,
                body=body,
                loop_vars=[i, current_theta, current_rho]
            )
            
            # calculate the final Hamiltonian and gradient
            H_final, grad_theta_final, grad_rho_final = self.hnn_model.compute_gradients(next_theta, next_rho)
            
            # remove the batch dimension
            next_state_parts = [tf.squeeze(next_theta, 0)]  # [13]
            next_momentum_parts = [tf.squeeze(next_rho, 0)]  # [13]
            next_target = tf.squeeze(H_final, 0)  # scalar
            
            # return the gradient for consistency with TFP interface
            next_target_grad_parts = [
                tf.squeeze(grad_theta_final, 0)  # [13]
            ]
            
            return next_momentum_parts, next_state_parts, next_target, next_target_grad_parts 