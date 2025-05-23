import tensorflow as tf

from tensorflow_probability.python.internal import tensorshape_util
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow_probability.python.mcmc.internal import leapfrog_integrator as leapfrog_impl

__all__ = [
    'PMLeapfrogIntegrator',
]


class PMLeapfrogIntegrator(leapfrog_impl.SimpleLeapfrogIntegrator):
    """Leapfrog integrator for PM-HMC algorithm.
    
    This integrator extends SimpleLeapfrogIntegrator to handle the joint state
    (theta + u) and momentum (rho + p) in PM-HMC. It implements the leapfrog steps
    with specific handling for parameter rotation and auxiliary variable rotation.
    
    PM-HMC leapfrog steps include:
    1. Half step A: Update theta and rotate (u,p)
    2. Full step B: Update momentum using gradients
    3. Half step A: Update theta again and rotate (u,p)
    """
    
    def __init__(self, target_fn, step_sizes, num_steps, rho_size=10.0):
        """Initialize PMLeapfrogIntegrator

        Args:
          target_fn: Target function that accepts joint state [theta, u] and returns log probability
          step_sizes: Step size for leapfrog integrator
          num_steps: Number of leapfrog integration steps
          rho_size: Scaling factor for theta momentum, default is 10.0
        """
        super(PMLeapfrogIntegrator, self).__init__(
            target_fn=target_fn,
            step_sizes=step_sizes,
            num_steps=num_steps
        )
        self._rho_size = rho_size
    
    @property
    def rho_size(self):
        """Returns rho_size parameter"""
        return self._rho_size
    
    def __call__(self,
                momentum_parts,
                state_parts,
                target=None,
                target_grad_parts=None,
                kinetic_energy_fn=None,
                name=None):
        """Execute PM-HMC leapfrog steps
        
        Args:
          momentum_parts: List containing two Tensors: [rho, p]
              rho: Momentum for theta, shape [D]
              p: Momentum for u, shape [T, N]
          state_parts: List containing two Tensors: [theta, u]
              theta: Parameters, shape [D]
              u: Auxiliary variables, shape [T, N]
          target: Scalar Tensor representing joint state log probability
          target_grad_parts: Gradient list [grad_theta, grad_u]
          kinetic_energy_fn: Kinetic energy function (optional)
          name: Operation name
        
        Returns:
          next_momentum_parts: [rho_new, p_new], updated momentum
          next_state_parts: [theta_new, u_new], updated state
          next_target: Updated log probability
          next_target_grad_parts: [grad_theta_new, grad_u_new], updated gradients
        """
        with tf.name_scope(name or 'pm_leapfrog_integrate'):
            # Process input parameters
            [
                momentum_parts,
                state_parts,
                target,
                target_grad_parts,
            ] = leapfrog_impl.process_args(
                self.target_fn,
                momentum_parts,
                state_parts,
                target,
                target_grad_parts)

            
            # Extract parameters and auxiliary variables
            theta, u = state_parts[0], state_parts[1]
            rho, p = momentum_parts[0], momentum_parts[1]
            
            # Initialize result variables
            current_theta, current_u = theta, u
            current_rho, current_p = rho, p
            current_target, current_target_grad_parts = target, target_grad_parts
            
            step_size = self.step_sizes[0]  # Assume all variables use the same step size
            half_step = 0.5 * step_size
            
            # cos and sin terms for auxiliary variable rotation
            cos_term = tf.cos(half_step)
            sin_term = tf.sin(half_step)
            
            # Execute num_steps complete Leapfrog steps
            for step_idx in range(self.num_steps):
                
                # Half step A: Update position (theta, u)
                               
                # Apply half-step update to theta
                half_theta = current_theta + _multiply(half_step * (1.0 / self.rho_size), current_rho, dtype=current_theta.dtype)
                
                # Apply rotation to u and p directly
                half_u = current_u * cos_term + current_p * sin_term
                half_p = current_p * cos_term - current_u * sin_term
                
                
                # Full step B: Update momentum (rho, p)
                               
                # Calculate gradients (at half-step position)
                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(half_theta)
                    tape.watch(half_u)
                    half_state = [half_theta, half_u]
                    half_target = self.target_fn(*half_state)
                
                # Calculate gradients
                grad_theta = tape.gradient(half_target, half_theta)
                grad_u = tape.gradient(half_target, half_u)
                del tape
                
                
                # Update momentum (full step)
                new_rho = current_rho + _multiply(step_size, grad_theta, dtype=current_rho.dtype)
                new_p = half_p + _multiply(step_size, grad_u, dtype=half_p.dtype)
                               
                # Half step A: Update position (theta, u)
                               
                # Apply half-step update to theta
                new_theta = half_theta + _multiply(half_step * (1.0 / self.rho_size), new_rho, dtype=half_theta.dtype)
                
                # Apply rotation to u and p
                new_u = half_u * cos_term + new_p * sin_term
                new_p = new_p * cos_term - half_u * sin_term
                
                # Update current state and momentum
                current_theta, current_u = new_theta, new_u
                current_rho, current_p = new_rho, new_p
            
            # Calculate final gradients and target function
            next_state_parts = [current_theta, current_u]
            next_momentum_parts = [current_rho, current_p]
            
            # Calculate new log probability and gradients
            next_target, next_target_grad_parts = mcmc_util.maybe_call_fn_and_grads(
                self.target_fn, next_state_parts)
            
            return (
                next_momentum_parts,
                next_state_parts,
                next_target,
                next_target_grad_parts,
            )


def _multiply(tensor, state_sized_tensor, dtype):
    """Multiply `tensor` by a "state sized" tensor and preserve shape."""
    # User should be using a step size that does not alter the state size. This
    # will fail noisily if that is not the case.
    result = tf.cast(tensor, dtype) * tf.cast(state_sized_tensor, dtype)
    tensorshape_util.set_shape(result, state_sized_tensor.shape)
    return result 