import collections
import tensorflow as tf
import tensorflow.compat.v2 as tf

from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc.internal import util as mcmc_util

from .hnn_leapfrog_integrator_original import HNNLeapfrogIntegrator

__all__ = [
    'HNNHamiltonianMonteCarlo',
    'UncalibratedHNNHMC',
]


class UncalibratedHNNHMCKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'UncalibratedHNNHMCKernelResults',
        [
            'log_acceptance_correction',
            'target_log_prob',        # negative Hamiltonian of the current state
            'grads_target_log_prob',  # gradient of the current state
            'initial_momentum',
            'final_momentum',
            'step_size',
            'num_leapfrog_steps',
            'rho_size',
            'seed',
        ])
    ):
  """Uncalibrated HNN-HMC kernel results"""
  __slots__ = ()


class UncalibratedHNNHMC(kernel_base.TransitionKernel):
    """Uncalibrated HNN-HMC transition kernel, using HNN model for Hamiltonian Monte Carlo sampling.
    
    Warning: This kernel will not produce a chain that converges to the target distribution.
    To obtain a convergent MCMC, please use HNNHamiltonianMonteCarlo(...) or MetropolisHastings(UncalibratedHNNHMC(...)).
    
    This implementation is based on the HNN model to calculate the Hamiltonian and gradient,
    using the negative Hamiltonian as the target log probability.
    """
    
    def __init__(self,
                 hnn_model,
                 step_size,
                 num_leapfrog_steps,
                 rho_size=10.0,
                 state_gradients_are_stopped=False,
                 store_parameters_in_results=False,
                 name=None):
        """Initialize the Uncalibrated HNN-HMC kernel.
        
        Args:
            hnn_model: HNN model instance, used to calculate the Hamiltonian and gradient
            step_size: float scalar or Tensor, representing the leapfrog step size
            num_leapfrog_steps: integer scalar or Tensor, representing the number of leapfrog steps
            rho_size: float scalar, representing the variance scaling factor for momentum sampling
            state_gradients_are_stopped: boolean, representing whether to stop state gradients
            store_parameters_in_results: boolean, whether to store parameters in kernel results
            name: prefix of the operation name
        """
        if not store_parameters_in_results:
            mcmc_util.warn_if_parameters_are_not_simple_tensors(
                dict(step_size=step_size, num_leapfrog_steps=num_leapfrog_steps, rho_size=rho_size))
        
        self._parameters = dict(
            hnn_model=hnn_model,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            rho_size=rho_size,
            state_gradients_are_stopped=state_gradients_are_stopped,
            name=name or 'uncalibrated_hnn_hmc',
            store_parameters_in_results=store_parameters_in_results,
        )
        self._momentum_dtype = None
        
        # initialize the custom leapfrog integrator
        self._integrator = HNNLeapfrogIntegrator(
            hnn_model=hnn_model,
            step_size=step_size,
            num_steps=num_leapfrog_steps,
            rho_size=rho_size
        )
    
    @property
    def hnn_model(self):
        """HNN model"""
        return self._parameters['hnn_model']
    
    @property
    def step_size(self):
        """step size"""
        return self._parameters['step_size']
    
    @property
    def num_leapfrog_steps(self):
        """number of leapfrog steps"""
        return self._parameters['num_leapfrog_steps']
    
    @property
    def rho_size(self):
        """rho_size"""
        return self._parameters['rho_size']
    
    @property
    def state_gradients_are_stopped(self):
        """state_gradients_are_stopped"""
        return self._parameters['state_gradients_are_stopped']
    
    @property
    def name(self):
        """name"""
        return self._parameters['name']
    
    @property
    def parameters(self):
        """parameters"""
        return self._parameters
    
    @property
    def is_calibrated(self):
        """is calibrated"""
        return False
    
    @property
    def _store_parameters_in_results(self):
        """store parameters in results"""
        return self._parameters['store_parameters_in_results']
    
    def _compute_neg_hamiltonian(self, state_parts, momentum_parts):
        """compute the negative Hamiltonian (used as target_log_prob)
        
        Args:
            state_parts: list of state, shape is [13]
            momentum_parts: list of momentum, shape is [13]
            
        Returns:
            negative Hamiltonian value (scalar)
        """
        # add batch dimension
        theta = tf.expand_dims(state_parts[0], 0)  # [1, 13]
        rho = tf.expand_dims(momentum_parts[0], 0)  # [1, 13]
        
        # compute Hamiltonian
        H, _, _ = self.hnn_model.compute_gradients(theta, rho)
        
        # return negative Hamiltonian (-H as log_prob)
        return -tf.squeeze(H, 0)
    
    def one_step(self, current_state, previous_kernel_results, seed=None):
        """execute one step of Uncalibrated HNN-HMC sampling
        
        Args:
            current_state: current state Tensor, shape is [13]
            previous_kernel_results: previous kernel results
            seed: random seed
            
        Returns:
            next_state: next state
            kernel_results: updated kernel results
        """
        with tf.name_scope(mcmc_util.make_name(self.name, 'hnn_hmc', 'one_step')):
            # get parameters
            if self._store_parameters_in_results:
                step_size = previous_kernel_results.step_size
                num_leapfrog_steps = previous_kernel_results.num_leapfrog_steps
                rho_size = previous_kernel_results.rho_size
            else:
                step_size = self.step_size
                num_leapfrog_steps = self.num_leapfrog_steps
                rho_size = self.rho_size
            
            # prepare current state
            current_state_parts = [tf.convert_to_tensor(current_state, dtype=tf.float32)]
            
            # handle random seed
            seed = samplers.sanitize_seed(seed)
            rho_seed = samplers.split_seed(seed)[0]
            
            # sample momentum
            current_momentum_parts = [
                samplers.normal(
                    shape=ps.shape(current_state),
                    dtype=current_state.dtype,
                    seed=rho_seed
                ) * tf.sqrt(rho_size)
            ]
            
            # use custom integrator to execute leapfrog steps
            [
                next_momentum_parts,
                next_state_parts,
                next_hamiltonian,
                next_target_grad_parts
            ] = self._integrator(
                current_momentum_parts,
                current_state_parts
            )
            
            # compute the negative Hamiltonian of the new state
            next_target_log_prob = -next_hamiltonian
            
            # set log_acceptance_correction for Metropolis-Hastings step
            # since we use -H as target log probability, the correction is 0
            log_acceptance_correction = tf.constant(0., dtype=current_state.dtype)
            
            # if needed, apply stop_gradient
            if self.state_gradients_are_stopped:
                next_state_parts = [tf.stop_gradient(x) for x in next_state_parts]
            
            def maybe_flatten(x):
                return x if mcmc_util.is_list_like(current_state) else x[0]
            
            # create kernel results
            new_kernel_results = previous_kernel_results._replace(
                log_acceptance_correction=log_acceptance_correction,
                target_log_prob=next_target_log_prob,
                grads_target_log_prob=next_target_grad_parts,
                initial_momentum=current_momentum_parts,
                final_momentum=next_momentum_parts,
                seed=seed,
            )
            
            return maybe_flatten(next_state_parts), new_kernel_results
    
    def bootstrap_results(self, init_state):
        """initialize kernel results
        
        Args:
            init_state: initial state, shape is [13]
            
        Returns:
            initialized kernel results
        """
        with tf.name_scope(mcmc_util.make_name(self.name, 'hnn_hmc', 'bootstrap_results')):
            # ensure the input is a list
            init_state_parts, _ = mcmc_util.prepare_state_parts(init_state)
            
            if self.state_gradients_are_stopped:
                init_state_parts = [tf.stop_gradient(x) for x in init_state_parts]
            
            # initialize zero momentum
            init_momentum_parts = [tf.zeros_like(x) for x in init_state_parts]
            
            # compute the negative Hamiltonian of the initial state
            init_target_log_prob = self._compute_neg_hamiltonian(
                init_state_parts, init_momentum_parts
            )
            
            # compute the initial gradient
            theta = tf.expand_dims(init_state_parts[0], 0)
            rho = tf.expand_dims(init_momentum_parts[0], 0)
            _, grad_theta, _ = self.hnn_model.compute_gradients(theta, rho)
            init_grads = [tf.squeeze(grad_theta, 0)]
            
            # create the result object
            result = UncalibratedHNNHMCKernelResults(
                log_acceptance_correction=tf.constant(0., dtype=init_state_parts[0].dtype),
                target_log_prob=init_target_log_prob,
                grads_target_log_prob=init_grads,
                initial_momentum=init_momentum_parts,
                final_momentum=init_momentum_parts,
                step_size=[],
                num_leapfrog_steps=[],
                rho_size=[],
                seed=samplers.zeros_seed()
            )
            
            # if needed, store parameters
            if self._store_parameters_in_results:
                result = result._replace(
                    step_size=tf.nest.map_structure(
                        lambda x: tf.convert_to_tensor(
                            x, dtype=init_target_log_prob.dtype, name='step_size'),
                        self.step_size),
                    num_leapfrog_steps=tf.convert_to_tensor(
                        self.num_leapfrog_steps, dtype=tf.int32, name='num_leapfrog_steps'),
                    rho_size=tf.convert_to_tensor(
                        self.rho_size, dtype=init_target_log_prob.dtype, name='rho_size')
                )
            
            return result


class HNNHamiltonianMonteCarlo(kernel_base.TransitionKernel):
    """HNN-HMC kernel based on HNN.
    
    Use HNN to calculate Hamiltonian and gradient, instead of the traditional target_log_prob_fn.
    This kernel uses MetropolisHastings wrapper to handle the accept/reject step.
    """
    
    def __init__(self,
                 hnn_model,
                 step_size,
                 num_leapfrog_steps,
                 rho_size=10.0,
                 state_gradients_are_stopped=False,
                 store_parameters_in_results=False,
                 name=None):
        """initialize HNN-HMC kernel.
        
        Args:
            hnn_model: HNN model instance, used to calculate the Hamiltonian and gradient
            step_size: float scalar or Tensor, representing the leapfrog step size
            num_leapfrog_steps: integer scalar or Tensor, representing the number of leapfrog steps
            rho_size: float scalar, representing the variance scaling factor for momentum sampling
            state_gradients_are_stopped: boolean, representing whether to stop state gradients
            store_parameters_in_results: boolean, whether to store parameters in kernel results
            name: prefix of the operation name
        """
        # use MetropolisHastings wrapper
        self._impl = metropolis_hastings.MetropolisHastings(
            inner_kernel=UncalibratedHNNHMC(
                hnn_model=hnn_model,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
                rho_size=rho_size,
                state_gradients_are_stopped=state_gradients_are_stopped,
                store_parameters_in_results=store_parameters_in_results,
                name=name or 'hnn_hmc',
            ))
        self._parameters = self._impl.inner_kernel.parameters.copy()
    
    @property
    def hnn_model(self):
        """return HNN model"""
        return self._impl.inner_kernel.hnn_model
    
    @property
    def step_size(self):
        """return step size"""
        return self._impl.inner_kernel.step_size
    
    @property
    def num_leapfrog_steps(self):
        """return number of leapfrog steps"""
        return self._impl.inner_kernel.num_leapfrog_steps
    
    @property
    def rho_size(self):
        """return rho_size"""
        return self._impl.inner_kernel.rho_size
    
    @property
    def state_gradients_are_stopped(self):
        """return state_gradients_are_stopped"""
        return self._impl.inner_kernel.state_gradients_are_stopped
    
    @property
    def name(self):
        """return name"""
        return self._impl.inner_kernel.name
    
    @property
    def parameters(self):
        """return parameters"""
        return self._parameters
    
    @property
    def is_calibrated(self):
        """return whether the kernel is calibrated, HMC is a calibrated kernel"""
        return True
    
    def one_step(self, current_state, previous_kernel_results, seed=None):
        """execute one step of HNN-HMC sampling
        
        Args:
            current_state: current state Tensor, shape is [13]
            previous_kernel_results: previous kernel results
            seed: random seed
            
        Returns:
            next_state: next state
            kernel_results: updated kernel results
        """
        return self._impl.one_step(
            current_state, previous_kernel_results, seed=seed)
    
    def bootstrap_results(self, init_state):
        """initialize kernel results
        
        Args:
            init_state: initial state, shape is [13]
            
        Returns:
            initialized kernel results
        """
        return self._impl.bootstrap_results(init_state) 