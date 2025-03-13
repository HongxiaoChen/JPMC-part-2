import collections
import tensorflow as tf

from tensorflow_probability.python.internal import distribute_lib
from tensorflow_probability.python.internal import dtype_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import samplers
from tensorflow_probability.python.mcmc import kernel as kernel_base
from tensorflow_probability.python.mcmc import metropolis_hastings
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from tensorflow.python.util import deprecation  # pylint: disable=g-direct-tensorflow-import

from .pm_leapfrog_integrator import PMLeapfrogIntegrator

__all__ = [
    'PMHMC',
    'UncalibratedPMHMC',
]


class UncalibratedPMHMCKernelResults(
    mcmc_util.PrettyNamedTupleMixin,
    collections.namedtuple(
        'UncalibratedPMHMCKernelResults',
        [
            'log_acceptance_correction',
            'target_log_prob',        # For "next_state".
            'grads_target_log_prob',  # For "next_state".
            'initial_momentum',
            'final_momentum',
            'step_size',
            'num_leapfrog_steps',
            'rho_size',
            'T',
            'N',
            # Seed received by one_step, to reproduce divergent transitions etc.
            'seed',
        ])
    ):
  """Internal state and diagnostics for Uncalibrated PM-HMC."""
  __slots__ = ()


class UncalibratedPMHMC(kernel_base.TransitionKernel):
    """Runs one step of Uncalibrated Pseudo-Marginal Hamiltonian Monte Carlo.

    Pseudo-Marginal Hamiltonian Monte Carlo (PM-HMC) is an extension of HMC 
    used for sampling from joint state spaces with auxiliary variables. It is particularly 
    useful for situations involving marginal likelihood estimation, such as 
    particle-based methods.

    This class implements a single-step sampling of PM-HMC, handling joint state [theta, u_flat], where:
    - theta is the model parameter
    - u_flat is the flattened auxiliary variable

    Key differences between PM-HMC and standard HMC:
    1. Joint state includes parameters theta and auxiliary variables u
    2. Momentum includes parameter momentum rho and auxiliary variable momentum p
    3. Uses a special leapfrog integrator implementing parameter updates and auxiliary variable rotation
    4. Acceptance rate calculation includes three parts of kinetic energy (parameter kinetic energy, 
       auxiliary variable kinetic energy, and auxiliary variable momentum kinetic energy)

    Warning: This kernel does not produce chains that converge to `target_log_prob`. 
    To get a convergent MCMC, use `PMHMC(...)` or 
    `MetropolisHastings(UncalibratedPMHMC(...))`.
    """

    def __init__(self,
                target_log_prob_fn,
                step_size,
                num_leapfrog_steps,
                T,
                N,
                rho_size=10.0,
                state_gradients_are_stopped=False,
                store_parameters_in_results=False,
                name=None):
        """Initialize the PM-HMC transition kernel.

        Args:
          target_log_prob_fn: Python callable which takes a `current_state`-like
            argument (or `*current_state` if it is a list) and returns its
            (possibly unnormalized) log-density under the target distribution.
          step_size: `Tensor` or Python `list` of `Tensor`s representing the step
            size for the leapfrog integrator. Must broadcast with the shape of
            `current_state`. Larger step sizes lead to faster progress, but
            too-large step sizes make rejection exponentially more likely.
          num_leapfrog_steps: Integer number of steps to run the leapfrog integrator
            for. Total progress per HMC step is roughly proportional to
            `step_size * num_leapfrog_steps`.
          T: First dimension of auxiliary variable u
          N: Second dimension of auxiliary variable u
          rho_size: Scaling factor for theta momentum, default is 10.0, affects momentum 
            generation and kinetic energy calculation.
          state_gradients_are_stopped: Python `bool` indicating whether gradients
            with respect to the state should be stopped. This is particularly useful
            when combining optimization with sampling.
            Default value: `False` (i.e., do not stop gradients).
          store_parameters_in_results: If `True`, then `step_size`, `num_leapfrog_steps`,
            and `rho_size` are written to and read from eponymous fields in the
            kernel results objects returned from `one_step` and
            `bootstrap_results`. This allows wrapper kernels to adjust those
            parameters on the fly.
          name: Python `str` name prefixed to Ops created by this function.
            Default value: `None` (i.e., 'pm_hmc_kernel').
        """
        if not store_parameters_in_results:
            mcmc_util.warn_if_parameters_are_not_simple_tensors(
                dict(step_size=step_size, num_leapfrog_steps=num_leapfrog_steps, rho_size=rho_size))
        self._parameters = dict(
            target_log_prob_fn=target_log_prob_fn,
            step_size=step_size,
            num_leapfrog_steps=num_leapfrog_steps,
            T=T,
            N=N,
            rho_size=rho_size,
            state_gradients_are_stopped=state_gradients_are_stopped,
            name=name or 'pm_hmc_kernel',
            store_parameters_in_results=store_parameters_in_results,
        )
        self._momentum_dtype = None

    @property
    def target_log_prob_fn(self):
        return self._parameters['target_log_prob_fn']

    @property
    def step_size(self):
        """Returns the step_size parameter"""
        return self._parameters['step_size']

    @property
    def num_leapfrog_steps(self):
        """Returns the num_leapfrog_steps parameter"""
        return self._parameters['num_leapfrog_steps']

    @property
    def T(self):
        """Returns the first dimension T of auxiliary variable u"""
        return self._parameters['T']
    
    @property
    def N(self):
        """Returns the second dimension N of auxiliary variable u"""
        return self._parameters['N']

    @property
    def rho_size(self):
        """Returns the rho_size parameter, scaling factor for theta momentum"""
        return self._parameters['rho_size']

    @property
    def state_gradients_are_stopped(self):
        return self._parameters['state_gradients_are_stopped']

    @property
    def name(self):
        return self._parameters['name']

    @property
    def parameters(self):
        """Return `dict` of ``__init__`` arguments and their values."""
        return self._parameters

    @property
    def is_calibrated(self):
        return False

    @property
    def _store_parameters_in_results(self):
        return self._parameters['store_parameters_in_results']

    def one_step(self, current_state, previous_kernel_results, seed=None):
        """Runs one iteration of PM-HMC.

        Args:
          current_state: `Tensor` or Python `list` of `Tensor`s representing the
            current state of the Markov chain, containing [theta, u_flat], where 
            theta is the parameter and u_flat is the flattened auxiliary variable.
          previous_kernel_results: `collections.namedtuple` containing `Tensor`s
            representing values from previous calls to this function (or from the
            `bootstrap_results` function).
          seed: PRNG seed; see `tfp.random.sanitize_seed` for details.

        Returns:
          next_state: `Tensor` or Python `list` of `Tensor`s representing the
            state of the Markov chain after taking exactly one step. Has same type
            and shape as `current_state`.
          kernel_results: `collections.namedtuple` of internal calculations used to
            advance the chain.

        Raises:
          ValueError: if there isn't one `step_size` or a list with same length as
            `current_state`.
        """
        with tf.name_scope(mcmc_util.make_name(self.name, 'pm_hmc', 'one_step')):
            if self._store_parameters_in_results:
                step_size = previous_kernel_results.step_size
                num_leapfrog_steps = previous_kernel_results.num_leapfrog_steps
                rho_size = previous_kernel_results.rho_size
                T = previous_kernel_results.T
                N = previous_kernel_results.N
            else:
                step_size = self.step_size
                num_leapfrog_steps = self.num_leapfrog_steps
                rho_size = self.rho_size
                T = self.T
                N = self.N

            [
                current_state_parts,
                step_sizes,
                current_target_log_prob,
                current_target_log_prob_grad_parts,
            ] = _prepare_args(
                self.target_log_prob_fn,
                current_state,
                step_size,
                previous_kernel_results.target_log_prob,
                previous_kernel_results.grads_target_log_prob,
                maybe_expand=True,
                state_gradients_are_stopped=self.state_gradients_are_stopped)

            seed = samplers.sanitize_seed(seed)  # Kept for diagnostics
            theta_seed, u_seed = samplers.split_seed(seed, n=2)

            # Extract parameter and auxiliary variable parts
            theta, u_flat = current_state_parts

            # Generate momentum
            current_theta_momentum = samplers.normal(
                shape=ps.shape(theta),
                dtype=dtype_util.base_dtype(theta.dtype),
                seed=theta_seed) * tf.sqrt(rho_size)

            current_u_momentum = samplers.normal(
                shape=ps.shape(u_flat),
                dtype=dtype_util.base_dtype(u_flat.dtype),
                seed=u_seed)

            current_momentum_parts = [current_theta_momentum, current_u_momentum]

            # Create custom leapfrog integrator, passing T and N parameters
            integrator = PMLeapfrogIntegrator(
                self.target_log_prob_fn, step_sizes, num_leapfrog_steps, T, N, rho_size)

            [
                next_momentum_parts,
                next_state_parts,
                next_target_log_prob,
                next_target_log_prob_grad_parts,
            ] = integrator(
                current_momentum_parts,
                current_state_parts,
                current_target_log_prob,
                current_target_log_prob_grad_parts)

            if self.state_gradients_are_stopped:
                next_state_parts = [tf.stop_gradient(x) for x in next_state_parts]

            def maybe_flatten(x):
                return x if mcmc_util.is_list_like(current_state) else x[0]

            # PM-HMC acceptance rate correction calculation
            # Here we need to consider three kinetic energy terms: theta kinetic energy, 
            # u_flat kinetic energy, and momentum kinetic energy
            independent_chain_ndims = ps.rank(current_target_log_prob)

            # Calculate acceptance rate correction
            log_acceptance_correction = _compute_log_acceptance_correction(
                current_momentum_parts, next_momentum_parts,
                current_state_parts, next_state_parts,
                independent_chain_ndims, rho_size)

            # Create new kernel_results
            new_kernel_results = previous_kernel_results._replace(
                log_acceptance_correction=log_acceptance_correction,
                target_log_prob=next_target_log_prob,
                grads_target_log_prob=next_target_log_prob_grad_parts,
                initial_momentum=current_momentum_parts,
                final_momentum=next_momentum_parts,
                seed=seed,
            )

            return maybe_flatten(next_state_parts), new_kernel_results

    def bootstrap_results(self, init_state):
        """Creates initial `previous_kernel_results` using a supplied `state`."""
        with tf.name_scope(
            mcmc_util.make_name(self.name, 'pm_hmc', 'bootstrap_results')):
            init_state, _ = mcmc_util.prepare_state_parts(init_state)
            
            if self.state_gradients_are_stopped:
                init_state = [tf.stop_gradient(x) for x in init_state]
                
            [
                init_target_log_prob,
                init_grads_target_log_prob,
            ] = mcmc_util.maybe_call_fn_and_grads(self.target_log_prob_fn, init_state)
            
            # Ensure we have two parts: theta and u_flat
            if len(init_state) != 2:
                raise ValueError(
                    "PM-HMC requires state to contain two parts: [theta, u_flat], "
                    f"but received {len(init_state)} parts."
                )
                
            # Initialize results object
            result = UncalibratedPMHMCKernelResults(
                log_acceptance_correction=tf.zeros_like(init_target_log_prob),
                target_log_prob=init_target_log_prob,
                grads_target_log_prob=init_grads_target_log_prob,
                initial_momentum=[
                    tf.zeros_like(init_state[0]),  # theta momentum (rho)
                    tf.zeros_like(init_state[1]),  # u_flat momentum (p_flat)
                ],
                final_momentum=[
                    tf.zeros_like(init_state[0]),  # theta momentum (rho)
                    tf.zeros_like(init_state[1]),  # u_flat momentum (p_flat)
                ],
                step_size=[],
                num_leapfrog_steps=[],
                rho_size=[],
                T=[],
                N=[],
                seed=samplers.zeros_seed())
                
            if self._store_parameters_in_results:
                result = result._replace(
                    step_size=tf.nest.map_structure(
                        lambda x: tf.convert_to_tensor(  # pylint: disable=g-long-lambda
                            x,
                            dtype=init_target_log_prob.dtype,
                            name='step_size'),
                        self.step_size),
                    num_leapfrog_steps=tf.convert_to_tensor(
                        self.num_leapfrog_steps,
                        dtype=tf.int32,
                        name='num_leapfrog_steps'),
                    rho_size=tf.convert_to_tensor(
                        self.rho_size,
                        dtype=init_target_log_prob.dtype,
                        name='rho_size'),
                    T=tf.convert_to_tensor(
                        self.T,
                        dtype=tf.int32,
                        name='T'),
                    N=tf.convert_to_tensor(
                        self.N,
                        dtype=tf.int32,
                        name='N')
                )
                        
            return result


class PMHMC(kernel_base.TransitionKernel):
    """Runs one step of Pseudo-Marginal Hamiltonian Monte Carlo (PM-HMC).

    PM-HMC is an MCMC method specifically designed for models with auxiliary variables. 
    It extends standard HMC by introducing auxiliary variables to handle marginal probability 
    estimation problems, particularly suitable for latent variable models such as 
    state-space models, hidden Markov models, etc.

    Compared to standard HMC, PM-HMC has the following characteristics:
    1. Sampling space includes parameters theta and auxiliary variables u
    2. Uses special leapfrog integrator handling parameter updates and auxiliary variable rotation
    3. Kinetic energy calculation considers multiple parts: parameter kinetic energy, 
       auxiliary variable kinetic energy, and auxiliary variable momentum kinetic energy

    This class implements a complete PM-HMC step, including momentum sampling, 
    leapfrog integration, and Metropolis accept/reject step.
    """

    def __init__(self,
               target_log_prob_fn,
               step_size,
               num_leapfrog_steps,
               T,
               N,
               rho_size=10.0,
               state_gradients_are_stopped=False,
               store_parameters_in_results=False,
               name=None):
        """Initialize the PM-HMC transition kernel.

        Args:
          target_log_prob_fn: Python callable that takes a `current_state`-like
            argument (or `*current_state` if it's a list) and returns its log-density
            under the target distribution.
          step_size: `Tensor` or Python `list` of `Tensor`s representing the step
            size for the leapfrog integrator.
          num_leapfrog_steps: Integer number of steps to run the leapfrog integrator.
          T: First dimension of auxiliary variable u
          N: Second dimension of auxiliary variable u
          rho_size: Scaling factor for theta momentum, default is 10.0
          state_gradients_are_stopped: Python `bool` indicating whether the proposed
            new state should be run through `tf.stop_gradient`.
          store_parameters_in_results: If `True`, parameters are written to kernel results.
          name: Python `str` prefix for Ops created by this function.
        """
        self._impl = metropolis_hastings.MetropolisHastings(
            inner_kernel=UncalibratedPMHMC(
                target_log_prob_fn=target_log_prob_fn,
                step_size=step_size,
                num_leapfrog_steps=num_leapfrog_steps,
                T=T,
                N=N,
                rho_size=rho_size,
                state_gradients_are_stopped=state_gradients_are_stopped,
                store_parameters_in_results=store_parameters_in_results,
                name=name or 'pm_hmc_kernel',
            ))
        self._parameters = self._impl.inner_kernel.parameters.copy()

    @property
    def target_log_prob_fn(self):
        return self._impl.inner_kernel.target_log_prob_fn

    @property
    def step_size(self):
        """Returns the step_size parameter"""
        return self._impl.inner_kernel.step_size

    @property
    def num_leapfrog_steps(self):
        """Returns the num_leapfrog_steps parameter"""
        return self._impl.inner_kernel.num_leapfrog_steps

    @property
    def T(self):
        """Returns the first dimension T of auxiliary variable u"""
        return self._impl.inner_kernel.T
    
    @property
    def N(self):
        """Returns the second dimension N of auxiliary variable u"""
        return self._impl.inner_kernel.N

    @property
    def rho_size(self):
        """Returns the rho_size parameter"""
        return self._impl.inner_kernel.rho_size

    @property
    def state_gradients_are_stopped(self):
        return self._impl.inner_kernel.state_gradients_are_stopped

    @property
    def name(self):
        return self._impl.inner_kernel.name

    @property
    def parameters(self):
        """Return `dict` of ``__init__`` arguments and their values."""
        return self._parameters

    @property
    def is_calibrated(self):
        return True

    def one_step(self, current_state, previous_kernel_results, seed=None):
        """Performs one step of PM-HMC sampling.

        Args:
          current_state: Current state, containing [theta, u_flat]
          previous_kernel_results: Previous kernel results
          seed: Random seed

        Returns:
          next_state: Next state
          kernel_results: Kernel results containing internal calculations
        """
        next_state, kernel_results = self._impl.one_step(
            current_state, previous_kernel_results, seed=seed)
        return next_state, kernel_results

    def bootstrap_results(self, init_state):
        """Creates initial `previous_kernel_results` using a supplied `state`."""
        return self._impl.bootstrap_results(init_state)


def _compute_log_acceptance_correction(current_momentums,
                                       proposed_momentums,
                                       current_state_parts,
                                       proposed_state_parts,
                                       independent_chain_ndims,
                                       rho_size=10.0,
                                       name=None):
    """Helper function to calculate PM-HMC acceptance rate correction
    
    PM-HMC acceptance rate correction considers three parts of kinetic energy:
    1. Parameter kinetic energy: 0.5 * sum(rho^2) / rho_size
    2. Auxiliary variable kinetic energy: 0.5 * sum(u^2)
    3. Auxiliary variable momentum kinetic energy: 0.5 * sum(p^2)

    Args:
        current_momentums: List containing current momentum [rho, p_flat]
        proposed_momentums: List containing proposed momentum [rho_new, p_flat_new]
        current_state_parts: List containing current state [theta, u_flat]
        proposed_state_parts: List containing proposed state [theta_new, u_flat_new]
        independent_chain_ndims: Scalar `int` `Tensor` representing number of independent chains
        rho_size: Scaling factor for parameter momentum
        name: Python `str` prefixed to Ops created by this function

    Returns:
        log_acceptance_correction: `Tensor` representing the log acceptance rate correction
    """
    with tf.name_scope(name or 'compute_log_acceptance_correction'):
        # Extract momentum parts
        current_rho, current_p_flat = current_momentums
        proposed_rho, proposed_p_flat = proposed_momentums
        
        # Extract state parts
        _, current_u_flat = current_state_parts
        _, proposed_u_flat = proposed_state_parts
        
        # Calculate parameter kinetic energy (with mass factor)
        current_rho_kinetic = 0.5 * tf.reduce_sum(
            tf.square(current_rho), 
            axis=ps.range(independent_chain_ndims, ps.rank(current_rho))
        ) / rho_size
        
        proposed_rho_kinetic = 0.5 * tf.reduce_sum(
            tf.square(proposed_rho), 
            axis=ps.range(independent_chain_ndims, ps.rank(proposed_rho))
        ) / rho_size
        
        # Calculate auxiliary variable u kinetic energy - added part
        current_u_kinetic = 0.5 * tf.reduce_sum(
            tf.square(current_u_flat), 
            axis=ps.range(independent_chain_ndims, ps.rank(current_u_flat))
        )
        
        proposed_u_kinetic = 0.5 * tf.reduce_sum(
            tf.square(proposed_u_flat), 
            axis=ps.range(independent_chain_ndims, ps.rank(proposed_u_flat))
        )
        
        # Calculate auxiliary variable momentum p kinetic energy
        current_p_kinetic = 0.5 * tf.reduce_sum(
            tf.square(current_p_flat), 
            axis=ps.range(independent_chain_ndims, ps.rank(current_p_flat))
        )
        
        proposed_p_kinetic = 0.5 * tf.reduce_sum(
            tf.square(proposed_p_flat), 
            axis=ps.range(independent_chain_ndims, ps.rank(proposed_p_flat))
        )
        
        # Calculate total kinetic energy difference, including three parts
        current_kinetic = current_rho_kinetic + current_u_kinetic + current_p_kinetic
        proposed_kinetic = proposed_rho_kinetic + proposed_u_kinetic + proposed_p_kinetic
        
        return current_kinetic - proposed_kinetic


def _prepare_args(target_log_prob_fn,
                 state,
                 step_size,
                 target_log_prob=None,
                 grads_target_log_prob=None,
                 maybe_expand=False,
                 state_gradients_are_stopped=False):
    """Process input arguments to meet list-form assumptions"""
    state_parts, _ = mcmc_util.prepare_state_parts(state, name='current_state')
    
    # Ensure state contains two parts: theta and u_flat
    if len(state_parts) != 2:
        raise ValueError(
            "PM-HMC requires state to contain two parts: [theta, u_flat], "
            f"but received {len(state_parts)} parts."
        )
    
    if state_gradients_are_stopped:
        state_parts = [tf.stop_gradient(x) for x in state_parts]
        
    target_log_prob, grads_target_log_prob = mcmc_util.maybe_call_fn_and_grads(
        target_log_prob_fn, state_parts, target_log_prob, grads_target_log_prob)
    

        
    step_sizes, _ = mcmc_util.prepare_state_parts(
        step_size, dtype=target_log_prob.dtype, name='step_size')
        
    if len(step_sizes) == 1:
        step_sizes = step_sizes * len(state_parts)
        
    if len(state_parts) != len(step_sizes):
        raise ValueError('There should be exactly one `step_size` or a list with same length as `current_state`.')
        
    def maybe_flatten(x):
        return x if maybe_expand or mcmc_util.is_list_like(state) else x[0]
        
    return [
        maybe_flatten(state_parts),
        maybe_flatten(step_sizes),
        target_log_prob,
        grads_target_log_prob,
    ] 