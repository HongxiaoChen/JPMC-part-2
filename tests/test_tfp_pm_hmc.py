import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'codes'))

import unittest
import tensorflow as tf


from codes.tfp_modified_kernels.pm_hmc import UncalibratedPMHMC, PMHMC
from codes.tfp_modified_kernels.pm_leapfrog_integrator import PMLeapfrogIntegrator



class TestPMHMC(unittest.TestCase):
    """Test suite for PM-HMC sampling functionality"""

    def setUp(self):
        """Set up test environment"""
        # Initialize test data
        self.T = 10  # Time steps
        self.N = 5   # Auxiliary variable dimension
        self.n = 3   # Observation dimension
        self.p_z = 2  # Covariate dimension
        
        # Create simulation data
        self.theta = tf.random.normal([13])  # Parameter vector
        self.u = tf.random.normal([self.T, self.N])  # Auxiliary variable
        self.y = tf.random.normal([self.T, self.n])  # Observations
        self.Z = tf.random.normal([self.T, self.n, self.p_z])  # Covariates
        
        # Set sampling parameters
        self.step_size = 0.01
        self.num_leapfrog_steps = 5
        self.rho_size = 10.0
        
        # Define target function
        self.target_log_prob_fn = lambda theta, u: -0.5 * tf.reduce_sum(tf.square(theta)) - 0.5 * tf.reduce_sum(tf.square(u))

    def test_uncalibrated_pmhmc_initialization(self):
        """Test initialization of UncalibratedPMHMC kernel"""
        # Initialize UncalibratedPMHMC kernel
        kernel = UncalibratedPMHMC(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=self.step_size,
            num_leapfrog_steps=self.num_leapfrog_steps,
            rho_size=self.rho_size
        )
        
        # Check if parameters are set correctly
        self.assertEqual(kernel.step_size, self.step_size)
        self.assertEqual(kernel.num_leapfrog_steps, self.num_leapfrog_steps)
        self.assertEqual(kernel.rho_size, self.rho_size)
        self.assertEqual(kernel.target_log_prob_fn, self.target_log_prob_fn)
        self.assertFalse(kernel.is_calibrated)

    def test_pmhmc_initialization(self):
        """Test initialization of PMHMC kernel"""
        # Initialize PMHMC kernel
        kernel = PMHMC(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=self.step_size,
            num_leapfrog_steps=self.num_leapfrog_steps,
            rho_size=self.rho_size
        )
        
        # Check if parameters are set correctly
        self.assertEqual(kernel.step_size, self.step_size)
        self.assertEqual(kernel.num_leapfrog_steps, self.num_leapfrog_steps)
        self.assertEqual(kernel.rho_size, self.rho_size)
        self.assertEqual(kernel.target_log_prob_fn, self.target_log_prob_fn)
        self.assertTrue(kernel.is_calibrated)

    def test_bootstrap_results(self):
        """Test bootstrap_results method"""
        # Initialize UncalibratedPMHMC kernel
        kernel = UncalibratedPMHMC(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=self.step_size,
            num_leapfrog_steps=self.num_leapfrog_steps,
            rho_size=self.rho_size,
            store_parameters_in_results=True
        )
        
        # Create initial state
        init_state = [self.theta, self.u]
        
        # Execute bootstrap_results
        results = kernel.bootstrap_results(init_state)
 
        
        # Verify that momentum parts have two components: rho and p
        self.assertEqual(len(results.initial_momentum), 2)
        self.assertEqual(results.initial_momentum[0].shape, self.theta.shape)
        self.assertEqual(results.initial_momentum[1].shape, self.u.shape)

    def test_one_step(self):
        """Test one_step method"""
        # Initialize UncalibratedPMHMC kernel
        kernel = UncalibratedPMHMC(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=self.step_size,
            num_leapfrog_steps=self.num_leapfrog_steps,
            rho_size=self.rho_size
        )
        
        # Create initial state and results
        init_state = [self.theta, self.u]
        init_results = kernel.bootstrap_results(init_state)
        
        # Execute one step sampling
        next_state, next_results = kernel.one_step(init_state, init_results, seed=42)
        
        # Check output shapes
        self.assertEqual(len(next_state), 2)
        self.assertEqual(next_state[0].shape, self.theta.shape)
        self.assertEqual(next_state[1].shape, self.u.shape)
        
        # Check if results contain necessary fields
        self.assertIn('log_acceptance_correction', next_results._fields)
        self.assertIn('target_log_prob', next_results._fields)
        self.assertIn('grads_target_log_prob', next_results._fields)
        self.assertIn('initial_momentum', next_results._fields)
        self.assertIn('final_momentum', next_results._fields)
        
        # Check if momentum parts are correct
        self.assertEqual(len(next_results.initial_momentum), 2)
        self.assertEqual(len(next_results.final_momentum), 2)
        self.assertEqual(next_results.initial_momentum[0].shape, self.theta.shape)
        self.assertEqual(next_results.initial_momentum[1].shape, self.u.shape)

    def test_pmhmc_step(self):
        """Test PMHMC step (including accept/reject)"""
        # Initialize PMHMC kernel
        kernel = PMHMC(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=self.step_size,
            num_leapfrog_steps=self.num_leapfrog_steps,
            rho_size=self.rho_size
        )
        
        # Create initial state and results
        init_state = [self.theta, self.u]
        init_results = kernel.bootstrap_results(init_state)
        
        # Execute one step sampling
        next_state, next_results = kernel.one_step(init_state, init_results, seed=42)
        
        # Check output shapes
        self.assertEqual(len(next_state), 2)
        self.assertEqual(next_state[0].shape, self.theta.shape)
        self.assertEqual(next_state[1].shape, self.u.shape)


    def test_pm_leapfrog_integrator(self):
        """Test PMLeapfrogIntegrator"""
        # Initialize integrator
        integrator = PMLeapfrogIntegrator(
            target_fn=self.target_log_prob_fn,
            step_sizes=[self.step_size, self.step_size],
            num_steps=self.num_leapfrog_steps,
            rho_size=self.rho_size
        )
        
        # Create momentum and state
        rho = tf.random.normal([13])
        p = tf.random.normal([self.T, self.N])
        theta = tf.random.normal([13])
        u = tf.random.normal([self.T, self.N])
        
        # Get initial target function value and gradients
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(theta)
            tape.watch(u)
            target = self.target_log_prob_fn(theta, u)
        
        grad_theta = tape.gradient(target, theta)
        grad_u = tape.gradient(target, u)
        
        # Execute integrator
        next_momentum_parts, next_state_parts, next_target, next_target_grad_parts = integrator(
            momentum_parts=[rho, p],
            state_parts=[theta, u],
            target=target,
            target_grad_parts=[grad_theta, grad_u]
        )
        
        # Check output shapes
        self.assertEqual(len(next_momentum_parts), 2)
        self.assertEqual(len(next_state_parts), 2)
        self.assertEqual(len(next_target_grad_parts), 2)
        
        self.assertEqual(next_momentum_parts[0].shape, rho.shape)
        self.assertEqual(next_momentum_parts[1].shape, p.shape)
        self.assertEqual(next_state_parts[0].shape, theta.shape)
        self.assertEqual(next_state_parts[1].shape, u.shape)
        
        # Check gradient shapes
        self.assertEqual(next_target_grad_parts[0].shape, grad_theta.shape)
        self.assertEqual(next_target_grad_parts[1].shape, grad_u.shape)

    def test_hamiltonian_conservation(self):
        """Test Hamiltonian conservation (with different step sizes)"""
        # Create initial state and momentum
        theta = tf.random.normal([13])
        u = tf.random.normal([self.T, self.N])
        rho = tf.random.normal([13])
        p = tf.random.normal([self.T, self.N])
        
        # Calculate initial Hamiltonian
        with tf.GradientTape() as tape:
            tape.watch(theta)
            tape.watch(u)
            target = self.target_log_prob_fn(theta, u)
        
        initial_H = target - (0.5 * tf.reduce_sum(tf.square(rho)) / self.rho_size + 
                           0.5 * tf.reduce_sum(tf.square(p)) + 
                           0.5 * tf.reduce_sum(tf.square(u)))
        
        # Test different step sizes
        step_sizes = [0.001, 0.01, 0.1]
        
        for step_size in step_sizes:
            # Create integrator
            integrator = PMLeapfrogIntegrator(
                target_fn=self.target_log_prob_fn,
                step_sizes=[step_size, step_size],
                num_steps=10,  # More steps to test conservation
                rho_size=self.rho_size
            )
            
            # Get initial target function value and gradients
            with tf.GradientTape(persistent=True) as tape:
                tape.watch(theta)
                tape.watch(u)
                target = self.target_log_prob_fn(theta, u)
            
            grad_theta = tape.gradient(target, theta)
            grad_u = tape.gradient(target, u)
            
            # Execute integrator
            next_momentum_parts, next_state_parts, next_target, _ = integrator(
                momentum_parts=[rho, p],
                state_parts=[theta, u],
                target=target,
                target_grad_parts=[grad_theta, grad_u]
            )
            
            # Calculate final Hamiltonian
            next_rho, next_p = next_momentum_parts
            next_theta, next_u = next_state_parts
            
            final_H = next_target - (0.5 * tf.reduce_sum(tf.square(next_rho)) / self.rho_size + 
                                  0.5 * tf.reduce_sum(tf.square(next_p)) + 
                                  0.5 * tf.reduce_sum(tf.square(next_u)))
            
            # Calculate relative error
            delta_H = tf.abs(final_H - initial_H)
            relative_error = delta_H / (tf.abs(initial_H) + 1e-10)
            
            print(f"\nStep size h={step_size}:")
            print(f"Initial H: {initial_H:.6f}")
            print(f"Final H: {final_H:.6f}")
            print(f"Absolute difference: {delta_H:.6f}")
            print(f"Relative error: {relative_error:.6f}")
            
            # Assert relative error is within tolerance
            # Larger step size, larger error, so use different tolerances
            if step_size <= 0.001:
                tolerance = 0.01  # 1% relative error tolerance
            elif step_size <= 0.01:
                tolerance = 0.05  # 5% relative error tolerance
            else:
                tolerance = 0.1   # 10% relative error tolerance
                
            self.assertTrue(
                relative_error < tolerance,
                f"Large Hamiltonian deviation for step size h={step_size}: relative error = {relative_error:.6f}"
            )

    def test_full_pm_hmc_chain(self):
        """Test full PM-HMC sampling chain"""
        import tensorflow_probability as tfp
        
        # Create simpler target function (Gaussian distribution)
        def simple_target_fn(theta, u):
            return -0.5 * tf.reduce_sum(tf.square(theta)) - 0.5 * tf.reduce_sum(tf.square(u))
        
        # Initialize PMHMC kernel
        kernel = PMHMC(
            target_log_prob_fn=simple_target_fn,
            step_size=0.1,
            num_leapfrog_steps=3,
            rho_size=self.rho_size
        )
        
        # Create initial state
        initial_theta = tf.zeros([5])  # 5-dimensional vector
        initial_u = tf.zeros([3, 2])   # Matrix with shape [3, 2]
        initial_state = [initial_theta, initial_u]
        
        # Set sampling parameters
        num_results = 10
        num_burnin_steps = 5
        
        # Use TFP's sample_chain function for sampling
        @tf.function
        def run_chain():
            samples = tfp.mcmc.sample_chain(
                num_results=num_results,
                current_state=initial_state,
                kernel=kernel,
                num_burnin_steps=num_burnin_steps,
                trace_fn=None
            )
            return samples
        
        # Execute sampling
        samples = run_chain()
        
        # Check output
        self.assertEqual(len(samples), 2)  # [theta_samples, u_samples]
        self.assertEqual(samples[0].shape, (num_results, 5))  # theta samples
        self.assertEqual(samples[1].shape, (num_results, 3, 2))  # u samples
        
        # Check if samples have finite values
        self.assertTrue(tf.reduce_all(tf.math.is_finite(samples[0])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(samples[1])))


if __name__ == '__main__':
    unittest.main()