import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'codes'))

import unittest
import tensorflow as tf
import tensorflow_probability as tfp

from codes.tfp_modified_kernels.NUTS import NoUTurnSampler


class TestPMNUTS(unittest.TestCase):
    """Test suite for PM-NUTS sampling functionality"""

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
        self.max_tree_depth = 5
        self.rho_size = 10.0
        
        # Define target function
        self.target_log_prob_fn = lambda theta, u: -0.5 * tf.reduce_sum(tf.square(theta)) - 0.5 * tf.reduce_sum(tf.square(u))

    def test_nuts_initialization(self):
        """Test initialization of NoUTurnSampler kernel"""
        # Initialize NoUTurnSampler kernel
        kernel = NoUTurnSampler(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=self.step_size,
            max_tree_depth=self.max_tree_depth,
            rho_size=self.rho_size
        )
        
        # Check if parameters are set correctly
        self.assertEqual(kernel.step_size, self.step_size)
        self.assertEqual(kernel.max_tree_depth, self.max_tree_depth)
        self.assertEqual(kernel.target_log_prob_fn, self.target_log_prob_fn)
        self.assertTrue(kernel.is_calibrated)

    def test_bootstrap_results(self):
        """Test bootstrap_results method"""
        # Initialize NUTS kernel
        kernel = NoUTurnSampler(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=self.step_size,
            max_tree_depth=self.max_tree_depth,
            rho_size=self.rho_size
        )
        
        # Create initial state
        init_state = [self.theta, self.u]
        
        # Execute bootstrap_results
        results = kernel.bootstrap_results(init_state)
        
        # Check results fields
        self.assertIn('target_log_prob', results._fields)
        self.assertIn('grads_target_log_prob', results._fields)
        self.assertIn('step_size', results._fields)
        self.assertIn('log_accept_ratio', results._fields)
        self.assertIn('leapfrogs_taken', results._fields)
        self.assertIn('is_accepted', results._fields)
        self.assertIn('energy', results._fields)
        
        # Check shapes
        self.assertEqual(results.target_log_prob.shape, ())
        self.assertEqual(len(results.grads_target_log_prob), 2)
        self.assertEqual(results.grads_target_log_prob[0].shape, self.theta.shape)
        self.assertEqual(results.grads_target_log_prob[1].shape, self.u.shape)

    def test_one_step(self):
        """Test one_step method"""
        # Initialize NUTS kernel with smaller max_tree_depth for faster testing
        kernel = NoUTurnSampler(
            target_log_prob_fn=self.target_log_prob_fn,
            step_size=self.step_size,
            max_tree_depth=3,  # Smaller for faster testing
            rho_size=self.rho_size
        )
        
        # Create initial state and results
        init_state = [self.theta, self.u]
        init_results = kernel.bootstrap_results(init_state)
        
        # Execute one step sampling
        next_state, next_results = kernel.one_step(init_state, init_results, seed=42)
        
        # Check output types and shapes
        self.assertEqual(len(next_state), 2)
        self.assertEqual(next_state[0].shape, self.theta.shape)
        self.assertEqual(next_state[1].shape, self.u.shape)
        
        # Check if results contain necessary fields
        self.assertIn('log_accept_ratio', next_results._fields)
        self.assertIn('target_log_prob', next_results._fields)
        self.assertIn('grads_target_log_prob', next_results._fields)
        self.assertIn('leapfrogs_taken', next_results._fields)
        self.assertIn('is_accepted', next_results._fields)
        
        # Check data types
        self.assertEqual(next_state[0].dtype, self.theta.dtype)
        self.assertEqual(next_state[1].dtype, self.u.dtype)
        
        # Check results shapes
        self.assertEqual(next_results.target_log_prob.shape, ())
        self.assertEqual(next_results.log_accept_ratio.shape, ())
        self.assertEqual(len(next_results.grads_target_log_prob), 2)
        self.assertEqual(next_results.grads_target_log_prob[0].shape, self.theta.shape)
        self.assertEqual(next_results.grads_target_log_prob[1].shape, self.u.shape)

    def test_full_pm_nuts_chain(self):
        """Test full PM-NUTS sampling chain"""
        # Create simpler target function for faster testing
        def simple_target_fn(theta, u):
            return -0.5 * tf.reduce_sum(tf.square(theta)) - 0.5 * tf.reduce_sum(tf.square(u))
        
        # Initialize NUTS kernel with smaller max_tree_depth for faster testing
        kernel = NoUTurnSampler(
            target_log_prob_fn=simple_target_fn,
            step_size=0.1,
            max_tree_depth=3,  # Smaller for faster testing
            rho_size=self.rho_size
        )
        
        # Create initial state
        initial_theta = tf.zeros([5])  # 5-dimensional vector
        initial_u = tf.zeros([3, 2])   # Matrix with shape [3, 2]
        initial_state = [initial_theta, initial_u]
        
        # Set sampling parameters
        num_results = 5  # Small number for testing
        num_burnin_steps = 2  # Small number for testing
        
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
        
        # Check output shape
        self.assertEqual(len(samples), 2)  # [theta_samples, u_samples]
        self.assertEqual(samples[0].shape, (num_results, 5))  # theta samples
        self.assertEqual(samples[1].shape, (num_results, 3, 2))  # u samples
        
        # Check if samples have finite values
        self.assertTrue(tf.reduce_all(tf.math.is_finite(samples[0])))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(samples[1])))
        


if __name__ == '__main__':
    unittest.main()
