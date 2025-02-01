import unittest
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

# Add project root and codes directory to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'codes'))

from codes.nuts_hnn_olm_complex import (
    single_leapfrog_update,
    build_tree,
    nuts_hnn_sample
)
from codes.generate_samples import generate_samples
from codes.params import NUTS_PARAMS, DATA_PARAMS, initialize_theta


class TestNUTS(unittest.TestCase):
    """Test suite for NUTS implementation"""

    def setUp(self):
        """Initialize test environment and set parameters"""
        # Set random seed for reproducibility
        tf.random.set_seed(42)

        # Generate synthetic test data
        self.T = 500
        self.N = 128
        self.n = 6
        self.num_features = 8

        self.Y, _, self.Z, self.beta = generate_samples(
            T=self.T, n=self.n, num_features=self.num_features
        )

        # Initialize parameters
        self.theta = initialize_theta()
        self.u = tf.random.normal([self.T, self.N])

        # NUTS parameters
        self.step_size = NUTS_PARAMS['nuts_step_size']
        self.rho_size = NUTS_PARAMS['rho_size']

        # Initialize momentum variables
        self.rho = tf.random.normal([len(self.theta)]) * tf.sqrt(self.rho_size)
        self.p = tf.random.normal(tf.shape(self.u))

    def test_single_leapfrog_update(self):
        """Test single leapfrog update step"""
        # Run single leapfrog update
        theta_new, rho_new, u_new, p_new = single_leapfrog_update(
            self.theta, self.rho, self.u, self.p,
            self.step_size, self.Y, self.Z,
            None, True  # No HNN model, traditional only
        )

        # Check output dimensions
        self.assertEqual(theta_new.shape, self.theta.shape)
        self.assertEqual(rho_new.shape, self.rho.shape)
        self.assertEqual(u_new.shape, self.u.shape)
        self.assertEqual(p_new.shape, self.p.shape)

        # Check finite values
        self.assertTrue(tf.reduce_all(tf.math.is_finite(theta_new)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(rho_new)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(u_new)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(p_new)))

    def test_build_tree_base_case(self):
        """Test build_tree function base case (j=0)"""
        # Initial Hamiltonian
        H0 = tf.constant(0.0)
        slice_variable = tf.exp(-H0) * 0.5  # Random slice between 0 and exp(-H0)

        # Run build_tree with j=0 (base case)
        results = build_tree(
            self.theta, self.rho, self.u, self.p,
            slice_variable, 1, 0, self.step_size, H0,
            self.Y, self.Z, None,
            len(self.theta),
            float('inf'), float('inf'),  # High thresholds
            False, True  # use_leapfrog=False, traditional_only=True
        )

        # Unpack results
        (theta_minus, rho_minus, theta_plus, rho_plus, theta_prime, rho_prime,
         u_minus, p_minus, u_plus, p_plus, u_prime, p_prime,
         n_prime, s_prime, alpha, n_alpha, error, use_leapfrog) = results

        # Check output dimensions
        self.assertEqual(theta_prime.shape, self.theta.shape)
        self.assertEqual(rho_prime.shape, self.rho.shape)
        self.assertEqual(u_prime.shape, self.u.shape)
        self.assertEqual(p_prime.shape, self.p.shape)

        # Check scalar outputs
        self.assertEqual(n_prime.shape, ())
        self.assertEqual(s_prime.shape, ())
        self.assertEqual(alpha.shape, ())

    def test_nuts_basic_sampling(self):
        """Test basic NUTS sampling functionality"""
        # Get related modules
        import codes.params as params_module
        import codes.nuts_hnn_olm_complex as nuts_module

        # Save original parameters
        original_params = params_module.NUTS_PARAMS.copy()

        try:
            # Modify parameters for testing
            num_samples = 3

            # Update both module parameters
            params_module.NUTS_PARAMS.update({
                'total_samples': num_samples,
                'burn_in': 1,
                'n_cooldown': 2,
                'max_depth': 5
            })

            nuts_module.NUTS_PARAMS = params_module.NUTS_PARAMS

            # Run NUTS sampling
            samples, acceptance, errors, final_state = nuts_module.nuts_hnn_sample(
                self.theta, self.u, self.Y, self.Z,
                None,  # No HNN model
                traditional_only=True,
                logger=None
            )

            # Check output dimensions
            self.assertEqual(samples.shape, (num_samples, len(self.theta)))
            self.assertEqual(acceptance.shape, (num_samples - params_module.NUTS_PARAMS['burn_in'],))
            self.assertEqual(errors.shape, (num_samples,))

            # Check finite values
            self.assertTrue(tf.reduce_all(tf.math.is_finite(samples)))
            self.assertTrue(tf.reduce_all(tf.math.is_finite(acceptance)))
            self.assertTrue(tf.reduce_all(tf.math.is_finite(errors)))

            # Check final state
            final_theta, final_u = final_state
            self.assertEqual(final_theta.shape, self.theta.shape)
            self.assertEqual(final_u.shape, self.u.shape)

            # Print sampling statistics
            print("\nNUTS Sampling Test Results:")
            print(f"Acceptance rate: {tf.reduce_mean(acceptance).numpy():.4f}")
            print(f"Average error: {tf.reduce_mean(errors).numpy():.4f}")

        finally:
            # Restore original parameters
            params_module.NUTS_PARAMS.clear()
            params_module.NUTS_PARAMS.update(original_params)
            nuts_module.NUTS_PARAMS = params_module.NUTS_PARAMS

    def tearDown(self):
        """Clean up test environment"""
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()