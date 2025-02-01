import unittest
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

# Get the project root path
PROJECT_ROOT = Path(__file__).parent.parent

# Add project root and codes directory to Python path
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'codes'))

from codes.log_likelihood_auto import (
    generate_X, compute_logg, compute_logf_components,
    compute_logq, compute_normalized_weights,
    compute_log_prior, compute_log_posterior,
    compute_log_likelihood_and_gradients_auto
)
from codes.log_likelihood_stable import compute_log_likelihood_and_gradients as compute_manual
from codes.generate_samples import generate_samples


class TestLogLikelihood(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        tf.random.set_seed(42)

        # Test dimensions
        self.T = 500
        self.N = 128
        self.n = 6
        self.num_features = 8

        # Generate test data
        self.Y, _, self.Z, self.beta = generate_samples(
            T=self.T, n=self.n, num_features=self.num_features
        )

        # Initialize parameters
        self.mu1 = tf.constant(0.0)
        self.mu2 = tf.constant(3.0)
        self.log_lambda1 = tf.constant(0.0)
        self.log_lambda2 = tf.constant(-2.3)
        self.eta = tf.constant(0.0)

        # Combine parameters
        self.theta = tf.concat([
            self.beta,
            [self.mu1, self.mu2, self.log_lambda1, self.log_lambda2, self.eta]
        ], axis=0)

        # Initialize auxiliary variables
        self.u = tf.random.normal([self.T, self.N])

    def test_generate_X(self):
        """Test generate_X function"""
        X = generate_X(self.u, self.eta, self.log_lambda1, self.log_lambda2)

        # Check shape
        self.assertEqual(X.shape, (self.T, self.N))

        # Check type
        self.assertEqual(X.dtype, tf.float32)

        # Check finite values
        self.assertTrue(tf.reduce_all(tf.math.is_finite(X)))

    def test_compute_logg(self):
        """Test compute_logg function"""
        # Generate X
        X = generate_X(self.u, self.eta, self.log_lambda1, self.log_lambda2)

        # Compute log g
        log_g = compute_logg(self.Y, X, self.Z, self.beta)

        # Check shape
        self.assertEqual(log_g.shape, (self.T, self.N))

        # Check type
        self.assertEqual(log_g.dtype, tf.float32)

        # Check finite values
        self.assertTrue(tf.reduce_all(tf.math.is_finite(log_g)))

    def test_compute_logf_components(self):
        """Test compute_logf_components function"""
        # Generate X
        X = generate_X(self.u, self.eta, self.log_lambda1, self.log_lambda2)

        # Compute log f components
        logf = compute_logf_components(X, self.mu1, self.mu2,
                                       self.log_lambda1, self.log_lambda2, self.eta)

        # Check shape
        self.assertEqual(logf.shape, (self.T, self.N))

        # Check type
        self.assertEqual(logf.dtype, tf.float32)

        # Check finite values
        self.assertTrue(tf.reduce_all(tf.math.is_finite(logf)))

    def test_compute_logq(self):
        """Test compute_logq function"""
        # Compute log q
        log_q = compute_logq(self.u)

        # Check shape
        self.assertEqual(log_q.shape, (self.T, self.N))

        # Check type
        self.assertEqual(log_q.dtype, tf.float32)

        # Check finite values
        self.assertTrue(tf.reduce_all(tf.math.is_finite(log_q)))

    def test_compute_normalized_weights(self):
        """Test compute_normalized_weights function"""
        # Generate log weights
        X = generate_X(self.u, self.eta, self.log_lambda1, self.log_lambda2)
        log_g = compute_logg(self.Y, X, self.Z, self.beta)
        log_f = compute_logf_components(X, self.mu1, self.mu2,
                                        self.log_lambda1, self.log_lambda2, self.eta)
        log_q = compute_logq(self.u)
        log_omega = log_g + log_f - log_q

        # Compute normalized weights
        norm_weights, log_likelihood = compute_normalized_weights(log_omega)

        # Check shapes
        self.assertEqual(norm_weights.shape, (self.T, self.N))
        self.assertEqual(log_likelihood.shape, ())

        # Check types
        self.assertEqual(norm_weights.dtype, tf.float32)
        self.assertEqual(log_likelihood.dtype, tf.float32)

        # Check weights sum to 1
        sum_weights = tf.reduce_sum(norm_weights, axis=1)
        self.assertTrue(tf.reduce_all(tf.abs(sum_weights - 1.0) < 1e-5))

    def test_compute_log_prior(self):
        """Test compute_log_prior function"""
        # Compute log prior
        log_prior = compute_log_prior(self.theta)

        # Check shape
        self.assertEqual(log_prior.shape, ())

        # Check type
        self.assertEqual(log_prior.dtype, tf.float32)

        # Check finite value
        self.assertTrue(tf.math.is_finite(log_prior))

    def test_auto_vs_manual_implementation(self):
        """Test automatic differentiation implementation against manual implementation"""
        # Compute using automatic differentiation
        ll_auto, grad_u_auto, grad_theta_auto = compute_log_likelihood_and_gradients_auto(
            self.theta, self.u, self.Y, self.Z
        )

        # Compute using manual implementation
        ll_manual, grad_u_manual, grad_theta_manual = compute_manual(
            self.theta, self.u, self.Y, self.Z
        )

        # Set tolerance
        tolerance = 1e-3

        # Check log likelihood
        ll_diff = tf.abs(ll_auto - ll_manual)
        self.assertLess(ll_diff, tolerance)

        # Check gradients
        grad_u_diff = tf.reduce_max(tf.abs(grad_u_auto - grad_u_manual))
        self.assertLess(grad_u_diff, tolerance)

        grad_theta_diff = tf.reduce_max(tf.abs(grad_theta_auto - grad_theta_manual))
        self.assertLess(grad_theta_diff, tolerance)

    def test_gradient_consistency(self):
        """Test gradient consistency using tf.GradientTape"""
        theta_var = tf.Variable(self.theta)
        u_var = tf.Variable(self.u)

        # Compute gradients using automatic differentiation
        with tf.GradientTape(persistent=True) as tape:
            ll = compute_log_posterior(theta_var, u_var, self.Y, self.Z)

        auto_grad_theta = tape.gradient(ll, theta_var)
        auto_grad_u = tape.gradient(ll, u_var)

        # Compute gradients using our implementation
        _, grad_u, grad_theta = compute_log_likelihood_and_gradients_auto(
            self.theta, self.u, self.Y, self.Z
        )

        # Check consistency
        tolerance = 1e-3
        theta_diff = tf.reduce_max(tf.abs(auto_grad_theta - grad_theta))
        u_diff = tf.reduce_max(tf.abs(auto_grad_u - grad_u))

        self.assertLess(theta_diff, tolerance)
        self.assertLess(u_diff, tolerance)

    def tearDown(self):
        """Clean up after tests"""
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()