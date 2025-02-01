import unittest
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

# Add project root and codes directory to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'codes'))

from codes.pm_hmc_steps import (
    full_step_A, full_step_B, leapfrog_step,
    compute_hamiltonian, metropolis_step,
    run_pm_hmc_iteration
)
from codes.generate_samples import generate_samples
from codes.params import (
    DATA_PARAMS, INIT_PARAMS, initialize_theta
)

class TestPMHMC(unittest.TestCase):
    """Test suite for Preconditioned Monte Carlo Hamiltonian steps"""

    def setUp(self):
        """Initialize test environment and set parameters"""
        # Set random seed for reproducibility
        tf.random.set_seed(42)

        # Define dimensions for testing
        self.T = 500  # Time steps
        self.N = 128  # Number of samples
        self.n = 6  # Number of observations
        self.num_features = 8  # Number of features

        # Generate synthetic test data
        self.Y, _, self.Z, self.beta = generate_samples(
            T=self.T, n=self.n, num_features=self.num_features
        )

        # Initialize model parameters
        self.mu1 = tf.constant(0.0)
        self.mu2 = tf.constant(3.0)
        self.log_lambda1 = tf.constant(0.0)
        self.log_lambda2 = tf.constant(-2.3)
        self.eta = tf.constant(0.0)

        # Combine all parameters into theta vector
        self.theta = tf.concat([
            self.beta,
            [self.mu1, self.mu2, self.log_lambda1, self.log_lambda2, self.eta]
        ], axis=0)

        # Initialize auxiliary variables
        self.u = tf.random.normal([self.T, self.N])

        # Set HMC parameters
        self.h = 0.001  # Step size
        self.L = 5  # Number of leapfrog steps
        self.rho_size = 1.0  # Momentum scale

        # Initialize momentum variables
        self.rho = tf.random.normal([len(self.theta)]) * tf.sqrt(self.rho_size)
        self.p = tf.random.normal(tf.shape(self.u))

    def test_dimensions(self):
        """Test output dimensions of all major functions"""
        # Test full_step_A
        theta_new, rho_new, u_new, p_new = full_step_A(
            self.theta, self.rho, self.u, self.p,
            self.h, self.Y, self.Z, self.rho_size
        )

        # Verify output shapes of step A
        self.assertEqual(theta_new.shape, self.theta.shape, "Step A: Wrong theta shape")
        self.assertEqual(rho_new.shape, self.rho.shape, "Step A: Wrong rho shape")
        self.assertEqual(u_new.shape, self.u.shape, "Step A: Wrong u shape")
        self.assertEqual(p_new.shape, self.p.shape, "Step A: Wrong p shape")

        # Test full_step_B
        theta_new, rho_new, u_new, p_new = full_step_B(
            self.theta, self.rho, self.u, self.p,
            self.h, self.Y, self.Z, self.rho_size,
            None, True
        )

        # Verify output shapes of step B
        self.assertEqual(theta_new.shape, self.theta.shape, "Step B: Wrong theta shape")
        self.assertEqual(rho_new.shape, self.rho.shape, "Step B: Wrong rho shape")
        self.assertEqual(u_new.shape, self.u.shape, "Step B: Wrong u shape")
        self.assertEqual(p_new.shape, self.p.shape, "Step B: Wrong p shape")

        # Test complete leapfrog step
        theta_new, rho_new, u_new, p_new = leapfrog_step(
            self.theta, self.rho, self.u, self.p,
            self.h, self.L, self.Y, self.Z, self.rho_size,
            None, True
        )

        # Verify output shapes of leapfrog step
        self.assertEqual(theta_new.shape, self.theta.shape, "Leapfrog: Wrong theta shape")
        self.assertEqual(rho_new.shape, self.rho.shape, "Leapfrog: Wrong rho shape")
        self.assertEqual(u_new.shape, self.u.shape, "Leapfrog: Wrong u shape")
        self.assertEqual(p_new.shape, self.p.shape, "Leapfrog: Wrong p shape")

    def test_energy_conservation(self):
        """Test energy conservation in leapfrog integration"""
        # Calculate initial Hamiltonian
        initial_H = compute_hamiltonian(
            self.theta, self.rho, self.u, self.p,
            self.Y, self.Z, self.rho_size
        )

        # Perform leapfrog integration
        theta_new, rho_new, u_new, p_new = leapfrog_step(
            self.theta, self.rho, self.u, self.p,
            self.h, self.L, self.Y, self.Z, self.rho_size,
            None, True
        )

        # Calculate final Hamiltonian
        final_H = compute_hamiltonian(
            theta_new, rho_new, u_new, p_new,
            self.Y, self.Z, self.rho_size
        )

        # Check absolute difference in energy
        abs_diff = tf.abs(final_H - initial_H)
        tolerance = 1.0

        # Print energy conservation details
        print("\nEnergy Conservation Details:")
        print(f"Initial Hamiltonian: {initial_H:.6f}")
        print(f"Final Hamiltonian: {final_H:.6f}")
        print(f"Absolute difference: {abs_diff:.2e}")

        self.assertLess(abs_diff, tolerance,
                        f"Energy conservation violated: absolute difference = {abs_diff:.2e}")

    def test_reversibility(self):
        """Test reversibility of the leapfrog integrator"""
        # Forward integration
        theta_new, rho_new, u_new, p_new = leapfrog_step(
            self.theta, self.rho, self.u, self.p,
            self.h, self.L, self.Y, self.Z, self.rho_size,
            None, True
        )

        # Backward integration
        theta_rev, rho_rev, u_rev, p_rev = leapfrog_step(
            theta_new, -rho_new, u_new, -p_new,
            -self.h, self.L, self.Y, self.Z, self.rho_size,
            None, True
        )

        # Calculate absolute differences
        theta_diff = tf.reduce_max(tf.abs(theta_rev - self.theta))
        u_diff = tf.reduce_max(tf.abs(u_rev - self.u))

        # Set tolerance and check differences
        tolerance = 1.0

        print("\nReversibility Test Results:")
        print(f"Maximum absolute difference in theta: {theta_diff:.2e}")
        print(f"Maximum absolute difference in u: {u_diff:.2e}")

        self.assertLess(theta_diff, tolerance, "Reversibility violated for theta")
        self.assertLess(u_diff, tolerance, "Reversibility violated for u")

    def test_metropolis_step(self):
        """Test Metropolis-Hastings accept/reject step"""
        # Generate proposal using leapfrog
        theta_new, rho_new, u_new, p_new = leapfrog_step(
            self.theta, self.rho, self.u, self.p,
            self.h, self.L, self.Y, self.Z, self.rho_size,
            None, True
        )

        # Perform Metropolis step
        accepted_theta, accepted_u, accepted = metropolis_step(
            self.theta, self.rho, self.u, self.p,
            theta_new, rho_new, u_new, p_new,
            self.Y, self.Z, self.rho_size
        )

        # Check output dimensions
        self.assertEqual(accepted_theta.shape, self.theta.shape, "Wrong accepted theta shape")
        self.assertEqual(accepted_u.shape, self.u.shape, "Wrong accepted u shape")
        self.assertEqual(accepted.shape, (), "Wrong acceptance flag shape")

        # Verify acceptance flag type
        accepted_np = accepted.numpy()
        self.assertTrue(isinstance(accepted_np, np.bool_), "Wrong acceptance flag type")

        # Verify state consistency
        if accepted_np:
            self.assertTrue(tf.reduce_all(tf.abs(accepted_theta - theta_new) < 1e-6),
                            "Inconsistent accepted theta state")
            self.assertTrue(tf.reduce_all(tf.abs(accepted_u - u_new) < 1e-6),
                            "Inconsistent accepted u state")
        else:
            self.assertTrue(tf.reduce_all(tf.abs(accepted_theta - self.theta) < 1e-6),
                            "Inconsistent rejected theta state")
            self.assertTrue(tf.reduce_all(tf.abs(accepted_u - self.u) < 1e-6),
                            "Inconsistent rejected u state")

    def test_full_pm_hmc_iteration(self):
        """Test complete PM-HMC iteration"""
        # Run one complete iteration
        theta_next, u_next, accepted = run_pm_hmc_iteration(
            self.theta, self.u, self.Y, self.Z,
            self.h, self.L, self.rho_size,
            None, True
        )

        # Check output dimensions
        self.assertEqual(theta_next.shape, self.theta.shape, "Wrong theta shape")
        self.assertEqual(u_next.shape, self.u.shape, "Wrong u shape")
        self.assertEqual(accepted.shape, (), "Wrong acceptance flag shape")

        # Check for finite values
        self.assertTrue(tf.reduce_all(tf.math.is_finite(theta_next)),
                        "Non-finite values in theta")
        self.assertTrue(tf.reduce_all(tf.math.is_finite(u_next)),
                        "Non-finite values in u")

        print("\nPM-HMC Iteration Results:")
        print(f"Accepted: {accepted.numpy()}")
        if accepted:
            theta_diff = tf.reduce_max(tf.abs(theta_next - self.theta))
            print(f"Maximum parameter change: {theta_diff:.2e}")

    def test_step_determinism(self):
        """Test deterministic behavior with fixed random seed"""
        tf.random.set_seed(42)

        # First run
        theta_next1, u_next1, accepted1 = run_pm_hmc_iteration(
            self.theta, self.u, self.Y, self.Z,
            self.h, self.L, self.rho_size,
            None, True
        )

        tf.random.set_seed(42)

        # Second run
        theta_next2, u_next2, accepted2 = run_pm_hmc_iteration(
            self.theta, self.u, self.Y, self.Z,
            self.h, self.L, self.rho_size,
            None, True
        )

        # Check for identical results
        self.assertTrue(tf.reduce_all(tf.equal(theta_next1, theta_next2)),
                        "Non-deterministic theta update")
        self.assertTrue(tf.reduce_all(tf.equal(u_next1, u_next2)),
                        "Non-deterministic u update")
        self.assertEqual(accepted1, accepted2,
                         "Non-deterministic acceptance")

    def test_initialize_theta(self):
        """Test theta initialization function"""
        # Reset random state
        tf.random.set_seed(42)

        # First run to let the decorator compile and trace the function
        _ = initialize_theta(seed=42)

        # Actual testing
        # Ensure running in the same execution context
        with tf.init_scope():
            tf.random.set_seed(42)
            theta1 = initialize_theta(seed=42)

            tf.random.set_seed(42)
            theta2 = initialize_theta(seed=42)

        # Check dimensions
        expected_dim = DATA_PARAMS['NUM_FEATURES'] + 5
        self.assertEqual(theta1.shape, (expected_dim,),
                         f"Wrong theta dimension. Expected {expected_dim}, got {theta1.shape[0]}")

        # Split theta into beta and other parameters
        beta1 = theta1[:DATA_PARAMS['NUM_FEATURES']]
        other_params1 = theta1[DATA_PARAMS['NUM_FEATURES']:]
        beta2 = theta2[:DATA_PARAMS['NUM_FEATURES']]
        other_params2 = theta2[DATA_PARAMS['NUM_FEATURES']:]

        # Print detailed information for debugging
        print("\nInitialization Analysis:")
        print("Beta1:", beta1.numpy())
        print("Beta2:", beta2.numpy())
        print(f"Maximum beta difference: {tf.reduce_max(tf.abs(beta1 - beta2)).numpy()}")

        # Check if other parameters match INIT_PARAMS
        expected_other_params = tf.constant([
            INIT_PARAMS['MU1'],
            INIT_PARAMS['MU2'],
            INIT_PARAMS['LOG_LAMBDA1'],
            INIT_PARAMS['LOG_LAMBDA2'],
            INIT_PARAMS['ETA']
        ], dtype=tf.float32)

        # Allow for some numerical error
        tolerance = 1e-5

        # Check other parameters
        self.assertTrue(
            tf.reduce_all(tf.abs(other_params1 - expected_other_params) < tolerance),
            "Other parameters do not match INIT_PARAMS"
        )

        # Check if results from two runs are within acceptable range
        # Note: With decorators, we might need to allow for larger tolerance
        beta_diff_tolerance = 1e-4
        max_beta_diff = tf.reduce_max(tf.abs(beta1 - beta2))
        self.assertTrue(
            max_beta_diff < beta_diff_tolerance,
            f"Beta values differ too much between runs: {max_beta_diff}"
        )

        # Check basic properties
        for beta in [beta1, beta2]:
            self.assertTrue(tf.reduce_all(tf.math.is_finite(beta)),
                            "Beta contains non-finite values")
            self.assertTrue(tf.reduce_all(tf.abs(beta) < 10.0),
                            "Beta contains unreasonably large values")

            # Check basic properties of normal distribution
            mean = tf.reduce_mean(beta)
            std = tf.math.reduce_std(beta)
            self.assertTrue(tf.abs(mean) < 2.0,
                            f"Beta mean ({mean:.2f}) too far from expected 0")
            self.assertTrue(tf.abs(std - 1.0) < 0.5,
                            f"Beta standard deviation ({std:.2f}) too far from expected 1.0")
    def tearDown(self):
        """Clean up test environment"""
        tf.keras.backend.clear_session()


if __name__ == '__main__':
    unittest.main()