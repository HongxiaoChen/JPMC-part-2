import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add project root and codes directory to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'codes'))
from datetime import datetime
import unittest
import tensorflow as tf
import numpy as np
import shutil
import os
from codes.run_hnn_hmc import (
    setup_logger,
    ensure_directories,
    plot_parameter_traces,
    leapfrog_steps,
    metropolis_step,
    run_hnn_hmc
)
from codes.hnn_architectures import HNN
from codes.params import MCMC_PARAMS, DATA_PARAMS


class TestHNNHMC(unittest.TestCase):
    """Test suite for HNN-HMC sampling functionality"""

    def setUp(self):
        """Set up test fixtures and ensure clean test environment"""
        # Define project paths
        self.current_dir = Path(__file__).parent
        self.project_root = self.current_dir.parent

        # Clean up any existing test directories first
        self.cleanup_directories()

        # Create test directories
        self.test_dirs = ['test_log', 'test_figures']
        for directory in self.test_dirs:
            Path(self.current_dir / directory).mkdir(exist_ok=True)

        # Initialize model
        self.model = HNN(activation='sin')

        # Create small test data
        self.feature_dim = DATA_PARAMS['NUM_FEATURES'] + 5
        self.test_samples = np.random.normal(size=(100, self.feature_dim))

        # Create a mock weights file for testing
        self.mock_weights_path = self.current_dir / 'test_weights.h5'
        self.model.save_weights(str(self.mock_weights_path))

        # Add HNN_WEIGHTS_PATH to MCMC_PARAMS
        MCMC_PARAMS['HNN_WEIGHTS_PATH'] = str(self.mock_weights_path)

    def tearDown(self):
        """Clean up test artifacts and ensure no residual files"""
        self.cleanup_directories()

        # Remove mock weights file
        if self.mock_weights_path.exists():
            self.mock_weights_path.unlink()

    def cleanup_directories(self):
        """Helper method to clean up all test-related directories"""
        directories_to_clean = [
            'test_log',
            'test_figures',
            'log',
            'figures'
        ]

        for directory in directories_to_clean:
            dir_path = self.current_dir / directory
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Warning: Failed to remove directory {directory}: {e}")

    def test_ensure_directories(self):
        """Test directory creation"""
        # Change to test directory
        original_dir = os.getcwd()
        os.chdir(str(self.current_dir))

        try:
            ensure_directories()

            # Verify directories were created
            for directory in ['log', 'figures']:
                self.assertTrue(Path(directory).exists())
                self.assertTrue(Path(directory).is_dir())
        finally:
            # Restore original directory
            os.chdir(original_dir)

    def test_plot_parameter_traces(self):
        """Test parameter trace plotting"""
        try:
            # Create figures directory if it doesn't exist
            figures_dir = self.current_dir / 'figures'
            figures_dir.mkdir(exist_ok=True)

            # Change working directory to test directory temporarily
            original_dir = os.getcwd()
            os.chdir(str(self.current_dir))

            try:
                plot_parameter_traces(
                    self.test_samples,
                    h=MCMC_PARAMS['H'],
                    L=MCMC_PARAMS['L'],
                    rho_size=MCMC_PARAMS['RHO_SIZE']
                )
                # Check if figure was created
                figures = list(Path('figures').glob('parameter_traces_*.png'))
                self.assertTrue(len(figures) > 0)
            finally:
                # Restore original working directory
                os.chdir(original_dir)

        except Exception as e:
            self.fail(f"Plot generation failed with error: {str(e)}")

    def test_leapfrog_steps(self):
        """Test leapfrog integration"""
        # Create test inputs
        theta = tf.random.normal([self.feature_dim])
        rho = tf.random.normal([self.feature_dim])
        h = MCMC_PARAMS['H']
        L = MCMC_PARAMS['L']

        # Run leapfrog integration
        final_theta, final_rho = leapfrog_steps(self.model, theta, rho, h, L)

        # Check shapes
        self.assertEqual(final_theta.shape, theta.shape)
        self.assertEqual(final_rho.shape, rho.shape)

        # Check for finite values
        self.assertTrue(tf.reduce_all(tf.math.is_finite(final_theta)))
        self.assertTrue(tf.reduce_all(tf.math.is_finite(final_rho)))

    def test_metropolis_step(self):
        """Test Metropolis accept-reject step"""
        # Create test inputs
        theta = tf.random.normal([self.feature_dim])
        rho = tf.random.normal([self.feature_dim])
        theta_new = tf.random.normal([self.feature_dim])
        rho_new = tf.random.normal([self.feature_dim])

        # Run Metropolis step
        accepted_theta, accepted = metropolis_step(
            self.model, theta, rho, theta_new, rho_new
        )

        # Check outputs
        self.assertEqual(accepted_theta.shape, theta.shape)
        self.assertTrue(isinstance(accepted, tf.Tensor))
        self.assertEqual(accepted.dtype, tf.bool)

        # Check if accepted_theta is either theta or theta_new
        is_original = tf.reduce_all(tf.equal(accepted_theta, theta))
        is_proposed = tf.reduce_all(tf.equal(accepted_theta, theta_new))
        self.assertTrue(is_original or is_proposed)

    def test_run_hnn_hmc(self):
        """Test complete HNN-HMC sampling"""
        # Get related modules
        import codes.params as params_module
        import codes.run_hnn_hmc as hmc_module

        # Store original parameters
        original_params = params_module.MCMC_PARAMS.copy()

        try:
            # Set test parameters
            test_params = {
                'M': 10,
                'BURN_IN': 2,
                'H': original_params['H'],
                'L': original_params['L'],
                'RHO_SIZE': original_params['RHO_SIZE'],
                'HNN_WEIGHTS_PATH': str(self.mock_weights_path)
            }

            # Update both module parameters
            params_module.MCMC_PARAMS.clear()
            params_module.MCMC_PARAMS.update(test_params)
            hmc_module.MCMC_PARAMS = params_module.MCMC_PARAMS

            # Change to test directory
            original_dir = os.getcwd()
            os.chdir(str(self.current_dir))

            try:
                # Run HNN-HMC
                results = hmc_module.run_hnn_hmc()

                # Unpack results
                (samples, acceptance_rate, posterior_mean_beta,
                 posterior_mean_mu, posterior_mean_lambda,
                 posterior_mean_w, logger) = results

                # Check outputs
                self.assertEqual(samples.shape[0], test_params['M'] - test_params['BURN_IN'])
                self.assertEqual(samples.shape[1], self.feature_dim)
                self.assertTrue(0 <= acceptance_rate <= 1)
                self.assertEqual(len(posterior_mean_beta), DATA_PARAMS['NUM_FEATURES'])
                self.assertEqual(len(posterior_mean_mu), 2)
                self.assertEqual(len(posterior_mean_lambda), 2)
                self.assertTrue(0 <= posterior_mean_w <= 1)
            finally:
                # Restore original directory
                os.chdir(original_dir)

        finally:
            # Restore original parameters
            params_module.MCMC_PARAMS.clear()
            params_module.MCMC_PARAMS.update(original_params)
            hmc_module.MCMC_PARAMS = params_module.MCMC_PARAMS

    def test_hamiltonian_conservation(self):
        """Test Hamiltonian conservation with different step sizes"""
        # Create test inputs
        theta = tf.random.normal([self.feature_dim])
        rho = tf.random.normal([self.feature_dim])

        # Test different step sizes
        step_sizes = [0.01, 0.1, 0.5, 1.0]
        L = MCMC_PARAMS['L']
        for h in step_sizes:
            # Initial Hamiltonian
            theta_expanded = tf.expand_dims(theta, 0)
            rho_expanded = tf.expand_dims(rho, 0)
            initial_H, _, _ = self.model.compute_gradients(theta_expanded, rho_expanded)
            initial_H = tf.squeeze(initial_H)

            # Perform leapfrog steps
            final_theta, final_rho = leapfrog_steps(self.model, theta, rho, h, L)

            # Final Hamiltonian
            final_theta_expanded = tf.expand_dims(final_theta, 0)
            final_rho_expanded = tf.expand_dims(final_rho, 0)
            final_H, _, _ = self.model.compute_gradients(final_theta_expanded, final_rho_expanded)
            final_H = tf.squeeze(final_H)

            # Check relative difference
            delta_H = tf.abs(final_H - initial_H)
            relative_error = delta_H / (tf.abs(initial_H) + 1e-10)

            print(f"\nStep size h={h}:")
            print(f"Initial H: {initial_H:.6f}")
            print(f"Final H: {final_H:.6f}")
            print(f"Absolute difference: {delta_H:.6f}")
            print(f"Relative error: {relative_error:.6f}")

            # Assert relative error is within tolerance
            tolerance = 0.1  # 10% relative error tolerance
            self.assertTrue(
                relative_error < tolerance,
                f"Large Hamiltonian deviation for h={h}: relative error = {relative_error:.6f}"
            )

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests have completed"""
        current_dir = Path(__file__).parent
        directories_to_clean = [
            'test_log',
            'test_figures',
            'log',
            'figures'
        ]

        for directory in directories_to_clean:
            dir_path = current_dir / directory
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Warning: Failed to remove directory {directory}: {e}")


if __name__ == '__main__':
    unittest.main()