from pathlib import Path
import sys

# Add project root and codes directory to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'codes'))

import unittest
import tensorflow as tf
import shutil
from codes.collect_hamiltonian_trajectories import (
    collect_single_trajectory,
    collect_chain_samples,
    collect_training_data,
    save_trajectories,
    setup_logger
)
from codes.generate_samples import generate_samples
from codes.params import DATA_PARAMS, TRAJ_PARAMS, MCMC_PARAMS


class TestTrajectoryCollection(unittest.TestCase):
    """Test suite for Hamiltonian trajectory collection functionality"""

    def setUp(self):
        """Set up test fixtures before each test method"""
        # Use the same dimensions as in DATA_PARAMS for consistency
        self.T = DATA_PARAMS['T']
        self.n = DATA_PARAMS['N_SUBJECTS']
        self.num_features = DATA_PARAMS['NUM_FEATURES']

        # Generate test data with correct dimensions
        self.Y, self.X, self.Z, self.beta = generate_samples(
            self.T, self.n, self.num_features
        )

        # Test parameters
        self.num_samples = 2
        self.L = 3
        self.num_chains = 2
        self.h = MCMC_PARAMS['H']
        self.rho_size = MCMC_PARAMS['RHO_SIZE']

        # Create test directory
        self.test_dir = Path('test_outputs')
        self.test_dir.mkdir(exist_ok=True)

        # Store original parameters
        self.original_save_dir = TRAJ_PARAMS['SAVE_DIR']
        TRAJ_PARAMS['SAVE_DIR'] = str(self.test_dir)

    def tearDown(self):
        """Clean up test artifacts"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        TRAJ_PARAMS['SAVE_DIR'] = self.original_save_dir

    def test_single_trajectory_collection(self):
        """Test collection of a single trajectory"""
        # Initialize with correct dimensions
        theta = tf.random.normal([DATA_PARAMS['NUM_FEATURES'] + 5])  # 13 features
        u = tf.random.normal([DATA_PARAMS['T'], MCMC_PARAMS['N']])

        results = collect_single_trajectory(
            theta, u, self.Y, self.Z,
            self.h, self.L, self.rho_size
        )

        thetas, rhos, hamiltonians, grad_thetas, grad_rhos, final_theta, final_u = results

        # Verify shapes
        feature_dim = DATA_PARAMS['NUM_FEATURES'] + 5
        self.assertEqual(thetas.shape, (self.L, feature_dim))
        self.assertEqual(rhos.shape, (self.L, feature_dim))
        self.assertEqual(hamiltonians.shape, (self.L,))
        self.assertEqual(grad_thetas.shape, (self.L, feature_dim))
        self.assertEqual(grad_rhos.shape, (self.L, feature_dim))
        self.assertEqual(final_theta.shape, (feature_dim,))
        self.assertEqual(final_u.shape, (DATA_PARAMS['T'], MCMC_PARAMS['N']))

        # Check for finite values
        for tensor in [thetas, rhos, hamiltonians, grad_thetas, grad_rhos]:
            self.assertTrue(tf.reduce_all(tf.math.is_finite(tensor)))

    def test_chain_samples_collection(self):
        """Test collection of chain samples"""
        feature_dim = DATA_PARAMS['NUM_FEATURES'] + 5
        initial_theta = tf.random.normal([feature_dim])
        initial_u = tf.random.normal([DATA_PARAMS['T'], MCMC_PARAMS['N']])
        start_time = tf.timestamp()

        results = collect_chain_samples(
            0, self.num_samples, initial_theta, initial_u,
            self.Y, self.Z, self.h, self.L, self.rho_size, start_time
        )

        # Verify shapes
        self.assertEqual(results[0].shape, (self.num_samples, self.L, feature_dim))
        self.assertEqual(results[1].shape, (self.num_samples, self.L, feature_dim))
        self.assertEqual(results[2].shape, (self.num_samples, self.L))
        self.assertEqual(results[3].shape, (self.num_samples, self.L, feature_dim))
        self.assertEqual(results[4].shape, (self.num_samples, self.L, feature_dim))

    def test_full_training_data_collection(self):
        """Test collection of complete training dataset"""
        try:
            training_data = collect_training_data(
                self.Y, self.Z,
                num_samples=self.num_samples,
                L=self.L,
                num_chains=self.num_chains
            )

            # Verify data structure
            expected_keys = ['thetas', 'rhos', 'hamiltonians', 'grad_thetas', 'grad_rhos']
            self.assertTrue(all(key in training_data for key in expected_keys))

            # Verify shapes
            feature_dim = DATA_PARAMS['NUM_FEATURES'] + 5
            expected_size = self.num_chains * self.num_samples * self.L
            self.assertEqual(training_data['thetas'].shape, (expected_size, feature_dim))
            self.assertEqual(training_data['rhos'].shape, (expected_size, feature_dim))
            self.assertEqual(training_data['hamiltonians'].shape, (expected_size,))
            self.assertEqual(training_data['grad_thetas'].shape, (expected_size, feature_dim))
            self.assertEqual(training_data['grad_rhos'].shape, (expected_size, feature_dim))

        except Exception as e:
            self.fail(f"Training data collection failed with error: {str(e)}")


if __name__ == '__main__':
    unittest.main()