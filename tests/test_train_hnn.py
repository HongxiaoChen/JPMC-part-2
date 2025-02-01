from pathlib import Path
import sys

# Add project root and codes directory to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'codes'))

import unittest
import tensorflow as tf
import numpy as np
import shutil
import time
from codes.generate_samples import generate_samples
from codes.params import DATA_PARAMS, TRAJ_PARAMS, MCMC_PARAMS, TRAIN_PARAMS, NN_PARAMS
from codes.train_hnn_hamiltonian import (
    setup_train_logger,
    setup_checkpoints,
    create_dataset,
    compute_loss,
    train_step,
    train_model
)
from codes.hnn_architectures import HNN


class TestHNNTraining(unittest.TestCase):
    """Test suite for HNN training functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create small test data
        self.T = 10
        self.n = 5
        self.num_features = 8
        self.Y, self.X, self.Z, self.beta = generate_samples(
            self.T, self.n, self.num_features
        )

        # Initialize model
        self.model = HNN(activation='sin')

        # Create temporary directories
        self.test_dir = Path('test_outputs')
        self.ckpt_dir = Path('test_ckpt')
        self.log_dir = Path('test_log')

        for directory in [self.test_dir, self.ckpt_dir, self.log_dir]:
            directory.mkdir(exist_ok=True)

    def tearDown(self):
        """Clean up test artifacts"""
        for directory in [self.test_dir, self.ckpt_dir, self.log_dir]:
            if directory.exists():
                shutil.rmtree(directory)

    def test_checkpoint_setup(self):
        """Test checkpoint directory setup"""
        ckpt_dir = setup_checkpoints()
        self.assertTrue(ckpt_dir.exists())
        self.assertTrue(ckpt_dir.is_dir())

    def test_dataset_creation(self):
        """Test creation of training dataset"""
        # Create sample data
        batch_size = 32
        num_samples = 100
        feature_dim = DATA_PARAMS['NUM_FEATURES'] + 5

        data_dict = {
            'thetas': tf.random.normal([num_samples, feature_dim]),
            'rhos': tf.random.normal([num_samples, feature_dim]),
            'hamiltonians': tf.random.normal([num_samples]),
            'grad_thetas': tf.random.normal([num_samples, feature_dim]),
            'grad_rhos': tf.random.normal([num_samples, feature_dim])
        }

        dataset = create_dataset(data_dict, batch_size)

        # Verify dataset properties
        self.assertIsInstance(dataset, tf.data.Dataset)

        # Check batch shape
        for inputs, targets in dataset.take(1):
            thetas, rhos = inputs
            hamiltonians, grad_thetas, grad_rhos = targets

            self.assertLessEqual(thetas.shape[0], batch_size)
            self.assertEqual(thetas.shape[1], feature_dim)
            self.assertEqual(rhos.shape[1], feature_dim)
            self.assertEqual(grad_thetas.shape[1], feature_dim)
            self.assertEqual(grad_rhos.shape[1], feature_dim)

    def test_loss_computation(self):
        """Test loss computation"""
        batch_size = 16
        feature_dim = DATA_PARAMS['NUM_FEATURES'] + 5

        # Create test inputs
        thetas = tf.random.normal([batch_size, feature_dim])
        rhos = tf.random.normal([batch_size, feature_dim])
        hamiltonians = tf.random.normal([batch_size])
        grad_thetas = tf.random.normal([batch_size, feature_dim])
        grad_rhos = tf.random.normal([batch_size, feature_dim])

        # Compute loss
        total_loss, _, theta_loss, rho_loss = compute_loss(
            self.model, thetas, rhos, hamiltonians, grad_thetas, grad_rhos
        )

        # Verify loss values
        self.assertTrue(tf.is_tensor(total_loss))
        self.assertTrue(tf.is_tensor(theta_loss))
        self.assertTrue(tf.is_tensor(rho_loss))

        # Check for finite values
        for loss in [total_loss, theta_loss, rho_loss]:
            self.assertTrue(tf.math.is_finite(loss))

    def test_train_step(self):
        """Test single training step"""
        batch_size = 16
        feature_dim = DATA_PARAMS['NUM_FEATURES'] + 5

        # Create test data
        thetas = tf.random.normal([batch_size, feature_dim])
        rhos = tf.random.normal([batch_size, feature_dim])
        hamiltonians = tf.random.normal([batch_size])
        grad_thetas = tf.random.normal([batch_size, feature_dim])
        grad_rhos = tf.random.normal([batch_size, feature_dim])

        # Create optimizer
        optimizer = tf.keras.optimizers.legacy.Adam(0.001)

        # Perform training step
        losses = train_step(
            self.model, optimizer, thetas, rhos,
            hamiltonians, grad_thetas, grad_rhos
        )

        # Verify outputs
        self.assertEqual(len(losses), 4)
        for loss in losses:
            self.assertTrue(tf.math.is_finite(loss))


if __name__ == '__main__':
    unittest.main()