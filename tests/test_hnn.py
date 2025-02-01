import unittest
import tensorflow as tf
import numpy as np
from pathlib import Path
import sys

# Add project root and codes directory to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / 'codes'))
from codes.hnn_architectures import HNN
from codes.params import DATA_PARAMS

class TestHNN(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.batch_size = 32
        self.input_dim = DATA_PARAMS['NUM_FEATURES'] + 5
        self.test_theta = tf.random.normal([self.batch_size, self.input_dim])
        self.test_rho = tf.random.normal([self.batch_size, self.input_dim])
        self.activations = ['relu', 'tanh', 'sin']

    def test_model_initialization(self):
        """Test model initialization with different activations"""
        for act in self.activations:
            hnn = HNN(activation=act, dropout_rate=0.2)
            self.assertIsInstance(hnn, tf.keras.Model)

        # Test invalid activation
        with self.assertRaises(ValueError):
            HNN(activation='invalid_activation')

    def test_forward_pass(self):
        """Test forward pass in both training and inference modes"""
        for act in self.activations:
            hnn = HNN(activation=act)

            for training in [True, False]:
                hamiltonian = hnn([self.test_theta, self.test_rho], training=training)

                # Check shapes
                self.assertEqual(hamiltonian.shape, (self.batch_size,))

                # Check for finite values
                self.assertTrue(tf.reduce_all(tf.math.is_finite(hamiltonian)))

    def test_gradient_computation(self):
        """Test gradient computation functionality"""
        for act in self.activations:
            hnn = HNN(activation=act)

            # Test with training=False
            hamiltonian, grad_theta, grad_rho = hnn.compute_gradients(
                self.test_theta, self.test_rho, training=False
            )

            # Check shapes
            self.assertEqual(hamiltonian.shape, (self.batch_size,))
            self.assertEqual(grad_theta.shape, self.test_theta.shape)
            self.assertEqual(grad_rho.shape, self.test_rho.shape)

            # Check for finite values
            self.assertTrue(tf.reduce_all(tf.math.is_finite(hamiltonian)))
            self.assertTrue(tf.reduce_all(tf.math.is_finite(grad_theta)))
            self.assertTrue(tf.reduce_all(tf.math.is_finite(grad_rho)))

    def test_batch_dimension_consistency(self):
        """Test model behavior with different batch sizes"""
        batch_sizes = [1, 16, 32, 64]
        hnn = HNN()

        for size in batch_sizes:
            theta = tf.random.normal([size, self.input_dim])
            rho = tf.random.normal([size, self.input_dim])

            # Test forward pass
            hamiltonian = hnn([theta, rho])
            self.assertEqual(hamiltonian.shape, (size,))

            # Test gradient computation
            hamiltonian, grad_theta, grad_rho = hnn.compute_gradients(theta, rho)
            self.assertEqual(hamiltonian.shape, (size,))
            self.assertEqual(grad_theta.shape, (size, self.input_dim))
            self.assertEqual(grad_rho.shape, (size, self.input_dim))

    def test_model_architecture(self):
        """Test model architecture and trainable parameters"""
        hnn = HNN()

        # Run model once to initialize weights
        _ = hnn([self.test_theta, self.test_rho])

        # Check layer dimensions
        self.assertEqual(hnn.hidden_dim, 128)
        self.assertEqual(hnn.output_layer.units, 20)

        # Check trainable parameters
        trainable_vars = hnn.trainable_variables
        self.assertTrue(len(trainable_vars) > 0)

        # Test if the model is trainable
        with tf.GradientTape() as tape:
            hamiltonian = hnn([self.test_theta, self.test_rho])
            loss = tf.reduce_mean(hamiltonian)

        gradients = tape.gradient(loss, trainable_vars)
        self.assertTrue(all(g is not None for g in gradients))

    def test_dropout_behavior(self):
        """Test dropout behavior in training and inference modes"""
        dropout_rate = 0.5
        hnn = HNN(dropout_rate=dropout_rate)

        # Run multiple forward passes in training mode
        results_training = [
            hnn([self.test_theta, self.test_rho], training=True)
            for _ in range(10)
        ]

        # Run multiple forward passes in inference mode
        results_inference = [
            hnn([self.test_theta, self.test_rho], training=False)
            for _ in range(10)
        ]

        # Training mode should give different results due to dropout
        training_var = tf.math.reduce_variance(tf.stack(results_training))

        # Inference mode should give consistent results
        inference_var = tf.math.reduce_variance(tf.stack(results_inference))

        self.assertGreater(training_var, inference_var)

    def test_input_validation(self):
        """Test model behavior with invalid inputs"""
        hnn = HNN()

        # Test with mismatched dimensions
        invalid_theta = tf.random.normal([self.batch_size, self.input_dim + 1])
        with self.assertRaises(Exception):
            hnn([invalid_theta, self.test_rho])

        # Test with invalid input types
        with self.assertRaises(Exception):
            hnn([None, self.test_rho])


if __name__ == '__main__':
    unittest.main()
