import tensorflow as tf
from params import NN_PARAMS, DATA_PARAMS


class HNN(tf.keras.Model):
    """
    Neural network that maps (theta, rho) to (hamiltonian, grad_theta, grad_rho)
    The gradients are computed via automatic differentiation
    Network structure: 3-layer MLP with customizable activation and dropout
    The final layer outputs 20 variables that are summed to get the Hamiltonian
    """

    def __init__(self, activation='relu', dropout_rate=0.2):
        super().__init__()

        # Select activation function
        if activation == 'relu':
            self.activation = tf.nn.relu
        elif activation == 'tanh':
            self.activation = tf.nn.tanh
        elif activation == 'sin':
            self.activation = tf.math.sin
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        # 3-layer MLP architecture with dropout
        self.hidden_dim = 128

        self.layer1 = tf.keras.layers.Dense(self.hidden_dim)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)

        self.layer2 = tf.keras.layers.Dense(self.hidden_dim)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

        self.layer3 = tf.keras.layers.Dense(self.hidden_dim)
        self.dropout3 = tf.keras.layers.Dropout(dropout_rate)

        self.output_layer = tf.keras.layers.Dense(20)  # Linear output for 20 components

    def call(self, inputs, training=False):
        """
        Forward pass

        Args:
            inputs: [theta, rho], each with shape [batch, 13]
            training: Boolean indicating whether the model is in training mode

        Returns:
            hamiltonian: Shape [batch], sum of 20 output components
        """
        # Unpack inputs
        theta, rho = inputs

        # Concatenate inputs
        x = tf.concat([theta, rho], axis=-1)  # [batch, 26]

        # Forward through MLP with dropout
        x = self.activation(self.layer1(x))
        x = self.dropout1(x, training=training)

        x = self.activation(self.layer2(x))
        x = self.dropout2(x, training=training)

        x = self.activation(self.layer3(x))
        x = self.dropout3(x, training=training)

        components = self.output_layer(x)  # [batch, 20]
        hamiltonian = tf.reduce_sum(components, axis=-1)  # [batch]

        return hamiltonian

    @tf.function
    def compute_gradients(self, theta, rho, training=False):
        """
        Compute Hamiltonian and its gradients with respect to theta and rho

        Args:
            theta: Parameter vector [batch, 13]
            rho: Momentum vector [batch, 13]
            training: Boolean indicating whether the model is in training mode

        Returns:
            tuple: (hamiltonian, grad_theta, grad_rho)
                - hamiltonian: Shape [batch]
                - grad_theta: Shape [batch, 13]
                - grad_rho: Shape [batch, 13]
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(theta)
            tape.watch(rho)
            hamiltonian = self.call([theta, rho], training=training)

        grad_theta = tape.gradient(hamiltonian, theta)
        grad_rho = tape.gradient(hamiltonian, rho)

        del tape  # Delete the persistent tape

        return hamiltonian, grad_theta, grad_rho


# Test code
if __name__ == "__main__":
    import numpy as np

    print("========== HamiltonianNN Dimension Test ==========")

    # Create test data
    batch_size = 32
    input_dim = DATA_PARAMS['NUM_FEATURES'] + 5  # theta dimension: random features + 5 fixed parameters
    test_theta = tf.random.normal([batch_size, input_dim])
    test_rho = tf.random.normal([batch_size, input_dim])

    print(f"\n1. Input Dimension Check:")
    print(f"theta shape: {test_theta.shape}")
    print(f"rho shape: {test_rho.shape}")
    print(f"- batch_size: {batch_size}")
    print(f"- input_dim: {input_dim} (NUM_FEATURES={DATA_PARAMS['NUM_FEATURES']}, fixed params=5)")

    # Test all activation functions
    activations = ['relu', 'tanh', 'sin']

    for act in activations:
        print(f"\n2. Testing activation {act}:")

        # Initialize model
        hnn = HNN(activation=act, dropout_rate=0.2)

        # Test forward pass
        # Test both training and inference modes
        print("\nForward Pass Test:")
        for training in [True, False]:
            mode = "Training" if training else "Inference"
            hamiltonian = hnn([test_theta, test_rho], training=training)
            print(f"\n{mode} mode:")
            print(f"- Hamiltonian shape: {hamiltonian.shape}")
            print(f"- Hamiltonian range: [{tf.reduce_min(hamiltonian):.4f}, {tf.reduce_max(hamiltonian):.4f}]")

        # Test gradient computation
        hamiltonian, grad_theta, grad_rho = hnn.compute_gradients(test_theta, test_rho, training=False)
        print("\nGradient Computation Test:")
        print(f"- Output hamiltonian shape: {hamiltonian.shape}")
        print(f"- Output grad_theta shape: {grad_theta.shape}")
        print(f"- Output grad_rho shape: {grad_rho.shape}")

        # Verify gradient dimensions match input dimensions
        assert grad_theta.shape == test_theta.shape, \
            f"Gradient theta shape {grad_theta.shape} doesn't match input shape {test_theta.shape}!"
        assert grad_rho.shape == test_rho.shape, \
            f"Gradient rho shape {grad_rho.shape} doesn't match input shape {test_rho.shape}!"

        # Check numerical validity
        print("\nNumerical Validity Check:")
        print(f"- Hamiltonian contains NaN: {tf.reduce_any(tf.math.is_nan(hamiltonian)).numpy()}")
        print(f"- grad_theta contains NaN: {tf.reduce_any(tf.math.is_nan(grad_theta)).numpy()}")
        print(f"- grad_rho contains NaN: {tf.reduce_any(tf.math.is_nan(grad_rho)).numpy()}")
        print(f"- grad_theta range: [{tf.reduce_min(grad_theta):.4f}, {tf.reduce_max(grad_theta):.4f}]")
        print(f"- grad_rho range: [{tf.reduce_min(grad_rho):.4f}, {tf.reduce_max(grad_rho):.4f}]")

        # Test batch dimension consistency
        single_theta = tf.random.normal([1, input_dim])
        single_rho = tf.random.normal([1, input_dim])
        single_hamiltonian = hnn([single_theta, single_rho], training=False)
        print("\nBatch Dimension Consistency Test:")
        print(f"- Single input theta shape: {single_theta.shape}")
        print(f"- Single input rho shape: {single_rho.shape}")
        print(f"- Single output shape: {single_hamiltonian.shape}")

        # Verify model trainability
        trainable_vars = hnn.trainable_variables
        print(f"\nModel Trainable Parameters:")
        print(f"- Number of parameter tensors: {len(trainable_vars)}")
        print(f"- Total parameters: {np.sum([np.prod(v.shape) for v in trainable_vars])}")

        print("\n" + "=" * 50)