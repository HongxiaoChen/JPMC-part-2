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

        Raises:
            ValueError: If inputs dimensions don't match or inputs are invalid
        """
        # Input validation
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("Inputs must be a list or tuple containing [theta, rho]")

        theta, rho = inputs

        # Check if inputs are valid tensors
        if not (isinstance(theta, tf.Tensor) and isinstance(rho, tf.Tensor)):
            raise ValueError("Both theta and rho must be tensorflow tensors")

        # Check shape compatibility
        if theta.shape != rho.shape:
            raise ValueError(f"Shape mismatch: theta shape {theta.shape} != rho shape {rho.shape}")

        if len(theta.shape) != 2:
            raise ValueError(f"Inputs must be 2D tensors, got theta shape {theta.shape}")

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

