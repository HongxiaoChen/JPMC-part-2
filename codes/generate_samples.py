import tensorflow as tf
from params import DATA_PARAMS


def generate_samples(T, n, num_features):
    """
    Generate synthetic data according to the specified model
    Returns:
        tuple: (Y, X, Z, beta) tensors
    """
    # Check if inputs are integers
    if not isinstance(T, int) or not isinstance(n, int) or not isinstance(num_features, int):
        raise tf.errors.InvalidArgumentError(
            None, None,
            f"All inputs must be integers, got types: T={type(T)}, n={type(n)}, num_features={type(num_features)}"
        )

    if T <= 0 or n <= 0:
        raise tf.errors.InvalidArgumentError(
            None, None,
            f"T and n must be positive integers, got T={T}, n={n}"
        )

    generator = tf.random.Generator.from_seed(42)
    # 1. Initialize beta and Z
    beta = generator.normal([num_features], mean=0.0, stddev=1.0)  # shape: [8]
    Z = generator.normal([T, n, num_features], mean=0.0, stddev=1.0)  # shape: [T, n, 8]

    # 2. Generate X as mixture of two normal distributions
    mu1 = DATA_PARAMS['MU1']
    mu2 = DATA_PARAMS['MU2']
    lambda1 = DATA_PARAMS['LAMBDA1']
    lambda2 = DATA_PARAMS['LAMBDA2']
    w1 = DATA_PARAMS['W1']
    w2 = 1.0 - w1

    # Convert precision to standard deviation
    std1 = tf.sqrt(1.0 / lambda1)
    std2 = tf.sqrt(1.0 / lambda2)
    # Generate indicator variables for mixture components
    z = generator.uniform([T]) < w1  # Bernoulli with p=w1

    # Generate samples from each component
    X1 = generator.normal([T], mean=mu1, stddev=std1)
    X2 = generator.normal([T], mean=mu2, stddev=std2)

    # Mix the components according to the indicator variables
    X = tf.where(z, X1, X2)
    #X = w1 * X1 + (1-w1) * X2  # shape: [T]

    # 3. Compute Z_ij^T * beta and combine with X
    # Reshape X from [T] to [T, 1]
    X_reshaped = tf.expand_dims(X, axis=1)  # shape: [T, 1]

    # Compute Z_ij^T * beta using batch matmul
    Z_beta = tf.einsum('tnf,f->tn', Z, beta)  # shape: [T, n]

    # Broadcasting will automatically handle the addition
    combined = X_reshaped + Z_beta  # shape: [T, n]

    # 4. Generate p_ij using logit^(-1) (sigmoid)
    p_ij = tf.sigmoid(combined)  # shape: [T, n]

    # 5. Generate Y from Bernoulli distribution
    Y = generator.uniform([T, n]) < p_ij  # shape: [T, n]
    Y = tf.cast(Y, tf.float32)

    return Y, X, Z, beta


if __name__ == '__main__':
    import os
    import matplotlib.pyplot as plt

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    # test distribution
    T = 10000
    _, X, _, _ = generate_samples(T, 10, 8)

    plt.hist(X.numpy(), bins=50, density=True, alpha=0.6, color='g')
    plt.title("Histogram of X")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()
