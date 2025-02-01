import tensorflow as tf
import numpy as np
from params import AUX_SCALE, PRIOR_VARIANCE
from generate_samples import generate_samples


@tf.function
def generate_X(u):
    """
    Generate latent variable X from auxiliary variable u

    Args:
        u: auxiliary variable [T, N, p]

    Returns:
        X: latent variable [T, N]
    """
    X = AUX_SCALE * u  # [T, N]
    return X


@tf.function
def compute_logg(y, X, Z, beta):
    """
    Compute log g_theta(y|X)

    Args:
        y: observations [T, n]
        X: latent variable [T, N]
        Z: covariates [T, n, p_z]
        beta: coefficients [p_z]

    Returns:
        log_g: log likelihood [T, N]
    """
    # Expand dimensions for broadcasting
    X_expanded = tf.expand_dims(X, axis=-1)  # [T, N, 1]
    Z_beta = tf.einsum('tnp,p->tn', Z, beta)  # [T, n]
    Z_beta_expanded = tf.expand_dims(Z_beta, axis=1)  # [T, 1, n]

    # Calculate logits
    logits = X_expanded + Z_beta_expanded  # [T, N, n]

    # Expand y
    y_expanded = tf.expand_dims(y, axis=1)  # [T, 1, n]

    # Calculate log probabilities
    #log_probs = y_expanded * (-tf.nn.softplus(-logits)) + \
    #            (1 - y_expanded) * (-tf.nn.softplus(logits))

    log_probs = y_expanded * tf.math.log(1 + tf.exp(-logits)) + (1 - y_expanded) * tf.math.log(1 + tf.exp(logits))

    # Sum over n dimension
    return -tf.reduce_sum(log_probs, axis=2)  # [T, N]


@tf.function
def compute_logq(u):
    """
    Compute log q_theta(X)

    Args:
        u: auxiliary variable [T, N, p]

    Returns:
        log_q: log importance density [T, N]
    """
    logqu = -0.5 * tf.math.log(2 * np.pi) - 0.5 * tf.math.log(9.0) - 0.5 * tf.square(u)
    return logqu  # [T, N]


@tf.function
def compute_logf_components(X, mu1, mu2, log_lambda1, log_lambda2, eta):
    """
    Compute components needed for log f_theta(X)

    Args:
        X: latent variable [T, N]
        mu1, mu2: means
        log_lambda1, log_lambda2: log precisions
        eta: logit(1-w1)

    Returns:
        z1, z2: log components [T, N]
        relative_weights: [T, N, 2]
    """
    # Calculate precisions and weights
    lambda1 = tf.exp(log_lambda1)
    lambda2 = tf.exp(log_lambda2)
    w2 = tf.sigmoid(eta)
    w1 = 1 - w2

    # Calculate z1, z2
    z1 = tf.math.log(w1) + 0.5 * log_lambda1 - 0.5 * tf.math.log(2.0) - 0.5 * tf.math.log(np.pi) \
         - 0.5 * lambda1 * tf.square(X - mu1)  # [T, N]

    z2 = tf.math.log(w2) + 0.5 * log_lambda2 - 0.5 * tf.math.log(2.0) - 0.5 * tf.math.log(np.pi) \
         - 0.5 * lambda2 * tf.square(X - mu2)  # [T, N]

    # Calculate logf using log-sum-exp trick
    maxz = tf.maximum(z1, z2)  # [T, N]
    logf = maxz + tf.math.log(tf.exp(z1 - maxz) + tf.exp(z2 - maxz))  # [T, N]

    # Calculate relative weights
    relative_weights = tf.stack([
        tf.exp(z1 - logf),
        tf.exp(z2 - logf)
    ], axis=-1)  # [T, N, 2]

    return z1, z2, relative_weights, logf


@tf.function
def compute_logf_gradients(X, mu1, mu2, log_lambda1, log_lambda2, eta, relative_weights, z1, z2, logf):
    """
    Compute gradients of log f_theta with respect to parameters

    Args:
        X: latent variable [T, N]
        mu1, mu2: means
        log_lambda1, log_lambda2: log precisions
        eta: logit(1-w1)
        relative_weights: weights from compute_logf_components [T, N, 2]
        z1, z2: log components [T, N]
        logf: log f_theta [T, N]

    Returns:
        grad_mu: gradients wrt mu1, mu2 [T, N, 2]
        grad_lambda: gradients wrt log_lambda1, log_lambda2 [T, N, 2]
        grad_eta: gradient wrt eta [T, N]
    """
    # Calculate precisions
    lambda1 = tf.exp(log_lambda1)
    lambda2 = tf.exp(log_lambda2)

    # Expand X for broadcasting
    X_expanded = tf.expand_dims(X, -1)  # [T, N, 1]

    # Gradients wrt mu1, mu2
    grad_mu = tf.stack([
        lambda1 * (X - mu1),
        lambda2 * (X - mu2)
    ], axis=2) * relative_weights  # [T, N, 2]

    # Gradients wrt log_lambda1, log_lambda2
    grad_lambda = tf.stack([
        0.5 - 0.5 * lambda1 * tf.square(X - mu1),
        0.5 - 0.5 * lambda2 * tf.square(X - mu2)
    ], axis=2) * relative_weights  # [T, N, 2]

    # Gradient wrt eta - corrected version according to PDF
    fct = 1 / (1 + tf.exp(eta))

    # Compute term1 and term2
    term1 = z2 + tf.math.log(fct)  # [T, N]
    term2 = z1 + tf.math.log(fct) + eta  # [T, N]

    # Compute gradient
    grad_eta = tf.exp(term1 - logf) - tf.exp(term2 - logf)  # [T, N]

    return grad_mu, grad_lambda, grad_eta


@tf.function
def compute_gradient_X(X, mu1, mu2, log_lambda1, log_lambda2, relative_weights):
    """
    Compute gradient of log f_theta with respect to X

    Args:
        X: latent variable [T, N]
        mu1, mu2: means
        log_lambda1, log_lambda2: log precisions
        relative_weights: weights [T, N, 2]

    Returns:
        grad_X: gradient [T, N]
    """
    # Calculate precisions
    lambda1 = tf.exp(log_lambda1)
    lambda2 = tf.exp(log_lambda2)

    # Calculate each component's gradient and multiply by its relative weight
    grad_X = -(lambda1 * (X - mu1) * relative_weights[..., 0] +
               lambda2 * (X - mu2) * relative_weights[..., 1])  # [T, N]

    return grad_X


@tf.function
def compute_gradient_u_logg(X, A):
    """
    Compute gradient of log g_theta with respect to u from auxiliary variable

    Args:
        X: latent variable [T, N]
        A: residuals [T, N, n]

    Returns:
        grad_u_g: gradient [T, N]
    """
    return -tf.reduce_sum(A, axis=2) * AUX_SCALE  # [T, N]


@tf.function
def compute_gradient_u_logf(X, mu1, mu2, log_lambda1, log_lambda2, relative_weights):
    """
    Compute gradient of log f_theta with respect to u

    Args:
        X: latent variable [T, N]
        mu1, mu2: means
        log_lambda1, log_lambda2: log precisions
        relative_weights: weights [T, N, 2]

    Returns:
        grad_u_f: gradient [T, N]
    """
    grad_X = compute_gradient_X(X, mu1, mu2, log_lambda1, log_lambda2, relative_weights)  # [T, N]
    return AUX_SCALE * grad_X  # [T, N]


@tf.function
def compute_gradient_u_logq(u):
    """
    Compute gradient of log q with respect to u

    Args:
        u: auxiliary variable [T, N, p]

    Returns:
        grad_u_q: gradient [T, N]
    """
    return -u  # [T, N]


@tf.function
def compute_gradient_u(u, X, mu1, mu2, log_lambda1, log_lambda2, A, relative_weights):
    """
    Compute total gradient with respect to u

    Args:
        u: auxiliary variable [T, N, p]
        X: latent variable [T, N]
        mu1, mu2: means
        log_lambda1, log_lambda2: log precisions
        A: residuals [T, N, n]
        relative_weights: weights [T, N, 2]

    Returns:
        grad_u: gradient [T, N]
    """
    grad_u_g = compute_gradient_u_logg(X, A)  # [T, N]
    grad_u_f = compute_gradient_u_logf(X, mu1, mu2, log_lambda1, log_lambda2, relative_weights)  # [T, N]
    grad_u_q = compute_gradient_u_logq(u)  # [T, N]

    return grad_u_g + grad_u_f - grad_u_q  # [T, N]


@tf.function
def compute_normalized_weights(log_omega):
    """
    Compute normalized importance weights

    Args:
        log_omega: log weights [T, N]

    Returns:
        normalized_weights: weights [T, N]
        log_likelihood: scalar
    """
    # Compute max for numerical stability
    max_omega = tf.reduce_max(log_omega, axis=1, keepdims=True)  # [T, 1]

    # Compute normalized weights
    omega = tf.exp(log_omega - max_omega)  # [T, N]
    sum_omega = tf.reduce_sum(omega, axis=1, keepdims=True)  # [T, 1]
    normalized_weights = omega / sum_omega  # [T, N]

    # Compute log likelihood
    log_likelihood = tf.reduce_sum(
        tf.math.log(sum_omega) + max_omega - tf.math.log(tf.cast(tf.shape(omega)[1], tf.float32))
    )  # scalar

    return normalized_weights, log_likelihood


@tf.function
def compute_log_prior(theta):
    """
    Compute log prior probability

    Args:
        theta: parameters vector

    Returns:
        log_prior: scalar
    """
    variance = PRIOR_VARIANCE
    log_prior = -0.5 * (tf.size(theta, out_type=tf.float32) * tf.math.log(2 * np.pi * variance) +
                        tf.reduce_sum(tf.square(theta)) / variance)
    grad_prior_theta = -theta / variance
    return log_prior, grad_prior_theta


@tf.function
def combine_parameter_gradients(normalized_weights, grad_beta, grad_mu, grad_lambda, grad_eta):
    """
    Combine gradients for all parameters

    Args:
        normalized_weights: [T, N]
        grad_beta: [T, N, p_z]
        grad_mu: [T, N, 2]
        grad_lambda: [T, N, 2]
        grad_eta: [T, N]

    Returns:
        grad_theta: combined gradients [p_z + 5]
    """
    # Add dimension to normalized_weights for broadcasting
    weights_expanded = tf.expand_dims(normalized_weights, -1)  # [T, N, 1]

    # Weight and sum gradients
    weighted_grad_beta = tf.reduce_sum(weights_expanded * grad_beta, axis=[0, 1])  # [p_z]
    weighted_grad_mu = tf.reduce_sum(weights_expanded * grad_mu, axis=[0, 1])  # [2]
    weighted_grad_lambda = tf.reduce_sum(weights_expanded * grad_lambda, axis=[0, 1])  # [2]
    weighted_grad_eta = tf.reduce_sum(normalized_weights * grad_eta, axis=[0, 1])  # scalar

    # Combine all gradients
    return tf.concat([
        weighted_grad_beta,  # [p_z]
        weighted_grad_mu,  # [2]
        weighted_grad_lambda,  # [2]
        weighted_grad_eta[None]  # [1]
    ], axis=0)  # [p_z + 5]


@tf.function
def compute_gradient_beta(y, X, Z, beta):
    """
    Compute gradient of log g_theta with respect to beta

    Args:
        y: observations [T, n]
        X: latent variable [T, N]
        Z: covariates [T, n, p_z]
        beta: coefficients [p_z]

    Returns:
        grad_beta: gradient [T, N, p_z]
    """
    # Calculate logits and probabilities
    X_expanded = tf.expand_dims(X, -1)  # [T, N, 1]
    Z_beta = tf.einsum('tnp,p->tn', Z, beta)  # [T, n]
    Z_beta_expanded = tf.expand_dims(Z_beta, 1)  # [T, 1, n]
    logits = X_expanded + Z_beta_expanded  # [T, N, n]

    probs = tf.sigmoid(logits)  # [T, N, n]

    # Calculate A
    y_expanded = tf.expand_dims(y, 1)  # [T, 1, n]
    A = (1 - y_expanded) * probs - y_expanded * (1 - probs)  # [T, N, n]

    # Reshape Z from [T, n, p_z] to [T, p_z, n]
    Z_reshaped = tf.transpose(Z, perm=[0, 2, 1])  # [T, p_z, n]

    # Expand dimensions for broadcasting
    A_expanded = tf.expand_dims(A, 2)  # [T, N, 1, n]
    Z_reshaped_expanded = tf.expand_dims(Z_reshaped, 1)  # [T, 1, p_z, n]

    # Calculate gradient using broadcasting
    grad_beta = -tf.reduce_sum(A_expanded * Z_reshaped_expanded, axis=3)  # [T, N, p_z]

    return grad_beta, A


@tf.function
def compute_gradient_beta_loop(y, X, Z, beta):
    """
    Compute gradient of log g_theta with respect to beta using loop structure

    Args:
        y: observations [T, n]
        X: latent variable [T, N]
        Z: covariates [T, n, p_z]
        beta: coefficients [p_z]

    Returns:
        grad_beta: gradient [T, N, p_z]
        A: residuals [T, N, n]
    """
    # Calculate logits and probabilities
    X_expanded = tf.expand_dims(X, -1)  # [T, N, 1]
    Z_beta = tf.einsum('tnp,p->tn', Z, beta)  # [T, n]
    Z_beta_expanded = tf.expand_dims(Z_beta, 1)  # [T, 1, n]
    logits = X_expanded + Z_beta_expanded  # [T, N, n]

    probs = tf.sigmoid(logits)  # [T, N, n]

    # Calculate A
    y_expanded = tf.expand_dims(y, 1)  # [T, 1, n]
    A = (1 - y_expanded) * probs - y_expanded * (1 - probs)  # [T, N, n]

    # Reshape Z from [T, n, p_z] to [T, p_z, n]
    Z_reshaped = tf.transpose(Z, perm=[0, 2, 1])  # [T, p_z, n]

    # Initialize TensorArray for grad_beta
    p_z = tf.shape(Z)[-1]
    grad_beta = tf.TensorArray(tf.float32, size=p_z)

    # Loop over p_z dimension
    for i in tf.range(p_z):
        # Get Z_i [T, n]
        Z_i = Z_reshaped[:, i, :]  # [T, n]
        # Expand Z_i for broadcasting
        Z_i_expanded = tf.expand_dims(Z_i, 1)  # [T, 1, n]
        # Calculate Z_i * A and sum over n dimension
        gbeta_i = -tf.reduce_sum(Z_i_expanded * A, axis=2)  # [T, N]
        # Write to TensorArray
        grad_beta = grad_beta.write(i, gbeta_i)

    # Stack results and transpose to get [T, N, p_z]
    grad_beta = tf.transpose(grad_beta.stack(), perm=[1, 2, 0])

    return grad_beta, A


@tf.function
def compute_gradient_beta_alt(y, X, Z, beta):
    """
    Compute gradient of log g_theta with respect to beta

    Args:
        y: observations [T, n]
        X: latent variable [T, N]
        Z: covariates [T, n, p_z]
        beta: coefficients [p_z]
    """
    # Calculate logits and probabilities
    X_expanded = tf.expand_dims(X, -1)  # [T, N, 1]
    Z_beta = tf.einsum('tnp,p->tn', Z, beta)  # [T, n]
    Z_beta_expanded = tf.expand_dims(Z_beta, 1)  # [T, 1, n]
    logits = X_expanded + Z_beta_expanded  # [T, N, n]

    probs = tf.sigmoid(logits)  # [T, N, n]

    # Calculate A
    y_expanded = tf.expand_dims(y, 1)  # [T, 1, n]
    A = (1 - y_expanded) * probs - y_expanded * (1 - probs)  # [T, N, n]

    # Method 1: Using einsum
    grad_beta = -tf.einsum('tnp,tNn->tNp', Z, A)

    return grad_beta, A  # [T, N, p_z]


@tf.function
def compute_log_likelihood_and_gradients(theta, u, y, Z):
    """
    Main function that computes log likelihood and its gradients

    Args:
        theta: parameters {beta, mu1, mu2, log_lambda1, log_lambda2, eta} [p_z + 5]
        u: auxiliary variable [T, N, p]
        y: observations [T, n]
        Z: covariates [T, n, p_z]

    Returns:
        prior_log_likelihood: scalar
        grad_u: gradient w.r.t u [T, N]
        grad_theta: gradient w.r.t theta [p_z + 5]
    """
    # Extract parameters from theta
    p_z = tf.shape(Z)[-1]
    beta = theta[:p_z]
    mu1 = theta[p_z]
    mu2 = theta[p_z + 1]
    log_lambda1 = theta[p_z + 2]
    log_lambda2 = theta[p_z + 3]
    eta = theta[p_z + 4]  # logit(1-w1)

    # 1. Generate X from u
    X = generate_X(u)  # [T, N]

    # 2. Calculate log g_theta and its gradients
    grad_beta, A = compute_gradient_beta_alt(y, X, Z, beta)  # [T, N, p_z], [T, N, n]
    log_g = compute_logg(y, X, Z, beta)  # [T, N]

    # 3. Calculate log f_theta components
    z1, z2, relative_weights, log_f = compute_logf_components(
        X, mu1, mu2, log_lambda1, log_lambda2, eta
    )  # [T, N], [T, N], [T, N, 2], [T, N]

    # 4. Calculate gradients of log f_theta
    grad_mu, grad_lambda, grad_eta = compute_logf_gradients(
        X, mu1, mu2, log_lambda1, log_lambda2, eta,
        relative_weights, z1, z2, log_f
    )  # [T, N, 2], [T, N, 2], [T, N]

    # 5. Calculate log q_theta
    log_q = compute_logq(u)  # [T, N]

    # 6. Compute log weights
    log_omega = log_g + log_f - log_q  # [T, N]

    # 7. Calculate normalized weights and log likelihood
    normalized_weights, log_likelihood = compute_normalized_weights(log_omega)  # [T, N], scalar

    # 8. Calculate gradients w.r.t u
    grad_u = compute_gradient_u(u, X, mu1, mu2, log_lambda1, log_lambda2, A, relative_weights)

    # Complete gradient calculation for u using normalized weights
    grad_u = normalized_weights * grad_u

    # 9. Combine all parameter gradients
    grad_theta = combine_parameter_gradients(
        normalized_weights, grad_beta, grad_mu, grad_lambda, grad_eta
    )  # [p_z + 5]

    # 10. Calculate log prior
    log_prior, grad_prior_theta = compute_log_prior(theta)  # scalar

    # Final prior log likelihood
    prior_log_likelihood = log_likelihood + log_prior  # scalar

    return prior_log_likelihood, grad_u, grad_theta + grad_prior_theta


def test_gradient_computation():
    """
    Test function to verify the gradient computation
    """
    # Set random seed
    tf.random.set_seed(42)

    # Generate test data
    T, N, n, p_z = 500, 128, 6, 8
    p = 1  # dimension of auxiliary variable

    # Initialize parameters and variables
    theta = tf.random.normal([p_z + 5])  # beta (p_z) + mu1 + mu2 + log_lambda1 + log_lambda2 + eta
    u = tf.random.normal([T, N, p])
    y = tf.cast(tf.random.uniform([T, n]) > 0.5, tf.float32)
    Z = tf.random.normal([T, n, p_z])

    # Compute log likelihood and gradients
    log_likelihood, grad_u, grad_theta = compute_log_likelihood_and_gradients(theta, u, y, Z)

    # Print results
    print("Test results:")
    print(f"Input shapes:")
    print(f"theta: {theta.shape}")
    print(f"u: {u.shape}")
    print(f"y: {y.shape}")
    print(f"Z: {Z.shape}")
    print(f"\nOutput shapes:")
    print(f"log_likelihood: scalar = {log_likelihood.numpy()}")
    print(f"grad_u: {grad_u.shape}")
    print(f"grad_theta: {grad_theta.shape}")
    print(f"\nGradient samples:")
    print(f"grad_u[0,0]: {grad_u[0, 0].numpy()}")
    print(f"grad_theta[:3]: {grad_theta[:3].numpy()}")

    return log_likelihood, grad_u, grad_theta


def test_gradient_beta_implementations():
    """
    Test and compare three implementations of gradient_beta computation
    """
    # Set random seed
    tf.random.set_seed(42)

    # Generate test data
    T, N, n, p_z = 500, 128, 6, 8

    # Initialize parameters
    X = tf.random.normal([T, N])
    y = tf.cast(tf.random.uniform([T, n]) > 0.5, tf.float32)
    Z = tf.random.normal([T, n, p_z])
    beta = tf.random.normal([p_z])

    # Compute gradients using all three methods
    grad1, A1 = compute_gradient_beta(y, X, Z, beta)  # Original broadcasting
    grad2, A2 = compute_gradient_beta_alt(y, X, Z, beta)  # Einsum
    grad3, A3 = compute_gradient_beta_loop(y, X, Z, beta)  # Loop

    # Compare results
    diff12 = tf.reduce_max(tf.abs(grad1 - grad2))
    diff13 = tf.reduce_max(tf.abs(grad1 - grad3))
    diff23 = tf.reduce_max(tf.abs(grad2 - grad3))

    # Time comparison
    num_runs = 100

    # Time first implementation (broadcasting)
    start_time = time.time()
    for _ in range(num_runs):
        grad1, _ = compute_gradient_beta(y, X, Z, beta)
    time1 = time.time() - start_time

    # Time second implementation (einsum)
    start_time = time.time()
    for _ in range(num_runs):
        grad2, _ = compute_gradient_beta_alt(y, X, Z, beta)
    time2 = time.time() - start_time

    # Time third implementation (loop)
    start_time = time.time()
    for _ in range(num_runs):
        grad3, _ = compute_gradient_beta_loop(y, X, Z, beta)
    time3 = time.time() - start_time

    # Print results
    print("\nTest results:")
    print(f"Input shapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")
    print(f"Z: {Z.shape}")
    print(f"beta: {beta.shape}")

    print(f"\nOutput shapes:")
    print(f"grad1: {grad1.shape}")
    print(f"grad2: {grad2.shape}")
    print(f"grad3: {grad3.shape}")

    print(f"\nMaximum absolute differences:")
    print(f"Between broadcast and einsum: {diff12.numpy():.2e}")
    print(f"Between broadcast and loop: {diff13.numpy():.2e}")
    print(f"Between einsum and loop: {diff23.numpy():.2e}")

    print(f"\nTiming results ({num_runs} runs):")
    print(f"Broadcasting: {time1 / num_runs * 1000:.2f} ms per run")
    print(f"Einsum: {time2 / num_runs * 1000:.2f} ms per run")
    print(f"Loop: {time3 / num_runs * 1000:.2f} ms per run")

    return grad1, grad2, grad3, A1, A2, A3


def test_relative_weights():
    """
    Test the computation and properties of relative weights
    """
    # Set random seed
    tf.random.set_seed(42)

    # Generate test data
    T, N = 3, 4  # Small numbers for easy visualization

    # Initialize parameters
    mu1 = 0.0
    mu2 = 3.0
    log_lambda1 = 0.0  # lambda1 = 1
    log_lambda2 = -2.3  # lambda2 = 0.1
    eta = 0.0  # w1 = w2 = 0.5

    # Generate X values around mu1 and mu2
    X = tf.constant([
        [0.1, 3.1, 0.2, 2.9],  # First batch
        [-0.1, 2.8, 0.3, 3.2],  # Second batch
        [0.0, 3.0, 0.1, 3.1]  # Third batch
    ], dtype=tf.float32)  # [T, N]

    # Compute components
    z1, z2, relative_weights, logf = compute_logf_components(
        X, mu1, mu2, log_lambda1, log_lambda2, eta
    )

    # Print shapes
    print("\nShape test:")
    print(f"X shape: {X.shape}")
    print(f"z1 shape: {z1.shape}")
    print(f"z2 shape: {z2.shape}")
    print(f"relative_weights shape: {relative_weights.shape}")
    print(f"logf shape: {logf.shape}")

    # Verify that relative weights sum to 1
    weights_sum = tf.reduce_sum(relative_weights, axis=-1)
    max_diff_from_one = tf.reduce_max(tf.abs(weights_sum - 1.0))
    print("\nWeights validation:")
    print(f"Maximum deviation from sum=1: {max_diff_from_one:.2e}")

    # Print actual values
    print("\nExample values:")
    print("X values:")
    print(X.numpy())
    print("\nRelative weights:")
    print(relative_weights.numpy())
    print("\nFirst component weights (should be close to 1 for X near mu1=0):")
    print(relative_weights[:, :, 0].numpy())
    print("\nSecond component weights (should be close to 1 for X near mu2=3):")
    print(relative_weights[:, :, 1].numpy())

    # Additional checks
    print("\nSpecific checks:")
    # Points near mu1 should have higher weight for first component
    near_mu1_mask = tf.abs(X - mu1) < 1.0
    weights_near_mu1 = tf.boolean_mask(relative_weights[:, :, 0], near_mu1_mask)
    print(f"Average weight for first component near mu1: {tf.reduce_mean(weights_near_mu1):.3f}")

    # Points near mu2 should have higher weight for second component
    near_mu2_mask = tf.abs(X - mu2) < 1.0
    weights_near_mu2 = tf.boolean_mask(relative_weights[:, :, 1], near_mu2_mask)
    print(f"Average weight for second component near mu2: {tf.reduce_mean(weights_near_mu2):.3f}")

    return z1, z2, relative_weights, logf


def test_gradient_distributions():
    """
    Test to analyze the distributions and ranges of gradients
    using generated samples from generate_samples
    """
    # Set random seed
    tf.random.set_seed(42)

    # Generate test data using generate_samples
    T, N, n = 500, 128, 6
    num_features = 8
    Y, _, Z, beta = generate_samples(T=T, n=n, num_features=num_features)

    # Initialize other parameters (using reasonable values)
    mu1 = tf.constant(0.0)
    mu2 = tf.constant(3.0)
    log_lambda1 = tf.constant(0.0)  # lambda1 = 1
    log_lambda2 = tf.constant(-2.3)  # lambda2 = 0.1
    eta = tf.constant(0.0)  # w1 = w2 = 0.5

    # Combine all parameters
    theta = tf.concat([
        beta,
        [mu1, mu2, log_lambda1, log_lambda2, eta]
    ], axis=0)

    # Initialize auxiliary variable
    u = tf.random.normal([T, N])

    # Compute gradients
    _, grad_u, grad_theta = compute_log_likelihood_and_gradients(theta, u, Y, Z)

    # Analyze gradients
    print("\nGradient Analysis:")

    # For grad_theta
    print("\ngrad_theta analysis:")
    print(f"Shape: {grad_theta.shape}")
    print(f"Mean: {tf.reduce_mean(grad_theta):.4f}")
    print(f"Std: {tf.math.reduce_std(grad_theta):.4f}")
    print(f"Min: {tf.reduce_min(grad_theta):.4f}")
    print(f"Max: {tf.reduce_max(grad_theta):.4f}")

    # Split grad_theta into components
    grad_beta = grad_theta[:num_features]
    grad_mu = grad_theta[num_features:num_features + 2]
    grad_lambda = grad_theta[num_features + 2:num_features + 4]
    grad_eta = grad_theta[-1]

    print("\ngrad_theta components:")
    components = {
        'grad_beta': grad_beta,
        'grad_mu': grad_mu,
        'grad_lambda': grad_lambda,
        'grad_eta': grad_eta
    }

    for name, grad in components.items():
        print(f"\n{name}:")
        print(f"Mean: {tf.reduce_mean(grad):.4f}")
        print(f"Std: {tf.math.reduce_std(grad):.4f}")
        print(f"Min: {tf.reduce_min(grad):.4f}")
        print(f"Max: {tf.reduce_max(grad):.4f}")

    # For grad_u
    print("\ngrad_u analysis:")
    print(f"Shape: {grad_u.shape}")
    print(f"Mean: {tf.reduce_mean(grad_u):.4f}")
    print(f"Std: {tf.math.reduce_std(grad_u):.4f}")
    print(f"Min: {tf.reduce_min(grad_u):.4f}")
    print(f"Max: {tf.reduce_max(grad_u):.4f}")

    # Optional: Create histograms if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        import os
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

        # Plot grad_theta components
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Gradient Distributions')

        axes[0, 0].hist(grad_beta.numpy(), bins=50)
        axes[0, 0].set_title('grad_beta')

        axes[0, 1].hist(grad_mu.numpy(), bins=50)
        axes[0, 1].set_title('grad_mu')

        axes[1, 0].hist(grad_lambda.numpy(), bins=50)
        axes[1, 0].set_title('grad_lambda')

        axes[1, 1].hist(tf.reshape(grad_eta, [1]).numpy(), bins=50)
        axes[1, 1].set_title('grad_eta')

        plt.tight_layout()
        plt.show()

        # Plot grad_u distribution
        plt.figure(figsize=(8, 6))
        plt.hist(tf.reshape(grad_u, [-1]).numpy(), bins=50)
        plt.title('grad_u distribution')
        plt.xlabel('Value')
        plt.ylabel('Count')
        plt.show()

    except ImportError:
        print("\nMatplotlib not available for plotting.")

    return grad_u, grad_theta, components


def test_gradient_consistency():
    """
    Test whether computed gradients are consistent with automatic differentiation.

    Returns:
        dict: Dictionary containing test results with the following keys:
            - auto_grad_theta: Gradients for theta computed by automatic differentiation
            - grad_theta_direct: Gradients for theta computed directly
            - auto_grad_u: Gradients for u computed by automatic differentiation
            - grad_u_direct: Gradients for u computed directly
            - theta_diff: Maximum absolute difference in theta gradients
            - u_diff: Maximum absolute difference in u gradients
            - is_passed: Boolean indicating whether all tests passed
    """
    # Set random seed
    tf.random.set_seed(42)

    # Generate test data
    T, N, n = 10, 32, 6  # Using smaller data for testing
    num_features = 8
    Y, _, Z, beta = generate_samples(T=T, n=n, num_features=num_features)

    # Initialize parameters
    mu1 = tf.constant(0.0)
    mu2 = tf.constant(3.0)
    log_lambda1 = tf.constant(0.0)
    log_lambda2 = tf.constant(-2.3)
    eta = tf.constant(0.0)

    # Combine parameters
    theta = tf.concat([
        beta,
        [mu1, mu2, log_lambda1, log_lambda2, eta]
    ], axis=0)

    # Initialize auxiliary variables
    u = tf.random.normal([T, N])

    # Convert to variables for automatic differentiation
    theta_var = tf.Variable(theta)
    u_var = tf.Variable(u)

    # Compute gradients using automatic differentiation
    with tf.GradientTape(persistent=True) as tape:
        ll, _, _ = compute_log_likelihood_and_gradients(theta_var, u_var, Y, Z)

    auto_grad_theta = tape.gradient(ll, theta_var)
    auto_grad_u = tape.gradient(ll, u_var)

    # Compute gradients directly from function
    ll_direct, grad_u_direct, grad_theta_direct = compute_log_likelihood_and_gradients(theta, u, Y, Z)

    # Calculate differences
    theta_diff = tf.reduce_max(tf.abs(auto_grad_theta - grad_theta_direct))
    u_diff = tf.reduce_max(tf.abs(auto_grad_u - grad_u_direct))

    print("\nGradient Consistency Test Results:")
    print(f"Maximum theta gradient difference: {theta_diff:.2e}")
    print(f"Maximum u gradient difference: {u_diff:.2e}")

    # Detailed component comparison
    print("\nDetailed Gradient Comparison:")
    print("\ntheta gradients:")
    print("Gradients computed by automatic differentiation:")
    print(auto_grad_theta.numpy())
    print("\nDirectly computed gradients:")
    print(grad_theta_direct.numpy())

    print("\nu gradients (partial display):")
    print("Gradients computed by automatic differentiation (first 3 rows):")
    print(auto_grad_u[:3].numpy())
    print("\nDirectly computed gradients (first 3 rows):")
    print(grad_u_direct[:3].numpy())

    # Check if within tolerance
    tolerance = 1e-5
    is_theta_close = theta_diff < tolerance
    is_u_close = u_diff < tolerance

    print("\nTest Conclusion:")
    print(f"theta gradient consistency: {'PASS' if is_theta_close else 'FAIL'}")
    print(f"u gradient consistency: {'PASS' if is_u_close else 'FAIL'}")

    return {
        'auto_grad_theta': auto_grad_theta,
        'grad_theta_direct': grad_theta_direct,
        'auto_grad_u': auto_grad_u,
        'grad_u_direct': grad_u_direct,
        'theta_diff': theta_diff,
        'u_diff': u_diff,
        'is_passed': is_theta_close and is_u_close
    }


if __name__ == "__main__":
    # Run tests
    import time

    try:
        results = test_gradient_consistency()
        if results['is_passed']:
            print("\nAll tests passed! ✓")
        else:
            print("\nTests failed. Please check gradient computation implementation.")
    except Exception as e:
        print(f"\nError occurred during testing: {str(e)}")
