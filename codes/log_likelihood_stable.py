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
