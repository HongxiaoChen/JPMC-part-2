import tensorflow as tf
import numpy as np
from params import AUX_SCALE, PRIOR_VARIANCE


@tf.function
def generate_X(u, eta, log_lambda1, log_lambda2):
    """
    Generate latent variable X from auxiliary variable u.

    Args:
        u: Auxiliary variable tensor

    Returns:
        X: Generated latent variable tensor
    """

    return AUX_SCALE * u


@tf.function
def compute_logg(y, X, Z, beta):
    """
    Compute log g_theta(y|X).

    Args:
        y: Observed data
        X: Latent variables
        Z: Covariates
        beta: Parameters

    Returns:
        tensor: Log likelihood of observations given latent variables
    """
    X_expanded = tf.expand_dims(X, axis=-1)
    Z_beta = tf.einsum('tnp,p->tn', Z, beta)
    Z_beta_expanded = tf.expand_dims(Z_beta, axis=1)
    logits = X_expanded + Z_beta_expanded

    y_expanded = tf.expand_dims(y, axis=1)
    log_probs = y_expanded * tf.math.log(1 + tf.exp(-logits)) + (1 - y_expanded) * tf.math.log(1 + tf.exp(logits))
    return -tf.reduce_sum(log_probs, axis=2)


@tf.function
def compute_logf_components(X, mu1, mu2, log_lambda1, log_lambda2, eta):
    """
    Compute components of log f_theta(X).

    Args:
        X: Latent variables
        mu1: Mean of first component
        mu2: Mean of second component
        log_lambda1: Log precision of first component
        log_lambda2: Log precision of second component
        eta: Logit of mixing proportion

    Returns:
        tensor: Log density of latent variables
    """
    lambda1 = tf.exp(log_lambda1)
    lambda2 = tf.exp(log_lambda2)

    log_w2 = -tf.nn.softplus(-eta)  # log(w2) = -softplus(-eta)
    log_w1 = -tf.nn.softplus(eta)  # log(w1) = -softplus(eta)

    z1 = log_w1 + 0.5 * log_lambda1 - 0.5 * tf.math.log(2.0) - 0.5 * tf.math.log(np.pi) \
         - 0.5 * lambda1 * tf.square(X - mu1)

    z2 = log_w2 + 0.5 * log_lambda2 - 0.5 * tf.math.log(2.0) - 0.5 * tf.math.log(np.pi) \
         - 0.5 * lambda2 * tf.square(X - mu2)

    maxz = tf.maximum(z1, z2)
    logf = maxz + tf.math.log(tf.exp(z1 - maxz) + tf.exp(z2 - maxz))

    return logf


@tf.function
def compute_logq(u):
    """
    Compute log q_theta(X). Since X is a function of u, I use u as the instead input.

    Args:
        u: Auxiliary variables

    Returns:
        tensor: Log density of auxiliary variables
    """
    logqu = -0.5 * tf.math.log(2 * np.pi) - 0.5 * tf.math.log(9.0) - 0.5 * tf.square(u)
    return logqu


@tf.function
def compute_normalized_weights(log_omega):
    """
    Compute normalized importance weights.

    Args:
        log_omega: Log of unnormalized weights

    Returns:
        tuple: (normalized_weights, log_likelihood)
            - normalized_weights: Normalized importance weights
            - log_likelihood: Log likelihood estimate
    """
    max_omega = tf.reduce_max(log_omega, axis=1, keepdims=True)
    omega = tf.exp(log_omega - max_omega)
    sum_omega = tf.reduce_sum(omega, axis=1, keepdims=True)
    normalized_weights = omega / sum_omega
    log_likelihood = tf.reduce_sum(
        tf.math.log(sum_omega) + max_omega - tf.math.log(tf.cast(tf.shape(omega)[1], tf.float32))
    )
    return normalized_weights, log_likelihood


@tf.function
def compute_log_prior(theta):
    """
    Compute log prior probability.

    Args:
        theta: Parameter vector

    Returns:
        float: Log prior probability
    """
    variance = PRIOR_VARIANCE
    log_prior = -0.5 * (tf.size(theta, out_type=tf.float32) * tf.math.log(2 * np.pi * variance) +
                        tf.reduce_sum(tf.square(theta)) / variance)
    return log_prior


@tf.function
def compute_log_posterior(theta_var, u_var, y, Z):
    """
    Compute log prior + log likelihood.

    Args:
        theta_var: Parameter vector
        u_var: Auxiliary variables
        y: Observed data
        Z: Covariates

    Returns:
        float: Log likelihood value plus log prior
    """
    # Extract parameters
    p_z = tf.shape(Z)[-1]
    beta = theta_var[:p_z]
    mu1 = theta_var[p_z]
    mu2 = theta_var[p_z + 1]
    log_lambda1 = theta_var[p_z + 2]
    log_lambda2 = theta_var[p_z + 3]
    eta = theta_var[p_z + 4]

    # Generate X
    X = generate_X(u_var, eta, log_lambda1, log_lambda2)

    # Compute log probabilities
    log_g = compute_logg(y, X, Z, beta)
    log_f = compute_logf_components(X, mu1, mu2, log_lambda1, log_lambda2, eta)
    log_q = compute_logq(u_var)

    # Compute log weights
    log_omega = log_g + log_f - log_q

    # Compute normalized weights and log likelihood
    _, log_likelihood = compute_normalized_weights(log_omega)

    # Compute prior
    log_prior = compute_log_prior(theta_var)

    # Total log likelihood
    return log_likelihood + log_prior


@tf.function
def compute_log_likelihood_and_gradients_auto(theta, u, y, Z):
    """
    Main function: Compute log likelihood and its gradients.

    Args:
        theta: Parameter vector
        u: Auxiliary variables
        y: Observed data
        Z: Covariates

    Returns:
        tuple: (prior_log_likelihood, grad_u, grad_theta)
            - prior_log_likelihood: Log likelihood value + log prior
            - grad_u: Gradient with respect to auxiliary variables
            - grad_theta: Gradient with respect to parameters
    """
    with tf.GradientTape(persistent=True) as tape:
        # Explicitly watch tensors for gradient computation
        tape.watch(theta)
        tape.watch(u)

        # Compute log prior +log likelihood
        log_prior_log_likelihood = compute_log_posterior(theta, u, y, Z)

    # Compute gradients using automatic differentiation
    grad_theta = tape.gradient(prior_log_likelihood, theta)
    grad_u = tape.gradient(prior_log_likelihood, u)

    del tape

    return log_prior_log_likelihood, grad_u, grad_theta
