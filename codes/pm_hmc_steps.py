import tensorflow as tf
from log_likelihood_auto import compute_log_likelihood_and_gradients_auto as compute_log_likelihood_and_gradients
#from log_likelihood_stable import compute_log_likelihood_and_gradients
from log_likelihood_auto import compute_log_posterior
from params import USE_HNN, DATA_PARAMS, MCMC_PARAMS, HNN_WEIGHTS_PATH
from hnn_architectures import HNN


def initialize_hnn():
    """
    Initialize HNN model and load weights.

    Returns:
        HNN: Initialized and loaded HNN model
    """
    hnn_model = HNN(T=DATA_PARAMS['T'], N=MCMC_PARAMS['N'])
    hnn_model.load_weights(HNN_WEIGHTS_PATH)
    return hnn_model


@tf.function
def compute_gradients_with_hnn(theta, u, y, Z, hnn_model):
    """
    Compute gradients using trained HNN.

    Args:
        theta: Parameter tensor
        u: Auxiliary variables
        y: Observed data
        Z: Covariates
        hnn_model: Trained HNN model

    Returns:
        tuple: (grad_u, grad_theta) Gradients with respect to u and theta
    """
    _, grad_theta, grad_u = hnn_model.compute_gradients(theta, u)
    return grad_u, grad_theta


@tf.function
def full_step_A(theta, rho, u, p, h, y, Z, rho_size):
    """
    Execute half step A: Update theta and rotate (u,p).

    Args:
        theta: Parameter vector [13]
        rho: Momentum vector [13]
        u: Auxiliary variables [T,N,p]
        p: Auxiliary momentum [T,N,p]
        h: Step size
        y: Observed data
        Z: Covariates
        rho_size: Scale factor for rho

    Returns:
        tuple: (theta_new, rho, u_new, p_new) Updated state variables
    """
    # Update theta
    theta_new = theta + (h) * rho * 1.0 / rho_size

    # Rotate (u,p)
    cos_term = tf.cos(h)
    sin_term = tf.sin(h)

    u_new = u * cos_term + p * sin_term
    p_new = p * cos_term - u * sin_term

    return theta_new, rho, u_new, p_new


@tf.function
def full_step_B(theta, rho, u, p, h, y, Z, rho_size, hnn_model, traditional_only, USE_HNN=USE_HNN):
    """
    Execute full step B: Update momentum.

    Args:
        theta: Current parameters [p_z + 5]
        rho: Parameter momentum [p_z + 5]
        u: Auxiliary variables [T, N, p]
        p: Auxiliary momentum [T, N, p]
        h: Step size
        y: Observed data [T,n]
        Z: Covariates [T,n,p_z]
        hnn_model: HNN model instance
        traditional_only: Flag for traditional gradient computation
        USE_HNN: Flag for using HNN

    Returns:
        tuple: (theta, rho_new, u, p_new) Updated state variables
    """
    if USE_HNN & (traditional_only == False):
        # Expand dimensions to match HNN input requirements
        theta_expanded = tf.expand_dims(theta, 0)  # [1, p_z + 5]
        u_expanded = tf.expand_dims(u, 0)  # [1, T, N, p]

        # Compute gradients using HNN
        grad_u_expanded, grad_theta_expanded = compute_gradients_with_hnn(theta_expanded, u_expanded, y, Z, hnn_model)

        # Remove extra dimensions
        grad_theta = tf.squeeze(grad_theta_expanded, 0)  # [p_z + 5]
        grad_u = tf.squeeze(grad_u_expanded, 0)  # [T, N, p]
    else:
        # Compute gradients using traditional method
        _, grad_u, grad_theta = compute_log_likelihood_and_gradients(theta, u, y, Z)

    # Update momentum
    rho_new = rho + h * grad_theta
    p_new = p + h * grad_u

    # Check gradients
    if tf.reduce_any(tf.math.is_nan(grad_theta)) or tf.reduce_any(tf.math.is_nan(grad_u)):
        tf.print("Warning: NaN gradients detected")
        tf.print("grad_theta:", grad_theta)

    return theta, rho_new, u, p_new


def leapfrog_step(theta, rho, u, p, h, L, y, Z, rho_size, hnn_model, traditional_only):
    """
    Execute L steps of Strang splitting/Leapfrog integration.

    Args:
        theta: Initial parameters
        rho: Initial momentum
        u: Initial auxiliary variables
        p: Initial auxiliary momentum
        h: Step size
        L: Number of steps
        y: Observed data
        Z: Covariates
        rho_size: Scale factor for rho
        hnn_model: HNN model instance
        traditional_only: Flag for traditional computation

    Returns:
        tuple: (current_theta, current_rho, current_u, current_p) Final state
    """
    # Save initial state
    current_theta, current_rho = theta, rho
    current_u, current_p = u, p

    for l in range(L):
        # Half step A
        current_theta, current_rho, current_u, current_p = full_step_A(
            current_theta, current_rho, current_u, current_p, h / 2, y, Z, rho_size
        )

        # Full step B
        current_theta, current_rho, current_u, current_p = full_step_B(
            current_theta, current_rho, current_u, current_p, h, y, Z, rho_size, hnn_model, traditional_only
        )

        # Half step A
        current_theta, current_rho, current_u, current_p = full_step_A(
            current_theta, current_rho, current_u, current_p, h / 2, y, Z, rho_size
        )

    return current_theta, current_rho, current_u, current_p


@tf.function
def compute_hamiltonian(theta, rho, u, p, y, Z, rho_size):
    """
    Compute Hamiltonian.

    Args:
        theta: Parameter vector [p_z + 5]
        rho: Momentum vector [p_z + 5]
        u: Auxiliary variables [T, N, p]
        p: Auxiliary momentum [T, N, p]
        y: Observed data [T,n]
        Z: Covariates [T,n,p_z]
        rho_size: Scale factor for rho

    Returns:
        float: Hamiltonian value (scalar)
    """
    # Compute potential energy: -log(posterior)
    prior_log_likelihood = compute_log_posterior(theta, u, y, Z)
    potential_energy = -prior_log_likelihood

    # Compute kinetic energy: 1/2(ρᵀρ + uᵀu + pᵀp)
    kinetic_energy_rho = 0.5 * tf.reduce_sum(tf.square(rho)) * 1.0 / rho_size  # Parameter kinetic energy
    kinetic_energy_u = 0.5 * tf.reduce_sum(tf.square(u))  # Auxiliary variable kinetic energy
    kinetic_energy_p = 0.5 * tf.reduce_sum(tf.square(p))  # Auxiliary momentum kinetic energy

    # Combine all energy terms
    hamiltonian = potential_energy + kinetic_energy_rho + kinetic_energy_u + kinetic_energy_p

    return hamiltonian


@tf.function
def metropolis_step(theta, rho, u, p, theta_new, rho_new, u_new, p_new, y, Z, rho_size):
    """
    Execute Metropolis-Hastings accept-reject step.

    Args:
        theta: Current parameters
        rho: Current momentum
        u: Current auxiliary variables
        p: Current auxiliary momentum
        theta_new: Proposed parameters
        rho_new: Proposed momentum
        u_new: Proposed auxiliary variables
        p_new: Proposed auxiliary momentum
        y: Observed data
        Z: Covariates
        rho_size: Scale factor for rho

    Returns:
        tuple: (new_theta, new_u, accepted) Updated state and acceptance flag
    """
    # Compute Hamiltonian difference
    current_H = compute_hamiltonian(theta, rho, u, p, y, Z, rho_size)
    proposed_H = compute_hamiltonian(theta_new, rho_new, u_new, p_new, y, Z, rho_size)

    # Compute acceptance probability
    delta_H = current_H - proposed_H

    accept_prob = tf.minimum(1.0, tf.exp(delta_H))
    # accept_prob = tf.constant(1.0)

    # Generate random number for acceptance decision
    random_uniform = tf.random.uniform([])

    # Return state based on acceptance probability
    accepted = random_uniform < accept_prob

    new_theta = tf.where(accepted, theta_new, theta)
    new_u = tf.where(accepted, u_new, u)

    return new_theta, new_u, accepted


def run_pm_hmc_iteration(theta, u, y, Z, h, L, rho_size, hnn_model, traditional_only=False):
    """
    Run one complete PM-HMC iteration.

    Args:
        theta: Current parameter state
        u: Current auxiliary variable state
        y: Observed data
        Z: Covariates
        h: Step size
        L: Number of leapfrog steps
        rho_size: Scale factor for rho
        hnn_model: HNN model instance
        traditional_only: Flag for traditional computation

    Returns:
        tuple: (theta_next, u_next, accepted) Updated state and acceptance flag
    """
    # Initialize momentum
    d = theta.shape[0]  # Parameter dimension
    D = tf.size(u)  # Total auxiliary variable dimension

    rho = tf.random.normal([d]) * tf.sqrt(rho_size)
    p = tf.random.normal(tf.shape(u))

    # Execute leapfrog steps
    theta_new, rho_new, u_new, p_new = leapfrog_step(theta, rho, u, p, h, L, y, Z, rho_size, hnn_model,
                                                     traditional_only)

    # Execute Metropolis-Hastings step
    theta_next, u_next, accepted = metropolis_step(
        theta, rho, u, p,
        theta_new, rho_new, u_new, p_new,
        y, Z, rho_size
    )

    return theta_next, u_next, accepted
