import tensorflow as tf
import time
from datetime import datetime
import logging
from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
from hnn_architectures import HNN
from params import MCMC_PARAMS, HNN_WEIGHTS_PATH, initialize_theta, DATA_PARAMS
from generate_samples import generate_samples


def setup_logger():
    """Set up a logger for training process"""
    log_dir = Path('log')
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'hnn_hmc_{timestamp}.log'

    logger = logging.getLogger('HNN_HMC')
    logger.setLevel(logging.INFO)

    if logger.handlers:
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    for handler in [fh, ch]:
        handler.setLevel(logging.INFO)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def ensure_directories():
    """Ensure necessary directories exist"""
    directories = ['log', 'figures']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def plot_parameter_traces(samples, h, L, rho_size):
    """
    Plot parameter trace plots with sampling parameters information.

    Args:
        samples: numpy array of shape [M, 13] containing all samples
        h: step size
        L: number of leapfrog steps
        rho_size: rho size parameter
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Left plot: trajectories of mu1 and mu2
    ax1.plot(samples[:, 8], color='blue', alpha=0.7, label='μ₁')
    ax1.plot(samples[:, 9], color='red', alpha=0.7, label='μ₂')
    ax1.axhline(y=np.mean(samples[:, 8]), color='blue', linestyle='--', alpha=0.5)
    ax1.axhline(y=np.mean(samples[:, 9]), color='red', linestyle='--', alpha=0.5)
    ax1.set_title('μ₁ and μ₂')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Value')
    ax1.set_ylim(-2, 6)
    ax1.legend()

    # Right plot: trajectories of 1/lambda1 and 1/lambda2
    inv_lambda1 = 1.0 / np.exp(samples[:, 10])
    inv_lambda2 = 1.0 / np.exp(samples[:, 11])

    mean_inv_lambda1 = np.mean(inv_lambda1)
    mean_inv_lambda2 = np.mean(inv_lambda2)

    ax2.plot(inv_lambda1, color='blue', alpha=0.7, label='1/λ₁')
    ax2.plot(inv_lambda2, color='red', alpha=0.7, label='1/λ₂')
    ax2.axhline(y=mean_inv_lambda1, color='blue', linestyle='--', alpha=0.5)
    ax2.axhline(y=mean_inv_lambda2, color='red', linestyle='--', alpha=0.5)
    ax2.set_title('1/λ₁ and 1/λ₂')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Value')
    ax2.set_ylim(0, 10)
    ax2.legend()

    # Set title with parameter information
    plt.suptitle(
        'HNN-HMC - '
        'Traces for parameters μ₁ and μ₂ (left) and 1/λ₁ and 1/λ₂ (right)\n'
        f'HNN-HMC sampler with h = {h}, L = {L}, ρ = {rho_size}\n'
        f'Total steps per iteration = {h * L:.4f}'
    )

    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'figures/parameter_traces_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


@tf.function
def leapfrog_steps(model, theta, rho, h, L):
    """
    Execute L steps of leapfrog integration using HNN

    Args:
        model: HNN model
        theta: Initial parameters [13]
        rho: Initial momentum [13]
        h: Step size
        L: Number of steps

    Returns:
        tuple: (final_theta, final_rho) with shape [13]
    """
    # Add batch dimension at the beginning
    current_theta = tf.expand_dims(theta, 0)  # [1, 13]
    current_rho = tf.expand_dims(rho, 0)  # [1, 13]

    for _ in tf.range(L):
        # Half step in theta using rho gradient
        _, _, grad_rho = model.compute_gradients(current_theta, current_rho)
        current_theta = current_theta + (h / 2) * grad_rho

        # Full step in rho using theta gradient
        _, grad_theta, _ = model.compute_gradients(current_theta, current_rho)
        current_rho = current_rho - h * grad_theta

        # Half step in theta using rho gradient
        _, _, grad_rho = model.compute_gradients(current_theta, current_rho)
        current_theta = current_theta + (h / 2) * grad_rho

    # Remove batch dimension before returning
    return tf.squeeze(current_theta, 0), tf.squeeze(current_rho, 0)


@tf.function
def metropolis_step(model, theta, rho, theta_new, rho_new):
    """
    Execute Metropolis accept-reject step

    Args:
        model: HNN model
        theta, rho: Current state with shape [13]
        theta_new, rho_new: Proposed state with shape [13]

    Returns:
        tuple: (accepted_theta, accepted)
            - accepted_theta: Final state with shape [13]
            - accepted: Boolean indicating if proposal was accepted
    """
    # Add batch dimension
    theta_expanded = tf.expand_dims(theta, 0)
    rho_expanded = tf.expand_dims(rho, 0)
    theta_new_expanded = tf.expand_dims(theta_new, 0)
    rho_new_expanded = tf.expand_dims(rho_new, 0)

    # Compute Hamiltonian difference using HNN
    H_current, _, _ = model.compute_gradients(theta_expanded, rho_expanded)
    H_proposed, _, _ = model.compute_gradients(theta_new_expanded, rho_new_expanded)

    # Remove batch dimension
    H_current = tf.squeeze(H_current, 0)
    H_proposed = tf.squeeze(H_proposed, 0)

    # Compute acceptance probability
    delta_H = H_current - H_proposed
    accept_prob = tf.minimum(1.0, tf.exp(delta_H)) * 0.7

    # Accept/reject
    uniform = tf.random.uniform([])
    accepted = uniform < accept_prob

    # Only update theta if accepted, discard rho either way
    accepted_theta = tf.where(accepted, theta_new, theta)

    return accepted_theta, accepted


def run_hnn_hmc():
    """Run HNN-HMC sampling"""
    # Setup
    logger = setup_logger()
    ensure_directories()

    # Generate true data for comparison
    T = DATA_PARAMS['T']
    n = DATA_PARAMS['N_SUBJECTS']
    num_features = DATA_PARAMS['NUM_FEATURES']
    Y, X, Z, beta_true = generate_samples(T, n, num_features)

    # Load model and weights
    model = HNN(activation='sin')
    model.load_weights(HNN_WEIGHTS_PATH)

    # Get parameters from MCMC_PARAMS
    M = MCMC_PARAMS['M']
    h = MCMC_PARAMS['H']
    L = MCMC_PARAMS['L']
    burn_in = MCMC_PARAMS['BURN_IN']
    rho_size = MCMC_PARAMS['RHO_SIZE']

    # Initialize state
    current_theta = initialize_theta()
    samples = []
    accept_count = 0

    # Record start time
    start_time = time.time()

    logger.info("Starting HNN-HMC sampling")
    logger.info(f"Parameters: M={M}, h={h}, L={L}, burn_in={burn_in}, rho_size={rho_size}")

    for m in range(M):
        # Initialize momentum with identity covariance and rho_size scaling
        current_rho = tf.random.normal(current_theta.shape) * tf.sqrt(rho_size)

        # Leapfrog integration
        proposed_theta, proposed_rho = leapfrog_steps(
            model, current_theta, current_rho, h, L
        )

        # Metropolis step
        current_theta, accepted = metropolis_step(
            model, current_theta, current_rho,
            proposed_theta, proposed_rho
        )

        # Store sample and update acceptance count
        if m >= burn_in:
            samples.append(current_theta.numpy())
        if accepted:
            accept_count += 1

        # Log progress every 100 iterations
        if (m + 1) % 100 == 0:
            elapsed_time = time.time() - start_time
            acceptance_rate = accept_count / (m + 1)
            remaining_iterations = M - (m + 1)
            estimated_time_left = (elapsed_time / (m + 1)) * remaining_iterations

            logger.info(
                f"Iteration {m + 1}/{M} ({(m + 1) / M * 100:.1f}%) | "
                f"Accept rate: {acceptance_rate:.3f} | "
                f"Elapsed: {elapsed_time / 60:.1f}min | "
                f"Est. remaining: {estimated_time_left / 60:.1f}min"
            )

    # Convert samples list to numpy array
    samples = np.array(samples)

    # Calculate posterior means
    posterior_mean_beta = np.mean(samples[:, :8], axis=0)
    posterior_mean_mu = np.mean(samples[:, 8:10], axis=0)

    # For lambda, calculate inverse means
    inv_lambda1 = 1.0 / np.exp(samples[:, 10])
    inv_lambda2 = 1.0 / np.exp(samples[:, 11])
    posterior_mean_lambda = np.array([
        1.0 / np.mean(inv_lambda1),
        1.0 / np.mean(inv_lambda2)
    ])

    posterior_mean_w = 1.0 - tf.sigmoid(np.mean(samples[:, 12])).numpy()

    # Final statistics
    final_acceptance_rate = accept_count / M
    total_time = time.time() - start_time

    # Log results
    logger.info("\nSampling completed:")
    logger.info(f"Total samples: {M}")
    logger.info(f"Burn-in samples: {burn_in}")
    logger.info(f"Final acceptance rate: {final_acceptance_rate:.3f}")
    logger.info(f"Total time: {total_time / 60:.1f}min")

    logger.info("\nParameter estimates:")
    logger.info(f"Beta (true): {beta_true.numpy().tolist()}")
    logger.info(f"Beta (estimated): {posterior_mean_beta.tolist()}")
    logger.info(f"mu1: {posterior_mean_mu[0]}")
    logger.info(f"mu2: {posterior_mean_mu[1]}")
    logger.info(f"lambda1: {posterior_mean_lambda[0]}")
    logger.info(f"lambda2: {posterior_mean_lambda[1]}")
    logger.info(f"w1: {posterior_mean_w}")

    # Generate trace plots
    logger.info("\nGenerating parameter trace plots...")
    plot_parameter_traces(samples, h, L, rho_size)
    logger.info("Parameter trace plots have been saved to the figures directory.")

    return (samples, final_acceptance_rate, posterior_mean_beta, posterior_mean_mu,
            posterior_mean_lambda, posterior_mean_w, logger)


if __name__ == "__main__":
    # Set random seed for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Run HNN-HMC
    (samples, acceptance_rate, posterior_mean_beta, posterior_mean_mu,
     posterior_mean_lambda, posterior_mean_w, logger) = run_hnn_hmc()

    # Print final results
    print("\nHNN-HMC sampling completed:")
    print(f"Final acceptance rate: {acceptance_rate:.3f}")
    print("\nParameter estimates:")
    print("Beta:", posterior_mean_beta)
    print("mu1:", posterior_mean_mu[0])
    print("mu2:", posterior_mean_mu[1])
    print("lambda1:", posterior_mean_lambda[0])
    print("lambda2:", posterior_mean_lambda[1])
    print("w1:", posterior_mean_w)