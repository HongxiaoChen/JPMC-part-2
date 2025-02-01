import tensorflow as tf
from generate_samples import generate_samples
from pm_hmc_steps import run_pm_hmc_iteration, initialize_hnn
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from params import MCMC_PARAMS, DATA_PARAMS, INIT_PARAMS, initialize_theta
from params import USE_HNN
import os


def ensure_directories():
    """
    Ensure necessary directories exist.

    Creates 'log' and 'figures' directories if they don't exist.
    """
    directories = ['log', 'figures']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def create_log_file():
    """
    Create a log file with timestamp.

    Returns:
        file: A file object for writing logs
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f'log/pm_hmc_{timestamp}.log'
    return open(log_filename, 'w')


def log_message(file, message):
    """
    Write and flush log message.

    Args:
        file: The log file object
        message: The message to be logged
    """
    file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
    file.flush()


def plot_parameter_traces(samples, h, L, N, rho_size):
    """
    Plot parameter trace plots with sampling parameters information.

    Args:
        samples: numpy array of shape [M, 13] containing all samples
        h: step size
        L: number of leapfrog steps
        N: number of auxiliary variables
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
    # Calculate all 1/lambda values
    inv_lambda1 = 1.0 / np.exp(samples[:, 10])
    inv_lambda2 = 1.0 / np.exp(samples[:, 11])

    # Calculate means of 1/lambda
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
        f"{'HNN' if USE_HNN else 'Traditional'} Mode - "
        'Traces for parameters μ₁ and μ₂ (left) and 1/λ₁ and 1/λ₂ (right)\n'
        f'PM-HMC sampler with N = {N}, h = {h}, L = {L}, ρ = {rho_size}\n'
        f'Total steps per iteration = {h * L:.4f}'
    )

    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'figures/parameter_traces_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.close()


def run_pm_hmc(Y, Z, M, h, L, N, p, initial_theta, hnn_model=None, rho_size=10, burn_in=1000):
    """
    Run complete PM-HMC sampling process.

    Args:
        Y: Observed data [T,n]
        Z: Covariates [T,n,8]
        M: Total number of samples
        h: Step size
        L: Number of leapfrog steps
        N: Number of auxiliary variables
        p: Dimension of auxiliary variables
        initial_theta: Initial parameter values
        hnn_model: HNN model instance (optional)
        rho_size: Scale factor for rho
        burn_in: Number of burn-in samples

    Returns:
        tuple: (samples, acceptance_rate, log_file)
            - samples: Collected parameter samples after burn-in
            - acceptance_rate: Acceptance rate of the sampling
            - log_file: Log file object
    """
    # Create log file
    log_file = create_log_file()
    log_message(log_file, f"Using {'HNN' if USE_HNN else 'Traditional'} mode for gradient computation")
    # Record initial parameters
    log_message(log_file, f"Starting PM-HMC sampling with parameters:")
    log_message(log_file, f"Total iterations: {M}")
    log_message(log_file, f"Step size (h): {h}")
    log_message(log_file, f"Leapfrog steps (L): {L}")
    log_message(log_file, f"Burn-in: {burn_in}")
    log_message(log_file, f"rho_size: {rho_size}")

    # Set parameter dimensions
    T, n = Y.shape

    # Initialize auxiliary variables u
    current_u = tf.random.normal([T, N])

    # Store sampling results
    samples = []
    accept_count = 0
    current_theta = initial_theta

    # Record start time
    start_time = time.time()
    last_update_time = start_time

    # Step 6: Repeat sampling M times
    for m in range(M):
        # Run one PM-HMC iteration
        next_theta, next_u, accepted = run_pm_hmc_iteration(
            current_theta, current_u, Y, Z, h, L, rho_size, hnn_model
        )

        # Update current state
        if accepted:
            accept_count += 1
            current_theta = next_theta
            current_u = next_u

        # Store sample
        samples.append(current_theta.numpy())

        # Update progress every 100 iterations
        if (m + 1) % 100 == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            iterations_left = M - (m + 1)
            time_per_iteration = elapsed_time / (m + 1)
            estimated_time_left = time_per_iteration * iterations_left

            progress_msg = (
                f"Iteration {m + 1}/{M} "
                f"({(m + 1) / M * 100:.1f}%) | "
                f"Accept rate: {accept_count / (m + 1):.3f} | "
                f"Elapsed: {elapsed_time / 60:.1f}min | "
                f"Est. remaining: {estimated_time_left / 60:.1f}min"
            )
            log_message(log_file, progress_msg)

    # Convert to numpy array
    samples = tf.stack(samples, axis=0).numpy()
    acceptance_rate = accept_count / M
    total_time = time.time() - start_time

    # Record completion information
    # Take mean for beta and mu directly
    posterior_mean_beta = np.mean(samples[burn_in:, :8], axis=0)
    posterior_mean_mu = np.mean(samples[burn_in:, 8:10], axis=0)

    lambda1 = np.exp(samples[burn_in:, 10])
    lambda2 = np.exp(samples[burn_in:, 11])

    log_lambda_1 = samples[burn_in:, 10]
    log_lambda_2 = samples[burn_in:, 11]

    inv_lambda1 = 1.0 / np.exp(samples[burn_in:, 10])
    inv_lambda2 = 1.0 / np.exp(samples[burn_in:, 11])

    posterior_mean_lambda = np.array([
        np.mean(log_lambda_1),
        np.mean(log_lambda_2)
    ])

    # Use sigmoid for w1
    posterior_mean_w = 1.0 - tf.sigmoid(np.mean(samples[burn_in:, 12])).numpy()

    log_message(log_file, "\nParameter estimates:")
    log_message(log_file, f"mu1: {posterior_mean_mu[0]}")
    log_message(log_file, f"mu2: {posterior_mean_mu[1]}")
    log_message(log_file, f"log(lambda1): {posterior_mean_lambda[0]}")
    log_message(log_file, f"log(lambda2): {posterior_mean_lambda[1]}")
    log_message(log_file, f"w1: {posterior_mean_w}")
    log_message(log_file, f"Beta(Pred): {posterior_mean_beta.tolist()}")
    return samples[burn_in:], acceptance_rate, log_file


if __name__ == '__main__':
    # Initialize HNN model if using HNN mode

    # Generate simulated data
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    T = DATA_PARAMS['T']
    n = DATA_PARAMS['N_SUBJECTS']
    num_features = DATA_PARAMS['NUM_FEATURES']
    if USE_HNN:
        hnn_model = initialize_hnn()
    else:
        hnn_model = None

    Y, X, Z, beta_true = generate_samples(T, n, num_features)

    # PM-HMC parameter settings
    M = MCMC_PARAMS['M']
    h = MCMC_PARAMS['H']
    L = MCMC_PARAMS['L']
    burn_in = MCMC_PARAMS['BURN_IN']
    N = MCMC_PARAMS['N']
    p = MCMC_PARAMS['P']
    rho_size = MCMC_PARAMS['RHO_SIZE']
    # Step 1: Initialize parameter theta
    initial_theta = initialize_theta()

    # Check directories
    ensure_directories()

    # Run PM-HMC
    samples, acceptance_rate, log_file = run_pm_hmc(Y, Z, M, h, L, N, p, initial_theta, hnn_model, rho_size, burn_in)

    # Plot parameter trajectories
    log_message(log_file, f"Beta(True): {beta_true.numpy()}")
    log_message(log_file, "\nGenerating parameter trace plots...")
    plot_parameter_traces(samples, h, L, N, rho_size)
    log_message(log_file, "Parameter trace plots have been saved to the figures directory.")

    # Close log file
    log_file.close()

    # Print results
    print("\nPM-HMC sampling completed:")
    print(f"Total samples: {M}")
    print(f"Burn-in samples: {burn_in}")
    print(f"Final acceptance rate: {acceptance_rate:.3f}")
    print(f"rho size: {rho_size:.1f}")

    # Calculate posterior means
    posterior_mean_beta = np.mean(samples[:, :8], axis=0)
    posterior_mean_mu = np.mean(samples[:, 8:10], axis=0)

    # For lambda, first calculate all 1/lambda values, take mean, then inverse
    inv_lambda1 = 1.0 / np.exp(samples[:, 10])
    inv_lambda2 = 1.0 / np.exp(samples[:, 11])
    posterior_mean_lambda = np.array([
        1.0 / np.mean(inv_lambda1),
        1.0 / np.mean(inv_lambda2)
    ])

    posterior_mean_w = 1.0 - tf.sigmoid(np.mean(samples[:, 12])).numpy()

    # Print parameter estimation results
    print("\nParameter estimates:")
    print("Beta (true):", beta_true.numpy())
    print("Beta (estimated):", posterior_mean_beta)
    print("mu1:", posterior_mean_mu[0])
    print("mu2:", posterior_mean_mu[1])
    print("lambda1:", posterior_mean_lambda[0])
    print("lambda2:", posterior_mean_lambda[1])
    print("w1:", posterior_mean_w)