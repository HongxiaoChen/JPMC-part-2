import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import os

# Custom module imports
from tfp_modified_kernels.pm_hmc import UncalibratedPMHMC, PMHMC
from tfp_modified_kernels.pm_leapfrog_integrator import PMLeapfrogIntegrator
from log_likelihood_auto import compute_log_likelihood_and_gradients_auto, compute_log_posterior
from params import MCMC_PARAMS, DATA_PARAMS, initialize_theta
from generate_samples import generate_samples

# Set environment variables to avoid library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def initialize_sample_data():
    """
    Initialize sample data.
    
    Returns:
        tuple: (Y, Z, beta_true, current_u, T, N)
            - Y: Observed data
            - Z: Covariates
            - beta_true: True beta values
            - current_u: Initialized auxiliary variables u
            - T: Time steps
            - N: Number of auxiliary variables
    """
    # Generate simulated data
    T = DATA_PARAMS['T']
    n = DATA_PARAMS['N_SUBJECTS']
    num_features = DATA_PARAMS['NUM_FEATURES']
    
    # Generate sample data
    Y, X, Z, beta_true = generate_samples(T, n, num_features)
    
    # Initialize auxiliary variables u
    N = MCMC_PARAMS['N']
    current_u = tf.random.normal([T, N])
    
    return Y, Z, beta_true, current_u, T, N


def joint_log_prob_fn(theta, u_flat, y, Z, T, N):
    """
    Calculate joint log probability function.
    
    Args:
        theta: Parameter vector
        u_flat: Flattened auxiliary variables
        y: Observed data
        Z: Covariates
        T: Time steps
        N: Number of auxiliary variables
        
    Returns:
        float: Log probability value
    """
    # Reshape flattened u to [T, N] shape
    u = tf.reshape(u_flat, [T, N])
    
    # Use compute_log_posterior to calculate log likelihood and its gradients
    log_prob = compute_log_posterior(theta, u, y, Z)
    
    return log_prob


def run_pm_hmc_sampling():
    """
    Run PM-HMC sampling.
    """
    # Initialize sample data
    print("Initializing sample data...")
    Y, Z, beta_true, current_u, T, N = initialize_sample_data()
    
    # Initialize parameter theta
    initial_theta = initialize_theta()
    theta_size = tf.size(initial_theta)
    
    # Define target_log_prob_fn - Note TFP will pass states as multiple parameters
    target_log_prob_fn = lambda theta, u_flat: joint_log_prob_fn(
        theta, u_flat, Y, Z, T, N
    )
    
    # Print confirmation information
    print("Successfully initialized target_log_prob_fn function")
    print(f"theta_size: {theta_size}")
    print(f"T: {T}, N: {N}")
    print(f"Y shape: {Y.shape}, Z shape: {Z.shape}")
    
    # Set sampling parameters
    num_results = MCMC_PARAMS['M']  # Number of samples
    num_burnin_steps = MCMC_PARAMS['BURN_IN']  # Burn-in steps
    h = MCMC_PARAMS['H']  # Step size
    L = MCMC_PARAMS['L']  # Leapfrog steps
    rho_size = MCMC_PARAMS['RHO_SIZE']  # rho_size parameter
    
    # Create PM-HMC kernel
    print(f"\nSampling using PM-HMC...")
    print(f"Step size h = {h}, leapfrog steps L = {L}, rho_size = {rho_size}")
    
    pm_hmc_kernel = PMHMC(
        target_log_prob_fn=target_log_prob_fn,
        step_size=h,
        num_leapfrog_steps=L,
        T=T,
        N=N,
        rho_size=rho_size
    )
    
    # Prepare initial state
    # Flatten u
    initial_u_flat = tf.reshape(current_u, [-1])
    initial_state = [initial_theta, initial_u_flat]
    
    # Record start time
    start_time = time.time()
    print(f"Starting sampling: {num_results} samples, burn-in steps={num_burnin_steps}...")
    
    # Use TFP's sample_chain function for sampling
    @tf.function
    def run_chain():
        samples = tfp.mcmc.sample_chain(
            num_results=num_results,
            current_state=initial_state,
            kernel=pm_hmc_kernel,
            num_burnin_steps=num_burnin_steps,
            trace_fn=None
        )
        return samples
    
    samples = run_chain()
    
    # Calculate sampling time
    sampling_time = time.time() - start_time
    print(f"Sampling complete! Time taken: {sampling_time:.2f} seconds")
    
    # Extract parameters from sampling results
    # samples is a list containing [theta_samples, u_flat_samples]
    theta_samples = samples[0].numpy()  # Shape: [num_results, 13]
    
    # Extract and calculate means
    param_8 = theta_samples[:, 8]  # mu1
    param_9 = theta_samples[:, 9]  # mu2
    
    # Calculate 1/lambda1 and 1/lambda2
    log_lambda1 = theta_samples[:, 10]
    log_lambda2 = theta_samples[:, 11]
    inv_lambda1 = 1.0 / np.exp(log_lambda1)
    inv_lambda2 = 1.0 / np.exp(log_lambda2)
    
    # Calculate means
    mean_inv_lambda1 = np.mean(inv_lambda1)
    mean_inv_lambda2 = np.mean(inv_lambda2)
    
    print(f"Mean of parameter 8 (mu1): {np.mean(param_8):.4f}")
    print(f"Mean of parameter 9 (mu2): {np.mean(param_9):.4f}")
    print(f"Mean of 1/lambda1: {mean_inv_lambda1:.4f}")
    print(f"Mean of 1/lambda2: {mean_inv_lambda2:.4f}")
    
    # Ensure figures directory exists
    if not os.path.exists('figures'):
        os.makedirs('figures')
    
    # Create a figure to display sampled parameters
    plt.figure(figsize=(12, 7))
    
    # Trace plot of mu1 and mu2
    plt.subplot(1, 2, 1)
    plt.plot(param_8, label='μ₁', color='blue', alpha=0.7)
    plt.plot(param_9, label='μ₂', color='red', alpha=0.7)
    plt.axhline(y=np.mean(param_8), color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=np.mean(param_9), color='red', linestyle='--', alpha=0.5)
    plt.title('μ₁ and μ₂ (post burn-in)')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.ylim(-2, 6)
    plt.legend()
    
    # Trace plot of 1/lambda1 and 1/lambda2
    plt.subplot(1, 2, 2)
    plt.plot(inv_lambda1, label='1/λ₁', color='blue', alpha=0.7)
    plt.plot(inv_lambda2, label='1/λ₂', color='red', alpha=0.7)
    plt.axhline(y=mean_inv_lambda1, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=mean_inv_lambda2, color='red', linestyle='--', alpha=0.5)
    plt.title('1/λ₁ and 1/λ₂ (post burn-in)')
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.ylim(0, 10)
    plt.legend()
    
    # Set main title
    plt.suptitle(
        f'PM-HMC Sampling Results\n'
        f'N = {N}, h = {h}, L = {L}, ρ mass = {rho_size}\n'
        f'Total steps per iteration = {h * L:.4f}',
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plt.savefig(f'figures/TFP_parameter_traces_{timestamp}.png', dpi=300, bbox_inches='tight')
    
    print(f"Figure saved to figures/TFP_parameter_traces_{timestamp}.png")
    
    return theta_samples


if __name__ == '__main__':
    # Run PM-HMC sampling
    theta_samples = run_pm_hmc_sampling()
    
    # Print results summary
    print("\nPM-HMC Sampling Summary:")
    print(f"Number of parameters: {theta_samples.shape[1]}")
    print(f"Number of samples: {theta_samples.shape[0]}")
    
    # Calculate posterior means and standard deviations for parameters
    posterior_means = np.mean(theta_samples, axis=0)
    posterior_stds = np.std(theta_samples, axis=0)
    
    # Display parameter means and standard deviations
    print("\nParameter posterior estimates:")
    for i, (mean, std) in enumerate(zip(posterior_means, posterior_stds)):
        print(f"Parameter {i}: mean = {mean:.4f}, std = {std:.4f}")
    
    # Focus on specific important parameters
    print("\nKey parameters:")
    # Beta parameters (indices 0-7)
    print(f"Beta: {posterior_means[:8]}")
    # mu1, mu2 (indices 8-9)
    print(f"mu1: {posterior_means[8]:.4f} ± {posterior_stds[8]:.4f}")
    print(f"mu2: {posterior_means[9]:.4f} ± {posterior_stds[9]:.4f}")
    # log(lambda1), log(lambda2) (indices 10-11)
    print(f"log(lambda1): {posterior_means[10]:.4f} ± {posterior_stds[10]:.4f}")
    print(f"log(lambda2): {posterior_means[11]:.4f} ± {posterior_stds[11]:.4f}")
    # w1 (converted from eta, index 12)
    w1 = 1.0 - tf.sigmoid(posterior_means[12]).numpy()
    print(f"w1: {w1:.4f}") 