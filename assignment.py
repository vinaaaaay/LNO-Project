import numpy as np
from PIL import Image
from numpy.linalg import norm
from scipy.fft import dct, idct
import matplotlib.pyplot as plt
import time
import sys

# ====================================
# SECTION 1: CORE ALGORITHM FUNCTIONS
# ====================================

def calculate_psnr(img_true, img_recon):
    """Calculates the Peak Signal-to-Noise Ratio (PSNR)."""
    N = img_true.size
    mse = np.sum((img_true - img_recon)**2) / N
    if mse == 0:
        return float('inf')
    max_intensity = np.max(img_true)
    psnr_val = 20 * np.log10(max_intensity / np.sqrt(mse))
    return psnr_val

def calculate_rel_error(img_true, img_recon):
    """Calculates the l2 relative error."""
    return norm(img_recon - img_true) / norm(img_true)

def calculate_objective(x, m, mask, lam, p_val, epsilon):
    """Calculates the objective function J(x)."""
    data_cost = np.sum((mask * x - m)**2)
    dct_x = dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho')
    sparsity_cost = lam * np.sum((epsilon + dct_x**2)**p_val)
    return data_cost + sparsity_cost

def create_random_mask(height, width, sampling_ratio):
    """Creates a random binary mask."""
    num_pixels = height * width
    num_samples = int(num_pixels * sampling_ratio)
    flat_indices = np.arange(num_pixels)
    np.random.shuffle(flat_indices)
    sample_indices = flat_indices[:num_samples]
    mask_flat = np.zeros(num_pixels)
    mask_flat[sample_indices] = 1.0
    return mask_flat.reshape((height, width))

def add_snr_noise(image_sampled, mask, target_snr_db=30):
    """Adds Gaussian noise to achieve a specific SNR."""
    signal_pixels = image_sampled[mask > 0]
    signal_norm = np.linalg.norm(signal_pixels)
    noise_norm = signal_norm / (10**(target_snr_db / 20.0))
    noise = np.random.randn(len(signal_pixels))
    scaled_noise = noise * (noise_norm / np.linalg.norm(noise))
    m = image_sampled.copy()
    m[mask > 0] += scaled_noise
    return m

def apply_M_operator(z, mask, weights, lam):
    """Applies the linear operator M(k) to an image z."""
    part1 = mask * z
    dct_z = dct(dct(z, axis=0, norm='ortho'), axis=1, norm='ortho')
    weighted_dct = weights * dct_z
    part2 = idct(idct(weighted_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
    return part1 + (lam * part2)

def preconditioned_cg(b, operator_func, x0, preconditioner, tol=1e-6, max_iter=50):
    """Solves M(x) = b using Preconditioned Conjugate Gradient."""
    x = x0.copy()
    r = b - operator_func(x)
    z = r / preconditioner
    p = z.copy()
    rz_old = np.sum(r * z)
    
    for i in range(max_iter):
        Ap = operator_func(p)
        alpha = rz_old / np.sum(p * Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        
        if np.sqrt(np.sum(r*r)) < tol:
            break
            
        z = r / preconditioner
        rz_new = np.sum(r * z)
        beta = rz_new / rz_old
        p = z + beta * p
        rz_old = rz_new
        
    return x, i + 1

def run_reconstruction(m, mask, x_true, lam, p_val, epsilon, max_mm_iter=50, mm_tol=1e-4, cg_tol=1e-6, max_cg_iter=50):
    """Runs the full MM-PCG reconstruction and returns history."""
    x_k = m.copy() # x^(0)
    
    history = {
        'obj': [], 'psnr': [], 'rel_change': [], 'cg_iters': []
    }
    
    # Record initial state
    obj_val = calculate_objective(x_k, m, mask, lam, p_val, epsilon)
    psnr_val = calculate_psnr(x_true, x_k)
    history['obj'].append(obj_val)
    history['psnr'].append(psnr_val)
    
    for k in range(max_mm_iter):
        dct_x = dct(dct(x_k, axis=0, norm='ortho'), axis=1, norm='ortho')
        weights = p_val * (epsilon + dct_x**2)**(p_val - 1)
        
        b = m
        operator_for_cg = lambda z: apply_M_operator(z, mask, weights, lam)
        
        max_weight = np.max(weights)
        preconditioner = mask + (lam * max_weight)
        preconditioner[preconditioner == 0] = 1e-6
        
        x_k_plus_1, cg_iters = preconditioned_cg(
            b, operator_for_cg, x_k, preconditioner,
            tol=cg_tol, max_iter=max_cg_iter
        )
        
        relative_change = norm(x_k_plus_1 - x_k) / norm(x_k)
        
        # Calculate and record diagnostics
        obj_val = calculate_objective(x_k_plus_1, m, mask, lam, p_val, epsilon)
        psnr_val = calculate_psnr(x_true, x_k_plus_1)
        history['obj'].append(obj_val)
        history['psnr'].append(psnr_val)
        history['rel_change'].append(relative_change)
        history['cg_iters'].append(cg_iters)
        
        x_k = x_k_plus_1
        
        if relative_change < mm_tol and k > 0:
            break
            
    # Add final metrics for easy access
    history['final_psnr'] = history['psnr'][-1]
    history['final_rel_err'] = calculate_rel_error(x_true, x_k)
    history['final_recon'] = x_k
        
    return history

def plot_diagnostic_graphs(history, x_true, m, p, r, lam):
    """
    Generates the 2x2 diagnostic plots (b, c, d, e, f)
    and the visual result plots.
    """
    
    print("\nGenerating diagnostic plots for best case...")
    
    x_recon = history['final_recon']
    psnr_input = calculate_psnr(x_true, m)
    psnr_final = history['final_psnr']
    mm_tol = 1e-4 # Define for plot
    
    # Plot 1: Visual Results (Original, Sampled, Recon, Residual)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'Visual Results (r={r}, p={p}, $\lambda$={lam})', fontsize=16)

    axes[0, 0].imshow(x_true, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title(f'(a) Original Image')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(m, cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title(f'(b) Sampled (PSNR: {psnr_input:.2f} dB)')
    axes[0, 1].axis('off')
    
    axes[1, 0].imshow(x_recon, cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title(f'(c) Reconstructed (PSNR: {psnr_final:.2f} dB)')
    axes[1, 0].axis('off')

    residual = x_true - x_recon
    vmax = np.max(np.abs(residual))
    im = axes[1, 1].imshow(residual, cmap='seismic', vmin=-vmax, vmax=vmax)
    axes[1, 1].set_title(f'(c) Residual Map (x* - x_recon)') 
    axes[1, 1].axis('off')
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    plt.savefig('visuals.png')
    
    # Plot 2: Diagnostic Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Diagnostic Plots (r={r}, p={p}, $\lambda$={lam})', fontsize=16)
    mm_iters = np.arange(len(history['obj']))

    # Plot (b) Convergence of J(x)
    axes[0, 0].plot(mm_iters, history['obj'], 'bo-')
    axes[0, 0].set_title(f'(b) Convergence of J(x)')
    axes[0, 0].set_xlabel('MM Iteration (k)')
    axes[0, 0].set_ylabel('Objective Value')
    axes[0, 0].grid(True)
    
    # Plot (d) DCT Histogram
    dct_true_flat = dct(dct(x_true, axis=0, norm='ortho'), axis=1, norm='ortho').ravel()
    dct_recon_flat = dct(dct(x_recon, axis=0, norm='ortho'), axis=1, norm='ortho').ravel()
    axes[0, 1].hist(dct_true_flat, bins=100, range=(-20, 100), log=True, 
                     alpha=0.7, label='Original DCT Coeffs')
    axes[0, 1].hist(dct_recon_flat, bins=100, range=(-20, 100), log=True, 
                     alpha=0.7, label='Reconstructed DCT Coeffs', color='red')
    axes[0, 1].set_title('(d) DCT Histogram')
    axes[0, 1].set_xlabel('Coefficient value')
    axes[0, 1].set_ylabel('Frequency (log scale)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, which='both', linestyle=':')

    # Plot (e) Relative Change
    axes[1, 0].plot(mm_iters[1:], history['rel_change'], 'ro-')
    axes[1, 0].set_title('(e) Relative Change vs. MM Iteration')
    axes[1, 0].set_xlabel('MM Iteration (k)')
    axes[1, 0].set_ylabel('Relative Change (log scale)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].axhline(y=mm_tol, color='k', linestyle='--', label=f'Tolerance ({mm_tol})')
    axes[1, 0].legend()
    axes[1, 0].grid(True, which='both', linestyle=':')

    # Plot (f) PCG Iterations
    axes[1, 1].bar(mm_iters[1:], history['cg_iters'])
    axes[1, 1].set_title('(f) PCG Iterations per MM Step')
    axes[1, 1].set_xlabel('MM Iteration (k)')
    axes[1, 1].set_ylabel('Number of PCG Iterations')
    axes[1, 1].grid(True, axis='y', linestyle=':')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('diagnostic_plots.png')

def save_table_as_figure(data_list_of_dicts, col_headers, title, filename):
    """
    Saves the final results table as a PNG image.
    """
    print(f"\nGenerating table image: {filename}...")
    
    # 1. Convert data from list-of-dicts to list-of-lists (strings)
    cell_text = []
    for run_data in data_list_of_dicts:
        row = [
            f"{run_data['r']:.1f}",
            f"{run_data['p']:.1f}",
            f"{run_data['best_lambda']:.2g}", # Use .2g for 0.01, 0.1
            f"{run_data['best_psnr']:.2f}",
            f"{run_data['rel_err']:.4f}",
            f"{run_data['runtime']:.2f}"
        ]
        cell_text.append(row)
    
    # 2. Create the figure and table
    # Adjust figsize; (width, height)
    fig, ax = plt.subplots(figsize=(12, len(cell_text) * 0.4 + 1)) 
    ax.axis('off') # Hide axes (x, y)
    ax.axis('tight')

    # Create the table
    table = ax.table(cellText=cell_text, 
                     colLabels=col_headers, 
                     loc='center', 
                     cellLoc='center')
    
    # 3. Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.1, 1.4) # Adjust scale (width, height)
    
    # Style header row
    for (i, j), cell in table.get_celld().items():
        if i == 0: # Header row
            cell.set_text_props(weight='bold')

    # 4. Add title
    plt.title(title, weight='bold', fontsize=12, y=1.08)
    
    # 5. Save the figure
    plt.savefig(filename, bbox_inches='tight', dpi=200, pad_inches=0.1)
    plt.close(fig) # Close the figure to free memory
    print(f"Saved table to {filename}")

# ========================================
# SECTION 2: MAIN EXPERIMENT SCRIPT
# ========================================
if __name__ == '__main__':
    
    # --- 1. Define All Experiment Parameters ---
    
    IMAGE_PATH = 'images/lena.png' 
    EPSILON = 1e-6
    MM_TOL = 1e-4
    MAX_MM_ITER = 50 # Set a fixed number for consistent timing
    
    # Parameters to sweep
    r_values = [0.1, 0.2, 0.3, 0.5]
    p_values = [0.3, 0.4, 0.5]
    lambda_values = [1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    col_headers = ["Sampling (r)", "p-value", "Best Lambda", 
                   "Best PSNR (dB)", "Rel. Error (l2)", "Runtime (s)"]

    # --- 2. Load the Original Image ---
    try:
        x_true = np.array(Image.open(IMAGE_PATH).convert('L')) / 255.0
    except FileNotFoundError:
        print(f"Error: Image '{IMAGE_PATH}' not found.")
        sys.exit()

    print(f"--- Running Full Experiment Sweep for '{IMAGE_PATH}' ---")
    
    # Print the table header
    print("="*70)
    print(f"{'Sampling (r)':<12} | {'p-value':<7} | {'Best Lambda':<11} | {'Best PSNR (dB)':<15} | {'Rel. Error (l2)':<15} | {'Runtime (s)':<10}")
    print("-"*70)

    # --- 3. Start the Triple Loop ---
    
    all_best_results = []
    # This will store the data needed for Plot (a)
    psnr_vs_lambda_data_r0_5_p0_5 = [] 
    # This will store the full history for the best run for Plots (b-f)
    best_history_r0_5_p0_5 = None
    best_lambda_for_plots = None
    best_psnr_for_plots = -1.0
    
    # --- Loop 1: Sampling Ratio (r) ---
    for r in r_values:
        
        # Create mask and noisy data ONCE for this r
        mask = create_random_mask(x_true.shape[0], x_true.shape[1], r)
        x_sampled = x_true * mask
        m = add_snr_noise(x_sampled, mask, target_snr_db=30)
        
        # --- Loop 2: p-value ---
        for p in p_values:
            
            sweep_results = []
            
            # --- Loop 3: Lambda ($\lambda$) ---
            for lam in lambda_values:
                
                print(f"  Running: r={r}, p={p}, \u03BB={lam:.1e}...")
                
                start_time = time.time()
                
                history = run_reconstruction(
                    m, mask, x_true, 
                    lam=lam, p_val=p, epsilon=EPSILON,
                    max_mm_iter=MAX_MM_ITER, mm_tol=MM_TOL
                )
                
                end_time = time.time()
                runtime = end_time - start_time
                
                # Store the results of this single run
                run_data = {
                    'lambda': lam,
                    'psnr': history['final_psnr'],
                    'rel_err': history['final_rel_err'],
                    'runtime': runtime,
                    'history': history # Store the full history
                }
                sweep_results.append(run_data)
                
                # --- Special step: Save data for the plots ---
                # If this is the specific case we want to plot (r=0.5, p=0.5)
                if r == 0.5 and p == 0.5:
                    psnr_vs_lambda_data_r0_5_p0_5.append(run_data)
                    
                    # Check if this is the best PSNR *for this case*
                    if run_data['psnr'] > best_psnr_for_plots:
                        best_psnr_for_plots = run_data['psnr']
                        best_history_r0_5_p0_5 = history
                        best_lambda_for_plots = lam
            
            # --- Find the BEST result from the lambda sweep ---
            best_run = max(sweep_results, key=lambda x: x['psnr'])
            table_row_data = {
                'r': r,
                'p': p,
                'best_lambda': best_run['lambda'],
                'best_psnr': best_run['psnr'],
                'rel_err': best_run['rel_err'],
                'runtime': best_run['runtime']
            }
            # Store the best one for the table
            all_best_results.append(table_row_data)
            
            # Print the row for the table
            print(f"{r:<12.1f} | {p:<7.1f} | {best_run['lambda']:<7.2g} | {best_run['psnr']:<15.2f} | {best_run['rel_err']:<15.4f} | {best_run['runtime']:<10.2f}")

    print("="*80)
    print("--- Experiment Sweep Complete ---")

    # --- 4. Generate Plot (a) ---
    print("\nGenerating 'psnr_vs_lambda_plot.png'...")
    
    # Extract data for plotting
    lambdas = [run['lambda'] for run in psnr_vs_lambda_data_r0_5_p0_5]
    psnrs = [run['psnr'] for run in psnr_vs_lambda_data_r0_5_p0_5]
    
    plt.figure(figsize=(8, 6))
    plt.plot(lambdas, psnrs, 'bo-')
    plt.title(f'(a) PSNR vs. $\lambda$ (r=0.5, p=0.5)')
    plt.xlabel('$\lambda$ (Lambda)')
    plt.ylabel('PSNR (dB)')
    plt.xscale('log')
    plt.grid(True, which='both', linestyle=':')
    plt.savefig('psnr_vs_lambda_plot.png')
    
    # --- 5. Generate Plots (b, c, d, e, f) ---
    if best_history_r0_5_p0_5:
        # Re-create the mask and noise for this specific case to plot
        r_plot, p_plot = 0.5, 0.5
        mask_plot = create_random_mask(x_true.shape[0], x_true.shape[1], r_plot)
        x_sampled_plot = x_true * mask_plot
        m_plot = add_snr_noise(x_sampled_plot, mask_plot, target_snr_db=30)
        
        plot_diagnostic_graphs(best_history_r0_5_p0_5, 
                               x_true, m_plot, 
                               p=p_plot, r=r_plot, lam=best_lambda_for_plots)
        print("Generated 'final_reconstruction_visuals.png' and 'final_diagnostic_plots.png'")
    
    # --- 6. NEW: Save the final table as a PNG ---
    table_title = f'Table: Best Reconstruction Metrics for {IMAGE_PATH}'
    save_table_as_figure(all_best_results, col_headers, table_title, 'results_table.png')
    
    print("\nAll tasks complete.")