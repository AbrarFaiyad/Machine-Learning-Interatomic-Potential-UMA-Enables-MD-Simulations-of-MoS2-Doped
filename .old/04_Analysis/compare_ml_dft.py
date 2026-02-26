#!/usr/bin/env python
"""
Compare ML vs DFT formation energies for MoS2 dopants.
Creates multiple visualizations to assess ML potential accuracy.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from scipy.stats import pearsonr
import matplotlib.patches as patches
from math import pi
from matplotlib.backends.backend_pdf import PdfPages

# =============================================================================
# PLOT CONFIGURATION - Set True/False to control which plots are generated
# =============================================================================
PLOT_CONFIG = {
    # Main comparison plots
    'individual_parity_plots': False,        # Section 1: Individual parity plots for each formation type
    'combined_parity_plot': False,           # Section 1b: Combined parity plot with all formation types
    
    # Error analysis plots
    'error_distribution': False,             # Section 2a: Error distribution by formation type
    'error_heatmap': False,                  # Section 2b: Error heatmap by dopant
    'error_by_category': True,              # Section 2c: Absolute error by dopant type (metal/nonmetal)
    'correlation_matrix': False,             # Section 2d: Correlation matrix
    
    # Detailed analysis plots
    'radar_plots': False,                    # Section 3: Individual radar plots for all dopants
    'stacked_bar_plot': True,               # Section 4: Stacked bar plot - errors by dopant
    
    # Property analysis plots
    'dopant_properties_analysis': False,     # Section 5: All dopant properties analysis plots
    'error_vs_atomic_radius': False,         # Section 5.1: Error vs atomic radius
    'error_vs_atomic_weight': False,         # Section 5.2: Error vs atomic weight
    'error_by_period': False,                # Section 5.3: Error by period
    'error_by_group': False,                 # Section 5.4: Error by group
    
    # Final outputs
    'publication_figure': False,             # Section 6: Publication-quality figure
    'summary_statistics': False,             # Section 4: Summary statistics table
    'pdf_report': False,                     # Generate comprehensive PDF report
}

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# -----------------------------------------------------------------------------
# Global plotting defaults (centralized style constants and rcParams)
# Change these values to adjust plot appearance project-wide.
# -----------------------------------------------------------------------------
FIGURE_DPI = 300                 # default figure DPI for on-screen
SAVEFIG_DPI = 600                # DPI used when saving high-resolution PNGs

# Font sizes (points) - dramatically increased for better readability
FONT_SIZE = 24                   # base font size
TITLE_FONT_SIZE = 28             # figure / subplot titles (will be removed per user request)
LABEL_FONT_SIZE = 26             # axis labels
TICK_LABEL_SIZE = 22             # tick label font size
LEGEND_FONT_SIZE = 20            # legend font size
ANNOTATION_FONT_SIZE = 18        # annotations and text

# Tick and line styling
TICK_SIZE = 8                     # major tick length (points)
TICK_WIDTH = 2.0
MINOR_TICK_SIZE = 4
LINE_WIDTH = 3.0
MARKER_SIZE = 150                 # default marker area used by scatter (s parameter)
MARKER_POINT = int(np.sqrt(MARKER_SIZE))  # approximate marker size in points for markersize

# Apply rcParams so defaults are used across the entire script
plt.rcParams.update({
    'figure.dpi': FIGURE_DPI,
    'savefig.dpi': SAVEFIG_DPI,
    'font.size': FONT_SIZE,
    'axes.titlesize': TITLE_FONT_SIZE,
    'axes.labelsize': LABEL_FONT_SIZE,
    'xtick.labelsize': TICK_LABEL_SIZE,
    'ytick.labelsize': TICK_LABEL_SIZE,
    'xtick.major.size': TICK_SIZE,
    'ytick.major.size': TICK_SIZE,
    'xtick.major.width': TICK_WIDTH,
    'ytick.major.width': TICK_WIDTH,
    'xtick.minor.size': MINOR_TICK_SIZE,
    'ytick.minor.size': MINOR_TICK_SIZE,
    'lines.linewidth': LINE_WIDTH,
    'legend.fontsize': LEGEND_FONT_SIZE,
    'savefig.facecolor': 'white'
})



# Always resolve paths relative to the repo root
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
ml_csv = REPO_ROOT / "01_DFT_MLIP_Validation" / "ml_formation_energies_s.csv"
dft_csv = REPO_ROOT / "01_DFT_MLIP_Validation" / "dft_formation_energies.csv"
ml_df = pd.read_csv(ml_csv)
dft_df = pd.read_csv(dft_csv)

DOPANTS = [
    'C', 'N', 'O', 'F', 'B', 'P', 'Te', 'Cl', 'Si', 'Li', 'Na', 'Al', 'Zn', 'V', 'Fe', 'Co', 'Ni', 'Cu', 'Nb', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Ta', 'W', 'Re', 'Ir', 'Pt', 'Au', 'Ti'
] # !!!!change as needed!!!!!

# Check which dopants are in each dataset
ml_dopants = set(ml_df['dopant'])
dft_dopants = set(dft_df['dopant'])

print(f"ML dopants: {sorted(ml_dopants)}")
print(f"DFT dopants: {sorted(dft_dopants)}")
print(f"Common dopants: {sorted(ml_dopants & dft_dopants)}")
print(f"Only in ML: {sorted(ml_dopants - dft_dopants)}")
print(f"Only in DFT: {sorted(dft_dopants - ml_dopants)}")

# Merge on dopant
df = pd.merge(ml_df, dft_df, on="dopant", suffixes=("_ml", "_dft"))

# Filter to only DOPANTS in the list
df = df[df['dopant'].isin(DOPANTS)].reset_index(drop=True)

# Report missing dopants
missing_in_ml = sorted(set(DOPANTS) - set(ml_df['dopant']))
missing_in_dft = sorted(set(DOPANTS) - set(dft_df['dopant']))
missing_in_both = sorted(set(DOPANTS) - set(df['dopant']))

if missing_in_ml:
    print(f"\n⚠ Dopants missing in ML data: {missing_in_ml}")
if missing_in_dft:
    print(f"\n⚠ Dopants missing in DFT data: {missing_in_dft}")
if missing_in_both:
    print(f"\n⚠ Dopants missing in both merged data: {missing_in_both}")

print(f"\nAnalysis will use {len(df)} dopants from the DOPANTS list.")

# Check for missing data
print(f"ML data: {len(ml_df)} dopants")
print(f"DFT data: {len(dft_df)} dopants")
print(f"Merged data: {len(df)} dopants")

# Check for NaN values
nan_check = df.isnull().sum()
if nan_check.sum() > 0:
    print("\nNaN values found:")
    print(nan_check[nan_check > 0])
    print("\nDropping rows with NaN values...")
    df = df.dropna()
    print(f"Final data: {len(df)} dopants")

# Formation energy columns to compare (base names)
formations = ["Eform_Ssub (eV)", "Eform_Mosub (eV)", "Eform_int (eV)"]
formation_labels = ["S-substitution", "Mo-substitution", "Intercalation"]

# Create output directory
output_dir = Path("ml_vs_dft_comparison_s_new")
output_dir.mkdir(exist_ok=True)

# Color schemes for different formations - distinct, colorblind-friendly palette
colors = ['#E31A1C', '#1F78B4', '#33A02C']  # Red, Blue, Green - highly distinct


##%# 1. PARITY PLOTS - Main comparison (One plot per figure)
stats_summary = []

# Calculate statistics for all formation types (needed for summary even if plots are disabled)
for i, (form, label, color) in enumerate(zip(formations, formation_labels, colors)):
    # Get ML and DFT values - use correct column names after merge
    ml_vals = df[f"{form}_ml"].values
    dft_vals = df[f"{form}_dft"].values
    
    # Calculate statistics
    mae = mean_absolute_error(dft_vals, ml_vals)
    rmse = np.sqrt(np.mean((ml_vals - dft_vals)**2))
    r2 = r2_score(dft_vals, ml_vals)
    pearson_r, _ = pearsonr(dft_vals, ml_vals)
    
    stats_summary.append({
        'Formation_Type': label,
        'MAE (eV)': mae,
        'RMSE (eV)': rmse,
        'R²': r2,
        'Pearson_r': pearson_r
    })

if PLOT_CONFIG['individual_parity_plots']:
    print("Generating individual parity plots...")

    for i, (form, label, color) in enumerate(zip(formations, formation_labels, colors)):
        # Create individual figure for each parity plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        
        # Get ML and DFT values - use correct column names after merge
        ml_vals = df[f"{form}_ml"].values
        dft_vals = df[f"{form}_dft"].values
        
        # Get statistics from pre-calculated stats_summary
        stats = stats_summary[i]
        mae = stats['MAE (eV)']
        r2 = stats['R²']
        pearson_r = stats['Pearson_r']
        
        # Scatter plot
        ax.scatter(dft_vals, ml_vals, alpha=0.7, s=MARKER_SIZE, color=color, edgecolors='white', linewidth=LINE_WIDTH)
        
        # Perfect correlation line
        min_val = min(min(dft_vals), min(ml_vals))
        max_val = max(max(dft_vals), max(ml_vals))
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=LINE_WIDTH, label='Perfect correlation')

        # Error bands (±0.3 eV, ±0.6 eV)
        x_range = np.linspace(min_val, max_val, 100)
        ax.fill_between(x_range, x_range - 0.3, x_range + 0.3, alpha=0.2, color='green', label='±0.3 eV')
        ax.fill_between(x_range, x_range - 0.6, x_range + 0.6, alpha=0.1, color='orange', label='±0.6 eV')

        # Annotations for outliers removed per user request
        
        ax.set_xlabel(f'DFT {label} (eV)', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(f'ML {label} (eV)', fontsize=LABEL_FONT_SIZE)
        # Remove title as requested by user
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        
        # Make axes equal
        ax.set_aspect('equal', adjustable='box')
        
        # Add statistics text box
        stats_text = f'MAE: {mae:.3f} eV\nR²: {r2:.3f}\nPearson r: {pearson_r:.3f}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=ANNOTATION_FONT_SIZE,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        filename = f"parity_plot_{form}.png"
        plt.savefig(output_dir / filename, dpi=SAVEFIG_DPI, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.show()
        plt.close()

# 1b. COMBINED PARITY PLOT - All three formation types in one figure
if PLOT_CONFIG['combined_parity_plot']:
    print("Generating combined parity plot...")
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))

    for i, (form, label, color) in enumerate(zip(formations, formation_labels, colors)):
        ml_vals = df[f"{form}_ml"].values
        dft_vals = df[f"{form}_dft"].values
        ax.scatter(dft_vals, ml_vals, alpha=0.7, s=MARKER_SIZE, color=color, 
                   edgecolors='white', linewidth=LINE_WIDTH, label=label)

    # Calculate overall range for perfect correlation line
    all_dft = []
    all_ml = []
    for form in formations:
        all_dft.extend(df[f"{form}_dft"].values)
        all_ml.extend(df[f"{form}_ml"].values)

    min_val = min(min(all_dft), min(all_ml))
    max_val = max(max(all_dft), max(all_ml))

    # Perfect correlation line
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=LINE_WIDTH, label='Perfect correlation')

    # Error bands
    x_range = np.linspace(min_val, max_val, 100)
    ax.fill_between(x_range, x_range - 0.3, x_range + 0.3, alpha=0.2, color='green', label='±0.3 eV')
    ax.fill_between(x_range, x_range - 0.6, x_range + 0.6, alpha=0.1, color='orange', label='±0.6 eV')

    ax.set_xlabel('DFT Formation Energy (eV)', fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel('ML Formation Energy (eV)', fontsize=LABEL_FONT_SIZE)
    ax.grid(True, alpha=0.3)

    # Create comprehensive legend with formation types and error bands
    ax.legend(fontsize=LEGEND_FONT_SIZE, loc='lower right')

    # Make axes equal
    ax.set_aspect('equal', adjustable='box')

    # Calculate overall statistics
    all_dft_flat = np.array(all_dft)
    all_ml_flat = np.array(all_ml)
    overall_mae = mean_absolute_error(all_dft_flat, all_ml_flat)
    overall_r2 = r2_score(all_dft_flat, all_ml_flat)
    overall_pearson_r, _ = pearsonr(all_dft_flat, all_ml_flat)

    # Add overall statistics text box
    stats_text = f'Overall MAE: {overall_mae:.3f} eV\nOverall R²: {overall_r2:.3f}\nOverall Pearson r: {overall_pearson_r:.3f}'
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=ANNOTATION_FONT_SIZE,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_dir / "combined_parity_plot.png", dpi=SAVEFIG_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# 2. ERROR ANALYSIS BY DOPANT (Individual plots)

# Calculate errors
for i, (form, label, color) in enumerate(zip(formations, formation_labels, colors)):
    df[f"{form}_error"] = df[f"{form}_ml"] - df[f"{form}_dft"]

# Calculate total errors and sort dopants from best to worst (needed for multiple plot types)
total_errors = []
for _, row in df.iterrows():
    total_error = sum([abs(row[f"{form}_error"]) for form in formations])
    total_errors.append((row['dopant'], total_error, row))

# Sort by total error (best performers first)
total_errors.sort(key=lambda x: x[1])

# Calculate global min/max for consistent scaling - use fixed scale
global_min = -5.1
global_max = 9.1

# Categorize dopants (needed by multiple sections)
metals = ['Au', 'Ta', 'Pd', 'V', 'Pt', 'Ag', 'Re', 'Ru', 'Nb', 'W', 'Fe', 'Cu', 'Ni', 'Al', 'Ir', 'Rh', 'Co', 'Zn', 'Cd', 'Ti'] # Add others as needed
nonmetals = ['C', 'Si', 'B', 'N', 'O', 'F', 'Cl', 'S', 'Se', 'Te'] # Add others as needed
df['dopant_type'] = df['dopant'].apply(lambda x: 'Metal' if x in metals else 'Nonmetal')

# 2a. Error distribution by formation type
if PLOT_CONFIG['error_distribution']:
    print("Generating error distribution plot...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    errors_data = [df[f"{form}_error"].values for form in formations]
    bp = ax.boxplot(errors_data, tick_labels=formation_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(0, color='black', linestyle='--', alpha=0.8)
    ax.set_ylabel('ML - DFT Error (eV)', fontsize=LABEL_FONT_SIZE)
    # No title as requested
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(output_dir / "error_distribution.png", dpi=SAVEFIG_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# 2b. Error heatmap by dopant
if PLOT_CONFIG['error_heatmap']:
    print("Generating error heatmap...")
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    error_matrix = df[['dopant'] + [f"{form}_error" for form in formations]].set_index('dopant')
    error_matrix.columns = formation_labels
    im = ax.imshow(error_matrix.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df['dopant'], rotation=45, ha='right', fontsize=TICK_LABEL_SIZE)
    ax.set_yticks(range(len(formation_labels)))
    ax.set_yticklabels(formation_labels, fontsize=TICK_LABEL_SIZE)
    # No title as requested
    plt.colorbar(im, ax=ax, label='ML - DFT Error (eV)')
    plt.tight_layout()
    plt.savefig(output_dir / "error_heatmap.png", dpi=SAVEFIG_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# 2c. Absolute error by dopant type
if PLOT_CONFIG['error_by_category']:
    print("Generating error by category plot...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for i, (form, label, color) in enumerate(zip(formations, formation_labels, colors)):
        abs_errors = df[f"{form}_error"].abs()
        metal_errors = abs_errors[df['dopant_type'] == 'Metal']
        nonmetal_errors = abs_errors[df['dopant_type'] == 'Nonmetal']
        
        x_pos = np.arange(2) + i * 0.25
        ax.bar(x_pos, [metal_errors.mean(), nonmetal_errors.mean()], 
               width=0.25, label=label, color=color, alpha=0.7)

    ax.set_xticks(np.arange(2) + 0.25)
    ax.set_xticklabels(['Metals', 'Nonmetals'])
    ax.set_ylabel('Mean Absolute Error (eV)', fontsize=LABEL_FONT_SIZE)
    # No title as requested
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(output_dir / "error_by_category.png", dpi=SAVEFIG_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# 2d. Correlation matrix
if PLOT_CONFIG['correlation_matrix']:
    print("Generating correlation matrix...")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    corr_data = []
    for form in formations:
        ml_vals = df[f"{form}_ml"].values
        dft_vals = df[f"{form}_dft"].values
        corr_data.append([pearsonr(dft_vals, ml_vals)[0]])

    corr_matrix = np.array([[pearsonr(df[f"{form1}_ml"], df[f"{form1}_dft"])[0] for form1 in formations]])
    im = ax.imshow(corr_matrix, vmin=0.8, vmax=1.0, cmap='viridis')
    ax.set_xticks(range(len(formation_labels)))
    ax.set_xticklabels(formation_labels, rotation=45, ha='right')
    ax.set_yticks([0])
    ax.set_yticklabels(['ML vs DFT'])
    # No title as requested

    # Add correlation values as text
    for i in range(len(formation_labels)):
        r_val = pearsonr(df[f"{formations[i]}_ml"], df[f"{formations[i]}_dft"])[0]
        ax.text(i, 0, f'{r_val:.3f}', ha='center', va='center', color='white', fontweight='bold', fontsize=ANNOTATION_FONT_SIZE)

    plt.colorbar(im, ax=ax, label='Pearson r')
    ax.tick_params(labelsize=TICK_LABEL_SIZE)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_matrix.png", dpi=SAVEFIG_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# 3. INDIVIDUAL RADAR PLOTS FOR ALL DOPANTS
if PLOT_CONFIG['radar_plots']:
    print("Generating radar plots...")

    # Create radar plots directory
    radar_dir = output_dir / "Radar_plots"
    radar_dir.mkdir(exist_ok=True)

    def create_radar_plot(dopant_data, dopant_name, global_min=-5.1, global_max=9.1):
        """Create enhanced radar plot for a specific dopant with all three formation types"""
        categories = []
        ml_values = []
        dft_values = []
        
        # Only include formation types where data exists
        for form, label in zip(formations, formation_labels):
            ml_col = f"{form}_ml"
            dft_col = f"{form}_dft"
            
            if ml_col in dopant_data and dft_col in dopant_data:
                if not (pd.isna(dopant_data[ml_col]) or pd.isna(dopant_data[dft_col])):
                    categories.append(label)
                    ml_values.append(dopant_data[ml_col])
                    dft_values.append(dopant_data[dft_col])
        
        if len(categories) == 0:
            print(f"Warning: No valid data for {dopant_name}")
            return None
        
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Add the first value to complete the circle
        ml_values += ml_values[:1]
        dft_values += dft_values[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        fig.patch.set_facecolor('white')
        
        # Colors
        ml_color = '#E74C3C'
        dft_color = '#3498DB'
        
        # Plot ML values
        ax.plot(angles, ml_values, 'o-', linewidth=LINE_WIDTH, label='ML (UMA)', 
                color=ml_color, alpha=0.9, markersize=MARKER_POINT, markerfacecolor=ml_color, 
                markeredgecolor='white', markeredgewidth=2)
        ax.fill(angles, ml_values, alpha=0.15, color=ml_color)
        
        # Plot DFT values
        ax.plot(angles, dft_values, 's-', linewidth=LINE_WIDTH, label='DFT (PBE)', 
                color=dft_color, alpha=0.9, markersize=MARKER_POINT, markerfacecolor=dft_color, 
                markeredgecolor='white', markeredgewidth=2)
        ax.fill(angles, dft_values, alpha=0.15, color=dft_color)
        
        # Set scale
        ax.set_ylim(global_min, global_max)
        ax.grid(True, color='#BDC3C7', linestyle='-', linewidth=1, alpha=0.6)
        
        # Calculate error for this dopant
        total_error = sum([abs(dopant_data.get(f"{form}_error", 0)) for form in formations 
                          if f"{form}_error" in dopant_data and not pd.isna(dopant_data[f"{form}_error"])])
        
        # Add category labels
        for angle, label in zip(angles[:-1], categories):
            ax.text(angle, global_max + 0.8, label, rotation=0, 
                    ha='center', va='center', fontsize=LABEL_FONT_SIZE + 2, fontweight='bold', 
                    color='#2C3E50', bbox=dict(boxstyle="round,pad=0.3", 
                    facecolor='white', edgecolor='none', alpha=0.8))
        
        ax.set_xticklabels([])
        
        # Title removed as requested by user
        
        # Legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=LEGEND_FONT_SIZE + 2, 
                  frameon=True, fancybox=True, shadow=True, facecolor='white', 
                  edgecolor='gray', framealpha=0.9)
        
        # Radial ticks
        rticks = np.linspace(global_min, global_max, 8)
        ax.set_rticks(rticks)
        ax.set_rgrids(rticks, labels=[f'{r:.1f}' for r in rticks], 
                      fontsize=TICK_LABEL_SIZE, color='#7F8C8D', alpha=0.8)
        
        return fig

    print(f"\nCreating individual radar plots for {len(total_errors)} dopants...")
    print("Dopant ranking (best to worst):")
    for i, (dopant, error, _) in enumerate(total_errors):
        print(f"{i+1:2d}. {dopant:2s} - Total Error: {error:.3f} eV")

    # Create individual radar plots
    individual_plots = []
    for i, (dopant_name, total_error, dopant_data) in enumerate(total_errors):
        print(f"Creating plot {i+1}/{len(total_errors)}: {dopant_name}")
        
        fig = create_radar_plot(dopant_data, dopant_name, global_min, global_max)
        
        # Save individual plot with high DPI
        plot_filename = f"{i+1:02d}_{dopant_name}_radar.png"
        fig.savefig(radar_dir / plot_filename, dpi=SAVEFIG_DPI, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        individual_plots.append((dopant_name, fig))
        
        # Close figure to save memory
        plt.close(fig)

    # Create PDF with all radar plots
    pdf_filename = radar_dir / "All_Dopant_Radar_Plots.pdf"
    print(f"\nCreating PDF with all radar plots: {pdf_filename}")

# 4. STACKED BAR PLOT - Errors by dopant ordered best to worst
if PLOT_CONFIG['stacked_bar_plot']:
    print("Generating stacked bar plot...")

    # Extract error data for each formation type
    error_data = []
    dopant_names = []

    for dopant_name, total_error, dopant_data in total_errors:
        # Get absolute errors for each formation type
        s_sub_error = abs(dopant_data.get(f"{formations[0]}_error", 0))  # S-substitution
        mo_sub_error = abs(dopant_data.get(f"{formations[1]}_error", 0))  # Mo-substitution
        int_error = abs(dopant_data.get(f"{formations[2]}_error", 0))  # Intercalation
        
        error_data.append([s_sub_error, mo_sub_error, int_error])
        dopant_names.append(dopant_name)

    # Convert to numpy array for easier manipulation
    error_data = np.array(error_data)

    # Create stacked bar plot
    fig, ax = plt.subplots(figsize=(20, 12))

    # Define colors for each formation type
    colors = ['#E31A1C', '#1F78B4', '#33A02C']  # Same colors as used in parity plots
    formation_labels_short = ['S-sub', 'Mo-sub', 'Intercalation']

    # Create the stacked bars
    x_positions = np.arange(len(dopant_names))
    bar_width = 0.8

    # Bottom values for stacking
    bottom1 = error_data[:, 0]  # S-substitution errors
    bottom2 = bottom1 + error_data[:, 1]  # S-sub + Mo-sub errors

    # Create bars
    bars1 = ax.bar(x_positions, error_data[:, 0], bar_width, 
                   label=formation_labels_short[0], color=colors[0], alpha=0.8, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x_positions, error_data[:, 1], bar_width, bottom=bottom1,
                   label=formation_labels_short[1], color=colors[1], alpha=0.8, edgecolor='white', linewidth=1)
    bars3 = ax.bar(x_positions, error_data[:, 2], bar_width, bottom=bottom2,
                   label=formation_labels_short[2], color=colors[2], alpha=0.8, edgecolor='white', linewidth=1)

    # Customize the plot
    ax.set_xlabel('Dopants (Best → Worst Performance)', fontsize=LABEL_FONT_SIZE*1.4, fontweight='bold')
    ax.set_ylabel('Absolute Error Formation Energy (eV)', fontsize=LABEL_FONT_SIZE*1.4, fontweight='bold')
    # Title removed as requested by user

    # Set x-axis labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dopant_names, fontsize=TICK_LABEL_SIZE * 1.4, fontweight='bold')

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

    # Add legend
    ax.legend(fontsize=LEGEND_FONT_SIZE * 1.4, loc='upper left', frameon=True, fancybox=True, 
              shadow=True, facecolor='white', edgecolor='gray', framealpha=0.9)

    # Add grid for better readability
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=1)
    ax.set_axisbelow(True)

    # Improve tick formatting
    ax.tick_params(axis='y', labelsize=TICK_LABEL_SIZE*1.4)
    ax.tick_params(axis='x', labelsize=TICK_LABEL_SIZE*1.4)

    # Add total error values on top of each bar
    total_errors_values = [sum(errors) for errors in error_data]
    for i, (total_err, dopant) in enumerate(zip(total_errors_values, dopant_names)):
        ax.text(i, total_err + 0.05, f'{total_err:.2f}', 
                ha='center', va='bottom', fontsize=ANNOTATION_FONT_SIZE*1.2, fontweight='bold', color='black')

    # Set y-axis limits with some padding
    ax.set_ylim(0, 5)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_SIZE*1.2)

    # Make the plot layout tight
    plt.tight_layout()

    # Save the stacked bar plot with high DPI
    plt.savefig(output_dir / "stacked_bar_errors.png", dpi=SAVEFIG_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

    # Create a summary table of the data
    error_summary_df = pd.DataFrame({
        'Dopant': dopant_names,
        'S-substitution_Error': error_data[:, 0],
        'Mo-substitution_Error': error_data[:, 1], 
        'Intercalation_Error': error_data[:, 2],
        'Total_Error': total_errors_values,
        'Rank': range(1, len(dopant_names) + 1)
    })

    # Save summary table
    error_summary_df.to_csv(output_dir / "dopant_error_ranking.csv", index=False)
    print(f"Error ranking saved to: {output_dir / 'dopant_error_ranking.csv'}")

    print(f"Stacked bar plot saved to: {output_dir / 'stacked_bar_errors.png'}")
    print(f"Best performing dopants: {', '.join(dopant_names[:5])}")
    print(f"Worst performing dopants: {', '.join(dopant_names[-5:])}")

# Create comprehensive PDF combining all existing high-quality plots
comprehensive_pdf = output_dir / "Complete_ML_DFT_Analysis.pdf"
print(f"\nCreating comprehensive PDF by combining existing plots: {comprehensive_pdf}")

if PLOT_CONFIG['pdf_report']:
    print("Generating comprehensive PDF report...")
    
    comprehensive_pdf = output_dir / "Comprehensive_ML_DFT_Report.pdf"
    
    with PdfPages(comprehensive_pdf) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.text(0.5, 0.8, 'ML vs DFT Formation Energy Analysis', 
            horizontalalignment='center', fontsize=TITLE_FONT_SIZE + 12, fontweight='bold', 
            transform=ax.transAxes, color='#2C3E50')
        ax.text(0.5, 0.7, 'Comprehensive Comparison Report', 
            horizontalalignment='center', fontsize=TITLE_FONT_SIZE + 4, transform=ax.transAxes, 
            color='#34495E')
        ax.text(0.5, 0.6, f'Total Dopants Analyzed: {len(total_errors)}', 
            horizontalalignment='center', fontsize=LABEL_FONT_SIZE + 4, transform=ax.transAxes)
    ax.text(0.5, 0.5, 'ML Potential: UMA (OMat24)', 
        horizontalalignment='center', fontsize=LABEL_FONT_SIZE + 2, transform=ax.transAxes)
    ax.text(0.5, 0.45, 'DFT: PBE with Quantum ESPRESSO', 
        horizontalalignment='center', fontsize=LABEL_FONT_SIZE + 2, transform=ax.transAxes)
    ax.text(0.5, 0.35, f'Formation Energy Scale: {global_min:.1f} to {global_max:.1f} eV', 
        horizontalalignment='center', fontsize=TICK_LABEL_SIZE + 2, transform=ax.transAxes)
    ax.text(0.5, 0.25, 'Analysis includes: S-substitution, Mo-substitution, and Intercalation', 
        horizontalalignment='center', fontsize=TICK_LABEL_SIZE + 2, transform=ax.transAxes)
    ax.text(0.5, 0.1, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}', 
        horizontalalignment='center', fontsize=ANNOTATION_FONT_SIZE, transform=ax.transAxes, 
        style='italic', color='gray')
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # Add existing high-quality plots by loading and displaying them
    from PIL import Image
    import matplotlib.image as mpimg
    
    # List of existing plot files to include
    plot_files = [
        ('parity_plots.png', 'Parity Plots: ML vs DFT'),
        ('error_analysis.png', 'Error Analysis by Dopant'),
        ('publication_figure.png', 'Publication Figure'),
        ('dopant_properties_analysis.png', 'Dopant Properties Analysis'),
        ('radar_plots_summary.png', 'Radar Plots Summary')
    ]
    
    # Add each existing plot to PDF
    for plot_file, title in plot_files:
        plot_path = output_dir / plot_file
        if plot_path.exists():
            print(f"Adding to PDF: {plot_file}")
            
            # Load the image
            img = mpimg.imread(str(plot_path))
            
            # Create figure and display image
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            print(f"Warning: Plot file not found: {plot_file}")
    
    # Individual radar plots section header
    fig, ax = plt.subplots(figsize=(11, 8.5))
    ax.text(0.5, 0.6, 'Individual Dopant Analysis', 
            horizontalalignment='center', fontsize=28, fontweight='bold', 
            transform=ax.transAxes, color='#2C3E50')
    ax.text(0.5, 0.5, 'Radar Plots for All Dopants', 
            horizontalalignment='center', fontsize=20, transform=ax.transAxes,
            color='#34495E')
    ax.text(0.5, 0.4, '(Ordered from Best to Worst Performance)', 
            horizontalalignment='center', fontsize=16, transform=ax.transAxes,
            style='italic', color='#7F8C8D')
    ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight', dpi=300)
    plt.close(fig)
    
    # Add all individual radar plots from the existing PNG files
    for i, (dopant_name, total_error, dopant_data) in enumerate(total_errors):
        radar_file = radar_dir / f"{i+1:02d}_{dopant_name}_radar.png"
        
        if radar_file.exists():
            print(f"Adding radar plot to PDF: {dopant_name}")
            
            # Load the radar plot image
            img = mpimg.imread(str(radar_file))
            
            # Create figure and display image
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis('off')
            
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)
        else:
            print(f"Warning: Radar plot not found: {radar_file}")

print(f"Comprehensive PDF created using existing high-quality plots: {comprehensive_pdf}")

# Create summary plot showing best and worst performers (only if needed for demonstration)
if PLOT_CONFIG['radar_plots']:  # Only create this summary if radar plots are enabled
    print("\nGenerating best/worst dopants radar summary...")
    # Create summary plot showing best and worst performers
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
    fig.suptitle('ML vs DFT: Best and Worst Performing Dopants', fontsize=16, fontweight='bold')

    # Best performers (first 3)
    best_dopants = total_errors[:3]
    for i, (dopant_name, total_error, dopant_data) in enumerate(best_dopants):
        ax = axes[0, i]
        
        categories = formation_labels
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        ml_values = [dopant_data[f"{form}_ml"] for form in formations]
        ml_values += ml_values[:1]
        dft_values = [dopant_data[f"{form}_dft"] for form in formations]
        dft_values += dft_values[:1]
        
        ax.plot(angles, ml_values, 'o-', linewidth=2, label='ML', color='red', alpha=0.8)
        ax.fill(angles, ml_values, alpha=0.25, color='red')
        ax.plot(angles, dft_values, 'o-', linewidth=2, label='DFT', color='blue', alpha=0.8)
        ax.fill(angles, dft_values, alpha=0.25, color='blue')
        
        ax.set_ylim(global_min, global_max)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        if i == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))

    # Worst performers (last 3)
    worst_dopants = total_errors[-3:]
    for i, (dopant_name, total_error, dopant_data) in enumerate(worst_dopants):
        ax = axes[1, i]
        
        categories = formation_labels
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]
        
        ml_values = [dopant_data[f"{form}_ml"] for form in formations]
        ml_values += ml_values[:1]
        dft_values = [dopant_data[f"{form}_dft"] for form in formations]
        dft_values += dft_values[:1]
        
        ax.plot(angles, ml_values, 'o-', linewidth=2, label='ML', color='red', alpha=0.8)
        ax.fill(angles, ml_values, alpha=0.25, color='red')
        ax.plot(angles, dft_values, 'o-', linewidth=2, label='DFT', color='blue', alpha=0.8)
        ax.fill(angles, dft_values, alpha=0.25, color='blue')
        
        ax.set_ylim(global_min, global_max)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "radar_plots_summary.png", dpi=SAVEFIG_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()

    # print(f"\nRadar plots completed!")
    # print(f"- Individual plots saved in: {radar_dir}")
    # print(f"- PDF with all plots: {pdf_filename}")
# print(f"- Summary plot: {output_dir / 'radar_plots_summary.png'}")

# 4. SUMMARY STATISTICS TABLE (moved before PDF creation)
stats_df = pd.DataFrame(stats_summary)
print("\n" + "="*60)
print("ML vs DFT FORMATION ENERGY COMPARISON SUMMARY")
print("="*60)
print(stats_df.to_string(index=False, float_format='%.4f'))
print("\n" + "="*60)

# Save summary statistics
stats_df.to_csv(output_dir / "summary_statistics.csv", index=False)

# 5. DOPANT PROPERTIES ANALYSIS
if PLOT_CONFIG['dopant_properties_analysis']:
    print("Generating dopant properties analysis...")

    # Define dopant properties for all DOPANTS
dopant_properties = {
    'C':  {'period': 2, 'group': 14, 'atomic_weight': 12.01,  'location': 'p-block'},
    'N':  {'period': 2, 'group': 15, 'atomic_weight': 14.01,  'location': 'p-block'},
    'O':  {'period': 2, 'group': 16, 'atomic_weight': 16.00,  'location': 'p-block'},
    'F':  {'period': 2, 'group': 17, 'atomic_weight': 19.00,  'location': 'p-block'},
    'B':  {'period': 2, 'group': 13, 'atomic_weight': 10.81,  'location': 'p-block'},
    'P':  {'period': 3, 'group': 15, 'atomic_weight': 30.97,  'location': 'p-block'},
    'Se': {'period': 4, 'group': 16, 'atomic_weight': 78.96,  'location': 'p-block'},
    'Te': {'period': 5, 'group': 16, 'atomic_weight': 127.60, 'location': 'p-block'},
    'Cl': {'period': 3, 'group': 17, 'atomic_weight': 35.45,  'location': 'p-block'},
    'Si': {'period': 3, 'group': 14, 'atomic_weight': 28.09,  'location': 'p-block'},
    'Li': {'period': 2, 'group': 1,  'atomic_weight': 6.94,   'location': 's-block'},
    'Na': {'period': 3, 'group': 1,  'atomic_weight': 22.99,  'location': 's-block'},
    'Al': {'period': 3, 'group': 13, 'atomic_weight': 26.98,  'location': 'p-block'},
    'Zn': {'period': 4, 'group': 12, 'atomic_weight': 65.38,  'location': 'd-block'},
    'V':  {'period': 4, 'group': 5,  'atomic_weight': 50.94,  'location': 'd-block'},
    'Mn': {'period': 4, 'group': 7,  'atomic_weight': 54.94,  'location': 'd-block'},
    'Fe': {'period': 4, 'group': 8,  'atomic_weight': 55.85,  'location': 'd-block'},
    'Co': {'period': 4, 'group': 9,  'atomic_weight': 58.93,  'location': 'd-block'},
    'Ni': {'period': 4, 'group': 10, 'atomic_weight': 58.69,  'location': 'd-block'},
    'Cu': {'period': 4, 'group': 11, 'atomic_weight': 63.55,  'location': 'd-block'},
    'Nb': {'period': 5, 'group': 5,  'atomic_weight': 92.91,  'location': 'd-block'},
    'Ru': {'period': 5, 'group': 8,  'atomic_weight': 101.07, 'location': 'd-block'},
    'Rh': {'period': 5, 'group': 9,  'atomic_weight': 102.91, 'location': 'd-block'},
    'Pd': {'period': 5, 'group': 10, 'atomic_weight': 106.42, 'location': 'd-block'},
    'Ag': {'period': 5, 'group': 11, 'atomic_weight': 107.87, 'location': 'd-block'},
    'Cd': {'period': 5, 'group': 12, 'atomic_weight': 112.41, 'location': 'd-block'},
    'Ta': {'period': 6, 'group': 5,  'atomic_weight': 180.95, 'location': 'd-block'},
    'W':  {'period': 6, 'group': 6,  'atomic_weight': 183.84, 'location': 'd-block'},
    'Re': {'period': 6, 'group': 7,  'atomic_weight': 186.21, 'location': 'd-block'},
    'Ir': {'period': 6, 'group': 9,  'atomic_weight': 192.22, 'location': 'd-block'},
    'Pt': {'period': 6, 'group': 10, 'atomic_weight': 195.08, 'location': 'd-block'},
    'Au': {'period': 6, 'group': 11, 'atomic_weight': 196.97, 'location': 'd-block'},
    'Ti': {'period': 4, 'group': 4,  'atomic_weight': 47.87,  'location': 'd-block'}
}

# Add properties to dataframe for all dopants in DOPANTS
for dopant in df['dopant']:
    props = dopant_properties.get(dopant)
    if props:
        df.loc[df['dopant'] == dopant, 'atomic_weight'] = props['atomic_weight']
        df.loc[df['dopant'] == dopant, 'location'] = props['location']
        df.loc[df['dopant'] == dopant, 'period'] = props['period']
        df.loc[df['dopant'] == dopant, 'group'] = props['group']
    else:
        print(f"⚠ Properties not defined for dopant: {dopant}")

# Calculate total absolute error for each dopant
df['total_abs_error'] = df[[f"{form}_error" for form in formations]].abs().sum(axis=1)

# Create the properties analysis plot
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Error Analysis by Dopant Properties', fontsize=16, fontweight='bold')

# 1. Error by location (d-block vs p-block)
ax = axes[0, 0]
location_data = []
location_labels = []
for loc in ['d-block', 'p-block']:
    subset = df[df['location'] == loc]
    if not subset.empty:
        location_data.append(subset['total_abs_error'].values)
        location_labels.append(f'{loc}\n(n={len(subset)})')

bp = ax.boxplot(location_data, tick_labels=location_labels, patch_artist=True)
colors = ['#3498DB', '#E74C3C']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Total Absolute Error (eV)', fontsize=12)
ax.grid(True, alpha=0.3)

# 2. Error vs Atomic Weight
ax = axes[0, 1]
scatter = ax.scatter(df['atomic_weight'], df['total_abs_error'], 
                    c=[colors[0] if loc == 'd-block' else colors[1] for loc in df['location']], 
                    alpha=0.7, s=80, edgecolors='white', linewidth=1)

# Add trend line
z = np.polyfit(df['atomic_weight'], df['total_abs_error'], 1)
p = np.poly1d(z)
ax.plot(df['atomic_weight'], p(df['atomic_weight']), "k--", alpha=0.8, linewidth=2)

# Annotate outliers
for i, row in df.iterrows():
    if row['total_abs_error'] > df['total_abs_error'].quantile(0.75):
        ax.annotate(row['dopant'], (row['atomic_weight'], row['total_abs_error']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

ax.set_xlabel('Atomic Weight (amu)', fontsize=12)
ax.set_ylabel('Total Absolute Error (eV)', fontsize=12)
ax.grid(True, alpha=0.3)

# Add correlation coefficient
corr_coef = np.corrcoef(df['atomic_weight'], df['total_abs_error'])[0, 1]
ax.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax.transAxes, 
        fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# 3. Error by Period
ax = axes[1, 0]
period_data = []
period_labels = []
for period in sorted(df['period'].unique()):
    subset = df[df['period'] == period]
    if not subset.empty:
        period_data.append(subset['total_abs_error'].values)
        period_labels.append(f'Period {int(period)}\n(n={len(subset)})')

bp = ax.boxplot(period_data, tick_labels=period_labels, patch_artist=True)
period_colors = plt.cm.viridis(np.linspace(0, 1, len(period_data)))
for patch, color in zip(bp['boxes'], period_colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax.set_ylabel('Total Absolute Error (eV)', fontsize=12)
ax.grid(True, alpha=0.3)

# 4. Error by Group (for common groups)
ax = axes[1, 1]
common_groups = df['group'].value_counts().head(6).index
group_data = []
group_labels = []
for group in sorted(common_groups):
    subset = df[df['group'] == group]
    if not subset.empty and len(subset) >= 2:  # Only include groups with 2+ elements
        group_data.append(subset['total_abs_error'].values)
        group_labels.append(f'Group {int(group)}\n(n={len(subset)})')

if group_data:
    bp = ax.boxplot(group_data, tick_labels=group_labels, patch_artist=True)
    group_colors = plt.cm.Set3(np.linspace(0, 1, len(group_data)))
    for patch, color in zip(bp['boxes'], group_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

ax.set_ylabel('Total Absolute Error (eV)', fontsize=12)
ax.grid(True, alpha=0.3)

# Add legend for d-block/p-block
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=colors[0], markersize=10, label='d-block'),
                  plt.Line2D([0], [0], marker='o', color='w', 
                             markerfacecolor=colors[1], markersize=10, label='p-block')]
axes[0, 1].legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig(output_dir / "dopant_properties_analysis.png", dpi=SAVEFIG_DPI, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# 6. PUBLICATION-QUALITY FIGURE (moved before PDF creation)
if PLOT_CONFIG['publication_figure']:
    print("Generating publication-quality figure...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('ML Potential Validation Against DFT', fontsize=16, fontweight='bold')

    # Combined parity plot
    ax = axes[0, 0]
    for i, (form, label, color) in enumerate(zip(formations, formation_labels, colors)):
        ml_vals = df[f"{form}_ml"].values
    dft_vals = df[f"{form}_dft"].values
    ax.scatter(dft_vals, ml_vals, alpha=0.7, s=60, color=color, label=label, edgecolors='white', linewidth=0.5)

all_dft = []
for form in formations:
    all_dft.extend(df[f"{form}_dft"].values)
min_val = min(all_dft)
max_val = max(all_dft)
ax.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.8, linewidth=2)
ax.fill_between([min_val, max_val], [min_val-0.1, max_val-0.1], [min_val+0.1, max_val+0.1], 
                alpha=0.2, color='gray', label='±0.1 eV')

ax.set_xlabel('DFT Formation Energy (eV)', fontsize=12)
ax.set_ylabel('ML Formation Energy (eV)', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)

# Error distribution
ax = axes[0, 1]
all_errors = []
error_labels = []
for form, label in zip(formations, formation_labels):
    errors = df[f"{form}_error"].values
    all_errors.extend(errors)
    error_labels.extend([label] * len(errors))

error_df = pd.DataFrame({'Error': all_errors, 'Type': error_labels})
sns.violinplot(data=error_df, x='Type', y='Error', ax=ax, inner='box')
ax.axhline(0, color='black', linestyle='--', alpha=0.8)
ax.set_ylabel('ML - DFT Error (eV)', fontsize=12)
ax.set_xlabel('')
ax.tick_params(axis='x', rotation=45)

# MAE comparison
ax = axes[1, 0]
mae_values = [stats_summary[i]['MAE (eV)'] for i in range(3)]
bars = ax.bar(formation_labels, mae_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax.set_ylabel('Mean Absolute Error (eV)', fontsize=12)
ax.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, mae_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

# R² comparison
ax = axes[1, 1]
r2_values = [stats_summary[i]['R²'] for i in range(3)]
bars = ax.bar(formation_labels, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
ax.set_ylabel('R² Score', fontsize=12)
ax.set_ylim(0, 1)
ax.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, value in zip(bars, r2_values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / "publication_figure.png", dpi=SAVEFIG_DPI, bbox_inches='tight', 
            facecolor='white', edgecolor='none')
plt.show()

# Note: Comprehensive PDF is created earlier using existing high-quality plots for efficiency

print("\nAnalysis complete. Check PNG files in output directory for results.")

if PLOT_CONFIG['pdf_report']:
    print("Generating comprehensive PDF report...")
    with PdfPages(comprehensive_pdf) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.text(0.5, 0.8, 'ML vs DFT Formation Energy Analysis', 
                horizontalalignment='center', fontsize=28, fontweight='bold', 
                transform=ax.transAxes, color='#2C3E50')
        ax.text(0.5, 0.7, 'Comprehensive Comparison Report', 
                horizontalalignment='center', fontsize=20, transform=ax.transAxes, 
                color='#34495E')
        ax.text(0.5, 0.6, f'Total Dopants Analyzed: {len(total_errors)}', 
                horizontalalignment='center', fontsize=16, transform=ax.transAxes)
        ax.text(0.5, 0.5, 'ML Potential: UMA (OMat24)', 
                horizontalalignment='center', fontsize=14, transform=ax.transAxes)
        ax.text(0.5, 0.45, 'DFT: PBE with Quantum ESPRESSO', 
                horizontalalignment='center', fontsize=14, transform=ax.transAxes)
        ax.text(0.5, 0.35, f'Formation Energy Scale: {global_min:.1f} to {global_max:.1f} eV', 
                horizontalalignment='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.25, 'Analysis includes: S-substitution, Mo-substitution, and Intercalation', 
                horizontalalignment='center', fontsize=12, transform=ax.transAxes)
        ax.text(0.5, 0.1, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                horizontalalignment='center', fontsize=10, transform=ax.transAxes, 
                style='italic', color='gray')
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        # Summary statistics page
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.text(0.5, 0.9, 'Summary Statistics', 
                horizontalalignment='center', fontsize=24, fontweight='bold', 
                transform=ax.transAxes, color='#2C3E50')
        
        # Create table
        stats_text = stats_df.to_string(index=False, float_format='%.4f')
        ax.text(0.5, 0.7, stats_text, horizontalalignment='center', 
                fontsize=12, transform=ax.transAxes, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='#F8F9FA', 
                         edgecolor='#DEE2E6', alpha=0.8))
        
        # Add interpretation
        ax.text(0.5, 0.4, 'Performance Ranking (Best to Worst):', 
                horizontalalignment='center', fontsize=16, fontweight='bold',
                transform=ax.transAxes, color='#2C3E50')
        
        ranking_text = ""
        for i, (dopant, error, _) in enumerate(total_errors[:10]):  # Top 10
            ranking_text += f"{i+1:2d}. {dopant:2s} (Error: {error:.3f} eV)\n"
        if len(total_errors) > 10:
            ranking_text += f"... and {len(total_errors)-10} more"
        
        ax.text(0.5, 0.15, ranking_text, horizontalalignment='center', 
                fontsize=11, transform=ax.transAxes, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='#E8F4FD', 
                         edgecolor='#2196F3', alpha=0.8))
        
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        # RECREATE PARITY PLOTS AT HIGH RESOLUTION
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('ML vs DFT Formation Energies: Parity Plots', fontsize=16, fontweight='bold')

        for i, (form, label, color) in enumerate(zip(formations, formation_labels, colors)):
            ax = axes[i]
            
            # Get ML and DFT values
            ml_vals = df[f"{form}_ml"].values
            dft_vals = df[f"{form}_dft"].values
            
            # Calculate statistics
            mae = mean_absolute_error(dft_vals, ml_vals)
            r2 = r2_score(dft_vals, ml_vals)
            
            # Scatter plot
            ax.scatter(dft_vals, ml_vals, alpha=0.7, s=MARKER_SIZE, color=color, edgecolors='white', linewidth=LINE_WIDTH)
            
            # Perfect correlation line
            min_val = min(min(dft_vals), min(ml_vals))
            max_val = max(max(dft_vals), max(ml_vals))
            ax.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.8, linewidth=LINE_WIDTH, label='Perfect correlation')
            
            # Error bands
            x_range = np.linspace(min_val, max_val, 100)
            ax.fill_between(x_range, x_range - 0.1, x_range + 0.1, alpha=0.2, color='green', label='±0.1 eV')
            ax.fill_between(x_range, x_range - 0.2, x_range + 0.2, alpha=0.1, color='orange', label='±0.2 eV')
            
            ax.set_xlabel(f'DFT {label} (eV)', fontsize=LABEL_FONT_SIZE)
            ax.set_ylabel(f'ML {label} (eV)', fontsize=LABEL_FONT_SIZE)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
            
            # Make axes equal
            ax.set_aspect('equal', adjustable='box')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=SAVEFIG_DPI)
        plt.close(fig)
        
        # RECREATE ERROR ANALYSIS AT HIGH RESOLUTION
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ML-DFT Error Analysis by Dopant', fontsize=16, fontweight='bold')

        # Error distribution by formation type
        ax = axes[0, 0]
        errors_data = [df[f"{form}_error"].values for form in formations]
        bp = ax.boxplot(errors_data, tick_labels=formation_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.axhline(0, color='black', linestyle='--', alpha=0.8)
        ax.set_ylabel('ML - DFT Error (eV)', fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)

        # Error heatmap by dopant
        ax = axes[0, 1]
        error_matrix = df[['dopant'] + [f"{form}_error" for form in formations]].set_index('dopant')
        error_matrix.columns = formation_labels
        im = ax.imshow(error_matrix.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df['dopant'], rotation=45, ha='right')
        ax.set_yticks(range(len(formation_labels)))
        ax.set_yticklabels(formation_labels)
        plt.colorbar(im, ax=ax, label='ML - DFT Error (eV)')

        # Absolute error by dopant type
        ax = axes[1, 0]
        metals = ['Au', 'Ta', 'Pd', 'V', 'Pt', 'Ag', 'Re', 'Ru', 'Nb', 'W', 'Fe', 'Cu', 'Ni', 'Al', 'Ir', 'Rh', 'Co', 'Zn', 'Cd', 'Ti']
        df['dopant_type'] = df['dopant'].apply(lambda x: 'Metal' if x in metals else 'Nonmetal')

        for i, (form, label, color) in enumerate(zip(formations, formation_labels, colors)):
            abs_errors = df[f"{form}_error"].abs()
            metal_errors = abs_errors[df['dopant_type'] == 'Metal']
            nonmetal_errors = abs_errors[df['dopant_type'] == 'Nonmetal']
            
            x_pos = np.arange(2) + i * 0.25
            ax.bar(x_pos, [metal_errors.mean(), nonmetal_errors.mean()], 
                   width=0.25, label=label, color=color, alpha=0.7)

        ax.set_xticks(np.arange(2) + 0.25)
        ax.set_xticklabels(['Metals', 'Nonmetals'])
        ax.set_ylabel('Mean Absolute Error (eV)', fontsize=LABEL_FONT_SIZE)
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, alpha=0.3)

        # Correlation matrix
        ax = axes[1, 1]
        corr_matrix = np.array([[pearsonr(df[f"{form1}_ml"], df[f"{form1}_dft"])[0] for form1 in formations]])
        im = ax.imshow(corr_matrix, vmin=0.8, vmax=1.0, cmap='viridis')
        ax.set_xticks(range(len(formation_labels)))
        ax.set_xticklabels(formation_labels, rotation=45, ha='right')
        ax.set_yticks([0])
        ax.set_yticklabels(['ML vs DFT'])

        for i in range(len(formation_labels)):
            r_val = pearsonr(df[f"{formations[i]}_ml"], df[f"{formations[i]}_dft"])[0]
            ax.text(i, 0, f'{r_val:.3f}', ha='center', va='center', color='white', fontweight='bold')

        plt.colorbar(im, ax=ax, label='Pearson r')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        plt.close(fig)
        
        # RECREATE PUBLICATION FIGURE AT HIGH RESOLUTION
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('ML Potential Validation Against DFT', fontsize=TITLE_FONT_SIZE, fontweight='bold')

        # Combined parity plot
        ax = axes[0, 0]
        for i, (form, label, color) in enumerate(zip(formations, formation_labels, colors)):
            ml_vals = df[f"{form}_ml"].values
            dft_vals = df[f"{form}_dft"].values
            ax.scatter(dft_vals, ml_vals, alpha=0.7, s=MARKER_SIZE, color=color, label=label, edgecolors='white', linewidth=LINE_WIDTH)

        all_dft = []
        for form in formations:
            all_dft.extend(df[f"{form}_dft"].values)
        min_val = min(all_dft)
        max_val = max(all_dft)
        ax.plot([min_val, max_val], [min_val, max_val], 'k-', alpha=0.8, linewidth=LINE_WIDTH)
        ax.fill_between([min_val, max_val], [min_val-0.1, max_val-0.1], [min_val+0.1, max_val+0.1], 
                        alpha=0.2, color='gray', label='±0.1 eV')

        ax.set_xlabel('DFT Formation Energy (eV)', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel('ML Formation Energy (eV)', fontsize=LABEL_FONT_SIZE)
        ax.legend(fontsize=LEGEND_FONT_SIZE)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=TICK_LABEL_SIZE)

        # Error distribution
        ax = axes[0, 1]
        all_errors = []
        error_labels = []
        for form, label in zip(formations, formation_labels):
            errors = df[f"{form}_error"].values
            all_errors.extend(errors)
            error_labels.extend([label] * len(errors))

        error_df = pd.DataFrame({'Error': all_errors, 'Type': error_labels})
        sns.violinplot(data=error_df, x='Type', y='Error', ax=ax, inner='box')
        ax.axhline(0, color='black', linestyle='--', alpha=0.8)
        ax.set_ylabel('ML - DFT Error (eV)', fontsize=LABEL_FONT_SIZE)
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45, labelsize=TICK_LABEL_SIZE)

        # MAE comparison
        ax = axes[1, 0]
        mae_values = [stats_summary[i]['MAE (eV)'] for i in range(3)]
        bars = ax.bar(formation_labels, mae_values, color=colors, alpha=0.7, edgecolor='black', linewidth=LINE_WIDTH)
        ax.set_ylabel('Mean Absolute Error (eV)', fontsize=LABEL_FONT_SIZE)
        ax.tick_params(axis='x', rotation=45, labelsize=TICK_LABEL_SIZE)

        for bar, value in zip(bars, mae_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        # R² comparison
        ax = axes[1, 1]
        r2_values = [stats_summary[i]['R²'] for i in range(3)]
        bars = ax.bar(formation_labels, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=LINE_WIDTH)
        ax.set_ylabel('R² Score', fontsize=LABEL_FONT_SIZE)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45, labelsize=TICK_LABEL_SIZE)

        for bar, value in zip(bars, r2_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=SAVEFIG_DPI)
        plt.close(fig)
        
        # RECREATE DOPANT PROPERTIES ANALYSIS AT HIGH RESOLUTION  
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Error Analysis by Dopant Properties', fontsize=TITLE_FONT_SIZE, fontweight='bold')

        # Error by location
        ax = axes[0, 0]
        location_data = []
        location_labels = []
        for loc in ['d-block', 'p-block']:
            subset = df[df['location'] == loc]
            if not subset.empty:
                location_data.append(subset['total_abs_error'].values)
                location_labels.append(f'{loc}\n(n={len(subset)})')

        bp = ax.boxplot(location_data, tick_labels=location_labels, patch_artist=True)
        colors_loc = ['#3498DB', '#E74C3C']
        for patch, color in zip(bp['boxes'], colors_loc):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Total Absolute Error (eV)', fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)

        # Error vs Atomic Weight
        ax = axes[0, 1]
        scatter = ax.scatter(df['atomic_weight'], df['total_abs_error'], 
                            c=[colors_loc[0] if loc == 'd-block' else colors_loc[1] for loc in df['location']], 
                            alpha=0.7, s=80, edgecolors='white', linewidth=1)

        z = np.polyfit(df['atomic_weight'], df['total_abs_error'], 1)
        p = np.poly1d(z)
        ax.plot(df['atomic_weight'], p(df['atomic_weight']), "k--", alpha=0.8, linewidth=2)

        ax.set_xlabel('Atomic Weight (amu)', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel('Total Absolute Error (eV)', fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)

        corr_coef = np.corrcoef(df['atomic_weight'], df['total_abs_error'])[0, 1]
        ax.text(0.05, 0.95, f'r = {corr_coef:.3f}', transform=ax.transAxes, 
                fontsize=ANNOTATION_FONT_SIZE, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Error by Period
        ax = axes[1, 0]
        period_data = []
        period_labels = []
        for period in sorted(df['period'].unique()):
            subset = df[df['period'] == period]
            if not subset.empty:
                period_data.append(subset['total_abs_error'].values)
                period_labels.append(f'Period {int(period)}\n(n={len(subset)})')

        bp = ax.boxplot(period_data, tick_labels=period_labels, patch_artist=True)
        period_colors = plt.cm.viridis(np.linspace(0, 1, len(period_data)))
        for patch, color in zip(bp['boxes'], period_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Total Absolute Error (eV)', fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)

        # Error by Group
        ax = axes[1, 1]
        common_groups = df['group'].value_counts().head(6).index
        group_data = []
        group_labels = []
        for group in sorted(common_groups):
            subset = df[df['group'] == group]
            if not subset.empty and len(subset) >= 2:
                group_data.append(subset['total_abs_error'].values)
                group_labels.append(f'Group {int(group)}\n(n={len(subset)})')

        if group_data:
            bp = ax.boxplot(group_data, tick_labels=group_labels, patch_artist=True)
            group_colors = plt.cm.Set3(np.linspace(0, 1, len(group_data)))
            for patch, color in zip(bp['boxes'], group_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_ylabel('Total Absolute Error (eV)', fontsize=LABEL_FONT_SIZE)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight', dpi=SAVEFIG_DPI)
        plt.close(fig)
        
        # Individual radar plots section header
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.text(0.5, 0.6, 'Individual Dopant Analysis', 
            horizontalalignment='center', fontsize=TITLE_FONT_SIZE, fontweight='bold', 
            transform=ax.transAxes, color='#2C3E50')
        ax.text(0.5, 0.5, 'Radar Plots for All Dopants', 
            horizontalalignment='center', fontsize=LABEL_FONT_SIZE, transform=ax.transAxes,
            color='#34495E')
        ax.text(0.5, 0.4, '(Ordered from Best to Worst Performance)', 
            horizontalalignment='center', fontsize=ANNOTATION_FONT_SIZE, transform=ax.transAxes,
            style='italic', color='#7F8C8D')
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight', dpi=SAVEFIG_DPI)
        plt.close(fig)
        
        # Add all individual radar plots
        for i, (dopant_name, total_error, dopant_data) in enumerate(total_errors):
            fig = create_radar_plot(dopant_data, dopant_name, global_min, global_max)
            pdf.savefig(fig, bbox_inches='tight', dpi=300)
            plt.close(fig)

        print(f"\nComprehensive PDF report created: {comprehensive_pdf}")

print("\nAnalysis complete! Check the output directory for all generated plots.")
print("\nTo control which plots are generated, modify the PLOT_CONFIG dictionary at the top of the script.")
