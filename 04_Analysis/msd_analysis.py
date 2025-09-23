#!/usr/bin/env python3
"""
Batch MSD Analysis Script for Multiple Dopants

Processes all dopant trajectory files and creates individual MSD plots for each.
Analyzes equilibration trajectories from multiple dopant systems.

Usage: python msd_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ase.io import read
from pathlib import Path
import pandas as pd
import os

# Configuration
BASE_DIR = "path_to/equilibriated_structures"
OUTPUT_DIR = "path_to/msd_results"

# Specific dopants to analyze
DOPANTS = [
    "Ir", "Re", "Rh", "Si", "Au", "Ru", "Al", "Pd", "Ti", "Ta", "Li", "Zn", "Nb", "Cu", "Ag", "Na", "C", "Pt", "V", "Cl", "O", "Fe", "N", "Te", "F"
]

# Create output directory if it doesn't exist
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Set style and plotting constants (matching dopant_interaction_comparison.py)
plt.style.use('seaborn-v0_8')

# Font sizes and formatting (matching dopant_interaction_comparison.py)
FIGURE_DPI = 300                 # default figure DPI for on-screen
SAVEFIG_DPI = 300                # DPI used when saving high-resolution PNGs

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

# Apply rcParams so defaults are used across the entire script (matching dopant_interaction_comparison.py)
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

def calculate_msd(positions, reference_positions=None):
    """
    Calculate Mean Square Displacement.
    
    Parameters:
    - positions: array of shape (n_frames, n_atoms, 3)
    - reference_positions: reference frame positions (default: first frame)
    
    Returns:
    - msd: array of shape (n_frames,) with MSD values
    """
    if reference_positions is None:
        reference_positions = positions[0]
    
    # Calculate displacement vectors
    displacements = positions - reference_positions[np.newaxis, :, :]
    
    # Calculate squared displacements
    squared_displacements = np.sum(displacements**2, axis=2)
    
    # Calculate mean over all atoms
    msd = np.mean(squared_displacements, axis=1)
    
    return msd

def load_trajectory(traj_file):
    """Load trajectory and extract positions."""
    print(f"Loading trajectory from: {traj_file}")
    
    # Read all frames
    atoms_list = read(traj_file, index=':')
    print(f"Loaded {len(atoms_list)} frames")
    
    # Extract basic info from first frame
    first_frame = atoms_list[0]
    print(f"System contains {len(first_frame)} atoms")
    print(f"Chemical symbols: {set(first_frame.get_chemical_symbols())}")
    
    # Extract positions for all frames
    positions = np.array([atoms.get_positions() for atoms in atoms_list])
    
    # Get chemical symbols
    symbols = first_frame.get_chemical_symbols()
    
    return positions, symbols, atoms_list

def analyze_species_msd(positions, symbols):
    """Calculate MSD for different atomic species."""
    unique_symbols = list(set(symbols))
    species_msd = {}
    
    for species in unique_symbols:
        # Get indices for this species
        indices = [i for i, sym in enumerate(symbols) if sym == species]
        
        if len(indices) > 0:
            # Extract positions for this species
            species_positions = positions[:, indices, :]
            # Calculate MSD
            species_msd[species] = calculate_msd(species_positions)
    
    return species_msd

def find_dopant_trajectories():
    """Find all dopant trajectory files in the base directory."""
    dopant_files = {}
    
    # Get all HEAT_* subfolders
    for subfolder in sorted(os.listdir(BASE_DIR)):
        if subfolder.startswith("HEAT_MoS2_8x8x4_5wt_"):
            # Extract dopant name (last part after the last underscore)
            dopant = subfolder.split("_")[-1]
            
            # Only process dopants in the specified list
            if dopant not in DOPANTS:
                continue
            
            # Construct trajectory file path
            traj_file = os.path.join(BASE_DIR, subfolder, f"nvt_equilibration_MoS2_8x8x4_5wt_{dopant}.traj")
            
            # Check if trajectory file exists
            if os.path.exists(traj_file):
                dopant_files[dopant] = traj_file
                print(f"Found trajectory for {dopant}: {traj_file}")
            else:
                print(f"Warning: Trajectory file not found for {dopant}")
    
    return dopant_files

def create_msd_plots(total_msd, species_msd, time_steps, symbols, dopant_name, output_dir, max_y_limit=None):
    """Create MSD visualization plots with publication-quality formatting."""
    
    # Create figure with reference-style size (matching dopant_interaction_comparison.py)
    plt.figure(figsize=(20, 12))
    
    # Fixed colors for Mo and S
    fixed_colors = {
        'Mo': '#2E86C1',  # Blue for Mo (always first)
        'S': '#E74C3C'    # Red for S (always second)
    }
    
    # Unique colors for each dopant (33 distinct colors)
    dopant_colors = {
        'Ag': '#9B59B6', 'Al': '#F39C12', 'Au': '#F1C40F', 'B': '#2ECC71', 'C': '#34495E',
        'Cd': '#E67E22', 'Cl': '#1ABC9C', 'Co': '#8E44AD', 'Cu': '#D35400', 'F': '#27AE60',
        'Fe': '#C0392B', 'Ir': '#16A085', 'Li': '#3498DB', 'Mn': '#9B59B6', 'N': '#2C3E50',
        'Na': '#F39C12', 'Nb': '#8B4513', 'Ni': '#17A2B8', 'O': '#E91E63', 'P': '#FF5722',
        'Pd': '#607D8B', 'Pt': '#795548', 'Re': '#FF9800', 'Rh': '#4CAF50', 'Ru': '#673AB7',
        'Se': '#009688', 'Si': '#CDDC39', 'Ta': '#FF6F00', 'Te': '#00BCD4', 'Ti': '#9C27B0',
        'V': '#4CAF50', 'W': '#FF5722', 'Zn': '#FFC107'
    } # Add others as needed
    
    # Define plot order: Mo, S, then dopant
    plot_order = ['Mo', 'S']
    
    # Add the dopant to the order (find which dopant is in this system)
    for species in species_msd.keys():
        if species not in plot_order:
            plot_order.append(species)
            break
    
    # Plot in the specified order
    for species in plot_order:
        if species in species_msd:
            atom_count = sum(1 for s in symbols if s == species)
            
            # Choose color based on species
            if species in fixed_colors:
                color = fixed_colors[species]
            else:
                color = dopant_colors.get(species, '#34495E')  # Default color if not found
            
            plt.plot(time_steps, species_msd[species], color=color, 
                    linewidth=LINE_WIDTH, label=f'{species} (n={atom_count})')
    
    plt.xlabel('Time (ps)', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    plt.ylabel('MSD (Å²)', fontsize=LABEL_FONT_SIZE, fontweight='bold')
    #plt.title(f'Species-Specific MSD - {dopant_name} Dopant', fontsize=TITLE_FONT_SIZE, fontweight='bold', pad=20)
    
    # Set tick label sizes to match reference
    plt.xticks(fontsize=TICK_LABEL_SIZE)
    plt.yticks(fontsize=TICK_LABEL_SIZE)
    
    # Set constant y-axis limits for all plots
    plt.ylim(0, max_y_limit if max_y_limit is not None else 70)
    
    # Add grid for better readability (reference style)
    plt.grid(True, alpha=0.3)
    
    # Improve legend to match reference style
    plt.legend(
        fontsize=LEGEND_FONT_SIZE,
        loc='upper left',
        frameon=True,
        fancybox=True,
        shadow=True,
        edgecolor='black',
        facecolor='white',
        framealpha=0.9
    )
    
    # Remove top and right spines for cleaner look
    sns.despine()
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot with high resolution (matching dopant_interaction_comparison.py)
    output_file = Path(output_dir) / f"msd_analysis_{dopant_name}.png"
    plt.savefig(output_file, dpi=SAVEFIG_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"MSD plot saved as: {output_file}")
    
    return plt.gcf()

def save_msd_data(total_msd, species_msd, time_steps, dopant_name, output_dir):
    """Save MSD data to CSV file."""
    
    # Create DataFrame
    data = {'Time_ps': time_steps, 'Total_MSD': total_msd}
    
    # Add species-specific MSD
    for species, msd in species_msd.items():
        data[f'{species}_MSD'] = msd
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = Path(output_dir) / f"msd_data_{dopant_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"MSD data saved as: {output_file}")
    
    return df

def calculate_all_msd_data(dopant_files):
    """Calculate MSD data for all dopants in one pass to avoid redundancy."""
    print("\nCalculating MSD data for all dopants...")
    max_msd_global = 0
    all_msd_data = {}
    
    for i, (dopant, traj_file) in enumerate(dopant_files.items(), 1):
        print(f"  Processing {i}/{len(dopant_files)}: {dopant}")
        
        try:
            # Load trajectory
            positions, symbols, atoms_list = load_trajectory(traj_file)
            
            # Create time array in picoseconds
            time_steps = np.arange(len(positions)) * 0.1  # Convert to picoseconds
            
            # Calculate total MSD
            total_msd = calculate_msd(positions)
            
            # Calculate species-specific MSD
            species_msd = analyze_species_msd(positions, symbols)
            
            # Calculate dopant slope
            dopant_slope = None
            for species, msd in species_msd.items():
                if species not in ['Mo', 'S'] and len(msd) > 10:
                    # Use last 50% of trajectory for linear fit
                    start_idx = len(msd) // 2
                    x = time_steps[start_idx:]
                    y = msd[start_idx:]
                    
                    # Linear fit: MSD = slope*t + intercept
                    slope, intercept = np.polyfit(x, y, 1)
                    dopant_slope = slope
                    break
            
            # Store all data for this dopant
            all_msd_data[dopant] = {
                'positions': positions,
                'symbols': symbols,
                'time_steps': time_steps,
                'total_msd': total_msd,
                'species_msd': species_msd,
                'dopant_slope': dopant_slope
            }
            
            # Track maximum MSD for y-axis scaling
            for species, msd_values in species_msd.items():
                if len(msd_values) > 0:
                    max_msd_this_species = np.max(msd_values)
                    max_msd_global = max(max_msd_global, max_msd_this_species)
                    
        except Exception as e:
            print(f"    Warning: Could not process {dopant}: {e}")
            continue
    
    print(f"  Maximum MSD found across all dopants: {max_msd_global:.3f} Å²")
    return all_msd_data, max_msd_global

def process_single_dopant_from_data(dopant_name, msd_data, output_dir, max_y_limit=None):
    """Process MSD analysis for a single dopant using pre-calculated data."""
    
    print(f"\n{'='*60}")
    print(f"PROCESSING DOPANT: {dopant_name}")
    print(f"{'='*60}")
    
    try:
        # Extract pre-calculated data
        positions = msd_data['positions']
        symbols = msd_data['symbols']
        time_steps = msd_data['time_steps']
        total_msd = msd_data['total_msd']
        species_msd = msd_data['species_msd']
        dopant_slope = msd_data['dopant_slope']
        
        # Create plots
        print("Creating MSD plots...")
        fig = create_msd_plots(total_msd, species_msd, time_steps, symbols, dopant_name, output_dir, max_y_limit)
        
        # Save data
        print("Saving MSD data...")
        df = save_msd_data(total_msd, species_msd, time_steps, dopant_name, output_dir)
        
        # Print statistics
        print_msd_statistics(total_msd, species_msd, time_steps, symbols, dopant_name)
        
        # Close plot to free memory
        plt.close(fig)
        
        return {
            'dopant': dopant_name,
            'total_msd_final': total_msd[-1],
            'species_msd_final': {species: msd[-1] for species, msd in species_msd.items()},
            'dopant_slope': dopant_slope,
            'n_frames': len(time_steps),
            'n_atoms': len(symbols)
        }
        
    except Exception as e:
        print(f"Error processing {dopant_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_msd_statistics(total_msd, species_msd, time_steps, symbols, dopant_name):
    """Print statistical summary of MSD analysis."""
    
    print("\n" + "="*60)
    print(f"MSD ANALYSIS SUMMARY - {dopant_name}")
    print("="*60)
    
    print(f"Total simulation time steps: {len(time_steps)}")
    print(f"Total simulation time: {time_steps[-1]:.1f} ps")
    print(f"Final total MSD: {total_msd[-1]:.3f} Å²")
    
    print("\nSpecies-specific final MSD:")
    dopant_slope = None
    for species, msd in species_msd.items():
        atom_count = sum(1 for s in symbols if s == species)
        print(f"  {species:>2}: {msd[-1]:8.3f} Å² (n={atom_count:>3} atoms)")
        
        # Calculate slope for dopant species (not Mo or S)
        if species not in ['Mo', 'S'] and len(msd) > 10:
            # Use last 50% of trajectory for linear fit
            start_idx = len(msd) // 2
            x = time_steps[start_idx:]
            y = msd[start_idx:]
            
            # Linear fit: MSD = slope*t + intercept
            slope, intercept = np.polyfit(x, y, 1)
            dopant_slope = slope
    
    # Calculate diffusion-like behavior (linear fit to later part)
    if len(total_msd) > 10:
        # Use last 50% of trajectory for linear fit
        start_idx = len(total_msd) // 2
        x = time_steps[start_idx:]
        y = total_msd[start_idx:]
        
        # Linear fit: MSD = 6*D*t + offset
        slope, intercept = np.polyfit(x, y, 1)
        
        print(f"\nLinear behavior analysis (last 50% of trajectory):")
        print(f"  Diffusivity: {slope:.6f} Å²/ps")
        print(f"  Estimated diffusion coefficient: {slope/6:.6f} Å²/ps")
    
    return dopant_slope

def create_summary_report(results_list, output_dir):
    """Create a summary report of all dopant MSD analyses."""
    
    # Filter out None results (failed analyses)
    valid_results = [r for r in results_list if r is not None]
    
    if not valid_results:
        print("No valid results to summarize!")
        return
    
    print(f"\n{'='*80}")
    print("BATCH MSD ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed {len(valid_results)} dopants")
    print(f"Failed analyses: {len(results_list) - len(valid_results)}")
    
    # Create summary DataFrame
    summary_data = []
    for result in valid_results:
        row = {
            'Dopant': result['dopant'],
            'Final_Total_MSD': result['total_msd_final'],
            'N_Frames': result['n_frames'],
            'N_Atoms': result['n_atoms']
        }
        # Add species-specific MSD
        for species, msd_val in result['species_msd_final'].items():
            row[f'{species}_MSD'] = msd_val
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_file = Path(output_dir) / "msd_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSummary saved as: {summary_file}")
    
    # Print top movers
    print("\nTop 5 dopants by final total MSD:")
    top_movers = summary_df.nlargest(5, 'Final_Total_MSD')
    for _, row in top_movers.iterrows():
        print(f"  {row['Dopant']:>2}: {row['Final_Total_MSD']:8.3f} Å²")
    
    return summary_df

def create_msd_slope_barplot(results_list, output_dir):
    """Create a bar plot comparing MSD slopes of dopants with density analysis formatting."""
    
    # Filter out None results and extract slope data
    slope_data = []
    for result in results_list:
        if result is not None and result['dopant_slope'] is not None:
            slope_data.append({
                'Dopant': result['dopant'],
                'MSD_Slope': result['dopant_slope']
            })
    
    if not slope_data:
        print("No valid slope data to plot!")
        return
    
    df = pd.DataFrame(slope_data)
    
    # Create figure with reference-style size (matching dopant_interaction_comparison.py)
    plt.figure(figsize=(20, 12))
    
    # Use different color from density analysis (density used green/blue)
    # Use orange/red theme for MSD slopes
    bar_color = "#FF6B35"  # Orange-red color, different from density analysis
    
    # Sort dopants by slope for better visualization (ascending order)
    df_sorted = df.sort_values('MSD_Slope', ascending=True)
    
    # Create barplot with sorted data
    ax = sns.barplot(
        data=df_sorted, 
        x="Dopant", 
        y="MSD_Slope",
        color=bar_color,
        alpha=0.8,
        edgecolor='white',
        linewidth=1
    )
    
    # Customize plot to match reference style (using dopant_interaction_comparison.py sizing)
    plt.xlabel("Dopant Element", fontsize=LABEL_FONT_SIZE*1.4, fontweight='bold')
    plt.ylabel("Diffusivity (Å²/ps)", fontsize=LABEL_FONT_SIZE*1.4, fontweight='bold')
    #plt.title("Dopant Mobility Comparison (MSD Slope)", fontweight='bold', pad=20)
    
    # Rotate x-axis labels for better readability (matching reference)
    plt.xticks(rotation=0, ha='center', fontsize=TICK_LABEL_SIZE*1.4, fontweight='bold')
    
    # Improve tick formatting (matching reference)
    ax.tick_params(axis='y', labelsize=TICK_LABEL_SIZE*1.2)
    ax.tick_params(axis='x', labelsize=TICK_LABEL_SIZE*1.4)
    
    # Add grid for better readability (reference style)
    ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=1)
    ax.set_axisbelow(True)
    
    # Remove top and right spines for cleaner look
    sns.despine()
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot with high resolution (matching dopant_interaction_comparison.py)
    output_file = Path(output_dir) / "msd_slope_comparison.png"
    plt.savefig(output_file, dpi=SAVEFIG_DPI, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"\nMSD slope comparison plot saved as: {output_file}")
    
    # Print top movers by slope
    print(f"\nTop 5 most mobile dopants by MSD slope:")
    df_sorted_desc = df_sorted.sort_values('MSD_Slope', ascending=False)
    for _, row in df_sorted_desc.head(5).iterrows():
        print(f"  {row['Dopant']:>2}: {row['MSD_Slope']:8.3f} Å²/ps")
    
    return ax

def main():
    """Main batch MSD analysis workflow."""
    
    print("Batch MSD Analysis for Multiple Dopants")
    print("="*60)
    
    # Find all dopant trajectory files
    print("Discovering dopant trajectory files...")
    dopant_files = find_dopant_trajectories()
    
    if not dopant_files:
        print("No trajectory files found!")
        return
    
    print(f"Found {len(dopant_files)} dopant trajectory files")
    
    # Single pass: Calculate all MSD data and find maximum for y-axis scaling
    all_msd_data, max_y_limit = calculate_all_msd_data(dopant_files)
    
    # Add a small margin (5%) to the maximum for better visualization
    max_y_limit_with_margin = max_y_limit * 1.05
    print(f"Using y-axis limit: 0 to {max_y_limit_with_margin:.1f} Å²")
    
    # Process each dopant for plotting and saving (using pre-calculated data)
    results = []
    for i, dopant in enumerate(dopant_files.keys(), 1):
        if dopant in all_msd_data:
            print(f"\nProcessing plots for {i}/{len(dopant_files)}: {dopant}")
            result = process_single_dopant_from_data(dopant, all_msd_data[dopant], OUTPUT_DIR, max_y_limit_with_margin)
            results.append(result)
    
    # Create summary report
    create_summary_report(results, OUTPUT_DIR)
    
    # Create MSD slope comparison plot (using pre-calculated slopes)
    create_msd_slope_barplot(results, OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print("BATCH ANALYSIS COMPLETED!")
    print(f"{'='*60}")
    print(f"Results saved in: {OUTPUT_DIR}")
    print(f"Individual plots: msd_analysis_{{dopant}}.png")
    print(f"Individual data: msd_data_{{dopant}}.csv")
    print(f"Summary report: msd_summary.csv")
    print(f"MSD slope comparison: msd_slope_comparison.png")

if __name__ == "__main__":
    main()