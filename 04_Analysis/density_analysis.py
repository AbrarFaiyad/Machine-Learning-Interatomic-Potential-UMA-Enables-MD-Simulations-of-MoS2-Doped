#!/usr/bin/env python3
"""
Simple script to analyze initial vs final densities for each dopant.
Reads lowtemp_output.txt files from each dopant subfolder and creates a barplot.
Extracts densities from lines 7-27 and averages the last 10 datapoints.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
BASE_DIR = "/home/afaiyad/borgstore/MoS2_Omat/05_Equlibriation/equilibriated_structures_doped"
OUTPUT_FILE = "dopant_density_comparison.png"

# Specific dopants to analyze
DOPANTS = [
        "Ir", "Re", "Rh", "Si", "Au", "Ru", "Al", "Pd", "Ti", "Ta", "Li", "Zn", "Nb", "Cu", "Ag", "Na", "C", "Pt", "V", "Cl", "O", "Fe", "N", "Te", "F"
    ]

def extract_densities():
    """Extract initial and final densities from all dopant subfolders."""
    data = []
    
    # Get all HEAT_* subfolders
    for subfolder in sorted(os.listdir(BASE_DIR)):
        if subfolder.startswith("HEAT_MoS2_8x8x4_5wt_"):
            # Extract dopant name (last part after the last underscore)
            dopant = subfolder.split("_")[-1]
            
            # Only process dopants in the specified list
            if dopant not in DOPANTS:
                continue
            
            # Path to lowtemp output file
            density_file = os.path.join(BASE_DIR, subfolder, "lowtemp_output.txt")
            
            if os.path.exists(density_file):
                try:
                    # Read the file
                    with open(density_file, 'r') as f:
                        lines = f.readlines()
                    
                    # Extract lines 7-27 (indices 6-26 in 0-based indexing)
                    if len(lines) >= 27:
                        target_lines = lines[6:27]  # Lines 7-27
                        
                        densities = []
                        for line in target_lines:
                            line = line.strip()
                            if 'ρ=' in line:
                                # Extract density value from ρ=X.XXXg/cm³ format
                                rho_part = line.split('ρ=')[1]
                                density_str = rho_part.split('g/cm³')[0]
                                density = float(density_str)
                                densities.append(density)
                        
                        if len(densities) >= 10:
                            # Average the last 10 density values
                            avg_density = sum(densities[-10:]) / 10
                            
                            # Add averaged density to data
                            data.append({"Dopant": dopant, "Density": avg_density})
                            
                            print(f"{dopant}: Last 10 avg density={avg_density:.6f} (from {len(densities)} points)")
                        else:
                            print(f"{dopant}: Insufficient density data points ({len(densities)} found)")

                except Exception as e:
                    print(f"Error processing {dopant}: {e}")
            else:
                print(f"No lowtemp_output.txt found for {dopant}")
    
    return pd.DataFrame(data)

def create_barplot(df):
    """Create publication-quality barplot using seaborn."""
    # Set style to match reference
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Apply rcParams to match reference formatting
    plt.rcParams.update({
        'figure.dpi': 300,
        'savefig.dpi': 600,
        'font.size': 24,
        'axes.titlesize': 28,
        'axes.labelsize': 26,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'xtick.major.size': 8,
        'ytick.major.size': 8,
        'xtick.major.width': 2.0,
        'ytick.major.width': 2.0,
        'xtick.minor.size': 4,
        'ytick.minor.size': 4,
        'lines.linewidth': 3.0,
        'legend.fontsize': 20,
        'savefig.facecolor': 'white'
    })
    
    # Create figure with reference-style size
    plt.figure(figsize=(16, 8))
    
    # Sort data by density in ascending order
    df_sorted = df.sort_values('Density', ascending=True)
    
    # Use single color since we only have equilibrated density
    bar_color = "#703C98"  # Green color from reference
    
    # Create barplot with sorted data (no hue since we only have one condition)
    ax = sns.barplot(
        data=df_sorted, 
        x="Dopant", 
        y="Density", 
        color=bar_color,
        alpha=0.8,
        edgecolor='white',
        linewidth=1
    )
    
    # Customize plot to match reference style
    plt.xlabel("Dopant Element", fontweight='bold')
    plt.ylabel("Equilibrated Density (g/cm³)", fontweight='bold')
    
    plt.xticks(fontweight='bold')
    
    # Add grid for better readability (reference style)
    ax.grid(True, alpha=0.3)

    # Remove top and right spines for cleaner look
    sns.despine()
    
    # Tight layout to prevent label cutoff
    plt.tight_layout()
    
    # Save plot with high resolution
    plt.savefig(OUTPUT_FILE, dpi=600, bbox_inches='tight', facecolor='white')
    print(f"\nPlot saved as: {OUTPUT_FILE}")
    
    # Show plot
    plt.show()
    
    return ax

def main():
    """Main function."""
    print("Extracting density data from dopant subfolders...")
    
    # Extract data
    df = extract_densities()
    
    if df.empty:
        print("No data found!")
        return
    
    print(f"\nFound data for {len(df['Dopant'].unique())} dopants")
    
    # Create plot
    create_barplot(df)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    summary = df['Density'].agg(['mean', 'std', 'min', 'max'])
    print(summary)

if __name__ == "__main__":
    main()