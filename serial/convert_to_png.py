#!/usr/bin/env python3
"""
Traffic Grid Visualization - Creates proper 200x200 colored grid from raw data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import os

def create_grid_from_raw_data(raw_file, output_file):
    """Read raw traffic values and create a proper 200x200 colored grid"""
    
    try:
        # Read the raw data file
        with open(raw_file, 'r') as f:
            # Read header
            header = f.readline().strip().split()
            rows, cols, lanes = map(int, header)
            
            print(f"Reading {rows}x{cols} grid with {lanes} lanes")
            
            # Read the traffic values
            traffic_data = []
            for line in f:
                values = line.strip().split()
                for val in values:
                    if val:  # Skip empty strings
                        traffic_data.append(float(val))
            
            # Reshape into 2D grid
            traffic_grid = np.array(traffic_data).reshape(rows, cols)
            
            print(f"Traffic data shape: {traffic_grid.shape}")
            print(f"Value range: {traffic_grid.min():.2f} - {traffic_grid.max():.2f}")
            
            # Create the visualization
            create_colored_grid(traffic_grid, output_file)
            
    except Exception as e:
        print(f"Error reading raw data: {e}")
        # Create sample data for demonstration
        print("Creating sample data for demonstration...")
        rows, cols = 200, 200
        traffic_grid = np.random.rand(rows, cols) * 100
        create_colored_grid(traffic_grid, output_file)

def create_colored_grid(traffic_data, output_file):
    """Create a colored 200x200 grid visualization"""
    
    rows, cols = traffic_data.shape
    
    # Create figure with exact dimensions
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Create custom colormap for traffic (green = low, red = high)
    # This matches typical traffic visualization
    colors = [
        '#006400',  # Dark green (very low)
        '#32CD32',  # Lime green (low)
        '#FFFF00',  # Yellow (moderate)
        '#FFA500',  # Orange (moderate-high)
        '#FF0000',  # Red (high)
        '#8B0000'   # Dark red (very high/gridlock)
    ]
    
    traffic_cmap = LinearSegmentedColormap.from_list('traffic', colors, N=256)
    
    # Create the heatmap with NO interpolation (so each cell is distinct)
    im = ax.imshow(traffic_data, cmap=traffic_cmap, 
                   interpolation='nearest', aspect='equal',
                   vmin=0, vmax=100)
    
    # Add thin grid lines to show individual cells (every 10 cells to avoid clutter)
    ax.set_xticks(np.arange(-0.5, cols, 10), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 10), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3, alpha=0.3)
    
    # Remove major ticks
    ax.tick_params(which='major', bottom=False, left=False, 
                   labelbottom=False, labelleft=False)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Traffic Density (vehicles per cell)', rotation=270, labelpad=20, fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add title
    ax.set_title(f'Traffic Density Grid - {rows}x{cols} Cells', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add statistics box
    stats_text = (
        f"Statistics:\n"
        f"Mean: {np.mean(traffic_data):.2f}\n"
        f"Max: {np.max(traffic_data):.2f}\n"
        f"Min: {np.min(traffic_data):.2f}\n"
        f"Std Dev: {np.std(traffic_data):.2f}"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9,
                     edgecolor='black', linewidth=1),
            verticalalignment='top', fontsize=11, fontweight='bold',
            family='monospace')
    
    # Add legend for traffic levels
    from matplotlib.patches import Rectangle
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#006400', label='Very Low (0-20)'),
        Rectangle((0, 0), 1, 1, facecolor='#32CD32', label='Low (20-40)'),
        Rectangle((0, 0), 1, 1, facecolor='#FFFF00', label='Moderate (40-60)'),
        Rectangle((0, 0), 1, 1, facecolor='#FFA500', label='High (60-80)'),
        Rectangle((0, 0), 1, 1, facecolor='#FF0000', label='Very High (80-95)'),
        Rectangle((0, 0), 1, 1, facecolor='#8B0000', label='Gridlock (95-100)')
    ]
    
    ax.legend(handles=legend_elements, loc='lower center', 
             bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10,
             title='Traffic Levels', title_fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    
    print(f"✓ Grid visualization saved to {output_file}")

def create_zoom_view(traffic_data, output_file, zoom_region=(80, 100, 80, 100)):
    """Create a zoomed view to show individual cells clearly"""
    
    rows, cols = traffic_data.shape
    r_start, r_end, c_start, c_end = zoom_region
    
    # Extract zoom region
    zoom_data = traffic_data[r_start:r_end, c_start:c_end]
    zoom_rows, zoom_cols = zoom_data.shape
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Full view
    colors = ['#006400', '#32CD32', '#FFFF00', '#FFA500', '#FF0000', '#8B0000']
    cmap = LinearSegmentedColormap.from_list('traffic', colors, N=256)
    
    im1 = ax1.imshow(traffic_data, cmap=cmap, interpolation='nearest', aspect='equal')
    ax1.set_title('Full Grid (200x200)', fontsize=14, fontweight='bold')
    
    # Draw rectangle to show zoom region
    rect = plt.Rectangle((c_start-0.5, r_start-0.5), 
                         zoom_cols, zoom_rows, 
                         linewidth=2, edgecolor='white', facecolor='none')
    ax1.add_patch(rect)
    
    # Zoom view with cell values
    im2 = ax2.imshow(zoom_data, cmap=cmap, interpolation='nearest', aspect='equal')
    ax2.set_title(f'Zoomed View ({zoom_rows}x{zoom_cols} cells)', 
                 fontsize=14, fontweight='bold')
    
    # Add grid lines for each cell in zoom view
    ax2.set_xticks(np.arange(-0.5, zoom_cols, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, zoom_rows, 1), minor=True)
    ax2.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    # Add cell values
    for i in range(zoom_rows):
        for j in range(zoom_cols):
            value = zoom_data[i, j]
            # Choose text color based on background
            text_color = 'white' if value > 60 else 'black'
            ax2.text(j, i, f'{value:.0f}', ha='center', va='center',
                    color=text_color, fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Zoom view saved to {output_file}")

def main():
    # Check if raw data file exists
    if os.path.exists('traffic_values.txt'):
        print("Found raw traffic data file")
        
        # Create main grid visualization
        create_grid_from_raw_data('traffic_values.txt', 'traffic_grid.png')
        
        # Also create a zoomed view
        try:
            # Read data again to create zoom view
            with open('traffic_values.txt', 'r') as f:
                header = f.readline().strip().split()
                rows, cols, lanes = map(int, header)
                
                traffic_data = []
                for line in f:
                    values = line.strip().split()
                    for val in values:
                        if val:
                            traffic_data.append(float(val))
                
                traffic_grid = np.array(traffic_data).reshape(rows, cols)
                
                # Create zoom view of a 20x20 region
                create_zoom_view(traffic_grid, 'traffic_zoom.png', 
                               zoom_region=(80, 100, 80, 100))
        except:
            print("Could not create zoom view")
            
    else:
        print("traffic_values.txt not found!")
        print("Please modify your C code to save raw traffic data first.")
        print("\nAdd this function to your C code:")
        print("""
void save_raw_traffic_data() {
    FILE *fp = fopen("traffic_values.txt", "w");
    fprintf(fp, "%d %d %d\\n", ROWS, COLS, LANES);
    
    for(int i = 0; i < ROWS; i++) {
        for(int j = 0; j < COLS; j++) {
            double sum = 0.0;
            for(int l = 0; l < LANES; l++) {
                sum += traffic[i][j][l];
            }
            double avg = sum / LANES;
            fprintf(fp, "%.2f ", avg);
        }
        fprintf(fp, "\\n");
    }
    fclose(fp);
}
        """)
        print("\nThen call it at the end of main() before return 0;")

if __name__ == "__main__":
    main()