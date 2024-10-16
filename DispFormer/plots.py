import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.interpolate import interp1d

def plot_vs(depth, vs, linecolor="k", linestyle='-', figsize=(4,6), label=""):
    """
    Plots shear wave velocity (Vs) against depth using a step plot.

    Parameters:
    depth (array-like): Depth values for the y-axis.
    vs (array-like): Shear wave velocity (Vs) values for the x-axis.
    linecolor (str): Color of the line (default is black, 'k').
    linestyle (str): Style of the line (default is solid line, '-').
    figsize (tuple): Size of the figure (default is (4,6)).
    label (str): Label for the plot; if provided, a legend will be displayed (default is empty string).

    Returns:
    None: Displays the plot.

    Notes:
    - The depth axis is inverted to show depth increasing downward.
    - If the label is provided, a legend appears in the upper right corner.
    """
    plt.figure(figsize=figsize)
    if label == "":
        plt.step(vs, depth, where='post', c=linecolor, linestyle=linestyle)
    else:
        plt.step(vs, depth, where='post', c=linecolor, linestyle=linestyle, label=label)
        plt.legend(loc='upper right')
    plt.gca().invert_yaxis()
    plt.tick_params(labelsize=15)
    plt.xlabel("Velocity (km/s)", fontsize=15)
    plt.ylabel("Depth (km)", fontsize=15)
    plt.show()

def plot_disp(period=[], phase_vel=None, group_vel=None, scatter=False):
    """
    Plots dispersion curves (phase and/or group velocity) against period.

    Parameters:
    period (array-like): Period values for the x-axis.
    phase_vel (array-like, optional): Phase velocity values for the y-axis (default is None).
    group_vel (array-like, optional): Group velocity values for the y-axis (default is None).
    scatter (bool): If True, scatter plot is used; otherwise, line plot is used (default is False).

    Returns:
    None: Displays the plot.

    Notes:
    - If both phase and group velocities are provided, they will be plotted in different colors.
    - If scatter is set to True, the points are plotted as scatter plots.
    """
    plt.figure()
    if phase_vel is not None:
        if scatter:
            plt.scatter(period, phase_vel, c='r', label="phase velocity")
        else:
            plt.plot(period, phase_vel, c='r', label="phase velocity")
    if group_vel is not None:
        if scatter:
            plt.scatter(period, group_vel, c='b', label="group velocity")
        else:
            plt.plot(period, group_vel, c='b', label="group velocity")
    plt.legend()
    plt.tick_params(labelsize=15)
    plt.xlabel("Period (s)", fontsize=15)
    plt.ylabel("Velocity (km/s)", fontsize=15)
    plt.show()

##################################################################################
#                               Loading the colormap
##################################################################################
# Parse .cpt file and generate a color dictionary
def load_cpt(file_path, num_colors=None, reverse=False):
    """
    Parses a .cpt (Color Palette Table) file and generates a colormap.

    Parameters:
    file_path (str): Path to the .cpt file.
    num_colors (int, optional): Number of colors for interpolation. If None, discrete colors are used without interpolation.
    reverse (bool, optional): If True, reverse the colormap. Default is False.

    Returns:
    ListedColormap or LinearSegmentedColormap: The colormap generated from the .cpt file.

    Notes:
    - If num_colors is None, the function returns a ListedColormap based on the exact colors from the file.
    - If num_colors is specified, the colormap is interpolated to generate the requested number of colors.
    """
    positions = []
    colors = []
    with open(file_path) as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.split()
            if len(parts) == 8:  # Regular color definition line
                zlow, r1, g1, b1, zhigh, r2, g2, b2 = map(float, parts)
                positions.append(zlow)
                colors.append([r1 / 255, g1 / 255, b1 / 255])
                positions.append(zhigh)
                colors.append([r2 / 255, g2 / 255, b2 / 255])

    positions = np.array(positions)
    colors = np.array(colors)

    # If num_colors is None, return a discrete ListedColormap without interpolation
    if num_colors is None:
        # Use only the distinct colors in the file
        unique_positions = np.unique(positions)
        num_colors = len(unique_positions) // 2  # Each segment defines two positions
        
        # Return a ListedColormap with the exact colors from the .cpt file
        return ListedColormap(colors)

    # If num_colors is specified, interpolate the colormap
    red_interp = interp1d(positions, colors[:, 0], kind='linear')
    green_interp = interp1d(positions, colors[:, 1], kind='linear')
    blue_interp = interp1d(positions, colors[:, 2], kind='linear')

    # Create the range of values for interpolation
    z_vals = np.linspace(min(positions), max(positions), num_colors)
    red_vals = red_interp(z_vals)
    green_vals = green_interp(z_vals)
    blue_vals = blue_interp(z_vals)

    # Build color dictionary for LinearSegmentedColormap
    cdict = {'red': [], 'green': [], 'blue': []}
    for z, r, g, b in zip(z_vals, red_vals, green_vals, blue_vals):
        norm_z = (z - z_vals.min()) / (z_vals.max() - z_vals.min())  # Normalize z values
        cdict['red'].append((norm_z, r, r))
        cdict['green'].append((norm_z, g, g))
        cdict['blue'].append((norm_z, b, b))

    cmap = LinearSegmentedColormap('custom_cpt', cdict)
    # Return a LinearSegmentedColormap
    if reverse:
        cmap = cmap.reversed()  # Reverse the colormap

    return cmap
