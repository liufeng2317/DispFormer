import numpy as np
import matplotlib.pyplot as plt

def plot_vs(depth,vs,linecolor="k",linestyle='-',figsize=(4,6),label=""):
    plt.figure(figsize=figsize)
    if label == "":
        plt.step(vs,depth,where='post',c=linecolor,linestyle=linestyle)
    else:
        plt.step(vs,depth,where='post',c=linecolor,linestyle=linestyle,label=label)
        plt.legend(loc='upper right')
    plt.gca().invert_yaxis()
    plt.tick_params(labelsize=15)
    plt.xlabel("Velocity (km/s)",fontsize=15)
    plt.ylabel("Depth (km)",fontsize=15)
    plt.show()
    
def plot_disp(period=[],phase_vel=None,group_vel=None,scatter=False):
    plt.figure()
    if phase_vel is not None:
        if scatter:
            plt.scatter(period,phase_vel,c='r',label="phase velocity")
        else:
            plt.plot(period,phase_vel,c='r',label="phase velocity")
    if group_vel is not None:
        if scatter:
            plt.scatter(period,group_vel,c='b',label="group velocity")
        else:
            plt.plot(period,group_vel,c='b',label="group velocity")
    plt.legend()
    plt.tick_params(labelsize=15)
    plt.xlabel("Period (s)",fontsize=15)
    plt.ylabel("Velocity (km/s)",fontsize=15)
    plt.show()

##################################################################################
#                               Loading the colormap
##################################################################################
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.interpolate import interp1d

# Parse .cpt file and generate a color dictionary
def load_cpt(file_path, num_colors=None, reverse=False):
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
