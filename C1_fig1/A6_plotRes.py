import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
import glob
import matplotlib.cm as cm
import os
import matplotlib

import matplotlib.ticker as ticker

# Function to generate a consistent color map based on unique labels
def generate_label_colormap(files):
    labels = set()
    for file in files:
        filename = os.path.basename(file)
        parts = filename.replace('.npy', '').split('_')

        # Determine RA/NRA/NA based on the first part
        if parts[2] == 'True':
            ra_nra = 'User-specified error'
        elif parts[2] == 'False':
            ra_nra = 'User-specified rank'
        else:
            ra_nra = 'BAD'

        # Determine PS/NPS/NA based on the second part
        if '-' in parts[3]:
            ps_nps = 'without priority sampling' if parts[3] == '1-0' else 'with priority sampling'
        else:
            ps_nps = 'BAD'

        # Construct the label
        label = f"{ra_nra}, {ps_nps}"
        labels.add(label)

    # Create a color map
    colormap = matplotlib.colormaps.get_cmap('tab10')
    label_to_color = {label: colormap(idx) for idx, label in enumerate(labels)}

    return label_to_color

# Define a function to plot the data with consistent colors and labels
def plot_error_vs_time(decayType, ax, label_to_color):
    # Get the list of .npy files
    files = glob.glob(f"tempErrorVsTime_{decayType}_*.npy")

    # Loop through each file and plot the data
    for file in files:
        # Load the data
        data = np.load(file)

        # Extract the filename components
        filename = os.path.basename(file)
        parts = filename.replace('.npy', '').split('_')

        # Determine RA/NRA/NA based on the first part
        if parts[2] == 'True':
            ra_nra = 'User-specified error'
        elif parts[2] == 'False':
            ra_nra = 'User-specified rank'
        else:
            ra_nra = 'BAD'

        # Determine PS/NPS/NA based on the second part
        if '-' in parts[3]:
            ps_nps = 'without priority sampling' if parts[3] == '1-0' else 'with priority sampling'
        else:
            ps_nps = 'BAD'

        # Construct the label
        label = f"{ra_nra}, {ps_nps}"

        # Extract the time and error values
        time_values = data[:, 0]
        time_order = np.argsort(time_values)
        error_values = data[:, 1][time_order]
        time_values = time_values[time_order]

        # Plot the data with a line and scatter points
        color = label_to_color[label]
        ax.semilogy(time_values, error_values, 'o-', label=label, color=color, alpha=0.75)


    # Add labels and grid
    if decayType=="mid":
        xlab = "exponential"
    elif decayType=="bot":
        xlab = "super-exponential"
    elif decayType=="top":
        xlab = "sub-exponential"
    else:
        xlab = "BAD"
    ax.set_xlabel(f'Runtime for {xlab} decaying singular values (seconds)', fontsize=22)
    ax.set_ylabel('(Log) Reconstruction error', fontsize=22)
    ax.grid(True)

# Get all .npy files to extract labels
all_files = glob.glob("tempErrorVsTime_*.npy")
label_to_color = generate_label_colormap(all_files)

# Create a figure and subplots
fig, axes = plt.subplots(2, 2, figsize=(24, 15))

# Plot the three decay types on the first three subplots
plot_error_vs_time("top", axes[0, 1], label_to_color)
plot_error_vs_time("mid", axes[1, 0], label_to_color)
plot_error_vs_time("bot", axes[1, 1], label_to_color)

######################################################
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

import numpy as np
import os

def compute_or_load_svd(data_path, u_path, s_path, vt_path):
    if os.path.exists(u_path) and os.path.exists(s_path) and os.path.exists(vt_path):
        u = np.load(u_path)
        s = np.load(s_path)
        vt = np.load(vt_path)
        print(f"Loaded SVD from {u_path}, {s_path}, and {vt_path}")
    else:
        data = np.load(data_path)
        u, s, vt = np.linalg.svd(data)
        np.save(u_path, u)
        np.save(s_path, s)
        np.save(vt_path, vt)
        print(f"Computed and saved SVD to {u_path}, {s_path}, and {vt_path}")
    return u, s, vt

# Paths to your data and SVD files
data_bot_path = '/sdf/home/w/winnicki/papertests_20240717/expDecayingSingularValues_bot.npy'
u_bot_path = '/sdf/home/w/winnicki/papertests_20240717/svd_bot_u.npy'
s_bot_path = '/sdf/home/w/winnicki/papertests_20240717/svd_bot_s.npy'
vt_bot_path = '/sdf/home/w/winnicki/papertests_20240717/svd_bot_vt.npy'

data_mid_path = '/sdf/home/w/winnicki/papertests_20240717/expDecayingSingularValues_mid.npy'
u_mid_path = '/sdf/home/w/winnicki/papertests_20240717/svd_mid_u.npy'
s_mid_path = '/sdf/home/w/winnicki/papertests_20240717/svd_mid_s.npy'
vt_mid_path = '/sdf/home/w/winnicki/papertests_20240717/svd_mid_vt.npy'

data_top_path = '/sdf/home/w/winnicki/papertests_20240717/expDecayingSingularValues_top.npy'
u_top_path = '/sdf/home/w/winnicki/papertests_20240717/svd_top_u.npy'
s_top_path = '/sdf/home/w/winnicki/papertests_20240717/svd_top_s.npy'
vt_top_path = '/sdf/home/w/winnicki/papertests_20240717/svd_top_vt.npy'

# Compute or load SVD
u_bot, s_bot, vt_bot = compute_or_load_svd(data_bot_path, u_bot_path, s_bot_path, vt_bot_path)
u_mid, s_mid, vt_mid = compute_or_load_svd(data_mid_path, u_mid_path, s_mid_path, vt_mid_path)
u_top, s_top, vt_top = compute_or_load_svd(data_top_path, u_top_path, s_top_path, vt_top_path)

print("Finished processing all!")

# plt.semilogy(np.arange(len(s)), s, 'k-', marker='o', markersize=3, linewidth=0.5, markevery=0.009)
ylabel1 = r'$2^{ -\left (\frac{j}{150} \right ) ^2}$'
ylabel2 = r'$2^{ \frac{-j}{22.5}}$'
ylabel3 = r'$2^{\left(\frac{1000}{150}  - \frac{j}{150}\right)^2 - \left(\frac{1000}{150}\right)^2}$'
axes[0, 0].semilogy(np.arange(len(s_top)), s_top, 'r-', marker='o', markersize=3, linewidth=0.5, markevery=0.01, label=ylabel1)
axes[0, 0].semilogy(np.arange(len(s_mid)), s_mid, 'b-', marker='o', markersize=3, linewidth=0.5, markevery=0.01, label=ylabel2)
axes[0, 0].semilogy(np.arange(len(s_bot)), s_bot, 'k-', marker='o', markersize=3, linewidth=0.5, markevery=0.01, label=ylabel3)

# Add labels and title with increased fontsize
axes[0, 0].set_xlabel('Index', fontsize=20)
axes[0, 0].set_ylabel('(Log) Singular value', fontsize=20)
axes[0, 0].legend(loc="upper right")

# Set tick parameters
axes[0, 0].tick_params(axis='both', which='major', labelsize=20)
######################################################

formatter = ticker.FuncFormatter(lambda val, pos: '{:.1f}'.format(val))

# Combine legends from the first three subplots
handles, labels = [], []
for ax in axes.flatten()[1:]:  # Exclude the scatter plot
    for handle, label in zip(*ax.get_legend_handles_labels()):
        if label not in labels:
            handles.append(handle)
            labels.append(label)
    # ax.yaxis.set_major_formatter(formatter)
    # ax.yaxis.set_minor_formatter(formatter)
    # ax.yaxis.set_minor_locator(ticker.MaxNLocator(nbins=5))

# Set up the legend in a 2x2 format
fig.legend(handles, labels, loc='lower center', fontsize='medium', ncol=2)
# fig.legend(handles, labels, loc='lower center', fontsize='medium', ncol=2, bbox_to_anchor=(0.5, -0.01))
# Adjust layout and save the figure

plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjusted to make more space for the legend
plt.savefig('combined_errorVsTimePlot.png', bbox_inches='tight', dpi=300)
