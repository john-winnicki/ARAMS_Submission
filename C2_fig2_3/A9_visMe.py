import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FixedLocator

# Read data from CSV files
runtimes_df = pd.read_csv('runtimes.csv')
errors_df = pd.read_csv('errors.csv')

# Function to set full number format and major ticks
def set_full_number_format_and_ticks(ax, x_ticks):
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    ax.ticklabel_format(style='plain', axis='both', useOffset=False)

# Define x-axis tick positions as powers of 2 up to 128
x_ticks = [2**i for i in range(8)]  # 1, 2, 4, 8, 16, 32, 64, 128

# Plot for runtime
plt.figure(figsize=(10, 6))
plt.loglog(runtimes_df['Cores'], runtimes_df['Serial'], 'o-', label='Serial merge')
plt.loglog(runtimes_df['Cores'], runtimes_df['Parallel'], 's-', label='Tree merge')
plt.xlabel('(Log) Number of cores', fontsize=20)
plt.ylabel('(Log) Runtime', fontsize=20)
plt.legend(fontsize=20)
plt.grid(True, which="both", ls="--")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax = plt.gca()
set_full_number_format_and_ticks(ax, x_ticks)
plt.tight_layout()
plt.savefig('ParallelEfficiency.png', dpi=300)

# Plot for error
plt.figure(figsize=(10, 6))
plt.loglog(errors_df['Cores'], errors_df['Serial'], 'o-', label='Serial merge')
plt.loglog(errors_df['Cores'], errors_df['Parallel'], 's-', label='Tree merge')
plt.xlabel('(Log) Number of cores', fontsize=20)
plt.ylabel('(Log) Reconstruction error', fontsize=20)
plt.legend(fontsize=20)
plt.grid(True, which="both", ls="--")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
ax = plt.gca()
set_full_number_format_and_ticks(ax, x_ticks)
plt.tight_layout()
plt.savefig('ParallelError.png', dpi=300)

