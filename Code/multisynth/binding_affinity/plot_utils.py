# standard imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# import custom modules
import sys
sys.path.append('../../..')
import project_config

def make_scatter(data, x, y, xlabel, ylabel, title = None, color = '#3498db'):

    # create title
    if title is None:
        title = xlabel + ' vs. ' + ylabel

    # set figure dimensions
    plt.figure(figsize=(10, 6))

    # plot data points
    plt.scatter(data[x], data[y], s=20, color=color)

    # set title and axis labels
    plt.title(title)
    plt.xlabel(xlabel, fontweight='bold', size=12)
    plt.ylabel(ylabel, fontweight='bold', size=12)

    # plot line of best fit
    m, b = np.polyfit(data[x], data[y], 1)
    plt.plot(data[x], m*data[x] + b, color='red', linewidth=2, linestyle='--', alpha=0.7)

    # add a gray dashed grid in the background
    plt.grid(axis = "both", color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().set_axisbelow(True)

    # save plot
    plt.savefig(project_config.RESULTS_DIR / str(xlabel + '_vs_' + ylabel + '.png'), dpi=600, bbox_inches='tight')
    plt.savefig(project_config.RESULTS_DIR / str(xlabel + '_vs_' + ylabel + '.pdf'), dpi=600, bbox_inches='tight')
    
    # return plot
    return plt