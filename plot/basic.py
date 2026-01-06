# flake8: noqa
# Plotting
import os
import matplotlib
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

from plot import style

style.set_style()

def plot_2d(variable_one, variable_two, range_one, range_two, name_one, name_two, title):
    fig, ax = plt.subplots(1, 1, figsize=(style.FIGURE_SIZE[0] + 2, style.FIGURE_SIZE[1]))

    hist2d = ax.hist2d(
        variable_one, variable_two, range=(range_one, range_two), bins=50, norm=matplotlib.colors.LogNorm(), cmap='jet'
    )
    ax.set_xlabel(name_one)
    ax.set_ylabel(name_two)
    cbar = plt.colorbar(hist2d[3], ax=ax)
    cbar.set_label('a.u.')
    plt.suptitle(title)
    return fig


def plot_histo(variable, name, title, xlabel, ylabel, log = 'log', x_range=(0, 1), bins = 50):
    plt.clf()
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    ## If we are histogramming by class and so want class colours
    if len(variable) > len(style.colours):
        colours = style.color_cycle
        linestyle = ['-' for i in range(len(variable))]
    else:
        colours = style.colours
        linestyle = style.LINESTYLES
    ax.hist(
            variable,
            bins=bins,
            range=x_range,
            histtype="step",
            stacked=False,
            color=[colours[i] for i in range(len(variable))],
            label=name,
            linewidth=style.LINEWIDTH - 1.5,
            linestyle=linestyle,
            density=True,
        )
    ax.grid(True)
    ax.set_yscale(log)
    ax.set_xlabel(xlabel, ha="right", x=1)
    ax.set_ylabel(ylabel, ha="right", y=1)
    ax.legend(loc='upper right')
    return fig

def error_residual(model_outputs,plot_dir):
    plot_histo(
                [model_outputs],
                ['outputs'],
                '',
                'MAE',
                'a.u',
                log='linear',
                x_range=(-1,1),
            )
    save_path = os.path.join(plot_dir, "error_residual")
    plt.savefig(f"{save_path}.png", bbox_inches='tight')
    plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    plt.close()    
    
def loss_history(plot_dir, loss_names, history):
    for metric in loss_names:
        #metric = metric + '_loss'

        fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
        ax.plot(history[metric], label='Train Loss', linewidth=style.LINEWIDTH)
        ax.plot(history['val_' + metric], label='Validation Loss', linewidth=style.LINEWIDTH)
        ax.grid(True)
        # ax.set_ylabel('Loss')
        ax.set_ylabel('Loss ' + metric)
        ax.set_xlabel('Epoch')
        ax.legend(loc='upper right')

        save_path = os.path.join(plot_dir, "loss_" + metric + "_history")
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')

        fig.clf()
        
def rates(model_name,sample_name,rates_outputs,plot_dir=None):

    hist, bin_edges = np.histogram(rates_outputs, range=(0,1), density=True, bins=100)
        
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    ax.plot(np.linspace(0,1,100),1 - (np.cumsum(hist))/100,label=model_name, linewidth=style.LINEWIDTH)
    ax.grid(True)
    ax.set_title(sample_name)
    ax.set_ylabel('Rate')
    ax.set_xlabel('Threshold')
    ax.legend(loc='upper right')
    
    if plot_dir != None:

        save_path = os.path.join(plot_dir, "rate")
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')

    return 1 - np.cumsum(hist)/100

def efficiency(model_name,sample_name,efficiencies_outputs,plot_dir=None):

    hist, bin_edges = np.histogram(efficiencies_outputs, range=(0,1), density=True, bins=100)
        
    fig, ax = plt.subplots(1, 1, figsize=style.FIGURE_SIZE)
    ax.plot(np.linspace(0,1,100),(np.cumsum(hist))/100, label=model_name, linewidth=style.LINEWIDTH)
    ax.set_title(sample_name)
    ax.grid(True)
    ax.set_ylabel('Efficiency')
    ax.set_xlabel('Threshold')
    ax.legend(loc='upper right')
    
    if plot_dir != None:

        save_path = os.path.join(plot_dir, sample_name+"_efficiency")
        plt.savefig(f"{save_path}.png", bbox_inches='tight')
        plt.savefig(f"{save_path}.pdf", bbox_inches='tight')
    
    return 1-np.cumsum(hist)/100
    
    
