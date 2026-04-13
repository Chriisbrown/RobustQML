import os
from argparse import ArgumentParser
from model.common import fromFolder
from data.EOSdataset import DataSet
from model.gpu_utils import setup_gpu_memory_growth

from plot import style

from basic import error_residual, plot_histo, rates,efficiency, clusters, plot_2d
import matplotlib
from sklearn.metrics import roc_curve, auc
from matplotlib.pyplot import cm
import itertools
from sklearn.decomposition import PCA

import itertools

import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

style.set_style()

def plot_latent(embedding, labels, label_style, plot_dir):
    
    embedding_dim = embedding.shape[1]
    embedding_dim_list = np.arange(embedding_dim)
    combinations = list(itertools.combinations(embedding_dim_list, 2))
    
    # Subsample to at most 1000 points
    n_samples = min(10000, len(embedding))
    sample_idx = np.random.choice(len(embedding), size=n_samples, replace=False)
    embedding_sample = embedding[sample_idx]
    labels_sample = labels[sample_idx]


    n = len(combinations) 
    titles =[str(combination[0]) + ' vs ' + str(combination[1]) for combination in combinations]
    # Determine grid dimensions
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    
    colormap = cm.get_cmap('Set1', len(label_style))
    figsize=(8 * ncols, 6 * nrows)
    fig, axes_grid = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        layout="constrained",
        squeeze=False,
    )
 
    axes = np.empty((nrows, ncols), dtype=object)

    for i in range(n):
        row, col = divmod(i, ncols)
        ax = axes_grid[row, col]
        axes[row, col] = ax

        scatter = ax.scatter(embedding_sample[:, combinations[i][0]], embedding_sample[:, combinations[i][1]], c=labels_sample, cmap= colormap, alpha=0.6, vmin=0, vmax=len(label_style))
        ax.set_title(titles[i], fontsize=20, pad=6)
    
    # Hide unused axes
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        fig.add_subplot(axes_grid[row, col]).set_visible(False)
 
    fig.draw_without_rendering()
    # Top of the top-left subplot, bottom of the bottom-right subplot
    # (in figure-fraction coordinates).
    top    = axes_grid[0, 0].get_position().y1
    bottom = axes_grid[nrows - 1, ncols - 1].get_position().y0
    right  = axes_grid[nrows - 1, ncols - 1].get_position().x1
 
    cbar_pad   = 0.01   # gap between grid and colorbar (figure fraction)
    cbar_width = 0.02

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=matplotlib.colors.Normalize(vmin=0, vmax=len(label_style)))
    sm.set_array([])
    cbar_ax = fig.add_axes([right + cbar_pad, bottom, cbar_width, top - bottom])
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=range(len(label_style)))
    cbar.set_ticklabels(label_style)
    cbar.ax.tick_params(labelsize=9)
    plt.savefig(plot_dir+'/latent_dims.png', dpi=100, bbox_inches="tight") 
    plt.close(fig)
    
    
def plot_latent_vs_variable(variable,embedding, labels, label_style, variable_name,plot_dir):
    
    embedding_dim = embedding.shape[1]
    embedding_dim_list = np.arange(embedding_dim)
    
    n = len(embedding_dim_list) 
    titles =[variable_name + ' vs latent ' + str(dim) for dim in embedding_dim_list]
    # Determine grid dimensions
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    
    colormap = cm.get_cmap('Set1', len(label_style))
    figsize=(8 * ncols, 6 * nrows)
    fig, axes_grid = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        layout="constrained",
        squeeze=False,
    )
 
    axes = np.empty((nrows, ncols), dtype=object)

    for i in range(n):
        row, col = divmod(i, ncols)
        ax = axes_grid[row, col]
        axes[row, col] = ax

        scatter = ax.scatter(variable, embedding[:,i], c=labels, cmap= colormap, alpha=0.6, vmin=0, vmax=len(label_style))
        ax.set_title(titles[i], fontsize=20, pad=6)
    
    # Hide unused axes
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        fig.add_subplot(axes_grid[row, col]).set_visible(False)
 
    fig.draw_without_rendering()
    # Top of the top-left subplot, bottom of the bottom-right subplot
    # (in figure-fraction coordinates).
    top    = axes_grid[0, 0].get_position().y1
    bottom = axes_grid[nrows - 1, ncols - 1].get_position().y0
    right  = axes_grid[nrows - 1, ncols - 1].get_position().x1
 
    cbar_pad   = 0.01   # gap between grid and colorbar (figure fraction)
    cbar_width = 0.02

    sm = plt.cm.ScalarMappable(cmap=colormap, norm=matplotlib.colors.Normalize(vmin=0, vmax=len(label_style)))
    sm.set_array([])
    cbar_ax = fig.add_axes([right + cbar_pad, bottom, cbar_width, top - bottom])
    cbar = fig.colorbar(sm, cax=cbar_ax, ticks=range(len(label_style)))
    cbar.set_ticklabels(label_style)
    cbar.ax.tick_params(labelsize=9)
    plt.savefig(plot_dir+'/latent_vs_'+variable_name+'.png', dpi=100, bbox_inches="tight") 
    plt.close(fig)
        
def plot_PCA(principle_components, labels, global_range,printing_labels,plot_dir):
    
    os.makedirs(plot_dir, exist_ok=True)
    
    colormap = cm.get_cmap('Set1', len(printing_labels))
    fig, ax = plt.subplots(1, 1, figsize=(style.FIGURE_SIZE[0]*1.2,style.FIGURE_SIZE[1]*1.2))
    scatter = ax.scatter(principle_components[:, 0], principle_components[:, 1], c=labels, cmap= colormap, alpha=0.6)
    cbar = plt.colorbar(scatter, ticks=range(len(printing_labels)))
    cbar.ax.set_yticklabels(printing_labels)
    
    ax.set_title("Principle Components of Pooling Layer embeddings",y=1.0, pad=84)
    
    plt.tight_layout()
    plt.savefig(plot_dir+'/PCA_scatter.png')
    
    n = len(printing_labels) 
    titles = list(printing_labels.keys())
    
    # Determine grid dimensions
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
        
    arrays = []
    for iclass in range(n):
        indices = np.squeeze(np.argwhere(labels==iclass))
        arrays.append((principle_components[indices,0], principle_components[indices,1]))

    ranges = [global_range] * len(arrays)
    
    all_counts = []
    for (x, y), rng in zip(arrays, ranges):
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if len(x) > 0:
            counts, _, _ = np.histogram2d(x, y, bins=50, range=rng)
            pos = counts[counts > 0]
            all_counts.append(pos)
 
    if all_counts:
        flat = np.concatenate(all_counts)
        vmin = float(flat.min())
        vmax = float(flat.max())
    else:
        vmin, vmax = 1, 10   # fallback for all-zero data
 
    norm = LogNorm(vmin=vmin, vmax=vmax)
    figsize=(10 * ncols, 8 * nrows)
     
    fig, ax = plt.subplots(1, 1, figsize=figsize)
 
    # constrained_layout automatically prevents tick labels / titles overlapping
    fig, axes_grid = plt.subplots(
        nrows, ncols,
        figsize=figsize,
        layout="constrained",
        squeeze=False,
    )
 
    axes = np.empty((nrows, ncols), dtype=object)
 
    for i, ((x, y), rng) in enumerate(zip(arrays, ranges)):
        row, col = divmod(i, ncols)
        ax = axes_grid[row, col]
        axes[row, col] = ax
        
        x = np.atleast_1d(x)
        y = np.atleast_1d(y)
        if len(x) > 0:
            ax.hist2d(x, y, range=rng, bins=50, norm=norm, cmap='jet')
 
        ax.set_title(printing_labels[titles[i]], fontsize=20, pad=6)
        ax.set_xlabel('PCA dim #1', fontsize=15)
        ax.set_ylabel('PCA dim #2', fontsize=15)
        ax.tick_params(labelsize=8)
 
    # Hide unused axes
    for i in range(n, nrows * ncols):
        row, col = divmod(i, ncols)
        fig.add_subplot(axes_grid[row, col]).set_visible(False)

 
    # ── 3. Shared colorbar in the reserved column ────────────────────────────
    fig.draw_without_rendering()
 
    # Top of the top-left subplot, bottom of the bottom-right subplot
    # (in figure-fraction coordinates).
    top    = axes_grid[0, 0].get_position().y1
    bottom = axes_grid[nrows - 1, ncols - 1].get_position().y0
    right  = axes_grid[nrows - 1, ncols - 1].get_position().x1
 
    cbar_pad   = 0.01   # gap between grid and colorbar (figure fraction)
    cbar_width = 0.02
 
    cbar_ax = fig.add_axes([right + cbar_pad, bottom, cbar_width, top - bottom])
    
    sm = plt.cm.ScalarMappable(cmap='jet', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('# Events', fontsize=20)
    cbar.ax.tick_params(labelsize=9)

    plt.savefig(plot_dir+'/PCA_2D.png', bbox_inches="tight")

if __name__ == "__main__":
    parser = ArgumentParser()
    # Training argument
    parser.add_argument(
        '-o', '--output', default='output/autoencoder', help='Output model directory path, also save evaluation plots'
    )
    
    parser.add_argument(
        '-e', '--events', default=-1, type=int,help='Number of the test set events to run over'
    )
    
    parser.add_argument(
        '-a', '--ad_dataset', action='store_true'
    )
    
    args = parser.parse_args()
    
    setup_gpu_memory_growth()
    
    model = fromFolder(args.output)

    plot_dir = os.path.join(model.output_directory, "plots/testing/latent")
    os.makedirs(plot_dir, exist_ok=True)
    
    if args.ad_dataset:
        event_labels = {"background" : 0, "ato4l" :1, "hChToTauNu" : 2, "hToTauTau" : 3, "leptoquark" : 4, "blackbox": 5}
        event_label_style = {"background" : 'SM background', "ato4l" :'A -> 4l', "hChToTauNu" : 'H+ -> Tau Nu', "hToTauTau" : 'h -> Tau Tau', "leptoquark" : 'leptoquark', "blackbox": 'blackbox'}
        path = '/eos/user/c/cebrown/RobustQML/AD_dataset/processed/'
        
    else:
        event_labels = {'minbias' :0,'QCD_HT50toInf' : 1,'HH_4b' : 2,'HH_bbgammagamma':3,'HH_bbtautau':4,'QCD_HT50tobb':5}
        event_label_style = {'minbias':'Minbias', 'QCD_HT50toInf': 'QCD_HT50toInf', 'HH_4b':'HH->bbbb','HH_bbgammagamma':'HH_bbgammagamma','HH_bbtautau':'HH_bbtautau','QCD_HT50tobb':'QCD_HT50tobb' }
        path = '/eos/user/c/cebrown/RobustQML/training_data/'
        
    latent_vector = []
    event_label_vector = []
    
    
    for datasets in event_labels.keys():
        data_test = DataSet.fromH5(path+datasets+'/test/')
        data_test.normalise()
        data_test_dataframe = data_test.data_frame.sample(n=1000)
        latent_representations = model.encoder_predict(data_test_dataframe,data_test.training_columns)
        latent_vector.append(latent_representations)
        event_label_vector.append(np.tile(event_labels[datasets],1000))
        
    latent_vector = np.concatenate(latent_vector, axis=0)
    event_label_vector = np.concatenate(event_label_vector, axis=0)
        
    plot_latent(latent_vector, event_label_vector, event_label_style, plot_dir)
    
    pca = PCA(n_components=2)
    event_principle_components = pca.fit_transform(latent_vector)

    event_xmax, event_xmin = np.percentile(event_principle_components[:,0],99), np.percentile(event_principle_components[:,0],1)
    event_ymax, event_ymin = np.percentile(event_principle_components[:,1],99), np.percentile(event_principle_components[:,1],1)
    
    plot_PCA(event_principle_components, event_label_vector,((event_xmin, event_xmax),(event_ymin, event_ymax)), event_label_style, plot_dir)