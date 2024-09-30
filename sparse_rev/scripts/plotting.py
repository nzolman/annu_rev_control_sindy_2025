import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Palatino']

colors = plt.cm.tab10.colors
colors = [colors[2], colors[4], colors[3], colors[1],  colors[0] , colors[5]]
cmap = sns.color_palette(colors, as_cmap=True)

from sparse_rev import _parent_dir

dt = 0.02
USE_LEGEND = False

# read csvs
df_t = pd.read_csv(os.path.join(_parent_dir, 'data', 'tmax', 'all_tmax.csv'))
df_noise = pd.read_csv(os.path.join(_parent_dir, 'data', 'noise', 'all_noise.csv'))


df_t['n_coll'] =  df_t['tmax']/dt 

df_noise['sigma_perc'] = df_noise['sigma'] * 100


fig = plt.figure(figsize=(5,5))
grid = sns.lineplot(df_t, x = 'n_coll', y = 'err', hue = 'model_name', 
            palette=cmap,
             legend=USE_LEGEND, 
             estimator='median',
             linewidth=3,
             marker='o',
             markersize=10,
             markeredgecolor=None,
             )
plt.yscale('log')
plt.xscale('log')
plt.xlim(10,None)
plt.xlabel(None)
plt.ylabel(None)
plt.tick_params('both', labelsize=15)
plt.ylim(0.3e-7, 10**(-1.5))

if USE_LEGEND:
    plt.legend(bbox_to_anchor = [1.0, 1.0])
    
plt.savefig(os.path.join(_parent_dir, 'data', 'tmax_errs.png'))
plt.close()

fig = plt.figure(figsize=(5,5))
sns.lineplot(df_noise, x = 'sigma_perc', y = 'err', hue = 'model_name',
             palette=cmap,
             legend=USE_LEGEND, 
             estimator="median",
             linewidth=3,
             marker='o',
             markersize=10,
             markeredgecolor=None,
             )

plt.yscale('log')
plt.xlabel(None)
plt.ylabel(None)
plt.tick_params('both', labelsize=15)
plt.xticks([0,5,10,15,20],
           ['0%','5%','10%','15%','20%'])
plt.ylim(0.3e-7, 10**(-1.5))

if USE_LEGEND:
    plt.legend(bbox_to_anchor = [1.0, 1.0])

plt.savefig(os.path.join(_parent_dir, 'data', 'noise_errs.png'))
plt.close()