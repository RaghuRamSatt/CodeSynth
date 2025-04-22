
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
_original_figure=plt.figure
_figs=[]
def _cf(*a,**k):
    fig=_original_figure(*a,**k)
    _figs.append(fig)
    return fig
plt.figure=_cf
_orig_sub=plt.subplots
def _cs(*a,**k):
    fig,ax=_orig_sub(*a,**k)
    _figs.append(fig)
    return fig,ax
plt.subplots=_cs

import pandas as pd

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('/sandbox/data.csv')

# Print the column names of the DataFrame
print(df.columns)

# save figures to disk
import os
os.makedirs('/sandbox/figs', exist_ok=True)
for i,fig in enumerate(_figs): fig.savefig(f'/sandbox/figs/fig{i}.png', dpi=100)
