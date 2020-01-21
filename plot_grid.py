import seaborn as sns
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

sns.set(style="darkgrid")
train_palette = sns.color_palette("Blues")
test_palette = sns.color_palette("Reds")

datas = []
for f in tqdm(glob("/home/jroy1/procgen_training/*/progress.csv")):
    if os.stat(f).st_size == 0:
        continue
    data = pd.read_csv(f)[["eprewmean", "eval_eprewmean", "misc/total_timesteps"]]
    data["misc/total_timesteps"] /= 1e6
    data = data.melt("misc/total_timesteps")
    sns.lineplot(x="misc/total_timesteps", y="value", hue="variable", data=data, legend=False)
    # sns.lineplot(x="misc/total_timesteps", y="eval_eprewmean", data=data, palette=test_palette)
plt.show()
