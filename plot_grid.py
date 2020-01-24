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

AVG_LEN = 10

datas = []
# coeffs = [47.0]
# for f in tqdm(["/home/jroy1/procgen_training_3/procgen_jumper_easy_disc_coeff_" + str(x) + "/progress.csv" for x in coeffs]):
for f in tqdm(glob("/home/jroy1/procgen_training_firstlayer_long_bigfish/*/progress.csv")):
# for f in tqdm(glob("/home/jroy1/procgen_training_firstlayer_long/*/progress.csv")):
    if os.stat(f).st_size == 0:
        continue
    if True:
        data = pd.read_csv(f)[["eprewmean", "eval_eprewmean", "misc/total_timesteps"]]
        if AVG_LEN > 1:
            data["eprewmean"] = data["eprewmean"].rolling(AVG_LEN).mean()
            data["eval_eprewmean"] = data["eval_eprewmean"].rolling(AVG_LEN).mean()
    else:
        loss = True
        if loss:
            data = pd.read_csv(f)[["loss/discriminator_accuracy", "loss/discriminator_loss", "misc/total_timesteps"]]
        else:
            data = pd.read_csv(f)[["loss/discriminator_accuracy", "misc/total_timesteps"]]
        if AVG_LEN > 1:
            data["loss/discriminator_accuracy"] = data["loss/discriminator_accuracy"].rolling(AVG_LEN).mean()
            if loss:
                data["loss/discriminator_loss"] = data["loss/discriminator_loss"].rolling(AVG_LEN).mean()
    # data = pd.read_csv(f)[["loss/pd_loss", "misc/total_timesteps"]]
    data["misc/total_timesteps"] /= 1e6
    if True:
        data = data.melt("misc/total_timesteps")
        sns.lineplot(x="misc/total_timesteps", y="value", hue="variable", data=data, legend=False, alpha=0.3)
    else:
        sns.lineplot(x="misc/total_timesteps", y="eprewmean", data=data, legend=False, alpha=0.3)
        sns.lineplot(x="misc/total_timesteps", y="eval_eprewmean", data=data, legend=False, alpha=0.3)

# plt.savefig("procgen_training_3_firstlayer_disc.png")
plt.show()
