import seaborn as sns
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

sns.set(style="darkgrid")


def plot_rewards(inputs, AVG_LEN, ax, ax2):
    data = inputs[["eprewmean", "eval_eprewmean", "misc/total_timesteps"]]
    if AVG_LEN > 1:
        data["eprewmean"] = data["eprewmean"].rolling(AVG_LEN).mean()
        data["eval_eprewmean"] = data["eval_eprewmean"].rolling(AVG_LEN).mean()
    data["eprewmean"] = data["eprewmean"].clip(lower=-10., upper=10.)
    data["eval_eprewmean"] = data["eval_eprewmean"].clip(lower=-10., upper=10.)
    newseries = data["eval_eprewmean"] / data["eprewmean"]
    newseries = newseries.clip(lower=-10., upper=10.)
    data = data.melt("misc/total_timesteps")
    sns.lineplot(x="misc/total_timesteps", y="value", hue="variable", data=data, alpha=1.0, ax=ax, ci='sd')
    ax.set_title("Rewards")
    sns.lineplot(data=newseries, ax=ax2, ci='sd')
    ax2.set_title("Reward Ratio")


def plot_discriminator_accuracy(inputs, AVG_LEN, ax):
    data = inputs[["loss/discriminator_accuracy", "misc/total_timesteps"]]
    if AVG_LEN > 1:
        data["loss/discriminator_accuracy"] = data["loss/discriminator_accuracy"].rolling(AVG_LEN).mean()
    data["loss/discriminator_accuracy"] = data["loss/discriminator_accuracy"].clip(lower=-10., upper=10.)
    data = data.melt("misc/total_timesteps")
    sns.lineplot(x="misc/total_timesteps", y="value", hue="variable", data=data, alpha=1.0, ax=ax, ci='sd')
    ax.set_title("Discriminator Accuracy")


def plot_discriminator_loss(inputs, AVG_LEN, ax):
    data = inputs[["loss/discriminator_loss", "misc/total_timesteps"]]
    if AVG_LEN > 1:
        data["loss/discriminator_loss"] = data["loss/discriminator_loss"].rolling(AVG_LEN).mean()
    data["loss/discriminator_loss"] = data["loss/discriminator_loss"].clip(lower=-10., upper=10.)
    data = data.melt("misc/total_timesteps")
    sns.lineplot(x="misc/total_timesteps", y="value", hue="variable", data=data, alpha=1.0, ax=ax, ci='sd')
    ax.set_title("Discriminator Loss")


def main_sweep():
    AVG_LEN = 500
    # for f in tqdm(glob("/home/jroy1/procgen_training_all_later_short_jumper/*/progress.csv")):
    # for f in tqdm(glob("/home/jroy1/procgen_training_all_later_hard_jumper/*/progress.csv")):
    for f in tqdm(glob("/home/jroy1/jumper_easy_batchsize/*/progress.csv")):
        if os.stat(f).st_size == 0:
            continue
        try:
            data = pd.read_csv(f)
            data["misc/total_timesteps"] /= 1e6
            # name = f[58:-13]
            name = str.split(f, "/")[4]

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
            fig.suptitle(name)

            plot_rewards(data, AVG_LEN, ax1, ax2)
            plot_discriminator_accuracy(data, AVG_LEN, ax3)
            plot_discriminator_loss(data, AVG_LEN, ax4)

            plt.savefig("figures/" + name + ".png")

            plt.close(fig)
        except Exception as e:
            print(f, "FAILED with ", e)
            continue

def main_trials():
    AVG_LEN = 10
    things = [str(0.01), str(0.02), str(0.03)]
    for thing in things:
        data = pd.DataFrame()
        for f in tqdm(glob("/home/jroy1/jumper_easy_gan_manytrials/*" + thing + "*/progress.csv")):
            if os.stat(f).st_size == 0:
                continue
            try:
                data_f = pd.read_csv(f)
                data = data.append(data_f)

            except Exception as e:
                print(f, "FAILED with ", e)
                continue

        data["misc/total_timesteps"] /= 1e6

        name = thing

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        fig.suptitle(name)

        plot_rewards(data, AVG_LEN, ax1, ax2)
        plot_discriminator_accuracy(data, AVG_LEN, ax3)
        plot_discriminator_loss(data, AVG_LEN, ax4)

        if True:
            plt.show()
        else:
            plt.savefig("figures/" + name + ".png")
            plt.close(fig)

if __name__ == "__main__":
    main_sweep()
