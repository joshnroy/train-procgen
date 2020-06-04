import seaborn as sns
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")
import os
from tqdm import tqdm
import sys

sns.set(style="darkgrid")


def plot_rewards(inputs, AVG_LEN, ax, ax2):
    data = inputs[["eprewmean", "eval_eprewmean", "misc/total_timesteps"]]
    if AVG_LEN > 1:
        data["eprewmean"] = data["eprewmean"].rolling(AVG_LEN, center=True).mean()
        data["eval_eprewmean"] = (
            data["eval_eprewmean"].rolling(AVG_LEN, center=True).mean()
        )
    # data["eprewmean"] = data["eprewmean"].clip(lower=-10., upper=10.)
    # data["eval_eprewmean"] = data["eval_eprewmean"].clip(lower=-10., upper=10.)
    newseries = data["eval_eprewmean"] / data["eprewmean"]
    newseries = newseries.clip(lower=-10.0, upper=10.0)
    data = data.melt("misc/total_timesteps")
    sns.lineplot(
        x="misc/total_timesteps",
        y="value",
        hue="variable",
        data=data,
        alpha=1.0,
        ax=ax,
        ci="sd",
    )
    ax.set_title("Rewards")
    sns.lineplot(data=newseries, ax=ax2, ci="sd")
    ax2.set_title("Reward Ratio")


def plot_critic_rating(inputs, AVG_LEN, ax):
    data = inputs[["loss/critic_min", "loss/critic_max", "misc/total_timesteps"]]
    if AVG_LEN > 1:
        data["loss/critic_min"] = (
            data["loss/critic_min"].rolling(AVG_LEN, center=True).mean()
        )
        data["loss/critic_max"] = (
            data["loss/critic_max"].rolling(AVG_LEN, center=True).mean()
        )
    data["loss/critic_min"] = data["loss/critic_min"].clip(lower=-10.0, upper=10.0)
    data["loss/critic_max"] = data["loss/critic_max"].clip(lower=-10.0, upper=10.0)
    data = data.melt("misc/total_timesteps")
    sns.lineplot(
        x="misc/total_timesteps",
        y="value",
        hue="variable",
        data=data,
        alpha=1.0,
        ax=ax,
        ci="sd",
    )
    ax.set_title("Rewards")


def plot_discriminator_accuracy(inputs, AVG_LEN, ax):
    data = inputs[["loss/discriminator_accuracy", "misc/total_timesteps"]]
    if AVG_LEN > 1:
        data["loss/discriminator_accuracy"] = (
            data["loss/discriminator_accuracy"].rolling(AVG_LEN, center=True).mean()
        )
    data["loss/discriminator_accuracy"] = data["loss/discriminator_accuracy"].clip(
        lower=-10.0, upper=10.0
    )
    data = data.melt("misc/total_timesteps")
    sns.lineplot(
        x="misc/total_timesteps",
        y="value",
        hue="variable",
        data=data,
        alpha=1.0,
        ax=ax,
        ci="sd",
    )
    ax.set_title("Discriminator Accuracy")


def plot_discriminator_loss(inputs, AVG_LEN, ax):
    data = inputs[["loss/discriminator_loss", "loss/pd_loss", "misc/total_timesteps"]]
    if AVG_LEN > 1:
        data["loss/discriminator_loss"] = (
            data["loss/discriminator_loss"].rolling(AVG_LEN, center=True).mean()
        )
        data["loss/pd_loss"] = data["loss/pd_loss"].rolling(AVG_LEN, center=True).mean()
    data["loss/discriminator_loss"] = data["loss/discriminator_loss"].clip(
        lower=-10.0, upper=10.0
    )
    data["loss/pd_loss"] = data["loss/pd_loss"].clip(lower=-10.0, upper=10.0)
    data = data.melt("misc/total_timesteps")
    sns.lineplot(
        x="misc/total_timesteps",
        y="value",
        hue="variable",
        data=data,
        alpha=1.0,
        ax=ax,
        ci="sd",
    )
    ax.set_title("Discriminator Loss")


def plot_value_loss(inputs, AVG_LEN, ax):
    data = inputs[["loss/value_loss", "misc/total_timesteps"]]
    if AVG_LEN > 1:
        data["loss/value_loss"] = (
            data["loss/value_loss"].rolling(AVG_LEN, center=True).mean()
        )
    data["loss/value_loss"] = data["loss/value_loss"].clip(lower=-1e3, upper=1e3)
    data = data.melt("misc/total_timesteps")
    sns.lineplot(
        x="misc/total_timesteps",
        y="value",
        hue="variable",
        data=data,
        alpha=1.0,
        ax=ax,
        ci="sd",
    )
    ax.set_title("value Loss")


def main_sweep():
    AVG_LEN = 30
    for f in tqdm(glob("/home/homedir/w_disc_again_easy_stuff/*/progress.csv")):
        if os.stat(f).st_size == 0:
            continue
        try:
            data = pd.read_csv(f)
            data["misc/total_timesteps"] /= 1e6
            # name = f[58:-13]
            name = str.split(f, "/")[4]

            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, figsize=(20, 20))
            fig.suptitle(name)

            plot_rewards(data, AVG_LEN, ax1, ax2)
            # plot_discriminator_accuracy(data, AVG_LEN, ax3)
            plot_discriminator_loss(data, AVG_LEN, ax3)
            plot_value_loss(data, AVG_LEN, ax4)
            plot_critic_rating(data, AVG_LEN, ax5)

            plt.savefig("figures/" + name + ".png")

            plt.close(fig)
        except Exception as e:
            print(f, "FAILED with ", e)
            continue


def main():
    AVG_LEN = 10
    all_data = []
    for f in tqdm(glob("visual-baselines-easy/*bigfish*/*.csv")):
        split = f.replace("/", "_").split("_")
        env_name = split[1]
        disc_coeff = split[4]
        num_levels = split[7]
        num_test_levels = split[16]
        trial_num = split[18]

        name = (
            env_name
            + " "
            + disc_coeff
            + " "
            + " "
            + num_levels
            + " "
            + num_test_levels
            + " "
            + trial_num
        )

        if os.stat(f).st_size == 0:
            continue
        try:
            data = pd.read_csv(f)
            data["num_levels"] = pd.Series(np.zeros(len(data)) + int(num_levels))
            all_data.append(data)

        except Exception as e:
            print(f, "FAILED with ", e)
            continue

    df = pd.concat(all_data)

    df["eprewmean"] = df["eprewmean"].clip(lower=-10.0, upper=50.0)
    df["eval_eprewmean"] = df["eval_eprewmean"].clip(lower=-10.0, upper=50.0)
    if AVG_LEN > 1:
        df["eprewmean"] = df["eprewmean"].ewm(span=AVG_LEN).mean()
        df["eval_eprewmean"] = df["eval_eprewmean"].ewm(span=AVG_LEN).mean()

    print("Starting plotting")
    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 20))
    fig.suptitle("Visual Easy BigFish, PPO")
    sns.lineplot(
        y="eprewmean",
        x="misc/total_timesteps",
        data=df,
        hue="num_levels",
        ci="sd",
        ax=ax1,
        palette=sns.color_palette(),
    )
    sns.lineplot(
        y="eval_eprewmean",
        x="misc/total_timesteps",
        data=df,
        hue="num_levels",
        ci="sd",
        ax=ax2,
        palette=sns.color_palette(),
    )
    print("Done plotting")
    plt.show()


if __name__ == "__main__":
    main()
