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
        data["eval_eprewmean"] = data["eval_eprewmean"].rolling(AVG_LEN, center=True).mean()
    # data["eprewmean"] = data["eprewmean"].clip(lower=-10., upper=10.)
    # data["eval_eprewmean"] = data["eval_eprewmean"].clip(lower=-10., upper=10.)
    newseries = data["eval_eprewmean"] / data["eprewmean"]
    newseries = newseries.clip(lower=-10., upper=10.)
    data = data.melt("misc/total_timesteps")
    sns.lineplot(x="misc/total_timesteps", y="value", hue="variable", data=data, alpha=1.0, ax=ax, ci='sd')
    ax.set_title("Rewards")
    sns.lineplot(data=newseries, ax=ax2, ci='sd')
    ax2.set_title("Reward Ratio")

def plot_critic_rating(inputs, AVG_LEN, ax):
    data = inputs[["loss/critic_min", "loss/critic_max", "misc/total_timesteps"]]
    if AVG_LEN > 1:
        data["loss/critic_min"] = data["loss/critic_min"].rolling(AVG_LEN, center=True).mean()
        data["loss/critic_max"] = data["loss/critic_max"].rolling(AVG_LEN, center=True).mean()
    data["loss/critic_min"] = data["loss/critic_min"].clip(lower=-10., upper=10.)
    data["loss/critic_max"] = data["loss/critic_max"].clip(lower=-10., upper=10.)
    data = data.melt("misc/total_timesteps")
    sns.lineplot(x="misc/total_timesteps", y="value", hue="variable", data=data, alpha=1.0, ax=ax, ci='sd')
    ax.set_title("Rewards")


def plot_discriminator_accuracy(inputs, AVG_LEN, ax):
    data = inputs[["loss/discriminator_accuracy", "misc/total_timesteps"]]
    if AVG_LEN > 1:
        data["loss/discriminator_accuracy"] = data["loss/discriminator_accuracy"].rolling(AVG_LEN, center=True).mean()
    data["loss/discriminator_accuracy"] = data["loss/discriminator_accuracy"].clip(lower=-10., upper=10.)
    data = data.melt("misc/total_timesteps")
    sns.lineplot(x="misc/total_timesteps", y="value", hue="variable", data=data, alpha=1.0, ax=ax, ci='sd')
    ax.set_title("Discriminator Accuracy")


def plot_discriminator_loss(inputs, AVG_LEN, ax):
    data = inputs[["loss/discriminator_loss", "loss/pd_loss", "misc/total_timesteps"]]
    if AVG_LEN > 1:
        data["loss/discriminator_loss"] = data["loss/discriminator_loss"].rolling(AVG_LEN, center=True).mean()
        data["loss/pd_loss"] = data["loss/pd_loss"].rolling(AVG_LEN, center=True).mean()
    data["loss/discriminator_loss"] = data["loss/discriminator_loss"].clip(lower=-10., upper=10.)
    data["loss/pd_loss"] = data["loss/pd_loss"].clip(lower=-10., upper=10.)
    data = data.melt("misc/total_timesteps")
    sns.lineplot(x="misc/total_timesteps", y="value", hue="variable", data=data, alpha=1.0, ax=ax, ci='sd')
    ax.set_title("Discriminator Loss")

def plot_value_loss(inputs, AVG_LEN, ax):
    data = inputs[["loss/value_loss", "misc/total_timesteps"]]
    if AVG_LEN > 1:
        data["loss/value_loss"] = data["loss/value_loss"].rolling(AVG_LEN, center=True).mean()
    data["loss/value_loss"] = data["loss/value_loss"].clip(lower=-1e3, upper=1e3)
    data = data.melt("misc/total_timesteps")
    sns.lineplot(x="misc/total_timesteps", y="value", hue="variable", data=data, alpha=1.0, ax=ax, ci='sd')
    ax.set_title("value Loss")

def main_sweep2():
    mins = {"coinrun": 5, "starpilot": 2.5, "caveflyer": 3.5, "dodgeball": 1.5, "fruitbot": -1.5, "chaser": .5, "miner": 1.5, "jumper": 3, "leaper": 3, "maze": 4, "bigfish": 1, "heist": 3.5, "climber": 2, "plunder": 4.5, "ninja": 3.5, "bossfight": .5}
    maxes = {"coinrun": 10, "starpilot": 64, "caveflyer": 12, "dodgeball": 19, "fruitbot": 32.4, "chaser": 13, "miner": 13, "jumper": 10, "leaper": 10, "maze": 10, "bigfish": 40, "heist": 10, "climber": 12.5, "plunder": 30, "ninja": 10, "bossfight": 13}
    AVG_LEN = 100
    my_normalized_returns_train = []
    reported_normalized_returns_train = []
    my_normalized_returns_test = []
    reported_normalized_returns_test = []
    for f in tqdm(glob("/home/josh/w_disc_again_easy_stuff/*/progress.csv")):
        if os.stat(f).st_size == 0:
            continue
        env_name = f.split("/")[4].split("_")[0]
        # if env_name in ["maze", "chaser", "dodgeball", "fruitbot", "miner", "heist", "coinrun", "caveflyer"]:
        # if env_name in ["coinrun"]:
            # continue
        procgen_report = "data/procgen_export_data/easy_gen_" + env_name + ".csv"
        try:
            data = pd.read_csv(f)
            data["misc/total_timesteps"] /= 1e6
            reported_data = pd.read_csv(procgen_report)
            # name = f[58:-13]
            name = str.split(f, "/")[4]

            if AVG_LEN > 1:
                data["eprewmean"] = data["eprewmean"].rolling(AVG_LEN, center=True).mean()
                data["eval_eprewmean"] = data["eval_eprewmean"].rolling(AVG_LEN, center=True).mean()

            if len(data["eprewmean"]) > 2000:
                data["eprewmean"] = data["eprewmean"].groupby(np.arange(len(data["eprewmean"]))//2).mean()
            data["eprewmean"] = (data["eprewmean"] - mins[env_name]) / (maxes[env_name] - mins[env_name])
            my_normalized_returns_train.append(data["eprewmean"])
            if len(data["eval_eprewmean"]) > 2000:
                data["eval_eprewmean"] = data["eval_eprewmean"].groupby(np.arange(len(data["eval_eprewmean"]))//2).mean()
            data["eval_eprewmean"] = (data["eval_eprewmean"] - mins[env_name]) / (maxes[env_name] - mins[env_name])
            my_normalized_returns_test.append(data["eval_eprewmean"])

            sns.lineplot(data=data[["eprewmean", "eval_eprewmean"]], palette=['blue', 'orange'])
            reported_data["train_mean0"] = (reported_data["train_mean0"] - mins[env_name]) / (maxes[env_name] - mins[env_name])
            reported_data["test_mean2"] = (reported_data["test_mean2"] - mins[env_name]) / (maxes[env_name] - mins[env_name])
            sns.lineplot(data=reported_data[["train_mean0", "test_mean2"]], palette=['green', 'red'])
            reported_normalized_returns_train.append(reported_data["train_mean0"])
            reported_normalized_returns_test.append(reported_data["test_mean2"])

            plt.savefig("figures/" + name + ".png")

            plt.close()
        except Exception as e:
            print(f, "FAILED with ", e)
            continue

    my_normalized_returns_train = pd.DataFrame.from_dict(map(dict, my_normalized_returns_train)).mean(axis=0)
    my_normalized_returns_test = pd.DataFrame.from_dict(map(dict, my_normalized_returns_test)).mean(axis=0)
    reported_normalized_returns_train = pd.DataFrame.from_dict(map(dict, reported_normalized_returns_train)).mean(axis=0)
    reported_normalized_returns_test = pd.DataFrame.from_dict(map(dict, reported_normalized_returns_test)).mean(axis=0)
    final_mean = pd.DataFrame.from_dict(([my_normalized_returns_train, my_normalized_returns_test, reported_normalized_returns_train, reported_normalized_returns_test]))
    final_mean = final_mean.transpose()
    print(final_mean)
    final_mean.columns = ["My Train", "My Test", "Reported Train", "Reported Test"]
    sns.lineplot(data=final_mean)
    plt.show()


def main_sweep():
    AVG_LEN = 30
    for f in tqdm(glob("/home/josh/w_disc_again_easy_stuff/*/progress.csv")):
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

def main_trials():
    AVG_LEN = 1
    things = [str(0.01), str(0.02), str(0.03)]
    for thing in things:
        data = pd.DataFrame()
        for f in tqdm(glob("/home/jroy1/jumper_easy_gan_manytrials/*" + thing + "*/progress.csv")):
            if os.stat(f).st_size == 0:
                continue
            print(f)
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
    main_sweep2()
