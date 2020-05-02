import seaborn as sns
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys
import matplotlib
matplotlib.use("Agg")

sns.set(style="darkgrid", rc={"lines.linewidth": 4.})


def plot_rewards(inputs, AVG_LEN, ax, colors=None):
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

def main_sweep_comparision():
    envs = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "fruitbot", "heist", "jumper", "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]
    norm_constants = [(0, 40), (0.5, 13), (2, 13.4), (0.5, 14.2), (1, 12.6), (5, 10), (1.5, 19), (-0.5, 27.2), (2, 10), (1, 10), (1.5, 10), (4, 10), (1.5, 20), (2, 10), (3, 30), (1.5, 35)]
    AVG_LEN = 200
    ci = 'sd'

    normalized_df = pd.DataFrame()
    for i, env in tqdm(enumerate(envs)):
        big_df = pd.DataFrame()
        fig, ax1 = plt.subplots(1, figsize=(20, 10))
        fig.suptitle(env + " Rewards")

        if len(glob("/home/jroy1/procgen_wconf_hard/*" + env + "*/progress.csv")) == 0:
            continue
        glob_files = list(glob("/home/jroy1/procgen_wconf_hard/*" + env + "*/progress.csv"))
        glob_files.sort()
        for f in glob_files:
            split = f.split("/")
            split = split[4].split("_")
            disc_coeff = float(split[3])
            if disc_coeff == 0.:
                disc_name = "vanilla"
            else:
                disc_name = "disc_coeff_" + str(disc_coeff)

            # name = env + "_" + disc_name

            if os.stat(f).st_size == 0:
                continue
            try:
                data = pd.read_csv(f)
                data['disc_name'] = pd.Series([disc_name for _ in range(len(data))], index=data.index)
                if AVG_LEN > 1:
                    data["eprewmean"] = data["eprewmean"].rolling(AVG_LEN).mean()
                    data["eval_eprewmean"] = data["eval_eprewmean"].rolling(AVG_LEN).mean()
                big_df = big_df.append(data, ignore_index=True)
                # plot_rewards(data, AVG_LEN, ax1, colors=colors)
                # names.append(name)

            except Exception as e:
                print(f, "FAILED with ", e)
                continue

        try:
            big_df = big_df[["eprewmean", "eval_eprewmean", "misc/total_timesteps", "disc_name"]]
            big_df_len = len(big_df)
            melted_big_df = big_df.melt(["misc/total_timesteps", "disc_name"])
            melted_big_df["identifier"] = melted_big_df["disc_name"] + "_" + melted_big_df["variable"]

            big_df["eprewmean"] -= norm_constants[i][0]
            big_df["eprewmean"] /= (norm_constants[i][1] - norm_constants[i][0])
            big_df["eval_eprewmean"] -= norm_constants[i][0]
            big_df["eval_eprewmean"] /= (norm_constants[i][1] - norm_constants[i][0])
            normalized_df = normalized_df.append(big_df, ignore_index=True)

            sns.lineplot(x="misc/total_timesteps", y="value", hue="variable", style="disc_name", data=melted_big_df, alpha=1.0, ax=ax1, ci=ci, legend="full")
            plt.savefig("figures/" + env + ".png")
        except Exception as e:
            print("Gg")

        plt.close(fig)

    melted_normalized_df = normalized_df.melt(["misc/total_timesteps", "disc_name"])
    fig, ax1 = plt.subplots(1, figsize=(20, 20))
    sns.lineplot(x="misc/total_timesteps", y="value", hue="variable", style="disc_name", data=melted_normalized_df, alpha=1.0, ci=ci, legend="full", ax=ax1)
    plt.savefig("figures/normalized_mean.png")
    plt.close(fig)

if __name__ == "__main__":
    main_sweep_comparision()
