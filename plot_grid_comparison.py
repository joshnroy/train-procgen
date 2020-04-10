import seaborn as sns
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

sns.set(style="darkgrid")


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
    AVG_LEN = 100
    for env in tqdm(envs):
        big_df = pd.DataFrame()
        fig, ax1 = plt.subplots(1, figsize=(20, 10))
        fig.suptitle(env + " Rewards")

        for f in glob("procgen_comparison/*" + env + "*/progress.csv"):
            split = f.split("/")
            split = split[1].split("_")
            disc_coeff = float(split[3])
            if disc_coeff == 0.:
                disc_name = "vanilla"
            else:
                disc_name = "disc_coeff_" + str(disc_coeff)

            name = env + "_" + disc_name

            if os.stat(f).st_size == 0:
                continue
            try:
                data = pd.read_csv(f)
                data['disc_name'] = pd.Series([name for _ in range(len(data))], index=data.index)
                if AVG_LEN > 1:
                    data["eprewmean"] = data["eprewmean"].rolling(AVG_LEN).mean()
                    data["eval_eprewmean"] = data["eval_eprewmean"].rolling(AVG_LEN).mean()
                big_df = big_df.append(data, ignore_index=True)
                # plot_rewards(data, AVG_LEN, ax1, colors=colors)
                # names.append(name)

            except Exception as e:
                print(f, "FAILED with ", e)
                continue

        big_df = big_df[["eprewmean", "eval_eprewmean", "misc/total_timesteps", "disc_name"]]
        big_df_len = len(big_df)
        big_df = big_df.melt(["misc/total_timesteps", "disc_name"])
        big_df["identifier"] = big_df["disc_name"] + "_" + big_df["variable"]
        # if big_df_len > len(data):
        #     print(big_df)
        #     sys.exit()
        sns.lineplot(x="misc/total_timesteps", y="value", hue="variable", style="disc_name", data=big_df, alpha=1.0, ax=ax1, ci='sd')
        plt.savefig("figures/" + env + ".png")

        plt.close(fig)

if __name__ == "__main__":
    main_sweep_comparision()
