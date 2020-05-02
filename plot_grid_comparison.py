import seaborn as sns
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys

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

def main_sweep_comparision(original_procgen=False):
    envs = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "fruitbot", "heist", "jumper", "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]
    # envs = ["visual-cartpole"]
    AVG_LEN = 10
    for env in tqdm(envs):
        big_df = pd.DataFrame()
        fig, ax1 = plt.subplots(1, figsize=(20, 10))
        fig.suptitle("Visual Cartpole Rewards")

        # files = "procgen_original_wconf_easy/*" + env + "*/progress.csv" if original_procgen else "procgen_comparison_easy/*" + env + "*/progress.csv"
        # files = "procgen_generalization_easy/*" + env + "*/progress.csv"
        # files = "vc_wconfgeneralization/*" + env + "*/progress.csv"
        # files = "visual-cartpole/*" + env + "*/progress.csv"
        # files = "vc_easy/*" + env + "*/progress.csv"
        files = "procgen_wconf_easy_2/*" + env + "*/progress.csv"

        if len(glob(files)) == 0:
            continue

        glob_files = list(glob(files))
        glob_files.sort()
        for f in glob_files:
            split = f.split("/")
            info = split[1]
            wdisc = False
            wconf = False
            if info[:6] == "wdisc_":
                wdisc = True
                info = info[6:]
            if info[:6] == "wconf_":
                wconf = True
                info = info[6:]
            split = info.split("_")
            disc_coeff = float(split[3])

            if disc_coeff == 0.:
                name = "Vanilla"
            elif wdisc:
                name = "Wasserstein Discriminator"
                continue
            elif wconf:
                name = "Adversarial PPO"
            elif "mmd" in f:
                name = "MMD"
            elif "robustdr" in f:
                name = "Robust Domain Randomization"
            elif "gp" in f:
                name = "Gradient Penalty"
            else:
                name = "Adversarial PPO"

            if disc_coeff > 0.:
                name += " disc coeff " + str(disc_coeff)

            # if "100k" in f:
            #     name += "_limited_100k"
            # if "10k" in f:
            #     name += "_limited_10k"
            # if "1k" in f:
            #     name += "_limited_1k"
            # if "hundred" in f:
            #     name += "_limited_hundred"

            if "olderbigger" in f:
                name += " Older Bigger"
            elif "older" in f:
                name += " Older"

            if "evenbiggercritic" in f:
                name += " Even Bigger"
            if "evenbigger" in f:
                name += " Even Bigger"
            if "disc5" in f:
                name += " Disc 5"

            # if "num_levels_3" in f:
            #     name += "num_levels_3"
            # if "num_levels_5" in f:
            #     name += "num_levels_5"

            # if "gp" in f:
            #     name += "gp"
            #     continue


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
        big_df.columns = ["Training Reward", "Testing Reward", "misc/total_timesteps", "disc_name"]
        big_df_len = len(big_df)

        if original_procgen:
            openai_data = pd.read_csv("OPENAI_DATA/easy_gen_" + env + ".csv")
            openai_data.columns = ["misc/total_timesteps", "eprewmean", "train_std", "eval_eprewmean", "test_std"]
            openai_data = openai_data[["misc/total_timesteps", "eprewmean", "eval_eprewmean"]]
            openai_data["disc_name"] = pd.Series(["openai_original" for _ in range(len(openai_data))])
            big_df = big_df.append(openai_data)

        big_df = big_df.melt(["misc/total_timesteps", "disc_name"])

        big_df.columns = ["Timesteps", "Algorithm", "Variable", "Reward"]


        sns.lineplot(x="Timesteps", y="Reward", style="Variable", hue="Algorithm", data=big_df, alpha=1.0, ax=ax1, ci='sd', legend="full")
        plt.savefig("figures/" + env + ".png", bbox_inches="tight")

        plt.close(fig)

if __name__ == "__main__":
    main_sweep_comparision(original_procgen=False)
