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

def main():
    AVG_LEN = 1
    my_returns_train = {"10": [], "5": [], "3": [], "1": []}
    ppo_returns_train = {"10": [], "5": [], "3": [], "1": []}
    my_returns_test = {"10": [], "5": [], "3": [], "1": []}
    ppo_returns_test = {"10": [], "5": [], "3": [], "1": []}
    for f in tqdm(glob("/home/josh/w_disc_againeasy/*/progress.csv")):
        if os.stat(f).st_size == 0:
            continue
        disc_coeff = f.split("/")[4].split("_")[3]
        num_training_levels = f.split("/")[4].split("_")[6]
        if disc_coeff == "0.0":
            training_list = ppo_returns_train[num_training_levels]
            testing_list = ppo_returns_test[num_training_levels]
        else:
            training_list = my_returns_train[num_training_levels]
            testing_list = my_returns_test[num_training_levels]
        try:
            data = pd.read_csv(f)
            data["misc/total_timesteps"] /= 1e6
            # name = f[58:-13]
            name = str.split(f, "/")[4]

            if AVG_LEN > 1:
                data["eprewmean"] = data["eprewmean"].rolling(AVG_LEN, center=True).mean()
                data["eval_eprewmean"] = data["eval_eprewmean"].rolling(AVG_LEN, center=True).mean()

            training_list.append(data[["eprewmean", "eval_eprewmean", "misc/total_timesteps"]])

        except Exception as e:
            print(f, "FAILED with ", e)
            continue

    for num_train_levels in my_returns_train:
        my_returns_train_level = pd.concat(my_returns_train[num_train_levels], axis=0)
        my_returns_train_level.columns = ["W Domain Confusion Train", "W Domain Confusion Test", "Environment Steps"]
        ppo_returns_train_level = pd.concat(ppo_returns_train[num_train_levels], axis=0)
        ppo_returns_train_level.columns = ["PPO Train", "PPO Test", "Environment Steps"]

        finals = pd.concat([my_returns_train_level, ppo_returns_train_level], axis=0)
        finals = pd.melt(finals, ["Environment Steps"])

        sns.lineplot(data=finals, hue="variable", y="value", x="Environment Steps")
        plt.show()

if __name__ == "__main__":
    main()
