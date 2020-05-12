import seaborn as sns
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys
import matplotlib.gridspec as gridspec
import matplotlib.style as style
from matplotlib import rcParams
import brewer2mpl

# sns.set(style="darkgrid", rc={"lines.linewidth": 4.})
style.use("seaborn-darkgrid")
rcParams['lines.linewidth'] = 4
rcParams['font.size'] = 12



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

    min_max_dict = {
        "coinrun": (5., 10.),
        "starpilot": (2.5, 64.),
        "caveflyer": (3.5, 12.),
        "dodgeball": (1.5, 19.),
        "fruitbot": (-1.5, 32.4),
        "chaser": (0.5, 13.),
        "miner": (1.5, 13.),
        "jumper": (3., 10.),
        "leaper": (3., 10.),
        "maze": (0., 10.),
        "bigfish": (1., 40.),
        "heist": (3.5, 10.),
        "climber": (2., 12.6),
        "plunder": (4.5, 30.),
        "ninja": (3.5, 10.),
        "bossfight": (0.5, 13)
    }

    AVG_LEN = 200
    overall_df = pd.DataFrame()
    square_len_envs = int(np.sqrt(len(envs)))
    fig = plt.figure(figsize=(17, 25))
    # fig.suptitle("Procgen Rewards")
    gridspec.GridSpec(square_len_envs*2, square_len_envs)
    artists = []
    colors = brewer2mpl.get_map('Set2', 'Qualitative', 3).mpl_colors
    for i_env, env in tqdm(enumerate(envs)):
        big_df = pd.DataFrame()
        # files = "procgen_original_wconf_easy/*" + env + "*/progress.csv" if original_procgen else "procgen_comparison_easy/*" + env + "*/progress.csv"
        # files = "procgen_generalization_easy/*" + env + "*/progress.csv"
        # files = "vc_wconfgeneralization/*" + env + "*/progress.csv"
        # files = "visual-cartpole/*" + env + "*/progress.csv"
        # files = "vc_easy/*" + env + "*/progress.csv"
        files = "procgen_wconf_easy_3/*" + env + "*/progress.csv"
        # files = "vc_easy/*" + env + "*/progress.csv"

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
                name = "PPO"
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

            # if disc_coeff > 0.:
            #     name += " disc coeff " + str(disc_coeff)

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

        try:
            big_df = big_df[["eprewmean", "eval_eprewmean", "misc/total_timesteps", "disc_name"]]
            big_df.columns = ["Training Reward", "Testing Reward", "misc/total_timesteps", "disc_name"]
            plt.subplot2grid((square_len_envs*2, square_len_envs), (i_env // square_len_envs, i_env % square_len_envs), colspan=1, rowspan=1)
            plt.title(env.title())
            disc_names = list(set(big_df["disc_name"]))
            disc_names.sort()
            # colors = ["#689C49", "#EB7A4D", "#414A9C"]
            for i_disc, disc_name in enumerate(disc_names):
                smol_data = big_df[big_df["disc_name"]==disc_name]
                grouped = smol_data.groupby("misc/total_timesteps")
                mean_training_reward = (grouped["Training Reward"].mean()).to_numpy()
                mean_testing_reward = (grouped["Testing Reward"].mean()).to_numpy()
                std_training_reward = (grouped["Training Reward"].std()).to_numpy()
                std_testing_reward = (grouped["Testing Reward"].std()).to_numpy()
                x = smol_data[:len(mean_training_reward)]["misc/total_timesteps"].to_numpy() / 1e6
                artists.append(plt.plot(x, mean_training_reward, linestyle='solid', color=colors[i_disc])[0])
                plt.fill_between(x, mean_training_reward - std_training_reward / 2., mean_training_reward + std_training_reward / 2., color=colors[i_disc], alpha=0.2)
                artists.append(plt.plot(x, mean_testing_reward, linestyle='dashed', color=colors[i_disc])[0])
                plt.fill_between(x, mean_testing_reward - std_testing_reward / 2., mean_testing_reward + std_testing_reward / 2., color=colors[i_disc], alpha=0.2)
        except Exception as e:
            continue

        if original_procgen:
            openai_data = pd.read_csv("OPENAI_DATA/easy_gen_" + env + ".csv")
            openai_data.columns = ["misc/total_timesteps", "eprewmean", "train_std", "eval_eprewmean", "test_std"]
            openai_data = openai_data[["misc/total_timesteps", "eprewmean", "eval_eprewmean"]]
            openai_data["disc_name"] = pd.Series(["openai_original" for _ in range(len(openai_data))])
            big_df = big_df.append(openai_data)

        # big_df = big_df.melt(["misc/total_timesteps", "disc_name"])

        # big_df.columns = ["Timesteps", "Algorithm", "Variable", "Reward"]


        # sns.lineplot(x="Timesteps", y="Reward", style="Variable", hue="Algorithm", data=big_df, alpha=1.0, ax=axes[i_env % square_len_envs, i_env // square_len_envs], ci='sd', legend="full")
        # plt.savefig("figures/" + env + ".png", bbox_inches="tight")

        # plt.close(fig)

        if env != "visual-cartpole":
            big_df["Normalized Training Reward"] = (big_df["Training Reward"] - min_max_dict[env][0]) / (min_max_dict[env][1] - min_max_dict[env][0])
            big_df["Normalized Testing Reward"] = (big_df["Testing Reward"] - min_max_dict[env][0]) / (min_max_dict[env][1] - min_max_dict[env][0])
            overall_df = overall_df.append(big_df, ignore_index=True)

    plt.subplot2grid((square_len_envs*2, square_len_envs), (square_len_envs, 0), colspan=16, rowspan=16)
    plt.title("Normalized Return")
    disc_names = list(set(big_df["disc_name"]))
    disc_names.sort()
    for i_disc, disc_name in enumerate(disc_names):
        smol_data = big_df[big_df["disc_name"]==disc_name]
        grouped = smol_data.groupby("misc/total_timesteps")
        mean_training_reward = (grouped["Normalized Training Reward"].mean()).to_numpy()
        mean_testing_reward = (grouped["Normalized Testing Reward"].mean()).to_numpy()
        std_training_reward = (grouped["Normalized Training Reward"].std()).to_numpy()
        std_testing_reward = (grouped["Normalized Testing Reward"].std()).to_numpy()
        x = smol_data[:len(mean_training_reward)]["misc/total_timesteps"].to_numpy() / 1e6
        artists.append(plt.plot(x, mean_training_reward, linestyle='solid', color=colors[i_disc], label=disc_name + " Training")[0])
        plt.fill_between(x, mean_training_reward - std_training_reward / 2., mean_training_reward + std_training_reward / 2., color=colors[i_disc], alpha=0.2)
        artists.append(plt.plot(x, mean_testing_reward, linestyle='dashed', color=colors[i_disc], label=disc_name + " Testing")[0])
        plt.fill_between(x, mean_testing_reward - std_testing_reward / 2., mean_testing_reward + std_testing_reward / 2., color=colors[i_disc], alpha=0.2)

    plt.xlabel("Timesteps (in millions)")
    plt.ylabel("Reward")
    lg = plt.legend()
    artists.append(lg)
    # plt.show()
    fig.savefig("figures/procgen-training.png", bbox_inches="tight")
if __name__ == "__main__":
    main_sweep_comparision(original_procgen=False)
