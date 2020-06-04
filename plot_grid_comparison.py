from glob import glob
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import sys
import matplotlib.gridspec as gridspec
import matplotlib.style as style
from matplotlib import rcParams
import brewer2mpl

style.use("seaborn-darkgrid")
rcParams["lines.linewidth"] = 4
rcParams["font.size"] = 12


def plot_rewards(inputs, AVG_LEN, ax, colors=None):
    data = inputs[["eprewmean", "eval_eprewmean", "misc/total_timesteps"]]
    if AVG_LEN > 1:
        data["eprewmean"] = data["eprewmean"].rolling(AVG_LEN).mean()
        data["eval_eprewmean"] = data["eval_eprewmean"].rolling(AVG_LEN).mean()
    data["eprewmean"] = data["eprewmean"].clip(lower=-10.0, upper=10.0)
    data["eval_eprewmean"] = data["eval_eprewmean"].clip(lower=-10.0, upper=10.0)
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


def main_sweep_comparision(original_procgen=False, vc=True, hard=False):
    if vc:
        envs = ["visual-cartpole"]
    else:
        envs = [
            "bigfish",
            "bossfight",
            "caveflyer",
            "chaser",
            "climber",
            "coinrun",
            "dodgeball",
            "fruitbot",
            "heist",
            "jumper",
            "leaper",
            "maze",
            "miner",
            "ninja",
            "plunder",
            "starpilot",
        ]

    min_max_dict = {
        "coinrun": (5.0, 10.0),
        "starpilot": (2.5, 64.0),
        "caveflyer": (3.5, 12.0),
        "dodgeball": (1.5, 19.0),
        "fruitbot": (-1.5, 32.4),
        "chaser": (0.5, 13.0),
        "miner": (1.5, 13.0),
        "jumper": (3.0, 10.0),
        "leaper": (3.0, 10.0),
        "maze": (5.0, 10.0),
        "bigfish": (1.0, 40.0),
        "heist": (3.5, 10.0),
        "climber": (2.0, 12.6),
        "plunder": (4.5, 30.0),
        "ninja": (3.5, 10.0),
        "bossfight": (0.5, 13),
    }

    if vc:
        AVG_LEN = 0.01
    else:
        if hard:
            AVG_LEN = 0.001
        else:
            AVG_LEN = 0.01
    overall_df = pd.DataFrame()
    square_len_envs = int(np.sqrt(len(envs)))
    figheight = 10
    fig = plt.figure(figsize=(figheight * 2, figheight))
    gridspec.GridSpec(square_len_envs, square_len_envs * 2)
    artists = []
    colors = brewer2mpl.get_map("Set2", "Qualitative", 4).mpl_colors
    colors.reverse()
    vr_goggles_plot_percentage = 1.0

    colors = {
        x: colors[i] for i, x in enumerate(["PPO", "RDR", "VR Goggles", "WAPPO",])
    }
    if vc:
        min_periods = 10
    else:
        min_periods = 10
    for i_env, env in tqdm(enumerate(envs), total=len(envs)):
        big_df = pd.DataFrame()

        if vc:
            files = "vc_easy/*" + env + "*/progress.csv"
            dist_mode = "easy"
            num_timesteps = 1.0
        else:
            if not hard:
                files = "procgen_wconf_easy_3/*" + env + "*/progress.csv"
                dist_mode = "easy"
                num_timesteps = 25.0
            else:
                files = "procgen_wconf_hard/*" + env + "*/progress.csv"
                dist_mode = "hard"
                num_timesteps = 200.0

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

            if disc_coeff == 0.0:
                name = "PPO"
            elif wdisc:
                name = "Wasserstein Discriminator"
                continue
            elif wconf:
                name = "WAPPO"
            elif "mmd" in f:
                name = "MMD"
            elif "robustdr" in f:
                name = "RDR"
            elif "gp" in f:
                name = "Gradient Penalty"
            else:
                name = "WAPPO"

            # if disc_coeff > 0.:
            #     name += " disc coeff " + str(disc_coeff)

            if "olderbigger" in f:
                name += " Older Bigger"
            elif "older" in f:
                name += " Older"

            if "evenbiggercritic" in f:
                name += " Even Bigger"
            if "evenbigger" in f:
                continue
                name += " Even Bigger"
            if "reallyhigher" in f:
                continue
                name += " Really Higher"

            if os.stat(f).st_size == 0:
                continue
            try:
                data = pd.read_csv(f)
                trial_num = int(f[-14:-13])
                if trial_num == 5 and dist_mode == "easy":
                    continue
                if (
                    dist_mode == "easy"
                    and env != "visual-cartpole"
                    and len(data) < 1525
                ):
                    continue
                if (
                    dist_mode == "hard"
                    and env not in ["visual-cartpole", "leaper"]
                    and len(data) < 12207
                ):
                    continue
                data["disc_name"] = pd.Series(
                    [name for _ in range(len(data))], index=data.index
                )
                data["trial_num"] = pd.Series(
                    [trial_num for _ in range(len(data))], index=data.index
                )
                if AVG_LEN != 1:
                    data["eprewmean"] = (
                        data["eprewmean"]
                        .ewm(alpha=AVG_LEN, min_periods=min_periods)
                        .mean()
                    )
                    data["eval_eprewmean"] = (
                        data["eval_eprewmean"]
                        .ewm(alpha=AVG_LEN, min_periods=min_periods)
                        .mean()
                    )
                big_df = big_df.append(data, ignore_index=True)

            except Exception as e:
                print(f, "FAILED with ", e)
                continue

        try:
            vr_data_raw = pd.read_csv(
                "../vrgoggles/vrgoggles_out_" + str(env) + "_" + dist_mode + ".csv"
            )
            vr_data = pd.DataFrame()
            if dist_mode == "hard":
                plot_len = 12207
            else:
                plot_len = len(data)
            plot_len = int(vr_goggles_plot_percentage * plot_len)
            for _, row in vr_data_raw.iterrows():
                smol_vr_data = pd.DataFrame()
                smol_vr_data["eprewmean"] = pd.Series(
                    [row[1]][0] for _ in range(plot_len)
                )
                smol_vr_data["eval_eprewmean"] = pd.Series(
                    [row[2]][0] for _ in range(plot_len)
                )
                smol_vr_data["disc_name"] = pd.Series(
                    ["VR Goggles" for _ in range(plot_len)]
                )
                smol_vr_data["trial_num"] = pd.Series(
                    [int(row[0]) for _ in range(plot_len)]
                )
                # smol_vr_data["misc/total_timesteps"] = data[
                #     "misc/total_timesteps"
                # ].to_numpy()[len(data) - plot_len :]
                smol_vr_data["misc/total_timesteps"] = np.arange(plot_len) * (
                    num_timesteps * 1e6 / plot_len
                )
                vr_data = vr_data.append(smol_vr_data, ignore_index=True)
            big_df = big_df.append(vr_data, ignore_index=True)
        except Exception as e:
            print("VRGOGGLES FAILED with", e)

        try:
            big_df = big_df[
                [
                    "eprewmean",
                    "eval_eprewmean",
                    "misc/total_timesteps",
                    "disc_name",
                    "trial_num",
                ]
            ]
            big_df.columns = [
                "Source Reward",
                "Target Reward",
                "misc/total_timesteps",
                "disc_name",
                "trial_num",
            ]
            plt.subplot2grid(
                (square_len_envs, square_len_envs * 2),
                (i_env // square_len_envs, i_env % square_len_envs),
                colspan=1,
                rowspan=1,
            )
            plt.title(env.title().replace("-", " "))
            disc_names = list(set(big_df["disc_name"]))
            disc_names.sort()
            # big_df = big_df.dropna()
            for i_disc, disc_name in enumerate(disc_names):
                smol_data = big_df[big_df["disc_name"] == disc_name]
                grouped = smol_data.groupby("misc/total_timesteps")
                mean_source_reward = (grouped["Source Reward"].mean()).to_numpy()
                mean_target_reward = (grouped["Target Reward"].mean()).to_numpy()
                std_source_reward = (grouped["Source Reward"].std()).to_numpy()
                std_target_reward = (grouped["Target Reward"].std()).to_numpy()
                x = grouped["misc/total_timesteps"].mean().to_numpy() / 1e6
                if disc_name == "WAPPO":
                    linewidth = 5
                else:
                    linewidth = 3
                if disc_name != "VR Goggles":
                    artists.append(
                        plt.plot(
                            x,
                            mean_source_reward,
                            linestyle="solid",
                            color=colors[disc_name],
                            label=disc_name + " Source",
                            linewidth=linewidth,
                        )[0]
                    )
                    plt.fill_between(
                        x,
                        mean_source_reward - std_source_reward / 2.0,
                        mean_source_reward + std_source_reward / 2.0,
                        color=colors[disc_name],
                        alpha=0.2,
                    )
                artists.append(
                    plt.plot(
                        x,
                        mean_target_reward,
                        linestyle="dashed",
                        color=colors[disc_name],
                        label=disc_name + " Target",
                        linewidth=linewidth,
                    )[0]
                )
                plt.fill_between(
                    x,
                    mean_target_reward - std_target_reward / 2.0,
                    mean_target_reward + std_target_reward / 2.0,
                    color=colors[disc_name],
                    alpha=0.2,
                )
        except Exception as e:
            print(e)
            continue

        if original_procgen:
            openai_data = pd.read_csv("OPENAI_DATA/easy_gen_" + env + ".csv")
            openai_data.columns = [
                "misc/total_timesteps",
                "eprewmean",
                "train_std",
                "eval_eprewmean",
                "test_std",
            ]
            openai_data = openai_data[
                ["misc/total_timesteps", "eprewmean", "eval_eprewmean"]
            ]
            openai_data["disc_name"] = pd.Series(
                ["openai_original" for _ in range(len(openai_data))]
            )
            big_df = big_df.append(openai_data)

        if env != "visual-cartpole":
            big_df["Normalized Source Reward"] = (
                (big_df["Source Reward"] - min_max_dict[env][0])
                / (min_max_dict[env][1] - min_max_dict[env][0])
            ).clip(0.0)
            big_df["Normalized Target Reward"] = (
                (big_df["Target Reward"] - min_max_dict[env][0])
                / (min_max_dict[env][1] - min_max_dict[env][0])
            ).clip(0.0)
            overall_df = overall_df.append(big_df, ignore_index=True)

    if not vc:
        plt.subplot2grid(
            (square_len_envs, square_len_envs * 2),
            (0, square_len_envs),
            colspan=square_len_envs,
            rowspan=square_len_envs,
        )
        plt.title("Normalized Return")
        disc_names = list(set(overall_df["disc_name"]))
        disc_names.sort()
        # overall_df = overall_df.dropna()
        for i_disc, disc_name in enumerate(disc_names):
            smol_data = overall_df[overall_df["disc_name"] == disc_name]
            grouped = smol_data.groupby(["trial_num", "misc/total_timesteps"]).mean()
            mean_source_reward = (
                (grouped["Normalized Source Reward"].mean(level="misc/total_timesteps"))
                .rolling(window=50)
                .mean()
                .to_numpy()
            )
            mean_target_reward = (
                (grouped["Normalized Target Reward"].mean(level="misc/total_timesteps"))
                .rolling(window=50)
                .mean()
                .to_numpy()
            )
            std_source_reward = (
                (grouped["Normalized Source Reward"].std(level="misc/total_timesteps"))
                .rolling(window=50)
                .mean()
                .to_numpy()
            )
            std_target_reward = (
                (grouped["Normalized Target Reward"].std(level="misc/total_timesteps"))
                .rolling(window=50)
                .mean()
                .to_numpy()
            )
            x = np.arange(len(mean_source_reward)) * (
                num_timesteps / len(mean_source_reward)
            )
            if disc_name == "WAPPO":
                linewidth = 5
            else:
                linewidth = 3
            if disc_name != "VR Goggles":
                artists.append(
                    plt.plot(
                        x,
                        mean_source_reward,
                        linestyle="solid",
                        color=colors[disc_name],
                        label=disc_name + " Source",
                        linewidth=linewidth,
                    )[0]
                )
                plt.fill_between(
                    x,
                    mean_source_reward - std_source_reward / 2.0,
                    mean_source_reward + std_source_reward / 2.0,
                    color=colors[disc_name],
                    alpha=0.2,
                )
            artists.append(
                plt.plot(
                    x,
                    mean_target_reward,
                    linestyle="dashed",
                    color=colors[disc_name],
                    label=disc_name + " Target",
                    linewidth=linewidth,
                )[0]
            )
            plt.fill_between(
                x,
                mean_target_reward - std_target_reward / 2.0,
                mean_target_reward + std_target_reward / 2.0,
                color=colors[disc_name],
                alpha=0.2,
            )

    plt.xlabel("Timesteps (in millions)")
    plt.ylabel("Reward")
    lg = plt.legend(loc="upper left")
    artists.append(lg)
    plt.tight_layout()
    if vc:
        fig.savefig("figures/vc-training.png", bbox_inches="tight")
    else:
        fig.savefig(
            "figures/procgen-training_" + dist_mode + ".png", bbox_inches="tight"
        )

    plt.close()


if __name__ == "__main__":
    main_sweep_comparision(original_procgen=False, vc=True)
    main_sweep_comparision(original_procgen=False, vc=False)
    main_sweep_comparision(original_procgen=False, vc=False, hard=True)
