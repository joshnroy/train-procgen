import tensorflow as tf
from baselines.adversarial_ppo2 import ppo2
from baselines.adversarial_ppo2.models import build_impala_cnn, nature_cnn
from baselines.common.mpi_util import setup_mpi_gpus
from procgen import ProcgenEnv
from baselines.common.vec_env import (
    VecExtractDictObs,
    VecMonitor,
    VecFrameStack,
    VecNormalize
)
from baselines import logger
from mpi4py import MPI
import argparse
import os
import sys
import gym
import gym_cartpole_visual
import numpy as np
from tqdm import tqdm
import torch

from pyvirtualdisplay import Display
display = Display(visible=0, size=(100, 100), backend="xvfb")
display.start()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.stats import gaussian_kde
import matplotlib.style as style
from matplotlib import rcParams
import brewer2mpl
from pathlib import Path

import random
random.seed(0)
np.random.seed(0)

style.use("seaborn-darkgrid")
rcParams['lines.linewidth'] = 1
rcParams['font.size'] = 12

def encircle(x,y, ax=None, **kw):
    if not ax: ax=plt.gca()
    p = np.c_[x,y]
    hull = ConvexHull(p)
    poly = plt.Polygon(p[hull.vertices,:], **kw)
    ax.add_patch(poly)

def density_estimation(m1, m2):
    print(m1.shape, m2.shape)
    xmin = -200.
    ymin = -200.
    xmax = 200.
    ymax = 200.
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z

def preprocess_fn(model, data, transform):
    # with torch.no_grad():
    data = torch.stack([transform(x) for x in data])
    # print(data.min(), data.max(), data.shape)
    # sys.exit()
    out = model.netG_A(data)
    # print(out.min(), out.max())
    out = ((out.permute(0, 2, 3, 1) + 1.) / 2. * 255.).detach().cpu().numpy()
    # print(out.min(), out.max())
    out = out.astype(np.uint8)
    return out


def main():
    num_envs = 9
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nminibatches = 8
    nsteps = (2048 // nminibatches)
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True
    dist_mode = "easy"

    visualization = False
    save_images = False
    vrgoggles = True

    # if int(os.environ["SGE_TASK_ID"]) not in [8, 10, 11, 12, 13, 14, 15, 16]:
    #     sys.exit()

    if "SGE_TASK_ID" in os.environ:
        indicator = int(os.environ["SGE_TASK_ID"]) - 1
        i_trial = indicator % len(target_levels)
        i_env = indicator // len(target_levels)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--i_trial", help="trial number", required=False, default=0)
        parser.add_argument("--i_env", help="env number", required=False, default=0)
        args = parser.parse_args()
        i_trial = int(args.i_trial)
        i_env = int(args.i_env)


    source_levels = [1543, 7991, 3671, 2336, 6420]
    source_level = source_levels[i_trial]

    target_levels = [7354, 9570, 6317, 6187, 8430]
    target_level = target_levels[i_trial]

    # env_names = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "fruitbot", "heist", "jumper", "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]

    env_name = "visual-cartpole"
    # env_name = env_names[i_env]
    num_frames = 1

    if env_name == "visual-cartpole":
        timesteps_per_proc = 1_000_000
        save_interval=30
    elif dist_mode == "easy":
        timesteps_per_proc = 25_000_000
        save_interval=100
    else:
        timesteps_per_proc = 200_000_000
        save_interval=100


    num_levels = 1
    num_test_levels = 1
    num_iterations = 2

    test_worker_interval = 0

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False

    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)

    mpi_rank_weight = 0 if is_test_worker else 1

    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout', 'tensorboard'] if log_comm.Get_rank() == 0 else []

    load_path = "train-procgen/vc_easy/visual-cartpole_disc_coeff_0.0_num_levels_1_nsteps_256_num_frames_1_num_test_levels_1_trial_" + str(i_trial) + "/checkpoints/00390"
    disc_coeff = float(load_path.split("_")[4])
    # load_path = "train-procgen/procgen_wconf_easy_3/" + env_name + "_disc_coeff_0.0_num_levels_1_nsteps_256_num_frames_1_num_test_levels_1_trial_0/checkpoints/01500"
    # disc_coeff = float(load_path.split("_")[6])

    preprocessor = None
    if vrgoggles:
        sys.path.insert(1, 'cyclegan')
        from models import create_model
        from options.test_options import TestOptions
        import torchvision.transforms as transforms
        opt_parser = TestOptions()
        opt = opt_parser.parse()
        opt.name = "visual-cartpole_trial_" + str(i_trial) + "_cyclegan"
        # opt.direction = "BtoA"
        opt.model = "cycle_gan"
        opt.num_threads = 0
        opt.batch_size=num_envs
        opt.serial_batches = True
        opt.no_flip = True
        opt.display_id = -1
        opt.checkpoints_dir = "./cyclegan/checkpoints/"
        opt.dataroot = "./cyclegan/datasets/vc_trial_" + str(i_trial)
        opt.no_dropout = True
        opt.isTrain = False

        opt_parser.print_options(opt)

        model = create_model(opt)
        model.setup(opt)

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        preprocessor = lambda x: preprocess_fn(model, x, transform)

    print(env_name, i_trial)
    with tf.Graph().as_default():
        print("Testing on Source")

        if env_name == "visual-cartpole":
            venv = gym.vector.make('cartpole-visual-v1', num_envs=num_envs, num_levels=num_levels, start_level=source_level)
            venv.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
            venv.action_space = gym.spaces.Discrete(2)
        else:
            venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=source_level, distribution_mode=dist_mode)
            venv = VecExtractDictObs(venv, "rgb")


        venv = VecMonitor(
            venv=venv, filename=None, keep_buf=100,
        )

        if num_frames > 1:
            venv = VecFrameStack(venv, num_frames)

        venv = VecNormalize(venv=venv, ob=False)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        sess = tf.Session(config=config)
        sess.__enter__()

        conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

        source_obs, source_latents, source_rewards, source_labels = ppo2.evaluate(
            env=venv,
            network=conv_fn,
            total_timesteps=timesteps_per_proc,
            save_interval=save_interval,
            nsteps=nsteps,
            nminibatches=nminibatches,
            lam=lam,
            gamma=gamma,
            noptepochs=ppo_epochs,
            log_interval=1,
            ent_coef=ent_coef,
            mpi_rank_weight=mpi_rank_weight,
            clip_vf=use_vf_clipping,
            comm=comm,
            lr=learning_rate,
            cliprange=clip_range,
            update_fn=None,
            init_fn=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
            num_levels=num_levels,
            disc_coeff=disc_coeff,
            load_path=load_path,
            num_iterations = num_iterations
        )
    with tf.Graph().as_default():
        print("Testing on Target")

        if env_name == "visual-cartpole":
            venv = gym.vector.make('cartpole-visual-v1', num_envs=num_envs, num_levels=num_levels, start_level=target_level)
            venv.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
            venv.action_space = gym.spaces.Discrete(2)
        else:
            venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=target_level, distribution_mode=dist_mode)
            venv = VecExtractDictObs(venv, "rgb")


        venv = VecMonitor(
            venv=venv, filename=None, keep_buf=100,
        )

        if num_frames > 1:
            venv = VecFrameStack(venv, num_frames)

        venv = VecNormalize(venv=venv, ob=False)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.9
        config.gpu_options.allow_growth = True #pylint: disable=E1101
        sess = tf.Session(config=config)
        sess.__enter__()

        conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

        target_obs, target_latents, target_rewards, _ = ppo2.evaluate(
            env=venv,
            network=conv_fn,
            total_timesteps=timesteps_per_proc,
            save_interval=save_interval,
            nsteps=nsteps,
            nminibatches=nminibatches,
            lam=lam,
            gamma=gamma,
            noptepochs=ppo_epochs,
            log_interval=1,
            ent_coef=ent_coef,
            mpi_rank_weight=mpi_rank_weight,
            clip_vf=use_vf_clipping,
            comm=comm,
            lr=learning_rate,
            cliprange=clip_range,
            update_fn=None,
            init_fn=None,
            vf_coef=0.5,
            max_grad_norm=0.5,
            num_levels=num_levels,
            disc_coeff=disc_coeff,
            load_path=load_path,
            num_iterations = num_iterations,
            preprocessor = preprocessor
        )


    fig, (ax1) = plt.subplots(1, figsize=(5, 5))

    if visualization:
        source_latents = pd.DataFrame(source_latents)
        target_latents = pd.DataFrame(target_latents)
        all_latents = pd.concat([source_latents, target_latents])

        source_latents = (source_latents - all_latents.min()) / (all_latents.max() - all_latents.min())
        target_latents = (target_latents - all_latents.min()) / (all_latents.max() - all_latents.min())
        all_latents = pd.concat([source_latents, target_latents])

        print(np.mean(source_rewards), np.mean(target_rewards))
        print("ABOUT TO TRANSFORM LATENTS")
        pca = PCA(n_components=50)
        tsne = TSNE(n_components=2, n_jobs=-1, perplexity=5)
        all_latents_transformed = pca.fit_transform(all_latents.to_numpy())
        print("PCA DONE")
        all_latents_transformed = tsne.fit_transform(all_latents_transformed)
        source_latents = pd.DataFrame(all_latents_transformed[:len(source_latents)])
        target_latents = pd.DataFrame(all_latents_transformed[len(source_latents):])

        # source_latents["Environment"] = pd.Series(["source " + str(x) for x in source_labels])
        # target_latents["Environment"] = pd.Series(["target" for _ in range(len(target_latents))])
        # all_latents = pd.concat([source_latents, target_latents])
        # all_latents.columns = ["Component 1", "Component 2", "Environment"]


        # sns.scatterplot(x="Component 1", y="Component 2", hue="Environment", data=all_latents, alpha=0.1, marker=False, ax=ax1)
        # source_latents_smol = source_latents[:500]
        # target_latents_smol = target_latents[:500]

        # artists = []

        print("ABOUT TO ESTIMATE DENSITY")
        target_x, target_y, target_z = density_estimation(target_latents[0], target_latents[1])
        source_x, source_y, source_z = density_estimation(source_latents[0], source_latents[1])

        print("PLOTTING")

        step = 0.1
        m = np.amax(target_z)
        source_levels = np.arange(0.0, m, step)
        plt.contourf(target_x, target_y, target_z, 5, cmap='Purples', alpha=1.)

        m = np.amax(source_z)
        target_levels = np.arange(0.0, m, step)
        plt.contourf(source_x, source_y, source_z, 5, cmap='Greys', alpha=0.5)

        plt.contour(target_x, target_y, target_z, 5, colors="Purple")
        plt.contour(source_x, source_y, source_z, 5, colors="Grey")

        # artists.append(plt.scatter(source_latents_smol[0], source_latents_smol[1], color="grey", alpha=0.5, marker=".", label="Source"))
        # artists.append(plt.scatter(target_latents_smol[0], target_latents_smol[1], color="purple", alpha=0.5, marker=".", label="Target"))
        # plt.legend(artists)

        proxy_source = plt.Rectangle((0, 0), 1, 1, fc='Grey')
        proxy_target = plt.Rectangle((0, 0), 1, 1, fc='Purple')
        plt.legend([proxy_source, proxy_target], ["Source", "Target"])

        plt.title("WAPPO Features")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.xlim(-200., 200.)
        plt.ylim(-200., 200.)
        print("SAVING", i_trial)
        plt.savefig("scatter.png", bbox_inches="tight")

    if save_images:
        source_obs = source_obs[:100_000]
        target_obs = target_obs[:100_000]
        print(source_obs.shape, target_obs.shape)

        dir_name = "cyclegan/datasets/" + env_name + "_trial_" + str(i_trial) + "/"
        # dir_name = "/home/josh/mnt/cyclegan_datasets/" + env_name + "_trial_" + str(i_trial) + "/"
        dir_name_a = dir_name + "trainA/"
        dir_name_b = dir_name + "trainB/"

        Path(dir_name_a).mkdir(parents=True, exist_ok=True)
        Path(dir_name_b).mkdir(parents=True, exist_ok=True)


        print("print saving images: source, latent", source_obs.shape, target_obs.shape)
        for i, s_o in tqdm(enumerate(source_obs), total=len(source_obs)):
            plt.imsave(dir_name_a + "img" + str(i) + ".png", s_o)
        for i, t_o in tqdm(enumerate(target_obs), total=len(target_obs)):
            plt.imsave(dir_name_b + "img" + str(i) + ".png", t_o)
    if vrgoggles: # Evaluate VRGoggles
        with open("vrgoggles_out_" + str(env_name) + "_" + dist_mode + ".txt", 'a') as outfile:
            outfile.write(str(i_trial) + ", " + str(np.mean(source_rewards)) + ", " + str(np.mean(target_rewards)) + "\n")
        print("Source Reward", np.mean(source_rewards), "Target Reward", np.mean(target_rewards))

if __name__ == '__main__':
    main()
