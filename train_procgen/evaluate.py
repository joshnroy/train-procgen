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

from pyvirtualdisplay import Display
display = Display(visible=0, size=(100, 100), backend="xvfb")
display.start()
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.decomposition import PCA

import random
random.seed(0)
np.random.seed(0)


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

    # if int(os.environ["SGE_TASK_ID"]) not in [8, 10, 11, 12, 13, 14, 15, 16]:
    #     sys.exit()

    if "SGE_TASK_ID" in os.environ:
        indicator = int(os.environ["SGE_TASK_ID"]) - 1
        i_trial = indicator % len(target_levels)
        i_env = indicator // len(target_levels)
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--i_trial", help="trial number", required=True)
        # parser.add_argument("--i_env", help="env number", required=True)
        args = parser.parse_args()
        i_trial = int(args.i_trial)
        # i_env = int(args.i_env)


    source_levels = [1543, 7991, 3671, 2336, 6420]
    source_level = source_levels[i_trial]

    target_levels = [7354, 9570, 6317, 6187, 8430]
    target_level = target_levels[i_trial]

    env_names = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "fruitbot", "heist", "jumper", "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]

    env_name = 'visual-cartpole'
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


    num_levels = 3
    num_test_levels = 0
    num_iterations = 200

    disc_coeff = 10.0

    test_worker_interval = 0

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False

    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)

    mpi_rank_weight = 0 if is_test_worker else 1

    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout', 'tensorboard'] if log_comm.Get_rank() == 0 else []

    load_path = "vc_cannonical_easy/visual-cartpole_disc_coeff_10.0_num_levels_3_nsteps_256_num_frames_1_num_test_levels_0_rmsprop_wgan_same_trainer_trial_0/checkpoints/00300"

    with tf.Graph().as_default():
        print("Testing on Source")

        if env_name == "visual-cartpole":
            venv = gym.vector.make('cartpole-visual-v1', num_envs=num_envs, num_levels=num_levels, start_level=1543)
            venv.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
            venv.action_space = gym.spaces.Discrete(2)
        else:
            venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_levels, start_level=1543, distribution_mode=dist_mode)
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

        _, source_latents, _, source_labels = ppo2.evaluate(
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

        _, target_latents, _, _ = ppo2.evaluate(
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
            num_iterations = num_iterations // num_levels
        )


    fig, (ax1) = plt.subplots(1, figsize=(10, 10))


    source_latents = pd.DataFrame(source_latents)
    target_latents = pd.DataFrame(target_latents)
    all_latents = pd.concat([source_latents, target_latents])

    source_latents = (source_latents - all_latents.min()) / (all_latents.max() - all_latents.min())
    target_latents = (target_latents - all_latents.min()) / (all_latents.max() - all_latents.min())
    all_latents = pd.concat([source_latents, target_latents])

    pca = PCA(n_components=2)
    pca.fit(all_latents.to_numpy())
    source_latents = pd.DataFrame(pca.transform(source_latents.to_numpy()))
    target_latents = pd.DataFrame(pca.transform(target_latents.to_numpy()))

    source_latents["Environment"] = pd.Series(["source " + str(x) for x in source_labels])
    target_latents["Environment"] = pd.Series(["target" for _ in range(len(target_latents))])
    all_latents = pd.concat([source_latents, target_latents])
    all_latents.columns = ["Component 1", "Component 2", "Environment"]


    sns.scatterplot(x="Component 1", y="Component 2", hue="Environment", data=all_latents, alpha=0.1, marker=False, ax=ax1)
    plt.savefig("scatter.png")




if __name__ == '__main__':
    main()
