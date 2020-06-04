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
# import gym_cartpole_visual
import numpy as np


def main():
    num_envs = 64
    learning_rate = 5e-4
    ent_coef = .01
    gamma = .999
    lam = .95
    nminibatches = 8
    nsteps = (2048 // nminibatches)
    ppo_epochs = 3
    clip_range = .2
    use_vf_clipping = True
    dist_mode = "hard"

    indicator = int(os.environ["SGE_TASK_ID"]) - 1

    source_levels = [1543, 7991, 3671, 2336, 6420]

    target_levels = [7354, 9570, 6317, 6187, 8430]
    env_names = ["bigfish", "bossfight", "caveflyer", "chaser", "climber", "coinrun", "dodgeball", "fruitbot", "heist", "jumper", "leaper", "maze", "miner", "ninja", "plunder", "starpilot"]

    i_trial = indicator % len(target_levels)
    if i_trial <= 2:
        sys.exit()
    i_trial_str = str(i_trial)

    source_level = source_levels[i_trial]
    target_level = target_levels[i_trial]


    i_env = indicator // len(target_levels)
    env_name = env_names[i_env]

    if env_name in ["maze", "plunder", "leaper", "miner"]:
        sys.exit()
    num_frames = 1

    disc_coeff = 10.

    if env_name == "visual-cartpole":
        timesteps_per_proc = 1_000_000
        save_interval=10
    elif dist_mode == "easy":
        timesteps_per_proc = 25_000_000
        save_interval=100
    else:
        timesteps_per_proc = 200_000_000
        save_interval=100


    num_levels = 1
    num_test_levels = 1

    # LOG_DIR = "/home/jroy1/procgen_combined_" + dist_mode + "/" + env_name + "_disc_coeff_" + str(disc_coeff) + "_num_levels_" + str(num_levels) + "_nsteps_" + str(nsteps) + "_num_frames_" + str(num_frames) + "_num_test_levels_" + str(num_test_levels)
    # LOG_DIR = "/home/jroy1/procgen_adaptive_sigmoid_" + dist_mode + "/" + env_name + "_num_levels_" + str(num_levels) + "_nsteps_" + str(nsteps) + "_num_frames_" + str(num_frames) + "_num_test_levels_" + str(num_test_levels)
    # LOG_DIR = "/home/jroy1/procgen_randomfeatures_" + dist_mode + "/" + env_name + "_num_levels_" + str(num_levels) + "_nsteps_" + str(nsteps) + "_num_frames_" + str(num_frames) + "_num_test_levels_" + str(num_test_levels)
    # LOG_DIR = "/home/jroy1/procgen_combined_testing_" + dist_mode + "/" + env_name + "_disc_coeff_" + str(disc_coeff) + "_num_levels_" + str(num_levels) + "_nsteps_" + str(nsteps) + "_num_frames_" + str(num_frames) + "_num_test_levels_" + str(num_test_levels)
    # LOG_DIR = "/home/jroy1/procgen_combined_testing_flipped_disc_coeff_" + dist_mode + "/" + env_name + "_disc_coeff_" + str(disc_coeff) + "_num_levels_" + str(num_levels) + "_nsteps_" + str(nsteps) + "_num_frames_" + str(num_frames) + "_num_test_levels_" + str(num_test_levels)
    # LOG_DIR = "/home/jroy1/procgen_vanilla" + dist_mode + "/" + env_name + "_disc_coeff_" + str(disc_coeff) + "_num_levels_" + str(num_levels) + "_nsteps_" + str(nsteps) + "_num_frames_" + str(num_frames) + "_num_test_levels_" + str(num_test_levels)
    LOG_DIR = "/data/people/jroy1/procgen_wconf_" + dist_mode + "/" + env_name + "_disc_coeff_" + str(disc_coeff) + "_num_levels_" + str(num_levels) + "_nsteps_" + str(nsteps) + "_num_frames_" + str(num_frames) + "_num_test_levels_" + str(num_test_levels)
    # LOG_DIR += "_robustdr"
    # LOG_DIR += "_rmsprop_wgan_same_trainer"
    # LOG_DIR += "_disc10"
    LOG_DIR += "_trial_" + i_trial_str

    test_worker_interval = 0

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    is_test_worker = False

    if test_worker_interval > 0:
        is_test_worker = comm.Get_rank() % test_worker_interval == (test_worker_interval - 1)

    mpi_rank_weight = 0 if is_test_worker else 1

    log_comm = comm.Split(1 if is_test_worker else 0, 0)
    format_strs = ['csv', 'stdout', 'tensorboard'] if log_comm.Get_rank() == 0 else []
    logger.configure(dir=LOG_DIR, format_strs=format_strs)

    logger.info("creating environment")

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

    if env_name == "visual-cartpole":
        test_venv = gym.vector.make('cartpole-visual-v1', num_envs=num_envs, num_levels=num_test_levels, start_level=target_level)
        test_venv.observation_space = gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        test_venv.action_space = gym.spaces.Discrete(2)
    else:
        test_venv = ProcgenEnv(num_envs=num_envs, env_name=env_name, num_levels=num_test_levels, start_level=target_level, distribution_mode=dist_mode)
        test_venv = VecExtractDictObs(test_venv, "rgb")

    if num_frames > 1:
        test_venv = VecFrameStack(test_venv, num_frames)

    test_venv = VecMonitor(
        venv=test_venv, filename=None, keep_buf=100,
    )
    test_venv = VecNormalize(venv=test_venv, ob=False)

    logger.info("creating tf session")
    # setup_mpi_gpus()
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True #pylint: disable=E1101
    sess = tf.Session(config=config)
    sess.__enter__()

    conv_fn = lambda x: build_impala_cnn(x, depths=[16,32,32], emb_size=256)

    logger.info("training")
    ppo2.learn(
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
        eval_env=test_venv,
        num_levels=num_levels,
        disc_coeff=disc_coeff,
    )

if __name__ == '__main__':
    main()
