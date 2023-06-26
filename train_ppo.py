import os
import time
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.ppo.policies import MlpPolicy as PPO_MlpPolicy
from torch import nn as nn

from src.utils.wrapper import CustomWrapper, ModelWrapper, make_env

OUTDIR = "./ppo_training/"
os.makedirs(OUTDIR, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
FILENAME = os.path.join(OUTDIR, f"result_{TIMESTAMP}.csv")
LOG_DIR = "./hypersearch_ppo7/"
OUT_FILE = os.path.join(OUTDIR, f"ppo_{TIMESTAMP}")
OUT_FILE_ENV = os.path.join(OUTDIR, f"ppo_{TIMESTAMP}_env.pkl")

# env parameters
N_PROCS = 128
DISCRETE_ACTION_SPACE = False
NEGATIVE_REWARD = True
NORMALIZE = False

# takes about 2h on 16 cores and rtx 3080ti
# model parameters
N_STEPS = 2048  # steps per environment
BATCH_SIZE = 512  # batch size for training
GAMMA = 0.999
POLICY_KWARGS = {"net_arch": [256, 256, 64]}
TOTAL_TIMESTEPS = 25_000_000


## 9.5600
DISCRETE_ACTION_SPACE = False
NEGATIVE_REWARD = False
NORMALIZE = True
# model parameters
N_STEPS = 256  # steps per environment
BATCH_SIZE = 512  # batch size for training
GAMMA = 0.99
POLICY_KWARGS = {
    "net_arch": {"pi": [64, 64], "vf": [64, 64]},
    "activation_fn": nn.LeakyReLU,
    "ortho_init": True,
}
TOTAL_TIMESTEPS = 15_000_000
N_EVALUATIONS = 15

EVAL_FREQ = int(TOTAL_TIMESTEPS / (N_EVALUATIONS * N_PROCS))


def train():
    start_time = time.time()
    train_env = SubprocVecEnv(
        [
            make_env(
                i,
                mode=None,
                discrete_action_space=DISCRETE_ACTION_SPACE,
                negativ_reward=NEGATIVE_REWARD,
            )
            for i in range(N_PROCS)
        ],
        start_method="fork",
    )
    eval_env = SubprocVecEnv(
        [
            make_env(
                i,
                mode=CustomWrapper.NORMAL,
                discrete_action_space=DISCRETE_ACTION_SPACE,
                negativ_reward=True,
            )
            for i in range(10)
        ],
        start_method="fork",
    )
    if NORMALIZE:
        train_env = VecNormalize(
            train_env,
            norm_obs=True,
            norm_reward=True,
        )
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            training=False,
        )

    train_env.reset()
    eval_env.reset()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=OUTDIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        deterministic=True,
        render=False,
        verbose=1,
        n_eval_episodes=500,
    )
    # model = PPO(PPO_MlpPolicy, train_env, gamma=GAMMA, n_epochs=3, verbose=1, n_steps=N_STEPS, batch_size=BATCH_SIZE, policy_kwargs=POLICY_KWARGS, tensorboard_log=LOG_DIR)
    model = PPO(
        PPO_MlpPolicy,
        train_env,
        vf_coef=0.6120,
        gae_lambda=0.9200,
        clip_range=0.1,
        n_epochs=5,
        ent_coef=7.582569237243832e-08,
        max_grad_norm=0.9,
        gamma=GAMMA,
        learning_rate=0.000880,
        verbose=1,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=LOG_DIR,
    )
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS, tb_log_name="optimized", callback=eval_callback
    )
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=1000)

    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    end_time = time.time()
    training_time = end_time - start_time
    training_time /= 60
    print(f"Training took {training_time} min")
    return model


if __name__ == "__main__":
    model = train()
    model.save(OUT_FILE)
    if NORMALIZE:
        model.get_vec_normalize_env().save(OUT_FILE_ENV)
    del model

    env = SubprocVecEnv(
        [
            make_env(
                i,
                mode=CustomWrapper.NORMAL,
                discrete_action_space=False,
                negativ_reward=True,
            )
            for i in range(2)
        ],
        start_method="fork",
    )
    env.render_mode = "human"
    obs = env.reset()
    model = PPO.load(OUT_FILE)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=1000)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    for i in range(100):
        for _ in range(10000):
            env.render()
            a1 = model.predict(obs)[0]
            obs, r, d, info = env.step(a1)
            if any(d):
                break
            time.sleep(0.01)
        obs = env.reset()
