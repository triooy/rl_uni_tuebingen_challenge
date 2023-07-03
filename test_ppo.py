import time

from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from torch import nn as nn

from src.utils.wrapper import CustomWrapper, ModelWrapper, make_env

ID = 11
OUT_DIR = f"serverlogs/selfplay3/trial_{ID}/"    #f"./selfplay/trial_{ID}/"
OUT_FILE = f"{OUT_DIR}best_model"
NORMALIZE = True
DISCRETE_ACTION_SPACE = False
NEGATIVE_REWARD = True
N_procs = 128

OPPNENT_ID = 12
OPPONENT_DIR = f"serverlogs/selfplay3/trial_{OPPNENT_ID}/"
OPPNENT_FILE = f"{OPPONENT_DIR}best_model"
OPPNENT_NORMALIZE = True
OPPONENT = True

if __name__ == "__main__":
    eval_env = SubprocVecEnv(
        [
            make_env(
                i,
                mode=CustomWrapper.NORMAL,
                discrete_action_space=DISCRETE_ACTION_SPACE,
                negativ_reward=True,
                weak=False,
            )
            for i in range(2)
        ],
        start_method="fork",
    )
    eval_env.render_mode = "human"
    obs = eval_env.reset()
    
    model = ModelWrapper.load(OUT_FILE, print_system_info=True, device="cuda")
    if NORMALIZE:
        model.load_env(
            NORMALIZE,
            path=OUT_DIR + "best_model_env.pkl",
        )
    if OPPONENT:
        eval_env.env_method("set_opponent", OPPNENT_FILE)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=300, deterministic=True
    )
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    
    for i in range(1000):
        for _ in range(10000):
            eval_env.render()
            a1 = model.predict(obs, deterministic=True)[0]
            obs, r, d, info = eval_env.step(a1)
            if any(d):
                break
            time.sleep(0.01)
        obs = eval_env.reset()
