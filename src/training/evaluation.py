import logging
import os
import pandas as pd
from src.utils.wrapper import get_env, CustomWrapper
from stable_baselines3.common.evaluation import evaluate_policy

logger = logging.getLogger(__name__)


def get_agent_folders(folder_to_agents):
    list_of_agents = [
        name
        for name in os.listdir(folder_to_agents)
        if os.path.isdir(os.path.join(folder_to_agents, name))
    ]
    for agent_folder in list_of_agents:
        # check if folder has file named "best_model.zip"
        if not os.path.isfile(
            os.path.join(folder_to_agents, agent_folder, "best_model.zip")
        ):
            logger.info(
                "Skipping {agent_folder} because it does not contain a best_model.zip file"
            )
            list_of_agents.remove(agent_folder)

        # check if folder has file that ends with .txt
        if not any(
            [
                file.endswith(".txt")
                for file in os.listdir(os.path.join(folder_to_agents, agent_folder))
            ]
        ):
            logger.info(
                f"Skipping {agent_folder} because it does not contain a txt file"
            )
            list_of_agents.remove(agent_folder)

    return list_of_agents


def get_all_possibilities(list_of_agents):
    model_vs_model = [
        (agent1, agent2, "weak")
        for agent1 in list_of_agents
        for agent2 in list_of_agents
    ]
    model_vs_strong = [(agent1, None, "strong") for agent1 in list_of_agents]
    model_vs_weak = [(agent1, None, "weak") for agent1 in list_of_agents]
    return model_vs_model + model_vs_strong + model_vs_weak


def evaluate_model_vs_model(
    agent_1, agent_2, config, DISCRETE_ACTION_SPACE, weak_opponent=True
):
    model1 = CustomWrapper.load_model_from_disk(agent_1)
    eval_env = get_env(
        mode=CustomWrapper.NORMAL,
        discrete_action_space=DISCRETE_ACTION_SPACE,
        negative_reward=True,
        weak=weak_opponent,
        n_envs=config["evaluation"]["n_eval_envs"],
        start_method=config["evaluation"]["start_method"],
    )
    eval_env.render_mode = config["evaluation"]["render_mode"]

    if isinstance(agent_2, str):
        eval_env.env_method("set_opponent", agent_2)
    else:
        agent_2 = "Weak Opponent" if weak_opponent else "Strong Opponent"

    obs = eval_env.reset()
    logger.info(f"Evaluating {agent_1} vs {agent_2}")
    mean_reward, std_reward = evaluate_policy(
        model1,
        eval_env,
        n_eval_episodes=config["evaluation"]["n_eval_episodes"],
        deterministic=True,
    )
    logger.info(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
    return mean_reward, std_reward


def evaluate(config):
    list_of_agents = get_agent_folders(config["evaluation"]["folder_to_agents"])
    DISCRETE_ACTION_SPACE = config["evaluation"]["discrete_action_space"]
    all_players = ["weak", "strong"] + list_of_agents
    df_mean = pd.DataFrame(columns=all_players, index=all_players)
    df_std = pd.DataFrame(columns=all_players, index=all_players)
    base_path = config["evaluation"]["folder_to_agents"]

    matches = get_all_possibilities(list_of_agents)
    for agent1, agent2, opponent in matches:
        if agent2 == None:
            agent1_path = f"{base_path}/{agent1}"
            print(agent1_path)
            mean_reward, std_reward = evaluate_model_vs_model(
                agent1_path, agent2, config, DISCRETE_ACTION_SPACE, opponent
            )
            df_mean.loc[agent1, opponent] = mean_reward
            df_mean.loc[opponent, agent1] = mean_reward
            df_std.loc[agent1, opponent] = std_reward
            df_std.loc[opponent, agent1] = std_reward
        else:
            agent1_path = f"{base_path}/{agent1}"
            agent2_path = f"{base_path}/{agent2}"
            print(agent1_path)
            print(agent2_path)
            mean_reward, std_reward = evaluate_model_vs_model(
                agent1_path, agent2_path, config, DISCRETE_ACTION_SPACE, opponent
            )
            df_mean.loc[agent1, agent2] = mean_reward
            df_mean.loc[agent2, agent1] = mean_reward
            df_std.loc[agent1, agent2] = std_reward
            df_std.loc[agent2, agent1] = std_reward

        save_location = config["evaluation"]["save_location"]
        # check if folder exists
        if not os.path.isdir(save_location):
            os.makedirs(save_location)

        save_path_mean = os.path.join(save_location, "df_mean.csv")
        save_path_std = os.path.join(save_location, "df_std.csv")
        df_mean.to_csv(save_path_mean)
        df_std.to_csv(save_path_std)
