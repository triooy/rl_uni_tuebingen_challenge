from src.training.trainer import Trainer
from torch import nn

activation_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "leaky_relu": nn.LeakyReLU}


def train(config) -> Trainer:
    agents_kwargs = config["agent_parameter"]
    hindsight_replay = (
        config["hindsight_replay"] if "hindsight_replay" in config else {}
    )
    trainer_kwargs = (
        config["training"]
        | config["agent"]
        | config["logs"]
        | config["selfplay"]
        | hindsight_replay
    )

    # check for custom policy kwargs
    if "policy_kwargs" in config["agent_parameter"]:
        policy_kwargs = config["agent_parameter"]["policy_kwargs"]
        if "activation_fn" in policy_kwargs:
            policy_kwargs["activation_fn"] = activation_fn[
                policy_kwargs["activation_fn"]
            ]
            agents_kwargs["policy_kwargs"] = policy_kwargs

    trainer_kwargs["agents_kwargs"] = agents_kwargs

    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    return trainer
