from src.training.trainer import Trainer


def train(config) -> Trainer:
    agents_kwargs = config["agent_parameter"]

    trainer_kwargs = (
        config["training"] | config["agent"] | config["logs"] | config["selfplay"]
    )

    trainer_kwargs["agents_kwargs"] = agents_kwargs
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    return trainer
