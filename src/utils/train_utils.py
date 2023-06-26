import ast
from typing import Any, Dict


def compare_params(params: Dict[str, Any], runs) -> bool:
    """Compares a current hyperparameter configuration with all previous runs. If a similar run is found, the reward of the previous run is returned. Precision for float values is 0.001."""
    for i in range(len(runs)):
        tmp = runs["params"].iloc[i]
        tmp = tmp.replace("'", '"').replace("<class ", "").replace(">", "")
        tmp = ast.literal_eval(tmp)
        tmp["normalize"] = runs["normalize"].iloc[i]
        tmp["negative_reward"] = runs["negativ_reward"].iloc[i]
        tmp["discrete_action_space"] = runs["discrete_action_space"].iloc[i]
        similar = compare_run(params, tmp)
        reward = runs["mean_reward"].iloc[i]
        if similar:
            return True, reward
    return False, 0


def compare_run(params: Dict[str, Any], run, precision=0.001) -> bool:
    if "trial_number" in params:
        del params["trial_number"]
    if "device" in params:
        del params["device"]
    if "steps" in params:
        del params["steps"]
    for key in params.keys():
        if key in run.keys():
            value = params[key]
            run_key = run[key]
            if isinstance(value, float):
                if abs(value - run_key) > precision:
                    return False
            elif isinstance(value, dict) and "net_arch" in value.keys():
                pi1, vf1 = value["net_arch"]["pi"], value["net_arch"]["vf"]
                pi2, vf2 = run_key["net_arch"]["pi"], run_key["net_arch"]["vf"]
                afn = (
                    str(value["activation_fn"]).replace("<class ", "").replace(">", "")
                )
                afn2 = run_key["activation_fn"]
                if pi1 != pi2 or vf1 != vf2 or afn != afn2:
                    return False
            elif value != run_key:
                return False
    print("------------------------------")
    print(f"Found similar run: {run}")
    print("------------------------------")
    return True
