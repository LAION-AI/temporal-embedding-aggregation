import json
from . import * # TODO: only take models


def get_model_string(m_dict):
    s = m_dict["type"]
    for k, v in m_dict["args"].items():
        s += f"_{k}_{v}"
    return s

def create_model(cfg_path):
    with open(cfg_path, "r") as f:
        model_dict = json.load(f)

    model_cls = eval(model_dict["type"])
    model = model_cls(**model_dict["args"])
    return model, get_model_string(model_dict)
