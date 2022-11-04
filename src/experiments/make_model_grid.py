"""Makes grid of model configs and writes to model_configs dir as json."""
import json

from itertools import product


EXPERIMENT_NAME = "H14_depth_run"
MODEL_GRID = {
    "type": "SelfAttentionalPooler",
    "args": {
        "dim": [1024],
        "context_dim": [1024],
        "seq_len": [200],
        "heads": [8],
        "dim_head": [64],
        "depth": [1, 2, 5, 10],
        "mlp_dim": [1024],
        "proj_dim": [-1],
        "dropout": [0.0],
    }
}


if __name__ == "__main__":
    model_type = MODEL_GRID["type"]
    params = MODEL_GRID["args"]
    keys, values = zip(*params.items()) 

    prod = product(*values)
    full_configs =  []
    i = 0
    for ps in prod:
        arg_conf = dict(zip(keys, ps))
        model_conf = {"type": model_type, "args": arg_conf}

        with open(f"model_configs/{EXPERIMENT_NAME}_{i}.json", "w") as out:
            json.dump(model_conf, out, indent=4)

        i += 1
