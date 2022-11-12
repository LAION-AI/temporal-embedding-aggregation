import os
import json
import logging

import torch
from aggregation.aggregator_wrapper import VideoCLIP
from . import * # TODO: only take models

def get_model_string(m_dict):
    s = m_dict["type"]
    for k, v in m_dict["args"].items():
        s += f"_{k}_{v}"
    return s


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    # resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(cfg_path, pretrained=''):
    with open(cfg_path, "r") as f:
        model_dict = json.load(f)

    model_name = model_dict["type"]
    model_cls = eval(model_name)
    model = model_cls(**model_dict["args"])

    # TODO: implement getting pretrained from releases like open_clip
    pretrained_cfg = {}
    if pretrained:
        checkpoint_path = ''
        if os.path.exists(pretrained):
            checkpoint_path = pretrained

        if checkpoint_path:
            logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
            load_checkpoint(model, checkpoint_path)
        else:
            logging.warning(f'Pretrained weights ({pretrained}) not found for model {model_name}.')
            raise RuntimeError(f'Pretrained weights ({pretrained}) not found for model {model_name}.')
    
    return VideoCLIP(model), get_model_string(model_dict)
