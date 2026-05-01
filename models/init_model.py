import torch
import numpy as np
import os

from .KeenKT import KEENKT

# from .folibikt import folibiKT

device = "cpu" if not torch.cuda.is_available() else "cuda"


def init_model(model_name, model_config, data_config, emb_type):
    if model_name == "dkt":
        model = DKT(data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "sakt":
        model = SAKT(data_config["num_c"],  **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "saint":
        model = SAINT(data_config["num_q"], data_config["num_c"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "akt":
        model = AKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "iekt":
        model = IEKT(num_q=data_config['num_q'], num_c=data_config['num_c'],
                max_concepts=data_config['max_concepts'], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"],device=device).to(device)
    elif model_name == "simplekt":
        model = simpleKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    elif model_name == "KeenKT":
        model = KEENKT(data_config["num_c"], data_config["num_q"], **model_config, emb_type=emb_type, emb_path=data_config["emb_path"]).to(device)
    else:
        # print("The wrong model name was used...")
        return None
    return model

def load_model(model_name, model_config, data_config, emb_type, ckpt_path):
    model = init_model(model_name, model_config, data_config, emb_type)
    pth_path = os.path.join(ckpt_path, emb_type + "_model.pth")
    ckpt_legacy_path = os.path.join(ckpt_path, emb_type + "_model.ckpt")

    if os.path.exists(pth_path):
        weight_path = pth_path
    elif os.path.exists(ckpt_legacy_path):
        weight_path = ckpt_legacy_path
    else:
        pth_candidates = sorted(
            [
                os.path.join(ckpt_path, fn)
                for fn in os.listdir(ckpt_path)
                if fn.endswith(".pth")
            ],
            key=os.path.getmtime,
            reverse=True,
        )
        if not pth_candidates:
            raise FileNotFoundError(
                f"No checkpoint found in {ckpt_path}. "
                f"Tried {pth_path}, {ckpt_legacy_path}, and any *.pth files."
            )
        weight_path = pth_candidates[0]

    net = torch.load(weight_path, map_location=device)
    model.load_state_dict(net)
    print(f"Loaded checkpoint: {weight_path}")
    return model
