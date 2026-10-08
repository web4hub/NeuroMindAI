"""Checkpoint persistence helpers."""
from pathlib import Path
import torch
def save_checkpoint(path,model,optimizer=None,step=0,**extra):
    payload={"config":vars(model.config),"state_dict":model.state_dict(),"step":step,**extra}
    if optimizer is not None: payload["optimizer_state_dict"]=optimizer.state_dict()
    Path(path).parent.mkdir(parents=True,exist_ok=True); torch.save(payload,path)
def load_checkpoint(path,model,optimizer=None,map_location="cpu"):
    p=torch.load(path,map_location=map_location,weights_only=False); model.load_state_dict(p["state_dict"])
    if optimizer is not None and "optimizer_state_dict" in p: optimizer.load_state_dict(p["optimizer_state_dict"])
    return p
