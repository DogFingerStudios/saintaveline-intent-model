import torch
import json
ckpt = torch.load("intent_model.pt", map_location="cpu")
with open("vocab.json", "w") as f:
    json.dump(ckpt["vocab"], f, indent=2)
