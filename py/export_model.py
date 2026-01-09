import torch
import numpy as np
from model import BowIntentNet


def main():
    ckpt = torch.load("intent_model.pt", map_location="cpu")

    vocab = ckpt["vocab"]
    hidden_dim = int(ckpt.get("hidden_dim", 128))

    model = BowIntentNet(
        vocab_size=len(vocab),
        hidden_dim=hidden_dim
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # AI: Dummy input = Bag-of-Words vector
    dummy_input = torch.zeros(1, len(vocab), dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        "intent_model.onnx",
        input_names=["bow_input"],
        output_names=["intent", "speed", "urgency"],
        dynamic_axes={
            "bow_input": {0: "batch"},
            "intent": {0: "batch"},
            "speed": {0: "batch"},
            "urgency": {0: "batch"}
        },
        opset_version=15
    )

    print("Exported intent_model.onnx")


if __name__ == "__main__":
    main()
