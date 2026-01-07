import json
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from model import BowIntentNet


INTENTS = ["go_to", "follow", "hold_position"]
SPEEDS = ["slow", "normal", "fast"]
URGENCY = ["low", "normal", "high"]


def softmax_confidence(logits: torch.Tensor):
    probs = torch.softmax(logits, dim=-1)
    conf, idx = torch.max(probs, dim=-1)
    return conf.item(), idx.item(), probs.squeeze(0).detach().cpu().numpy()


def main():
    ckpt = torch.load("intent_model.pt", map_location="cpu")

    vocab = ckpt["vocab"]
    hidden_dim = int(ckpt.get("hidden_dim", 128))

    vectorizer = CountVectorizer(vocabulary=vocab)

    model = BowIntentNet(vocab_size=len(vocab), hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("Type a command (or 'exit'):\n")

    while True:
        text = input("> ").strip()
        if text.lower() in ["exit", "quit"]:
            break

        bow = vectorizer.transform([text]).toarray().astype(np.float32)
        x = torch.from_numpy(bow)

        with torch.no_grad():
            out = model(x)

        intent_conf, intent_idx, intent_probs = softmax_confidence(out["intent"])
        speed_conf, speed_idx, speed_probs = softmax_confidence(out["speed"])
        urg_conf, urg_idx, urg_probs = softmax_confidence(out["urgency"])

        result = {
            "text": text,
            "intent": {
                "label": INTENTS[intent_idx],
                "confidence": round(intent_conf, 4),
                "probs": {INTENTS[i]: round(float(intent_probs[i]), 4) for i in range(len(INTENTS))}
            },
            "speed": {
                "label": SPEEDS[speed_idx],
                "confidence": round(speed_conf, 4),
                "probs": {SPEEDS[i]: round(float(speed_probs[i]), 4) for i in range(len(SPEEDS))}
            },
            "urgency": {
                "label": URGENCY[urg_idx],
                "confidence": round(urg_conf, 4),
                "probs": {URGENCY[i]: round(float(urg_probs[i]), 4) for i in range(len(URGENCY))}
            }
        }

        print(json.dumps(result, indent=2))
        print("")


if __name__ == "__main__":
    main()
