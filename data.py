import random
from typing import List, Dict, Tuple


INTENTS = ["go_to", "follow", "hold_position"]
SPEEDS = ["slow", "normal", "fast"]
URGENCY = ["low", "normal", "high"]

VERBS = {
    "go_to": ["go to", "head to", "move to", "get to"],
    "follow": ["follow", "stay with", "stick with"],
    "hold_position": ["hold", "stay", "hold position"]
}

SPEED_TERMS = {
    "slow": ["slowly", "carefully", "quietly"],
    "normal": [""],
    "fast": ["quickly", "hurry", "run", "now"]
}

URGENCY_TERMS = {
    "low": ["when you can", "no rush"],
    "normal": [""],
    "high": ["now", "immediately", "right now"]
}


def generate_example() -> Tuple[str, Dict[str, int]]:
    intent = random.choice(INTENTS)
    speed = random.choice(SPEEDS)
    urgency = random.choice(URGENCY)

    verb = random.choice(VERBS[intent])
    speed_term = random.choice(SPEED_TERMS[speed])
    urgency_term = random.choice(URGENCY_TERMS[urgency])

    parts = [speed_term, verb, "Waypoint1", urgency_term]
    text = " ".join(p for p in parts if p).strip()

    labels = {
        "intent": INTENTS.index(intent),
        "speed": SPEEDS.index(speed),
        "urgency": URGENCY.index(urgency)
    }

    return text, labels


def build_dataset(n: int = 2000) -> Tuple[List[str], Dict[str, List[int]]]:
    texts = []
    labels = {
        "intent": [],
        "speed": [],
        "urgency": []
    }

    for _ in range(n):
        text, lab = generate_example()
        texts.append(text)
        for k in labels:
            labels[k].append(lab[k])

    return texts, labels
