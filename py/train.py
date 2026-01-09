import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from data import build_dataset
from model import BowIntentNet


EPOCHS = 10
BATCH_SIZE = 32
HIDDEN_DIM = 128
LR = 0.01


def main():
    texts, labels = build_dataset(4000)

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts).toarray().astype(np.float32)

    y_intent = np.array(labels["intent"], dtype=np.int64)
    y_speed = np.array(labels["speed"], dtype=np.int64)
    y_urgency = np.array(labels["urgency"], dtype=np.int64)

    X = torch.from_numpy(X)
    y_intent = torch.from_numpy(y_intent)
    y_speed = torch.from_numpy(y_speed)
    y_urgency = torch.from_numpy(y_urgency)

    model = BowIntentNet(vocab_size=X.shape[1], hidden_dim=HIDDEN_DIM)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        perm = torch.randperm(X.shape[0])
        total_loss = 0.0

        for start in range(0, X.shape[0], BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            xb = X[idx]
            yi = y_intent[idx]
            ys = y_speed[idx]
            yu = y_urgency[idx]

            out = model(xb)

            loss = (
                loss_fn(out["intent"], yi) +
                loss_fn(out["speed"], ys) +
                loss_fn(out["urgency"], yu)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{EPOCHS} loss={total_loss:.3f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab": vectorizer.vocabulary_,
            "hidden_dim": HIDDEN_DIM
        },
        "intent_model.pt"
    )
    print("Saved intent_model.pt")


if __name__ == "__main__":
    main()
