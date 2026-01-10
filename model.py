import torch
import torch.nn as nn


class BowIntentNet(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int = 128):
        super().__init__()

        self.fc1 = nn.Linear(vocab_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        self.intent_head = nn.Linear(hidden_dim, 3)
        self.speed_head = nn.Linear(hidden_dim, 3)
        self.urgency_head = nn.Linear(hidden_dim, 3)

    def forward(self, x):
        # x: [batch, vocab_size] float
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))

        return {
            "intent": self.intent_head(h),
            "speed": self.speed_head(h),
            "urgency": self.urgency_head(h),
        }
