import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# NeoMind neural network
# ============================================================

class NeoMind(nn.Module):
    def __init__(
        self,
        input_size=10,
        hidden1=32,
        hidden2=16,
        output_size=2,
    ):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

        # Brand-new Xavier initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# Create model
# ============================================================

model = NeoMind().to(device)

print(model)


# ============================================================
# Dataset
# ============================================================

# 100 samples
# 10 input features
# 2 target values

x = torch.randn(100, 10, device=device)
y = torch.randn(100, 2, device=device)


# ============================================================
# Training configuration
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
)

criterion = nn.MSELoss()

epochs = 100


# ============================================================
# Training loop
# ============================================================

model.train()

for epoch in range(epochs):

    optimizer.zero_grad()

    output = model(x)

    loss = criterion(output, y)

    loss.backward()

    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Loss: {loss.item():.6f}"
        )


# ============================================================
# Save trained weights
# ============================================================

weights_path = "neomind_weights.pth"

torch.save(
    model.state_dict(),
    weights_path,
)

print()
print("Training complete.")
print("Final loss:", loss.item())
print("Weights saved to:", weights_path)


# ============================================================
# Loading the trained weights later
# ============================================================

loaded_model = NeoMind().to(device)

loaded_model.load_state_dict(
    torch.load(
        weights_path,
        map_location=device,
        weights_only=True,
    )
)

loaded_model.eval()

print("Trained NeoMind weights loaded successfully.")
