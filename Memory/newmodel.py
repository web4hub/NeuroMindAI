import torch, torch.nn as nn, torch.optim as optim

model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
opt = optim.Adam(model.parameters(), lr=1e-3)

for step in range(1000):
    x = torch.randn(16, 10)
    y = x.sum(dim=1, keepdim=True)
    loss = ((model(x) - y)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()

model = MyNeuralNet()
# weights are random now
train(model, my_dataset)  # trains and updates the weights
torch.save(model.state_dict(), "my_new_weights.pth")

import torch
import torch.nn as nn
import torch.nn.functional as F

class NeoMind(nn.Module):
    def __init__(self, input_size=10, hidden1=32, hidden2=16, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

        # initialize brand-new weights (not copied from anywhere)
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

model = NeoMind()
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# fake dataset: 100 samples of 10 features → 2 targets
x = torch.randn(100, 10)
y = torch.randn(100, 2)

for epoch in range(100):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

print("Training complete, final loss:", loss.item())
