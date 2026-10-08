# =====================================================
# EDQ + Brain Hybrid AI with Quantum-Inspired Entropy
# Dynamic multi-repo training from scratch
# Author: Seriki Yakub
# =====================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import re

# Optional quantum simulation
use_qiskit = False
try:
    from qiskit import QuantumCircuit, Aer, execute
    use_qiskit = True
except:
    use_qiskit = False

# ------------------------------
# 1️⃣ Dynamic repos list
# ------------------------------
REPO_URLS = [
    "https://github.com/Web4application/EDQ-AI",
    "https://github.com/Web4application/Brain",
    "https://github.com/Web4application/enclov-AI",
    "https://github.com/Web4application/swiftbot",
    "https://github.com/Web4application/reallms",
    "https://github.com/Web4application/kubu-hai.model.h5",
    "https://github.com/Web4application/SERAI",
    "https://github.com/Web4application/pilot_ai",
    "https://github.com/Web4application/RODAAI",
    "https://github.com/Web4application/congen-ai",
    "https://github.com/Web4application/Lola-AI-Assistant",
    "https://github.com/Web4application/project-upgrader-ai",
    "https://github.com/Web4application/qubuhub-voice-narrator",
    "https://github.com/Web4application/Ogugu",
    "https://github.com/Web4application/AgbakoAI",
    "https://github.com/Web4application/Neuralog",
    "https://github.com/Web4application/Model",
    "https://github.com/Web4application/roda_prompt_generator",
    "https://github.com/Web4application/Lola"
]

def clone_repos_dynamic(base_dir="repos"):
    os.makedirs(base_dir, exist_ok=True)
    for url in REPO_URLS:
        repo_name = url.rstrip("/").split("/")[-1]
        repo_path = os.path.join(base_dir, repo_name)
        if not os.path.exists(repo_path):
            os.system(f"git clone {url} {repo_path}")
        else:
            os.system(f"cd {repo_path} && git pull")
    print(f"✅ {len(REPO_URLS)} repos cloned/synced successfully")

clone_repos_dynamic()

# ------------------------------
# 2️⃣ Load and tokenize repo text
# ------------------------------
def read_texts_from_repo(repo_path):
    texts = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(('.txt', '.md', '.py', '.json')):
                try:
                    content = open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore').read()
                    texts.append(content)
                except:
                    pass
    return texts

def tokenize_texts(texts, vocab=None, max_len=32):
    if vocab is None:
        vocab = {}
    tokenized = []
    for text in texts:
        words = re.findall(r'\b\w+\b', text.lower())
        encoded = [vocab.setdefault(w, len(vocab)+1) for w in words]
        tokenized.append(encoded[:max_len])
    return tokenized, vocab

def generate_repo_data():
    all_texts = []
    for url in REPO_URLS:
        repo_name = url.rstrip("/").split("/")[-1]
        repo_path = os.path.join("repos", repo_name)
        all_texts.extend(read_texts_from_repo(repo_path))
    tokenized, vocab = tokenize_texts(all_texts)
    vocab_size = len(vocab)+1
    max_len = max(len(t) for t in tokenized)
    text_data = torch.zeros(len(tokenized), max_len, dtype=torch.long)
    for i, t in enumerate(tokenized):
        text_data[i, :len(t)] = torch.tensor(t)
    numeric_data = torch.randn(len(tokenized), 16)
    targets = torch.randn(len(tokenized), 16)
    return numeric_data, text_data, targets, vocab_size

numeric_data, text_data, targets, vocab_size = generate_repo_data()
print(f"Loaded {len(text_data)} samples, vocab size={vocab_size}")

# ------------------------------
# 3️⃣ Quantum-inspired entropy
# ------------------------------
def quantum_entropy_scalar(bits=2):
    if use_qiskit:
        qc = QuantumCircuit(bits, bits)
        for i in range(bits):
            qc.h(i)
        qc.measure(range(bits), range(bits))
        backend = Aer.get_backend("qasm_simulator")
        result = execute(qc, backend, shots=256).result()
        counts = result.get_counts()
        total = sum(counts.values())
        probs = [v/total for v in counts.values()]
        entropy = -sum(p * torch.log(torch.tensor(max(p,1e-12))) for p in probs)
        return 0.5 + (entropy.item()/bits)*0.5
    else:
        return 0.75 + (random.random()*0.5)

# ------------------------------
# 4️⃣ Neural network from scratch
# ------------------------------
class EDQBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(16, 128)
        self.l2 = nn.Linear(128, 64)
        for l in (self.l1, self.l2):
            nn.init.xavier_uniform_(l.weight)
            nn.init.zeros_(l.bias)
    def forward(self, x):
        return F.gelu(self.l2(F.gelu(self.l1(x))))

class BrainBranch(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.rnn = nn.GRU(64, 64, batch_first=True)
        nn.init.xavier_uniform_(self.rnn.weight_ih_l0)
        nn.init.zeros_(self.rnn.bias_ih_l0)
    def forward(self, x):
        x = self.embed(x)
        _, h = self.rnn(x)
        return F.gelu(h.squeeze(0))

class FusionNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.edq = EDQBranch()
        self.brain = BrainBranch(vocab_size)
        self.fuse1 = nn.Linear(64+64, 128)
        self.fuse2 = nn.Linear(128, 16)
        for l in (self.fuse1, self.fuse2):
            nn.init.kaiming_normal_(l.weight)
            nn.init.zeros_(l.bias)
    def forward(self, x_num, x_text):
        q = quantum_entropy_scalar()
        n_feat = self.edq(x_num) * q
        t_feat = self.brain(x_text)
        fused = torch.cat([n_feat, t_feat], dim=1)
        x = F.gelu(self.fuse1(fused))
        x = self.fuse2(x)
        return x

# ------------------------------
# 5️⃣ Training
# ------------------------------
def train(model, X_num, X_txt, Y, epochs=20, batch_size=32, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = X_num.size(0)
    indices = list(range(n))
    for epoch in range(epochs):
        random.shuffle(indices)
        epoch_loss = 0.0
        model.train()
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i+batch_size]
            xb_num = X_num[batch_idx].to(device)
            xb_txt = X_txt[batch_idx].to(device)
            yb = Y[batch_idx].to(device)
            optimizer.zero_grad()
            out = model(xb_num, xb_txt)
            loss = loss_fn(out, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb_num.size(0)
        epoch_loss /= n
        print(f"Epoch {epoch+1:03d} | Loss: {epoch_loss:.6f}")
    torch.save(model.state_dict(), "EDQ_Brain_Quantum_multi_repo_weights.pth")
    print("✅ Weights saved: EDQ_Brain_Quantum_multi_repo_weights.pth")
    return model

# ------------------------------
# 6️⃣ Entrypoint
# ------------------------------
if __name__ == "__main__":
    model = FusionNet(vocab_size)
    model = train(model, numeric_data, text_data, targets, epochs=30, batch_size=32)
    # quick sample inference
    model.eval()
    with torch.no_grad():
        sample_num = torch.randn(1,16)
        sample_txt = torch.randint(1, vocab_size, (1,text_data.size(1)))
        out = model(sample_num, sample_txt)
        print("Sample output:", out.cpu().numpy()[0][:8])