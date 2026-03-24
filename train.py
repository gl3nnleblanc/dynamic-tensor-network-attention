"""
L0-Gated Tensor Network GPT
============================
Karpathy's minimal GPT (https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
ported to PyTorch with the multi-head attention block replaced by TNAttention.

Training task: character-level name generation (same as the original gist).
"""

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tn_gpt import TNAttention

random.seed(42)
torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

if not os.path.exists('names.txt'):
    import urllib.request
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt',
        'names.txt'
    )
docs = [line.strip() for line in open('names.txt') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars     = sorted(set(''.join(docs)))
BOS        = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

n_layer    = 1
n_embd     = 16
block_size = 16
bond_dim   = 8       # TN bond dimension D
lambda_l0  = 1e-4   # L0 regularization weight (increase → sparser graph)
num_steps  = 1000
lr         = 0.01

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def rmsnorm(x: Tensor) -> Tensor:
    return x * (x.pow(2).mean() + 1e-5).rsqrt()


class GPTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn    = TNAttention(n_embd, bond_dim)
        self.mlp_fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.mlp_fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x: Tensor, pos: int, hidden_cache: list) -> Tensor:
        x = x + self.attn(rmsnorm(x), pos, hidden_cache)
        h = F.relu(self.mlp_fc1(rmsnorm(x)))
        x = x + self.mlp_fc2(h)
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte     = nn.Embedding(vocab_size, n_embd)
        self.wpe     = nn.Embedding(block_size, n_embd)
        self.blocks  = nn.ModuleList([GPTBlock() for _ in range(n_layer)])
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, token_id: int, pos_id: int, hidden_caches: list) -> Tensor:
        x = self.wte(torch.tensor(token_id)) + self.wpe(torch.tensor(pos_id))
        x = rmsnorm(x)
        for i, block in enumerate(self.blocks):
            x = block(x, pos_id, hidden_caches[i])
        return self.lm_head(x)   # (vocab_size,)

    def l0_loss(self) -> Tensor:
        return sum(b.attn.expected_L0() for b in self.blocks)

    def active_edge_count(self) -> int:
        return sum(b.attn.adjacency.active_edge_count() for b in self.blocks)


model     = GPT()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.85, 0.99))
print(f"num params (initial): {sum(p.numel() for p in model.parameters())}")

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

model.train()
for step in range(num_steps):
    doc    = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(c) for c in doc] + [BOS]
    n      = min(block_size, len(tokens) - 1)

    hidden_caches = [[] for _ in range(n_layer)]
    losses: list[Tensor] = []

    for pos_id in range(n):
        logits = model(tokens[pos_id], pos_id, hidden_caches)
        target = torch.tensor([tokens[pos_id + 1]])
        losses.append(F.cross_entropy(logits.unsqueeze(0), target))

    task_loss  = sum(losses) / n
    total_loss = task_loss + lambda_l0 * model.l0_loss()

    # Adam with linear LR decay (matches Karpathy)
    lr_t = lr * (1 - step / num_steps)
    for pg in optimizer.param_groups:
        pg['lr'] = lr_t

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    n_active = model.active_edge_count()
    print(
        f"step {step+1:4d}/{num_steps} | "
        f"loss {task_loss.item():.4f} | "
        f"l0 {model.l0_loss().item():.2f} | "
        f"active edges {n_active}",
        end='\r'
    )

print()  # newline after training

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

model.eval()
temperature = 0.5
print("\n--- inference (hallucinated names) ---")
with torch.no_grad():
    for sample_idx in range(20):
        hidden_caches = [[] for _ in range(n_layer)]
        token_id      = BOS
        sample        = []
        for pos_id in range(block_size):
            logits   = model(token_id, pos_id, hidden_caches)
            probs    = F.softmax(logits / temperature, dim=-1)
            token_id = torch.multinomial(probs, 1).item()
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        print(f"  {sample_idx+1:2d}: {''.join(sample)}")

# ---------------------------------------------------------------------------
# Graph summary
# ---------------------------------------------------------------------------

print("\n--- tensor network graph ---")
for i, block in enumerate(model.blocks):
    print(f"  layer {i}: {block.attn.adjacency}")
