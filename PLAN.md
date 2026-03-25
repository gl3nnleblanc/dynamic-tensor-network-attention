# Rebuild Plan: L0-Gated Tensor Network Attention

## What We Are Building

A replacement for the multi-head attention block in a transformer. The replacement is
a tensor network (TN) where:

- Each sequence position t has a **site tensor** that projects its input embedding into
  a lower-dimensional bond space.
- **Edges** between positions correspond to tensor contractions over a shared bond index
  of dimension `bond_dim`. This is the Stoudenmire & Schwab (1603.03039) formulation:
  adding an edge means adding a new contracted dimension of size D to two tensors and
  summing over it during inference.
- The **adjacency matrix** is a real `(N, N)` parameter tensor of logits. Hard Concrete
  gates applied element-wise give the gate matrix Z. This IS the adjacency — not a dict,
  not a ModuleDict, a tensor.
- L0 regularization on the expected gate values drives sparsity.
- `google/TensorNetwork` is used for the actual contractions.

---

## Mathematical Formulation

### Inputs

At position t, we have been given `x_t ∈ R^{n_embd}`.

### Step 1 — Site projections

Each position t has a learned matrix `A_t ∈ R^{n_embd × D}`.

```
h_t = A_t.T @ x_t          shape: (D,)
```

`h_t` is position t's embedding in bond space. It participates as one leg in every
contraction involving position t.

### Step 2 — Bond tensor

A single bond tensor `W ∈ R^{D × D × D}` is shared across all edges. Its three indices:
- index 0 (size D): contracts with h_j  (source position)
- index 1 (size D): contracts with h_t  (target/query position)
- index 2 (size D): FREE — this is the output bond, left dangling

```
c_jt[k] = sum_{a,b} W[a, b, k] * h_j[a] * h_t[b]      shape: (D,)
```

This is a proper TN contraction. After contracting both physical legs (h_j and h_t),
exactly ONE free leg remains per edge: the output bond of dimension D.

Using tensornetwork:
```python
W_node  = tn.Node(W,   backend='pytorch')   # shape (D, D, D)
hj_node = tn.Node(h_j, backend='pytorch')   # shape (D,)
ht_node = tn.Node(h_t, backend='pytorch')   # shape (D,)
tn.connect(hj_node[0], W_node[0])
tn.connect(ht_node[0], W_node[1])
result  = tn.contractors.greedy(
    [W_node, hj_node, ht_node],
    output_edge_order=[W_node[2]]
)                                            # result.tensor shape: (D,)
```

### Step 3 — Gated aggregation

The adjacency gate matrix Z gates each edge's contribution:

```
output_t = sum_{j=0}^{t} Z[t,j] * c_jt
```

Z is computed from the learnable logit matrix log_alpha using Hard Concrete:
- During training: stochastic sample (stretched Gumbel, clamped to [0,1])
- During eval:     sigmoid(log_alpha)
- Causal mask applied: Z[t,j] = 0 for j > t (upper triangle is ignored)

### Step 4 — Output projection

```
y_t = output_proj(output_t)      shape: (n_embd,)
```

output_proj is a Linear(D, n_embd, bias=False).

---

## Data Structures

| Name          | Shape              | Type             | Notes                                      |
|---------------|--------------------|------------------|--------------------------------------------|
| `log_alpha`   | (N, N)             | nn.Parameter     | THE adjacency matrix. logits for H.C. gate |
| `A`           | (N, n_embd, D)     | nn.Parameter     | Per-position site projections              |
| `W`           | (D, D, D)          | nn.Parameter     | Shared bond tensor                         |
| `output_proj` | Linear(D, n_embd)  | nn.Module        | Projects output bond back to embedding dim |

N = block_size (preallocated; all positions exist from the start, no dynamic growth needed
during a training run since sequence length is bounded by block_size).

Parameter count at n_embd=128, D=32, N=16:
- log_alpha: 256
- A:         16 × 128 × 32 = 65,536
- W:         32 × 32 × 32  = 32,768
- output_proj: 32 × 128    = 4,096
- Total TN attention: ~102k params per layer

---

## The Adjacency Matrix Is log_alpha

```python
self.log_alpha = nn.Parameter(torch.zeros(block_size, block_size))
```

- `log_alpha[i,j]` is the logit for the Hard Concrete gate on edge (i,j)
- Initialize chain edges (j == i-1) to `log_alpha_chain_init` (e.g., 0.0)
- Initialize all other edges to a large negative value (e.g., -10.0) so they
  start inactive
- The causal mask (tril) is applied at forward time, not stored in log_alpha
- L0 loss = `expected_L0(log_alpha).tril().sum()` (lower triangle only, causal)

Visualization: log_alpha IS the adjacency matrix. Save it as a heatmap image, not
a reconstructed graph from a dict.

---

## File Layout

### `tn_gpt/l0_gate.py`
Keep the constants (BETA, GAMMA, ZETA). Replace the Module with two pure functions:

```python
def hard_concrete_sample(log_alpha: Tensor) -> Tensor:
    """Element-wise Hard Concrete sample. Works on tensors of any shape."""
    ...

def expected_l0(log_alpha: Tensor) -> Tensor:
    """Element-wise expected gate value (differentiable). Any shape."""
    ...
```

No class needed. These are applied to the full log_alpha matrix at once.

### `tn_gpt/tn_attention.py`
Completely rewritten. One class: `TNAttention`.

```python
class TNAttention(nn.Module):
    def __init__(self, n_embd: int, bond_dim: int, block_size: int):
        self.log_alpha    # (block_size, block_size) — THE adjacency matrix
        self.A            # (block_size, n_embd, bond_dim) — site projections
        self.W            # (bond_dim, bond_dim, bond_dim) — shared bond tensor
        self.output_proj  # Linear(bond_dim, n_embd)

    def forward(self, x: Tensor, pos: int, h_cache: list) -> Tensor:
        # x: (n_embd,), pos: int, h_cache: list of (D,) tensors for positions 0..pos-1
        # 1. Project x to bond space: h_t = A[pos].T @ x          → (D,)
        # 2. Append h_t to h_cache
        # 3. Sample gate row: z = hard_concrete_sample(log_alpha[pos, :pos+1])  → (pos+1,)
        # 4. For each j where z[j] > 0:
        #       contract W with h_cache[j] and h_t via tensornetwork → (D,)
        #       accumulate z[j] * result into output
        # 5. return output_proj(output)

    def l0_loss(self) -> Tensor:
        # expected_l0(log_alpha).tril().sum()

    def gate_matrix(self) -> Tensor:
        # sigmoid(log_alpha) — for visualization, this IS the adjacency matrix
```

### `tn_gpt/adjacency.py`
DELETE. It is replaced by `log_alpha` inside `TNAttention`.

### `train.py`
Changes:
- Remove the `ensure_nodes` pre-warm call (no longer needed)
- Pass `block_size` to `TNAttention.__init__`
- The graph visualization uses `model.blocks[i].attn.gate_matrix()` directly —
  this is a real (N, N) tensor, just call `plt.imshow` on it

---

## Initialization

```python
# log_alpha: chain edges active, everything else off
log_alpha = torch.full((block_size, block_size), -10.0)
for i in range(1, block_size):
    log_alpha[i, i-1] = 0.0          # chain edge (i-1) → i starts at P=0.5
self.log_alpha = nn.Parameter(log_alpha)

# A: small random (Xavier-style)
self.A = nn.Parameter(torch.randn(block_size, n_embd, bond_dim) * 0.02)

# W: small random
self.W = nn.Parameter(torch.randn(bond_dim, bond_dim, bond_dim) * 0.02)
```

---

## Visualization

After training, `model.blocks[0].attn.gate_matrix()` returns a `(16, 16)` tensor.
`plt.imshow` it directly — no graph reconstruction needed. The image IS the adjacency
matrix. Bright = active edge, dark = pruned.

---

## What This Is NOT

- NOT message-passing / GNN. There are no "node features flowing along edges". The
  contraction `W[a,b,c] * h_j[a] * h_t[b]` is a mathematical operation on tensors.
  The output is a vector in R^D. The graph structure determines which contractions happen.
- NOT the old 3-tensor W_ij per edge. The bond tensor W is shared and the per-edge
  variation comes entirely from the site projections A[j] and A[t].
- NOT storing the adjacency in a Python dict. It is a Parameter tensor.
