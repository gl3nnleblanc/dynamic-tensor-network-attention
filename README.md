# Dynamic Tensor Network Attention

A character-level language model that replaces the transformer's multi-head attention
mechanism with an **L0-gated tensor network** whose graph structure is learned during
training.

Based on Karpathy's minimal GPT
([gist](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)), ported to
PyTorch. Inspired by Stoudenmire & Schwab,
[Supervised Learning with Tensor Networks](https://arxiv.org/abs/1603.03039) (NeurIPS 2016).

Code written by Claude with substantial hand-holding + correcting. Probably still contains slop.

---

## What is a tensor network?

A tensor network is a graph where:
- **Nodes** are tensors (multi-dimensional arrays)
- **Edges** are contractions — shared indices that are summed over

When two nodes share an edge of dimension `D`, you sum over that index during the forward
pass. Adding an edge means adding a new dimension of size `D` to both tensors and
contracting over it at inference time. The **bond dimension** `D` controls how much
information can flow across an edge.

This is a strict generalisation of matrix multiplication: a single edge between two
rank-1 tensors with bond dimension `D` is just a dot product; the full network is a
composition of such contractions.

---

## What replaces attention here?

Standard multi-head attention computes, for each output position `t`:

```
output_t = sum_j  softmax(q_t · k_j / sqrt(d)) * v_j
```

This is O(n²) in sequence length and uses a fixed, dense attention pattern.

Here, attention is replaced by a tensor network over sequence positions. For each output
position `t`, the model contracts a shared bond tensor `W` with the bond-space embeddings
of position `j` (source) and position `t` (query):

```
c_jt[k] = sum_{a,b}  W[a, b, k] * h_j[a] * h_t[b]

output_t = output_proj( sum_j  Z[t,j] * c_jt )
```

where:
- `h_t = A[t].T @ x_t` — each position's embedding projected into bond space (dim `D`)
- `W ∈ R^{D × D × D}` — the shared bond tensor; indices are (source, query, output)
- `Z[t,j]` — the gate value for edge `(j → t)`, drawn from the learned adjacency matrix
- One free leg `k` remains after the contraction, giving a `D`-dimensional output per edge

Contractions are performed using [google/TensorNetwork](https://github.com/google/TensorNetwork).

---

## The adjacency matrix

The attention pattern is governed by `log_alpha ∈ R^{N × N}`, a learnable parameter
tensor where `N = block_size`. This **is** the adjacency matrix of the tensor network
graph.

- `log_alpha[t, j]` is the logit for the Hard Concrete gate on edge `(j → t)`
- The gate matrix `Z = hard_concrete(log_alpha)` has entries in `[0, 1]`
- A causal mask enforces `Z[t, j] = 0` for `j > t`

### Edge types

**Chain edges** (`j == t-1`): nearest-neighbour connections, always active, excluded from
regularisation. These form the base MPS-style attention — a linear chain over the sequence.

**Long-range edges** (`j < t-1`): pre-seeded at initialisation with a small negative
logit. These are the edges the model can grow or prune. L0 regularisation penalises
their expected activation count, driving the graph toward a sparse skeleton of only the
long-range dependencies that are genuinely useful.

After training, `plt.imshow(model.blocks[0].attn.gate_matrix())` shows the learned
attention pattern directly — bright = active edge, dark = pruned.

---

## L0 regularisation and the Hard Concrete distribution

Discrete graph structure (edge present / absent) is incompatible with gradient descent.
The solution: treat the parameter space as **infinite but sparse**. Every potential edge
always exists, but most are clamped near zero by a learned gate.

The **Hard Concrete distribution** (Louizos et al., 2018) provides a continuous
relaxation of Bernoulli that places actual probability mass at exactly 0 and 1:

```
u ~ Uniform(0, 1)
s = sigmoid((log u - log(1-u) + log_alpha) / beta)
z = clamp(s * (zeta - gamma) + gamma,  0,  1)
```

During training, `z` is sampled stochastically; at eval, `z = sigmoid(log_alpha)`.
Because the stretched distribution has mass at exactly 0, the expected number of active
edges `sum_ij P(z_ij > 0)` is differentiable and can be added directly to the loss as an
L0 penalty — no surrogate, no annealing required.

The gradient of the task loss pushes useful gates open; the L0 penalty pushes all gates
closed. The graph structure that emerges is whatever sparsity level balances these two
forces.

---

## Architecture

```
Input tokens
    │
    ▼
wte (Embedding)  +  wpe (Embedding)
    │
    ▼  ─────────────────────────────────── x n_layer
┌─────────────────────────────────────┐
│  RMSNorm                            │
│      │                              │
│  TNAttention  ──────────────────────┤  (residual)
│      │                              │
│  RMSNorm                            │
│      │                              │
│  MLP (Linear → ReLU → Linear) ──────┤  (residual)
└─────────────────────────────────────┘
    │
    ▼
lm_head (Linear → logits over vocab)
```

### TNAttention internals

| Parameter    | Shape                  | Role                                          |
|--------------|------------------------|-----------------------------------------------|
| `log_alpha`  | `(N, N)`               | Adjacency matrix logits                       |
| `A`          | `(N, n_embd, D)`       | Per-position site projections to bond space   |
| `W`          | `(D, D, D)`            | Shared bond tensor                            |
| `output_proj`| `(D, n_embd)`          | Projects bond output back to embedding space  |

---

## Training data

Downloaded automatically on first run from Project Gutenberg:

| File | Source |
|------|--------|
| Shakespeare complete works | PG #100 |
| War and Peace | PG #2600 |
| Mark Twain collected works | PG #3200 |
| Don Quixote | PG #996 |
| Bible (KJV) | PG #10 |
| Moby Dick | PG #2701 |
| Les Misérables | PG #135 |

All files are concatenated, shuffled line-by-line, and used as a character-level corpus.

---

## Usage

```bash
pip install torch tensornetwork networkx matplotlib
python train.py
```

Outputs:
- Training loss, L0 penalty, and active edge count printed each step
- 20 sampled character sequences after training
- `tn_graph_layer{i}.png` — heatmap of the learned adjacency matrix per layer

---

## Key hyperparameters

| Parameter     | Default | Effect                                                    |
|---------------|---------|-----------------------------------------------------------|
| `n_embd`      | 128     | Token embedding dimension                                 |
| `bond_dim`    | 32      | TN bond dimension D — capacity per edge                   |
| `block_size`  | 128     | Context length and TN graph size                          |
| `lambda_l0`   | 1e-4    | L0 regularisation weight (÷400 applied in loss)           |
| `n_layer`     | 1       | Number of transformer blocks                              |

---

## References

- Stoudenmire & Schwab, [Supervised Learning with Tensor Networks](https://arxiv.org/abs/1603.03039), NeurIPS 2016
- Louizos, Welling & Kingma, [Learning Sparse Neural Networks through L0 Regularization](https://arxiv.org/abs/1712.01312), ICLR 2018
- Karpathy, [The unreasonable effectiveness of char-rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
