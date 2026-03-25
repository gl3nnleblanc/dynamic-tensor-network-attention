# Dynamic Tensor Network Attention

A character-level language model that replaces the transformer's multi-head attention
mechanism with an **L0-gated tensor network** whose "graph structure" is learned during
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
contracting over it at inference time.

This is a strict generalisation of matrix multiplication: a single edge between two
rank-1 tensors with bond dimension `D` is just a dot product; the full network is a
composition of such contractions.

The **bond dimension** `D` controls how much information can flow across an edge by limiting the maximum [entanglement entropy](https://en.wikipedia.org/wiki/Entropy_of_entanglement) between tensors.

---

## What replaces attention here?

Standard multi-head attention computes, for each output position `t`:

$$\text{output}^t = \sum_j \text{softmax}\left(\frac{q^t \cdot k^j}{\sqrt{d}}\right) v^j$$

This is $\mathcal{O}(n^2)$ in sequence length and uses a fixed, dense attention pattern.

Here, attention is replaced by a **Matrix Product State (MPS) recurrence** over the
sequence. Each position $t$ has a site tensor $A^t \in \mathbb{R}^{D \times n_{\text{embd}} \times D}$
that is contracted with the token embedding $x^t$ to produce a transfer matrix:

$$M^t = I_D + \sum_s A^t_{lsr}\ x^t_s \qquad (M^t \in \mathbb{R}^{D \times D})$$

The residual " $I_D +$ " keeps singular values of $M^t$ near 1 at initialisation so the
recurrence is stable from step 0.

A bond state $v \in \mathbb{R}^D$ is propagated left-to-right through the sequence.
At each position, the nearest-neighbour (chain) bond is always applied; long-range bonds
from any earlier position $i < t-1$ are gated by the learned adjacency matrix:

$$v^t = \tanh\left(v^{t-1} M^t\ +\ \sum_{i=0}^{t-2} Z_{it}\ \bigl(v^i\, B^{it}\bigr)\right)$$

$$\text{output}^t = W_{\text{out}}\ v^t$$

where:
- $v^0 = v_0$ — a learned left boundary state (dim $D$), initialized as $\mathcal{N}(0,\, 1/D)$ and updated by the optimizer
- $B^{it} \in \mathbb{R}^{D \times D}$ — long-range bond matrix for edge $(i \to t)$
- $Z_{it}$ — the Hard Concrete gate for edge $(i \to t)$, from the adjacency matrix
- $W_{\text{out}} \in \mathbb{R}^{D \times n_{\text{embd}}}$ — projects the bond state back to embedding space

---

## The adjacency matrix

The attention pattern is governed by $\log\alpha \in \mathbb{R}^{N \times N}$, a learnable
parameter tensor where $N$ = `block_size`. This **is** the adjacency matrix of the tensor
network graph.

- $\log\alpha_{tj}$ is the logit for the Hard Concrete gate on edge $(j \to t)$
- The gate matrix $Z = \text{hard}_\text{concrete}(\log\alpha)$ has entries in $[0, 1]$
- A causal mask enforces $Z_{tj} = 0$ for $j \geq t$

### Edge types

**Chain edges** ($j = t-1$): nearest-neighbour connections, always active, excluded from
regularisation. These form the base MPS-style recurrence — a linear chain over the sequence.

**Long-range edges** ($j < t-1$): initialized with $\log\alpha_{tj} = 0$, giving
$P(z_{tj} > 0) \approx 83\%$ so gradients flow freely from the start. L0 regularisation
penalises their expected activation count, driving the graph toward a sparse skeleton of
only the long-range dependencies that are genuinely useful.

After training, `plt.imshow(model.blocks[0].attn.gate_matrix())` shows the learned
attention pattern directly — bright = active edge, dark = pruned.

---

## L0 regularisation and the Hard Concrete distribution

Discrete graph structure (edge present / absent) is incompatible with gradient descent.
We treat the parameter space as **infinite but sparse**. Every potential edge
always exists, but most are clamped near zero by a learned gate.

The **Hard Concrete distribution** (Louizos et al., 2018) provides a continuous
relaxation of Bernoulli that places actual probability mass at exactly 0 and 1:

$$u \sim \text{Uniform}(0,\, 1)$$

$$s = \sigma\left(\frac{\log u - \log(1-u) + \log\alpha}{\beta}\right)$$

$$z = \text{clamp}\left(s\,(\zeta - \gamma) + \gamma,\; 0,\; 1\right)$$

During training, `z` is sampled stochastically; at eval, `z = sigmoid(log_alpha)`.
Because the stretched distribution has mass at exactly 0, the expected number of active
edges `sum_ij P(z_ij > 0)` is differentiable and can be added directly to the loss as an
L0 penalty — no surrogate, no annealing required.

The gradient of the task loss pushes useful gates open; the L0 penalty pushes all gates
closed. The graph structure that emerges is whatever sparsity level balances these two
forces.

---

### TNAttention internals

| Parameter    | Shape                  | Role                                                        |
|--------------|------------------------|-------------------------------------------------------------|
| `log_alpha`  | `(N, N)`               | Adjacency matrix logits for long-range gates                |
| `A`          | `(N, D, n_embd, D)`    | Site tensors — contracted with $x^t$ to form transfer matrix $M^t$ |
| `B`          | `(N, N, D, D)`         | Long-range bond matrices $B^{it}$                           |
| `v0`         | `(D,)`                 | Learned left boundary state                                 |
| `output_proj`| `(D, n_embd)`          | Projects bond state $v^t$ back to embedding space           |

---

## Training data

The model is trained on a **human-style reading curriculum** — the same progression a
student might follow from learning to read through graduate school. Each stage accumulates
all prior stages into the training pool, so the model never catastrophically forgets
earlier material.

All texts are downloaded automatically on first run (mostly from Project Gutenberg) and
split into paragraphs. The final stage streams physics abstracts from the HuggingFace
`scientific_papers` dataset.

| Stage | Content |
|-------|---------|
| **Preschool** | McGuffey First Eclectic Reader |
| **Grades 1–3** | Bible (KJV), McGuffey readers 2–3, Aesop's Fables, Grimm's Fairy Tales, Wind in the Willows, Ray's Primary & Intellectual Arithmetic |
| **Grades 4–6** | Alice in Wonderland, Wizard of Oz, Jungle Book, Peter Pan, Robin Hood, McGuffey readers 4–5, Ray's Practical Arithmetic |
| **Grades 7–9** | Tom Sawyer, Huckleberry Finn, Treasure Island, Sherlock Holmes, Frankenstein, Dracula, Jekyll & Hyde, The Time Machine, Ray's Algebra |
| **Grades 10–12** | Shakespeare (complete), Great Expectations, Pride and Prejudice, Tale of Two Cities, Paradise Lost, Canterbury Tales, projective geometry |
| **College** | Moby Dick, Ulysses, War and Peace, Brothers Karamazov, Les Misérables, Origin of Species, Wealth of Nations, The Republic, Calculus Made Easy |
| **arXiv physics** | ~50k abstracts from hep-th, hep-ph, gr-qc, quant-ph, cond-mat, astro-ph |

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
| `lambda_l0`   | 1e-3    | L0 regularisation weight (fraction of active edges)       |
| `n_layer`     | 1       | Number of transformer blocks                              |

---

## References

- Stoudenmire & Schwab, [Supervised Learning with Tensor Networks](https://arxiv.org/abs/1603.03039), NeurIPS 2016
- Louizos, Welling & Kingma, [Learning Sparse Neural Networks through L0 Regularization](https://arxiv.org/abs/1712.01312), ICLR 2018
- Karpathy, [The unreasonable effectiveness of char-rnn](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
