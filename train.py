"""
L0-Gated Tensor Network GPT
============================
Karpathy's minimal GPT (https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)
ported to PyTorch with the multi-head attention block replaced by TNAttention.
"""

import os
import re
import sys
import math
import random
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tn_gpt import TNAttention

random.seed(42)
torch.manual_seed(42)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"device: {device}")

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Curriculum: ordered by reading level, each stage accumulates prior stages.
# arXiv physics is streamed separately at the end via HuggingFace datasets.
# ---------------------------------------------------------------------------

DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

CURRICULUM = {
    'preschool': [
        # McGuffey First Eclectic Reader — "The cat is fat. Is the cat fat?"
        ('mcguffey_first.txt',     'https://www.gutenberg.org/files/14640/14640-0.txt'),
    ],
    'grades_1_3': [
        ('bible.txt',              'https://www.gutenberg.org/cache/epub/1582/pg1582.txt'),
        ('mcguffey_second.txt',    'https://www.gutenberg.org/files/14668/14668-0.txt'),
        ('mcguffey_third.txt',     'https://www.gutenberg.org/files/14766/14766-0.txt'),
        ('aesop_fables.txt',       'https://www.gutenberg.org/files/21/21-0.txt'),
        ('grimm_fairy_tales.txt',  'https://www.gutenberg.org/files/2591/2591-0.txt'),
        ('summers_readers.txt',    'https://www.gutenberg.org/cache/epub/67302/pg67302.txt'),
        ('kittens_first_reader.txt','https://www.gutenberg.org/cache/epub/61852/pg61852.txt'),
        ('wind_in_the_willows.txt','https://www.gutenberg.org/cache/epub/289/pg289.txt'),
        # Math — Internet Archive OCR'd djvu.txt (Ray's Arithmetic series)
        ('rays_primary_arith.txt', 'https://archive.org/download/raysnewprimarya00raygoog/raysnewprimarya00raygoog_djvu.txt'),
        ('rays_intellectual_arith.txt', 'https://archive.org/download/raysnewintellect00rayjrich/raysnewintellect00rayjrich_djvu.txt'),
    ],
    'grades_4_6': [
        ('america_first.txt',      'https://www.gutenberg.org/cache/epub/24798/pg24798.txt'),
        ('elson_grammar_school.txt', 'https://www.gutenberg.org/cache/epub/6963/pg6963.txt'),
        ('mcguffey_fourth.txt',    'https://www.gutenberg.org/files/14880/14880.txt'),
        ('mcguffey_fifth.txt',     'https://www.gutenberg.org/files/15040/15040.txt'),
        ('wizard_of_oz.txt',       'https://www.gutenberg.org/files/55/55-0.txt'),
        ('alice_wonderland.txt',   'https://www.gutenberg.org/files/11/11-0.txt'),
        ('jungle_book.txt',        'https://www.gutenberg.org/files/236/236-0.txt'),
        ('peter_pan.txt',          'https://www.gutenberg.org/files/16/16-0.txt'),
        ('robin_hood.txt',         'https://www.gutenberg.org/files/1256/1256-0.txt'),
        ('jack_and_jill.txt',      'https://www.gutenberg.org/cache/epub/2786/pg2786.txt'),
        # Math (Internet Archive OCR — Ray's Practical Arithmetic)
        ('rays_practical_arith.txt', 'https://archive.org/download/raysnewpractical00rayj/raysnewpractical00rayj_djvu.txt'),
    ],
    'grades_7_9': [
        ('little_women.txt',       'https://www.gutenberg.org/files/514/514-0.txt'),
        ('tom_sawyer.txt',         'https://www.gutenberg.org/files/74/74-0.txt'),
        ('huckleberry_finn.txt',   'https://www.gutenberg.org/files/76/76-0.txt'),
        ('treasure_island.txt',    'https://www.gutenberg.org/files/120/120-0.txt'),
        ('sherlock_holmes.txt',    'https://www.gutenberg.org/files/1661/1661-0.txt'),
        ('time_machine.txt',       'https://www.gutenberg.org/files/35/35-0.txt'),
        ('war_of_worlds.txt',      'https://www.gutenberg.org/files/36/36-0.txt'),
        ('jekyll_hyde.txt',        'https://www.gutenberg.org/files/42/42-0.txt'),
        ('frankenstein.txt',       'https://www.gutenberg.org/files/84/84-0.txt'),
        ('dracula.txt',            'https://www.gutenberg.org/files/345/345-0.txt'),
        # Math (Internet Archive OCR — Ray's Algebra)
        ('rays_algebra.txt',       'https://archive.org/download/raysalgebrapartf00rayjrich/raysalgebrapartf00rayjrich_djvu.txt'),
    ],
    'grades_10_12': [
        ('communist_manifesto.txt','https://www.gutenberg.org/cache/epub/61/pg61.txt'),
        ('elements_of_style.txt',  'https://www.gutenberg.org/cache/epub/37134/pg37134.txt'),
        ('shakespeare_complete.txt', 'https://www.gutenberg.org/files/100/100-0.txt'),
        ('great_expectations.txt', 'https://www.gutenberg.org/files/1400/1400-0.txt'),
        ('jane_eyre.txt',          'https://www.gutenberg.org/files/1260/1260-0.txt'),
        ('scarlet_letter.txt',     'https://www.gutenberg.org/files/25344/25344-0.txt'),
        ('tale_of_two_cities.txt', 'https://www.gutenberg.org/files/98/98-0.txt'),
        ('pride_and_prejudice.txt','https://www.gutenberg.org/files/1342/1342-0.txt'),
        ('dorian_gray.txt',        'https://www.gutenberg.org/files/174/174-0.txt'),
        ('heart_of_darkness.txt',  'https://www.gutenberg.org/files/219/219-0.txt'),
        ('red_badge_courage.txt',  'https://www.gutenberg.org/files/73/73-0.txt'),
        ('canterbury_tales.txt',   'https://www.gutenberg.org/files/2383/2383-0.txt'),
        ('paradise_lost.txt',      'https://www.gutenberg.org/files/26/26-0.txt'),
        # Math (Internet Archive OCR — plane geometry, trigonometry)
        ('projective_geometry.txt','https://www.gutenberg.org/cache/epub/17001/pg17001.txt'),
    ],
    'college': [
        ('moby_dick.txt',          'https://www.gutenberg.org/files/2701/2701-0.txt'),
        ('ulysses.txt',            'https://www.gutenberg.org/files/4300/4300-0.txt'),
        ('war_and_peace.txt',      'https://www.gutenberg.org/files/2600/2600-0.txt'),
        ('brothers_karamazov.txt', 'https://www.gutenberg.org/files/28054/28054-0.txt'),
        ('crime_punishment.txt',   'https://www.gutenberg.org/files/2554/2554-0.txt'),
        ('les_miserables.txt',     'https://www.gutenberg.org/files/135/135-0.txt'),
        ('middlemarch.txt',        'https://www.gutenberg.org/files/145/145-0.txt'),
        ('origin_of_species.txt',  'https://www.gutenberg.org/files/1228/1228-0.txt'),
        ('wealth_of_nations.txt',  'https://www.gutenberg.org/files/3300/3300-0.txt'),
        ('federalist_papers.txt',  'https://www.gutenberg.org/files/18/18-0.txt'),
        ('divine_comedy.txt',      'https://www.gutenberg.org/files/8800/8800-0.txt'),
        ('iliad.txt',              'https://www.gutenberg.org/files/6150/6150-0.txt'),
        ('odyssey.txt',            'https://www.gutenberg.org/files/1727/1727-0.txt'),
        ('faust.txt',              'https://www.gutenberg.org/files/14591/14591-0.txt'),
        ('notes_underground.txt',  'https://www.gutenberg.org/files/600/600-0.txt'),
        ('republic_plato.txt',     'https://www.gutenberg.org/files/1497/1497-0.txt'),
        # Math textbooks (Project Gutenberg)
        ('calculus_made_easy.txt', 'https://www.gutenberg.org/files/33283/33283-0.txt'),
        ('hardy_pure_math.txt',    'https://www.gutenberg.org/files/29785/29785-0.txt'),
    ],
    # arXiv physics loaded separately via streaming — see load_arxiv_physics()
}

PHYSICS_CATS = {'hep-th', 'hep-ph', 'gr-qc', 'quant-ph', 'cond-mat', 'astro-ph', 'physics'}


def strip_gutenberg_boilerplate(text):
    """Remove PG header and footer, keeping only the actual book content."""
    start_marker = '*** START OF THE PROJECT GUTENBERG EBOOK'
    end_marker   = '*** END OF THE PROJECT GUTENBERG EBOOK'
    # Case-insensitive search
    upper = text.upper()
    start = upper.find(start_marker)
    if start != -1:
        # Skip to end of the start marker line
        text = text[text.index('\n', start) + 1:]
    end = text.upper().find(end_marker)
    if end != -1:
        text = text[:end]
    return text


def load_paragraphs(path, min_len=40):
    """Strip PG boilerplate, then split into paragraphs (blank-line separated)."""
    raw  = open(path, encoding='utf-8', errors='ignore').read()
    text = strip_gutenberg_boilerplate(raw)
    paragraphs, current = [], []
    for line in text.splitlines():
        line = line.strip()
        line = re.sub(r'^\d+:\d+\.?\s*', '', line)   # strip verse/stanza refs (e.g. "3:2. ")
        if line:
            current.append(line)
        elif current:
            para = re.sub(r'[ \t]+', ' ', ' '.join(current)).strip()
            if len(para) >= min_len and 'ILLUSTRATION' not in para.upper():
                paragraphs.append(para)
            current = []
    if current:
        para = re.sub(r'[ \t]+', ' ', ' '.join(current)).strip()
        if len(para) >= min_len:
            paragraphs.append(para)
    return paragraphs


def load_arxiv_physics(n=50_000):
    """Stream physics abstracts from HuggingFace arxiv dataset."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("  (install 'datasets' for arXiv physics stage: pip install datasets)")
        return []
    print("streaming arXiv physics abstracts...")
    ds = load_dataset('Cornell-University/arxiv', split='train', streaming=True,
                      trust_remote_code=True)
    docs = []
    for ex in ds:
        if any(cat in ex.get('categories', '') for cat in PHYSICS_CATS):
            abstract = ex.get('abstract', '').replace('\n', ' ').strip()
            if len(abstract) >= 80:
                docs.append(abstract)
                if len(docs) >= n:
                    break
    print(f"  loaded {len(docs)} physics abstracts")
    return docs


# Download and load all Gutenberg levels
level_docs = {}
for level, sources in CURRICULUM.items():
    pool = []
    # Early readers use shorter min_len since sentences are intentionally brief
    min_len = 20 if level == 'preschool' else 40 if level == 'grades_1_3' else 80
    for fname, url in sources:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"downloading {fname} ...")
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as resp, open(fpath, 'wb') as f:
                    f.write(resp.read())
            except Exception as e:
                print(f"  WARNING: skipping {fname} ({e})")
                continue
        pool.extend(load_paragraphs(fpath, min_len=min_len))
    level_docs[level] = pool
    print(f"  {level}: {len(pool)} paragraphs")

level_docs['arxiv_physics'] = load_arxiv_physics()

# Build full vocab across all levels
all_text = ''.join(p for docs in level_docs.values() for p in docs)
uchars     = sorted(set(all_text))
BOS        = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# Ordered curriculum stages — each stage includes all prior stages pooled together
# so the model doesn't catastrophically forget earlier material.
LEVELS = list(CURRICULUM.keys()) + ['arxiv_physics']

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

n_layer    = 1
n_embd     = 128
block_size = 128
bond_dim   = 32      # TN bond dimension D
lambda_l0  = 0.001  # per-edge L0 cost; total penalty = lambda_l0 * expected_active_edges
num_epochs = 5      # full passes over the corpus
lr         = 1e-3

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def rmsnorm(x: Tensor) -> Tensor:
    return x * (x.pow(2).mean(-1, keepdim=True) + 1e-5).rsqrt()


class GPTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn    = TNAttention(n_embd, bond_dim, block_size)
        self.mlp_fc1 = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.mlp_fc2 = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: (T, n_embd)
        x = x + self.attn(rmsnorm(x))
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

    def forward(self, token_ids: Tensor) -> Tensor:
        # token_ids: (T,) — full sequence
        T = token_ids.shape[0]
        x = self.wte(token_ids) + self.wpe(torch.arange(T, device=token_ids.device))  # (T, n_embd)
        x = rmsnorm(x)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(x)   # (T, vocab_size)

    def l0_loss(self) -> Tensor:
        return sum(b.attn.l0_loss() for b in self.blocks)

    def active_edge_count(self) -> int:
        return sum(b.attn.active_edge_count() for b in self.blocks)


model = GPT().to(device)
# log_alpha parameters learn graph structure, not weights — needs a much higher lr
# to move meaningfully within a training run. Everything else uses the base lr.
log_alpha_params = [p for n, p in model.named_parameters() if 'log_alpha' in n]
other_params     = [p for n, p in model.named_parameters() if 'log_alpha' not in n]
optimizer = torch.optim.Adam(
    [{'params': other_params,     'lr': lr},
     {'params': log_alpha_params, 'lr': lr * 100}],
    betas=(0.85, 0.99),
)
print(f"num params: {sum(p.numel() for p in model.parameters())}")

# ---------------------------------------------------------------------------
# Terminal visualization
# ---------------------------------------------------------------------------

def _viridis(t: float):
    """Approximate viridis: t in [0,1] -> (R,G,B)."""
    keys = [(68,1,84),(58,82,139),(32,144,140),(94,201,98),(253,231,37)]
    t = max(0.0, min(1.0, t))
    x = t * (len(keys) - 1)
    lo, hi = int(x), min(int(x) + 1, len(keys) - 1)
    f = x - lo
    return tuple(int(keys[lo][i] + f * (keys[hi][i] - keys[lo][i])) for i in range(3))

def sample(n_samples=3, max_len=120, temperature=0.7):
    """Generate n_samples strings from the current model state."""
    model.eval()
    results = []
    with torch.no_grad():
        for _ in range(n_samples):
            tokens = [BOS]
            for _ in range(max_len):
                logits = model(torch.tensor(tokens, device=device))
                probs  = F.softmax(logits[-1] / temperature, dim=-1).cpu()
                tok    = torch.multinomial(probs, 1).item()
                if tok == BOS:
                    break
                tokens.append(tok)
            results.append(''.join(uchars[t] for t in tokens[1:]))
    model.train()
    return results


TIMELAPSE_DIR = os.path.expanduser('~/Desktop/timelapse')
os.makedirs(TIMELAPSE_DIR, exist_ok=True)

def save_timelapse_frame(gate_mat, level, step, task_loss, l0, n_active, samples):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        fig = plt.figure(figsize=(10, 7), facecolor='#1a1a2e')
        gs  = gridspec.GridSpec(2, 1, height_ratios=[4, 1], hspace=0.35)

        ax_mat = fig.add_subplot(gs[0])
        ax_mat.imshow(gate_mat.numpy(), vmin=0, vmax=1, cmap='viridis', origin='upper', aspect='auto')
        ax_mat.set_title(
            f'[{level}] step {step} | loss {task_loss:.4f} | l0 {l0:.4f} | active edges {n_active}',
            color='white', fontsize=9, pad=6
        )
        ax_mat.set_xlabel('source position j', color='#aaaaaa', fontsize=8)
        ax_mat.set_ylabel('target position t', color='#aaaaaa', fontsize=8)
        ax_mat.tick_params(colors='#aaaaaa', labelsize=7)
        for spine in ax_mat.spines.values():
            spine.set_edgecolor('#444444')

        ax_txt = fig.add_subplot(gs[1])
        ax_txt.axis('off')
        text = '\n'.join(f'[{i+1}] {s}' for i, s in enumerate(samples))
        ax_txt.text(0.01, 0.95, text, transform=ax_txt.transAxes,
                    color='#ccffcc', fontsize=7.5, fontfamily='monospace',
                    verticalalignment='top', wrap=True)

        path = os.path.join(TIMELAPSE_DIR, f'step_{step:06d}.png')
        plt.savefig(path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close(fig)
    except ImportError:
        pass


def render(gate_mat, level, step, total, task_loss, l0, n_active, mean_la, max_la, samples):
    """
    Render the NxN gate matrix as a live ANSI truecolor display.
    Uses the half-block trick (▄) to pack two matrix rows into one terminal row,
    giving an N x N/2 display that fits comfortably on screen.
    """
    N = gate_mat.shape[0]
    buf = ['\033[H\033[2J\033[3J']     # cursor home + clear screen + clear scrollback

    for row in range(0, N, 2):
        line = []
        for col in range(N):
            top = gate_mat[row,   col].item()
            bot = gate_mat[row+1, col].item() if row + 1 < N else 0.0
            tr, tg, tb = _viridis(top)
            br, bg, bb = _viridis(bot)
            line.append(f'\033[48;2;{tr};{tg};{tb}m\033[38;2;{br};{bg};{bb}m▄')
        buf.append(''.join(line) + '\033[0m\n')

    buf.append(
        f'\033[0m'
        f'[{level}] step {step}/{total} | '
        f'loss {task_loss:.4f} | '
        f'l0 {l0:.4f} | '
        f'log_alpha mean {mean_la:+.2f} max {max_la:+.2f} | '
        f'active edges {n_active}\n'
    )
    buf.append('\n')
    for i, s in enumerate(samples):
        buf.append(f'  [{i+1}] {s}\n')

    sys.stdout.write(''.join(buf))
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

# Total step count for LR decay: each stage trains on the full accumulated pool,
# so later stages repeat earlier docs. Sum accumulated pool sizes × passes.
passes_per_level = num_epochs
acc, total_steps = 0, 0
for level in LEVELS:
    pool = level_docs.get(level, [])
    if pool:
        acc += len(pool)
        total_steps += acc * passes_per_level
print(f"total training steps: {total_steps}  (capture every {max(1, total_steps // 14900)} steps to stay under 1GB)")
step = 0
samples = []

def train_on_pool(pool, level, passes=1):
    global step, samples
    for _ in range(passes):
        epoch_pool = list(pool)
        random.shuffle(epoch_pool)
        for doc in epoch_pool:
            tokens = [BOS] + [uchars.index(c) for c in doc] + [BOS]
            T      = min(block_size, len(tokens) - 1)
            if T < 2:
                continue

            inp     = torch.tensor(tokens[:T],     device=device)
            targets = torch.tensor(tokens[1:T + 1], device=device)

            logits    = model(inp)
            task_loss = F.cross_entropy(logits, targets)
            total_loss = task_loss + lambda_l0 * model.l0_loss()

            decay = max(0.0, 1 - step / total_steps)
            optimizer.param_groups[0]['lr'] = lr * decay
            optimizer.param_groups[1]['lr'] = lr * 100 * decay

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(other_params, 1.0)
            torch.nn.utils.clip_grad_norm_(log_alpha_params, 10.0)
            optimizer.step()
            step += 1

            if step % 500 == 0:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'uchars': uchars, 'BOS': BOS, 'vocab_size': vocab_size,
                    'n_layer': n_layer, 'n_embd': n_embd,
                    'bond_dim': bond_dim, 'block_size': block_size,
                    'step': step,
                }, 'tn_gpt.pt')

            with torch.no_grad():
                gate_mat = model.blocks[0].attn.gate_matrix()
                N = gate_mat.shape[0]
                for t in range(1, N):
                    gate_mat[t, t-1] = 1.0
            n_active = model.active_edge_count()
            mean_la  = sum(b.attn.log_alpha.mean().item() for b in model.blocks) / len(model.blocks)
            max_la   = max(b.attn.log_alpha.max().item()  for b in model.blocks)
            if step % 200 == 0 or step == 1:
                samples = sample()
                save_timelapse_frame(gate_mat, level, step,
                                     task_loss.item(), model.l0_loss().item(), n_active, samples)
            render(gate_mat, level, step, total_steps,
                   task_loss.item(), model.l0_loss().item(), n_active, mean_la, max_la, samples)


model.train()

# Accumulated pool — each stage adds its docs to the running pool so the model
# doesn't catastrophically forget earlier material (like a real student).
accumulated = []
for level in LEVELS:
    pool = level_docs[level]
    if not pool:
        continue
    accumulated.extend(pool)
    print(f"\n=== curriculum stage: {level} ({len(pool)} new docs, {len(accumulated)} total) ===")
    train_on_pool(accumulated, level, passes=passes_per_level)

print()  # newline after training

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

model.eval()
temperature = 0.5
print("\n--- inference (sampled text) ---")
with torch.no_grad():
    for sample_idx in range(20):
        tokens = [BOS]
        for _ in range(block_size - 1):
            inp    = torch.tensor(tokens)
            logits = model(inp)              # (T, vocab_size)
            probs  = F.softmax(logits[-1] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            if next_token == BOS:
                break
            tokens.append(next_token)
        sample = ''.join(uchars[t] for t in tokens[1:])
        print(f"  {sample_idx+1:2d}: {sample}")

# ---------------------------------------------------------------------------
# Graph summary
# ---------------------------------------------------------------------------

print("\n--- tensor network graph ---")
for i, block in enumerate(model.blocks):
    print(f"  layer {i}: {block.attn}")

# ---------------------------------------------------------------------------
# Graph visualization — log_alpha IS the adjacency matrix, just imshow it
# ---------------------------------------------------------------------------

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    for layer_idx, block in enumerate(model.blocks):
        gate_mat = block.attn.gate_matrix()   # (N, N) tensor of P(gate active)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(gate_mat, vmin=0, vmax=1, cmap='viridis', origin='upper')
        plt.colorbar(im, ax=ax, label='P(gate active)')
        ax.set_xlabel('source position j')
        ax.set_ylabel('target position t')
        ax.set_title(f'TN Adjacency Matrix — layer {layer_idx} '
                     f'({block.attn.active_edge_count()} active edges)')
        plt.tight_layout()
        path = f'tn_graph_layer{layer_idx}.png'
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  saved {path}")

except ImportError:
    print("  (install matplotlib to save adjacency image)")

# ---------------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------------
ckpt = {
    'model_state_dict': model.state_dict(),
    'uchars': uchars,
    'BOS': BOS,
    'vocab_size': vocab_size,
    'n_layer': n_layer,
    'n_embd': n_embd,
    'bond_dim': bond_dim,
    'block_size': block_size,
}
torch.save(ckpt, 'tn_gpt.pt')
print("checkpoint saved to tn_gpt.pt")
