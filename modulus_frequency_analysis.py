"""
Multi-Modulus Fourier-Analyse
=============================
Trainiert Grokking-Transformer auf verschiedenen Primzahlen p und
extrahiert jeweils die Key-Frequenzen. Ziel: Sind die Frequenzen
strukturell zu p gebunden, oder zufällig?

Hypothese (aus Nanda et al. 2023):
    Das Modell wählt Frequenzen k die "gut zu p passen" —
    d.h. k und p sind teilerfremd, und k liegt nahe an p/m
    für kleine m (harmonische Struktur).

Verwendung
----------
  python modulus_frequency_analysis.py

    --moduli      Komma-getrennte Liste von Primzahlen (default: 41,59,71,83,97,113)
    --steps       Trainingsschritte pro Modulus (default: 15000)
    --train-frac  Anteil Trainingsdaten (default: 0.30)
    --wd          Weight Decay (default: 1.0)
    --seeds       Anzahl Seeds pro Modulus für Robustheit (default: 3)
    --out-dir     Output-Ordner (default: multimod_results)

Outputs
-------
  multimod_results/
    train_p{p}_s{seed}.pt          -- Checkpoints
    multimod_01_frequencies.png    -- Key-Freq. pro p
    multimod_02_ratios.png         -- k/p Verhältnisse
    multimod_03_heatmap.png        -- Freq-Matrix über p × seed
    multimod_summary.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, TensorDataset, random_split

# Farben
C = ['#7c6af7', '#e05a6a', '#4caf7d', '#f7a46a', '#4ab8d4',
     '#c77dff', '#f4a261', '#2a9d8f']
C_BG = '#f9f9fb'

# ══════════════════════════════════════════════════════════════
# Minimaler Transformer — unabhängig von grokking_correct.py
# (damit das Script überall läuft)
# ══════════════════════════════════════════════════════════════

class MinimalConfig:
    def __init__(self, p, d_model=128, n_heads=4, n_layers=1,
                 batch_size=512, lr=1e-3):
        self.p          = p
        self.vocab_size = p + 1        # 0..p-1 + "=" Token
        self.d_model    = d_model
        self.n_heads    = n_heads
        self.n_layers   = n_layers
        self.batch_size = batch_size
        self.lr         = lr
        self.device     = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.betas      = (0.9, 0.98)


class MLP(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 4 * d),
            torch.nn.GELU(),
            torch.nn.Linear(4 * d, d),
        )
    def forward(self, x):
        return self.net(x)


class TransformerBlock(torch.nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.mlp  = MLP(d)
        self.ln1  = torch.nn.LayerNorm(d)
        self.ln2  = torch.nn.LayerNorm(d)

    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                           need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class GrokkingModel(torch.nn.Module):
    def __init__(self, cfg: MinimalConfig):
        super().__init__()
        self.cfg     = cfg
        self.tok_emb = torch.nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = torch.nn.Embedding(3, cfg.d_model)
        self.blocks  = torch.nn.ModuleList(
            [TransformerBlock(cfg.d_model, cfg.n_heads)
             for _ in range(cfg.n_layers)]
        )
        self.ln_f    = torch.nn.LayerNorm(cfg.d_model)
        self.head    = torch.nn.Linear(cfg.d_model, cfg.p, bias=False)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h   = self.tok_emb(x) + self.pos_emb(pos)
        for blk in self.blocks:
            h = blk(h)
        h = self.ln_f(h)
        return self.head(h[:, -1])   # letztes Token → Logits


# ══════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════

def make_dataset(p, train_frac, seed, batch_size):
    pairs  = [(a, b, (a + b) % p) for a in range(p) for b in range(p)]
    inputs = torch.tensor([[a, b, p] for a, b, _ in pairs], dtype=torch.long)
    labels = torch.tensor([c for _, _, c in pairs],         dtype=torch.long)
    ds     = TensorDataset(inputs, labels)
    n_tr   = int(len(ds) * train_frac)
    torch.manual_seed(seed)
    tr, va = random_split(ds, [n_tr, len(ds) - n_tr])
    return (DataLoader(tr, batch_size, shuffle=True),
            DataLoader(va, batch_size, shuffle=False))


# ══════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    c, t = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        c += (model(x).argmax(-1) == y).sum().item()
        t += y.size(0)
    return c / t


def train(cfg: MinimalConfig, train_frac: float, weight_decay: float,
          n_steps: int, seed: int, ckpt_path: Path,
          target_val_acc: float = 0.97) -> dict:
    """
    Trainiert bis n_steps oder bis val_acc >= target_val_acc.
    Gibt {'val_acc', 'train_acc', 'step', 'grokked'} zurück.
    """
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        print(f"    [geladen] step={ckpt['step']}  val={ckpt['val_acc']:.3f}")
        return ckpt

    torch.manual_seed(seed)
    tr_ld, va_ld = make_dataset(cfg.p, train_frac, seed, cfg.batch_size)
    model = GrokkingModel(cfg).to(cfg.device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                             weight_decay=weight_decay, betas=cfg.betas)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500))

    def inf(ld):
        while True: yield from ld

    step, best_val = 0, 0.0
    log_every = max(1000, n_steps // 10)

    for x, y in inf(tr_ld):
        if step >= n_steps:
            break
        model.train()
        x, y = x.to(cfg.device), y.to(cfg.device)
        F.cross_entropy(model(x), y).backward()
        opt.step(); opt.zero_grad(); sched.step()
        step += 1

        if step % log_every == 0 or step == n_steps:
            va = eval_acc(model, va_ld, cfg.device)
            tr = eval_acc(model, tr_ld, cfg.device)
            best_val = max(best_val, va)
            print(f"    step {step:6d}  train={tr:.3f}  val={va:.3f}")
            if va >= target_val_acc:
                print(f"    → Grokked bei step {step}!")
                break

    va  = eval_acc(model, va_ld, cfg.device)
    tr  = eval_acc(model, tr_ld, cfg.device)
    result = {
        'step':        step,
        'seed':        seed,
        'p':           cfg.p,
        'val_acc':     va,
        'train_acc':   tr,
        'grokked':     va >= target_val_acc,
        'model_state': model.state_dict(),
    }
    torch.save(result, ckpt_path)
    print(f"    → Fertig: val={va:.3f}  grokked={result['grokked']}")
    return result


# ══════════════════════════════════════════════════════════════
# Fourier-Analyse
# ══════════════════════════════════════════════════════════════

def get_key_frequencies(model_state: dict, cfg: MinimalConfig,
                         top_k: int = 5) -> np.ndarray:
    """Extrahiert Top-k Fourier-Frequenzen aus W_E."""
    model = GrokkingModel(cfg)
    model.load_state_dict(model_state)
    W_E = model.tok_emb.weight[:cfg.p].detach().numpy()

    fft_power  = np.abs(np.fft.fft(W_E, axis=0)) ** 2
    mean_power = fft_power.mean(axis=1)
    half       = cfg.p // 2
    top_idx    = np.argsort(mean_power[1:half])[::-1][:top_k] + 1
    return top_idx


def gini(arr: np.ndarray) -> float:
    s = np.sort(arr[arr > 0])
    n = len(s)
    if n == 0: return 0.0
    return (2 * np.sum(np.arange(1, n+1) * s) / (n * s.sum())) - (n+1)/n


def get_embedding_gini(model_state: dict, cfg: MinimalConfig) -> float:
    model = GrokkingModel(cfg)
    model.load_state_dict(model_state)
    W_E = model.tok_emb.weight[:cfg.p].detach().numpy()
    F_E = np.fft.fft(W_E, axis=0)
    norms = np.linalg.norm(np.abs(F_E), axis=1)[:cfg.p // 2]
    return gini(norms[1:])


# ══════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════

def plot_frequencies(results: dict, out_dir: Path):
    """
    results: {p: {'seeds': [seed,...], 'key_freqs': [[k1,k2,...], ...],
                  'val_accs': [...], 'ginis': [...]}}
    """
    moduli = sorted(results.keys())
    n_p    = len(moduli)

    fig = plt.figure(figsize=(15, 4 * n_p), facecolor=C_BG)
    fig.suptitle('Key-Frequenzen pro Modulus p\n(Fourier-Multiplikations-Algorithmus)',
                 fontsize=14, fontweight='bold', y=1.01)

    for i, p in enumerate(moduli):
        ax = fig.add_subplot(n_p, 1, i + 1)
        ax.set_facecolor(C_BG)
        half = p // 2
        freqs = np.arange(half)

        # Aggregiere Power über Seeds
        all_powers = []
        for seed_idx, key_f in enumerate(results[p]['key_freqs']):
            model_state = results[p]['model_states'][seed_idx]
            cfg = MinimalConfig(p)
            model = GrokkingModel(cfg)
            model.load_state_dict(model_state)
            W_E = model.tok_emb.weight[:p].detach().numpy()
            fft_power = np.abs(np.fft.fft(W_E, axis=0)) ** 2
            all_powers.append(fft_power.mean(axis=1)[:half])

        mean_power = np.mean(all_powers, axis=0)
        mean_power /= mean_power[1:].max()   # normalisieren

        # Konsens-Key-Frequenzen (in ≥ 50% der Seeds)
        from collections import Counter
        all_kf = [k for kf in results[p]['key_freqs'] for k in kf]
        freq_count = Counter(all_kf)
        n_seeds = len(results[p]['seeds'])
        consensus_k = {k for k, cnt in freq_count.items()
                       if cnt >= max(1, n_seeds // 2)}

        # Bars
        bar_colors = [C[0] if k in consensus_k else '#cccccc'
                      for k in freqs]
        ax.bar(freqs, mean_power, color=bar_colors, width=0.8, alpha=0.85)

        # Annotate consensus keys
        for k in sorted(consensus_k):
            if k < half:
                ax.text(k, mean_power[k] + 0.02, f'k={k}',
                        ha='center', va='bottom', fontsize=7,
                        color=C[0], fontweight='bold')

        # k/p Verhältnisse als sekundäre Info
        ratio_str = '  '.join(
            f'k={k} → k/p={k/p:.3f}'
            for k in sorted(consensus_k) if k < half
        )
        val_mean = np.mean(results[p]['val_accs'])
        gini_mean = np.mean(results[p]['ginis'])
        ax.set_title(
            f'p={p}  |  val_acc={val_mean:.3f}  |  Gini={gini_mean:.3f}\n'
            f'{ratio_str}',
            fontsize=9, loc='left'
        )
        ax.set_xlabel('Frequenz k', fontsize=9)
        ax.set_ylabel('Norm. Power', fontsize=9)
        ax.set_xlim(-0.5, half - 0.5)
        ax.grid(axis='y', alpha=0.2, linestyle='--')

    plt.tight_layout()
    out = out_dir / 'multimod_01_frequencies.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {out.name}")


def plot_ratios(results: dict, out_dir: Path):
    """
    Scatter: k/p über alle p und Seeds.
    Zeigt ob Frequenzen bei bestimmten Verhältnissen clustern.
    """
    moduli = sorted(results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor=C_BG)
    fig.suptitle('k/p Verhältnisse — Sind Key-Frequenzen strukturell gebunden?',
                 fontsize=13, fontweight='bold')

    # ── Links: Scatter k/p über p ─────────────────────────
    ax = axes[0]
    ax.set_facecolor(C_BG)

    for i, p in enumerate(moduli):
        for seed_idx, key_f in enumerate(results[p]['key_freqs']):
            if results[p]['val_accs'][seed_idx] < 0.90:
                continue  # nur grokkte Modelle
            ratios = np.array(key_f) / p
            ax.scatter([p] * len(ratios), ratios,
                       color=C[i % len(C)], s=60, alpha=0.75,
                       zorder=3, edgecolors='white', linewidths=0.5)

    # Horizontale Linien bei einfachen Brüchen
    for num, den, label in [
        (1, 4, '1/4'), (1, 3, '1/3'), (1, 2, '1/2'),
        (2, 5, '2/5'), (3, 7, '3/7'), (1, 7, '1/7'),
        (1, 14, '1/14'), (3, 8, '3/8'),
    ]:
        val = num / den
        ax.axhline(val, color='#cccccc', linewidth=0.8,
                   linestyle='--', alpha=0.7)
        ax.text(moduli[-1] + 1, val, f' {label}',
                fontsize=7.5, va='center', color='#999999')

    ax.set_xlabel('Modulus p', fontsize=11)
    ax.set_ylabel('k / p', fontsize=11)
    ax.set_title('Key-Frequenz-Verhältnisse k/p\n'
                 '(Cluster = strukturelle Bindung an p)', fontsize=10)
    ax.set_ylim(0, 0.55)
    ax.grid(alpha=0.15, linestyle='--')

    # ── Rechts: Histogram der k/p Werte (alle p zusammen) ─
    ax2 = axes[1]
    ax2.set_facecolor(C_BG)

    all_ratios = []
    for p in moduli:
        for seed_idx, key_f in enumerate(results[p]['key_freqs']):
            if results[p]['val_accs'][seed_idx] < 0.90:
                continue
            all_ratios.extend([k / p for k in key_f])

    if all_ratios:
        ax2.hist(all_ratios, bins=25, color=C[0], alpha=0.75,
                 edgecolor='white', linewidth=0.5)
        # Vertikale Linien bei einfachen Brüchen
        for num, den, label in [
            (1, 4, '1/4'), (1, 3, '1/3'), (1, 2, '1/2'),
            (2, 5, '2/5'), (1, 7, '1/7'),
        ]:
            ax2.axvline(num/den, color=C[1], linewidth=1.5,
                        linestyle='--', alpha=0.8)
            ax2.text(num/den, ax2.get_ylim()[1] * 0.9 if ax2.get_ylim()[1] > 0 else 1,
                     f' {label}', fontsize=8, color=C[1], va='top')

        ax2.set_xlabel('k / p', fontsize=11)
        ax2.set_ylabel('Anzahl Key-Frequenzen', fontsize=11)
        ax2.set_title('Verteilung aller k/p Verhältnisse\n'
                      '(Peaks = bevorzugte harmonische Verhältnisse)', fontsize=10)
        ax2.grid(axis='y', alpha=0.2, linestyle='--')

    plt.tight_layout()
    out = out_dir / 'multimod_02_ratios.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {out.name}")


def plot_heatmap(results: dict, out_dir: Path, top_k: int = 5):
    """
    Heatmap: Zeilen = p, Spalten = k/p bins.
    Helligkeit = Power in diesem Frequenz-Bereich.
    """
    moduli = sorted(results.keys())
    n_bins = 24
    bins   = np.linspace(0, 0.5, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    matrix = np.zeros((len(moduli), n_bins))

    for i, p in enumerate(moduli):
        all_powers_binned = np.zeros(n_bins)
        count = 0
        for seed_idx, key_f in enumerate(results[p]['key_freqs']):
            if results[p]['val_accs'][seed_idx] < 0.90:
                continue
            model_state = results[p]['model_states'][seed_idx]
            cfg = MinimalConfig(p)
            model = GrokkingModel(cfg)
            model.load_state_dict(model_state)
            W_E = model.tok_emb.weight[:p].detach().numpy()
            fft_power = np.abs(np.fft.fft(W_E, axis=0)) ** 2
            mean_power = fft_power.mean(axis=1)
            half = p // 2

            # Normalisieren und in Bins
            norm_power = mean_power[1:half] / mean_power[1:half].sum()
            ratios = np.arange(1, half) / p
            for j, (r, pw) in enumerate(zip(ratios, norm_power)):
                bin_idx = np.searchsorted(bins, r) - 1
                if 0 <= bin_idx < n_bins:
                    all_powers_binned[bin_idx] += pw
            count += 1

        if count > 0:
            matrix[i] = all_powers_binned / count

    fig, ax = plt.subplots(figsize=(14, 5), facecolor=C_BG)
    ax.set_facecolor(C_BG)
    fig.suptitle('Fourier-Power-Verteilung über Moduli\n'
                 '(Helle Spalten = bevorzugte k/p-Verhältnisse)',
                 fontsize=13, fontweight='bold')

    im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
                   interpolation='nearest',
                   extent=[0, 0.5, len(moduli) - 0.5, -0.5])
    plt.colorbar(im, ax=ax, label='Norm. Fourier-Power', shrink=0.8)

    ax.set_yticks(np.arange(len(moduli)))
    ax.set_yticklabels([f'p={p}' for p in moduli], fontsize=10)
    ax.set_xlabel('k / p', fontsize=11)

    # Markiere einfache Brüche
    for num, den, label in [
        (1, 7, '1/7'), (1, 4, '¼'), (1, 3, '⅓'),
        (2, 5, '2/5'), (1, 2, '½'),
    ]:
        ax.axvline(num/den, color='#4ab8d4', linewidth=1.2,
                   linestyle='--', alpha=0.7)
        ax.text(num/den, -0.6, label, ha='center',
                fontsize=8, color='#4ab8d4')

    plt.tight_layout()
    out = out_dir / 'multimod_03_heatmap.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → {out.name}")


def write_summary(results: dict, out_dir: Path, moduli, n_steps, train_frac, wd):
    lines = [
        "Multi-Modulus Fourier-Analyse — Summary",
        "=" * 55,
        f"Moduli:       {moduli}",
        f"Steps:        {n_steps}",
        f"Train Frac:   {train_frac}",
        f"Weight Decay: {wd}",
        "",
        f"{'p':>5}  {'grokked':>8}  {'val_acc':>8}  "
        f"{'gini':>6}  {'key_freqs (consensus)'}",
        "─" * 70,
    ]

    from collections import Counter
    all_ratios_global = []

    for p in sorted(results.keys()):
        r = results[p]
        n_seeds   = len(r['seeds'])
        n_grokked = sum(1 for v in r['val_accs'] if v >= 0.90)
        val_mean  = np.mean(r['val_accs'])
        gini_mean = np.mean(r['ginis'])

        all_kf    = [k for kf in r['key_freqs'] for k in kf]
        cnt       = Counter(all_kf)
        consensus = sorted(k for k, c in cnt.items()
                           if c >= max(1, n_seeds // 2))
        ratio_str = ' '.join(f'{k}({k/p:.3f})' for k in consensus)

        lines.append(
            f"{p:>5}  {n_grokked}/{n_seeds:>5}  {val_mean:>8.3f}  "
            f"{gini_mean:>6.3f}  {ratio_str}"
        )

        for v, kf in zip(r['val_accs'], r['key_freqs']):
            if v >= 0.90:
                all_ratios_global.extend([k/p for k in kf])

    if all_ratios_global:
        lines += [
            "",
            "k/p Statistik (alle gegrookkten Modelle)",
            f"  Mean:   {np.mean(all_ratios_global):.4f}",
            f"  Std:    {np.std(all_ratios_global):.4f}",
            f"  Median: {np.median(all_ratios_global):.4f}",
            f"  Min:    {np.min(all_ratios_global):.4f}",
            f"  Max:    {np.max(all_ratios_global):.4f}",
        ]

        # Cluster-Analyse
        lines += ["", "Häufige k/p-Verhältnisse (±0.02 Toleranz):"]
        checkpoints_ratios = [
            (1/7, '1/7≈0.143'), (1/6, '1/6≈0.167'), (1/5, '1/5=0.200'),
            (1/4, '1/4=0.250'), (2/7, '2/7≈0.286'), (1/3, '1/3≈0.333'),
            (3/8, '3/8=0.375'), (2/5, '2/5=0.400'), (3/7, '3/7≈0.429'),
            (1/2, '1/2=0.500'),
        ]
        for target, label in checkpoints_ratios:
            close = sum(1 for r in all_ratios_global if abs(r - target) < 0.02)
            if close > 0:
                pct = close / len(all_ratios_global) * 100
                lines.append(f"  {label:>12}:  {close:3d} Freq. ({pct:4.1f}%)")

    lines += [
        "",
        "Referenz: Nanda et al. (2023), ICLR",
    ]

    out = out_dir / 'multimod_summary.txt'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"  → {out.name}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--moduli',     default='41,59,71,83,97,113',
                        help='Komma-getrennte Primzahlen')
    parser.add_argument('--steps',      type=int, default=15000)
    parser.add_argument('--train-frac', type=float, default=0.30)
    parser.add_argument('--wd',         type=float, default=1.0)
    parser.add_argument('--seeds',      type=int, default=3)
    parser.add_argument('--out-dir',    type=Path,
                        default=Path('multimod_results'))
    args = parser.parse_args()

    moduli    = [int(x) for x in args.moduli.split(',')]
    out_dir   = args.out_dir
    out_dir.mkdir(exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Moduli: {moduli}  |  Steps: {args.steps}  "
          f"|  Seeds: {args.seeds}  |  WD: {args.wd}\n")

    # ── Training ─────────────────────────────────────────────
    results = {}

    for p in moduli:
        print(f"{'━'*55}")
        print(f"p = {p}")
        print(f"{'━'*55}")
        results[p] = {
            'seeds':        [],
            'key_freqs':    [],
            'val_accs':     [],
            'ginis':        [],
            'model_states': [],
        }

        for s in range(args.seeds):
            seed = 42 + s
            print(f"  Seed {seed}:")
            ckpt_path = out_dir / f"train_p{p}_s{seed}.pt"
            cfg = MinimalConfig(p)

            ckpt = train(cfg, args.train_frac, args.wd,
                         args.steps, seed, ckpt_path)

            key_f = get_key_frequencies(ckpt['model_state'], cfg)
            g     = get_embedding_gini(ckpt['model_state'], cfg)

            results[p]['seeds'].append(seed)
            results[p]['key_freqs'].append(list(key_f))
            results[p]['val_accs'].append(ckpt['val_acc'])
            results[p]['ginis'].append(g)
            results[p]['model_states'].append(ckpt['model_state'])

            print(f"    key_freqs={list(key_f)}  "
                  f"ratios={[round(k/p,3) for k in key_f]}  "
                  f"gini={g:.3f}")

    # ── Plots ────────────────────────────────────────────────
    print(f"\n{'━'*55}")
    print("Plots werden erstellt ...")
    print(f"{'━'*55}")

    plot_frequencies(results, out_dir)
    plot_ratios(results, out_dir)
    plot_heatmap(results, out_dir)
    write_summary(results, out_dir, moduli,
                  args.steps, args.train_frac, args.wd)

    print(f"\n✓ Fertig. Outputs in: {out_dir}/\n")


if __name__ == '__main__':
    main()