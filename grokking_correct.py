"""
Grokking Phasendiagramm: train_frac × weight_decay
====================================================
Misst für jede Kombination:
  - ob Grokking auftritt (Val Acc > 95%)
  - wann es auftritt (Grokking-Step)
  - ob direkte Generalisierung auftritt (Val Acc > 95% innerhalb der ersten 10% der Steps)

Ergebnis: 2D-Heatmap die zeigt wo die Grenze zwischen
  MEMORISIERUNG / GROKKING / DIREKTE GENERALISIERUNG liegt.

Laufzeit (CPU, 5×5 Grid, 30k steps):  ~60–90 Minuten
Laufzeit (GPU):                        ~10–15 Minuten
"""

import math
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ══════════════════════════════════════════════════════════════
# Konfiguration
# ══════════════════════════════════════════════════════════════

@dataclass
class Config:
    p: int = 97
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    dropout: float = 0.0
    lr: float = 1e-3
    betas: tuple = (0.9, 0.98)
    num_steps: int = 15_000      # pro Run — erhöhen für vollständigeres Bild
    batch_size: int = 512
    log_interval: int = 250
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# Phasendiagramm-Achsen
TRAIN_FRACS    = [0.10, 0.20, 0.30, 0.40, 0.50]   # x-Achse
WEIGHT_DECAYS  = [0.1,  0.5,  1.0,  2.0,  5.0]    # y-Achse

# Schwellenwerte
GROK_THRESHOLD       = 0.95   # Val Acc ab dem Grokking als eingetreten gilt
DIRECT_GEN_CUTOFF    = 0.10   # Anteil der Steps innerhalb dem direkte Gen. zählt


# ══════════════════════════════════════════════════════════════
# Modell (identisch zu grokking_correct.py)
# ══════════════════════════════════════════════════════════════

class GrokkingTransformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        vocab_size = cfg.p + 1
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(3, cfg.d_model)
        self.attn    = nn.MultiheadAttention(cfg.d_model, cfg.n_heads,
                                             dropout=cfg.dropout, batch_first=True)
        self.norm1   = nn.LayerNorm(cfg.d_model)
        self.norm2   = nn.LayerNorm(cfg.d_model)
        self.ff      = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )
        self.head = nn.Linear(cfg.d_model, cfg.p)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, T = x.shape
        pos  = torch.arange(T, device=x.device).unsqueeze(0)
        h    = self.tok_emb(x) + self.pos_emb(pos)
        h2, _ = self.attn(h, h, h)
        h    = self.norm1(h + h2)
        h    = self.norm2(h + self.ff(h))
        return self.head(h[:, -1, :])


# ══════════════════════════════════════════════════════════════
# Datensatz
# ══════════════════════════════════════════════════════════════

def make_dataset(cfg: Config, train_frac: float):
    pairs   = [(a, b, (a + b) % cfg.p)
               for a in range(cfg.p) for b in range(cfg.p)]
    inputs  = torch.tensor([[a, b, cfg.p] for a, b, _ in pairs], dtype=torch.long)
    labels  = torch.tensor([c for _, _, c in pairs],             dtype=torch.long)
    dataset = TensorDataset(inputs, labels)
    n_train = int(len(dataset) * train_frac)
    n_val   = len(dataset) - n_train
    torch.manual_seed(cfg.seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    return (DataLoader(train_ds, cfg.batch_size, shuffle=True),
            DataLoader(val_ds,   cfg.batch_size, shuffle=False))


@torch.no_grad()
def eval_acc(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(-1) == y).sum().item()
        total   += y.size(0)
    return correct / total


# ══════════════════════════════════════════════════════════════
# Einzelner Run
# ══════════════════════════════════════════════════════════════

@dataclass
class RunResult:
    train_frac:   float
    weight_decay: float
    grok_step:    Optional[int]    # None = kein Grokking innerhalb num_steps
    direct_gen:   bool             # True = sofort generalisiert
    final_val:    float
    val_curve:    list = field(default_factory=list)


def run_single(cfg: Config, train_frac: float, weight_decay: float) -> RunResult:
    torch.manual_seed(cfg.seed)
    train_loader, val_loader = make_dataset(cfg, train_frac)
    model = GrokkingTransformer(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=cfg.lr,
                                  weight_decay=weight_decay,
                                  betas=cfg.betas)

    def lr_lambda(s):
        warmup = 500
        # Linearer Decay nach Warmup
        return min(1.0, s / warmup) * max(0.1, 1.0 - s / cfg.num_steps)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    val_curve  = []
    grok_step  = None
    direct_gen = False
    direct_cutoff_step = int(cfg.num_steps * DIRECT_GEN_CUTOFF)

    def infinite(loader):
        while True:
            yield from loader

    step = 0
    for x, y in infinite(train_loader):
        if step >= cfg.num_steps:
            break
        model.train()
        x, y = x.to(cfg.device), y.to(cfg.device)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1

        if step % cfg.log_interval == 0:
            v_acc = eval_acc(model, val_loader, cfg.device)
            val_curve.append(v_acc)

            # Direkte Generalisierung: Val > Threshold früh
            if v_acc >= GROK_THRESHOLD and step <= direct_cutoff_step:
                direct_gen = True

            # Grokking: Val > Threshold spät (nach direct_cutoff_step)
            if v_acc >= GROK_THRESHOLD and grok_step is None and not direct_gen:
                grok_step = step

    final_val = val_curve[-1] if val_curve else 0.0
    return RunResult(train_frac, weight_decay, grok_step, direct_gen, final_val, val_curve)


# ══════════════════════════════════════════════════════════════
# Phasendiagramm
# ══════════════════════════════════════════════════════════════

def classify_run(r: RunResult) -> str:
    """
    Drei Regime:
      DIRECT   — sofort generalisiert (kein Memorisierungs-Plateau)
      GROKKING — Plateau dann Sprung
      STUCK    — kein Grokking innerhalb num_steps
    """
    if r.direct_gen:
        return 'DIRECT'
    elif r.grok_step is not None:
        return 'GROKKING'
    else:
        return 'STUCK'


def run_phase_diagram(cfg: Config):
    n_frac = len(TRAIN_FRACS)
    n_wd   = len(WEIGHT_DECAYS)
    total  = n_frac * n_wd

    # Ergebnismatrizen
    phase_matrix    = np.empty((n_wd, n_frac), dtype=object)
    grokstep_matrix = np.full((n_wd, n_frac), np.nan)
    finalval_matrix = np.zeros((n_wd, n_frac))

    print(f"Phasendiagramm: {n_frac}×{n_wd} = {total} Runs")
    print(f"Steps pro Run: {cfg.num_steps:,}  |  Device: {cfg.device}")
    print(f"{'Run':>4}  {'train_frac':>10}  {'wd':>6}  {'Phase':>10}  "
          f"{'grok_step':>10}  {'final_val':>10}")
    print("─" * 58)

    run_idx = 0
    for j, tf in enumerate(TRAIN_FRACS):
        for i, wd in enumerate(WEIGHT_DECAYS):
            run_idx += 1
            r = run_single(cfg, tf, wd)
            phase = classify_run(r)
            phase_matrix[i, j]    = phase
            finalval_matrix[i, j] = r.final_val
            if r.grok_step:
                grokstep_matrix[i, j] = r.grok_step

            gstep_str = f"{r.grok_step:,}" if r.grok_step else "—"
            print(f"{run_idx:4d}  {tf:10.2f}  {wd:6.1f}  {phase:>10}  "
                  f"{gstep_str:>10}  {r.final_val*100:9.1f}%")

    return phase_matrix, grokstep_matrix, finalval_matrix


# ══════════════════════════════════════════════════════════════
# Visualisierung
# ══════════════════════════════════════════════════════════════

def plot_phase_diagram(phase_matrix, grokstep_matrix, finalval_matrix, cfg):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Grokking Phasendiagramm — (a+b) mod {cfg.p}  '
                 f'({cfg.num_steps:,} steps)',
                 fontsize=13, fontweight='bold')

    x_labels = [f'{f:.0%}' for f in TRAIN_FRACS]
    y_labels  = [str(w) for w in WEIGHT_DECAYS]

    # ── Panel 1: Phase ─────────────────────────────────────
    ax = axes[0]
    color_map = {'DIRECT': 0, 'GROKKING': 1, 'STUCK': 2}
    colors    = ['#4caf7d', '#7c6af7', '#e05a6a']
    cmap_disc = mcolors.ListedColormap(colors)
    numeric   = np.array([[color_map[phase_matrix[i, j]]
                           for j in range(len(TRAIN_FRACS))]
                          for i in range(len(WEIGHT_DECAYS))], dtype=float)
    im = ax.imshow(numeric, cmap=cmap_disc, vmin=-0.5, vmax=2.5,
                   aspect='auto', origin='lower')
    ax.set_xticks(range(len(TRAIN_FRACS)));   ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(WEIGHT_DECAYS))); ax.set_yticklabels(y_labels)
    ax.set_xlabel('Train Fraction', fontsize=11)
    ax.set_ylabel('Weight Decay',   fontsize=11)
    ax.set_title('Regime', fontsize=12, fontweight='bold')
    for i in range(len(WEIGHT_DECAYS)):
        for j in range(len(TRAIN_FRACS)):
            ax.text(j, i, phase_matrix[i, j][:4],
                    ha='center', va='center', fontsize=8,
                    color='white', fontweight='bold')
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[0], label='Direct Gen.'),
                       Patch(facecolor=colors[1], label='Grokking'),
                       Patch(facecolor=colors[2], label='Stuck')]
    ax.legend(handles=legend_elements, loc='upper right',
              fontsize=8, framealpha=0.8)

    # ── Panel 2: Grokking-Zeitpunkt ────────────────────────
    ax = axes[1]
    im2 = ax.imshow(grokstep_matrix, cmap='plasma_r', aspect='auto',
                    origin='lower', vmin=0, vmax=cfg.num_steps)
    ax.set_xticks(range(len(TRAIN_FRACS)));   ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(WEIGHT_DECAYS))); ax.set_yticklabels(y_labels)
    ax.set_xlabel('Train Fraction', fontsize=11)
    ax.set_ylabel('Weight Decay',   fontsize=11)
    ax.set_title('Grokking-Zeitpunkt (steps)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=ax, label='Steps bis Grokking')
    for i in range(len(WEIGHT_DECAYS)):
        for j in range(len(TRAIN_FRACS)):
            val = grokstep_matrix[i, j]
            txt = f'{int(val/1000)}k' if not np.isnan(val) else '—'
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

    # ── Panel 3: Final Val Acc ─────────────────────────────
    ax = axes[2]
    im3 = ax.imshow(finalval_matrix * 100, cmap='RdYlGn', aspect='auto',
                    origin='lower', vmin=0, vmax=100)
    ax.set_xticks(range(len(TRAIN_FRACS)));   ax.set_xticklabels(x_labels)
    ax.set_yticks(range(len(WEIGHT_DECAYS))); ax.set_yticklabels(y_labels)
    ax.set_xlabel('Train Fraction', fontsize=11)
    ax.set_ylabel('Weight Decay',   fontsize=11)
    ax.set_title(f'Final Val Acc nach {cfg.num_steps:,} steps',
                 fontsize=12, fontweight='bold')
    plt.colorbar(im3, ax=ax, label='Val Accuracy (%)')
    for i in range(len(WEIGHT_DECAYS)):
        for j in range(len(TRAIN_FRACS)):
            v = finalval_matrix[i, j]
            col = 'white' if v < 0.5 else 'black'
            ax.text(j, i, f'{v*100:.0f}%',
                    ha='center', va='center', fontsize=8,
                    color=col, fontweight='bold')

    plt.tight_layout()
    plt.savefig('grokking_phase_diagram.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Gespeichert: grokking_phase_diagram.png")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    cfg = Config()
    print(f"Config: p={cfg.p}, steps={cfg.num_steps:,}, device={cfg.device}")
    print(f"Grid:   train_fracs={TRAIN_FRACS}")
    print(f"        weight_decays={WEIGHT_DECAYS}\n")

    phase_matrix, grokstep_matrix, finalval_matrix = run_phase_diagram(cfg)
    plot_phase_diagram(phase_matrix, grokstep_matrix, finalval_matrix, cfg)

    # Zusammenfassung
    phases_flat = phase_matrix.flatten()
    n_direct  = (phases_flat == 'DIRECT').sum()
    n_grok    = (phases_flat == 'GROKKING').sum()
    n_stuck   = (phases_flat == 'STUCK').sum()
    print(f"\nZusammenfassung ({len(phases_flat)} Runs):")
    print(f"  DIRECT:   {n_direct:2d}  ({n_direct/len(phases_flat)*100:.0f}%)")
    print(f"  GROKKING: {n_grok:2d}  ({n_grok/len(phases_flat)*100:.0f}%)")
    print(f"  STUCK:    {n_stuck:2d}  ({n_stuck/len(phases_flat)*100:.0f}%)")