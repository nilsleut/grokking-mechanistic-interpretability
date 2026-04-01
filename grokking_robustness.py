"""
Grokking Robustness-Analyse: Dominante Frequenz über Seeds
===========================================================
Prüft ob der Frequenzwechsel (z.B. k=45 → k=14) robust über Seeds ist.

Verwendung
----------
  # Schritt 1: paralleles Training — je Seed ein Terminal
  python grokking_robustness.py --seed 42
  python grokking_robustness.py --seed 43
  python grokking_robustness.py --seed 44
  python grokking_robustness.py --seed 45
  python grokking_robustness.py --seed 46

  # Schritt 2: Plot aller Seeds (wenn alle fertig)
  python grokking_robustness.py --plot
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from torch.utils.data import DataLoader, TensorDataset, random_split

from grokking_correct import GrokkingTransformer, Config


# ══════════════════════════════════════════════════════════════
# Globale Parameter
# ══════════════════════════════════════════════════════════════

SEEDS = [42, 43, 44, 45, 46]

CHECKPOINT_STEPS = [
    500, 700, 900, 1100, 1300, 1500,
    1700, 1900, 2100, 2300, 2500, 3000, 5000,
]

TRAIN_FRAC   = 0.20
WEIGHT_DECAY = 1.0

# Farben pro Seed (colorblind-freundliche Palette)
SEED_COLORS = {
    42: '#7c6af7',  # lila
    43: '#e05a6a',  # rot
    44: '#4caf7d',  # grün
    45: '#f7a46a',  # orange
    46: '#4ab8d4',  # blau
}


def checkpoint_dir(seed: int) -> Path:
    return Path(f"checkpoints_seed{seed}")


# ══════════════════════════════════════════════════════════════
# Dataset & Evaluation
# ══════════════════════════════════════════════════════════════

def make_dataset(cfg: Config, train_frac: float, seed: int):
    pairs  = [(a, b, (a + b) % cfg.p)
              for a in range(cfg.p) for b in range(cfg.p)]
    inputs = torch.tensor([[a, b, cfg.p] for a, b, _ in pairs], dtype=torch.long)
    labels = torch.tensor([c for _, _, c in pairs],             dtype=torch.long)
    dataset = TensorDataset(inputs, labels)
    n_train = int(len(dataset) * train_frac)
    n_val   = len(dataset) - n_train
    torch.manual_seed(seed)
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
# Training mit Checkpoints
# ══════════════════════════════════════════════════════════════

def train_with_checkpoints(cfg: Config, seed: int):
    ckpt_dir = checkpoint_dir(seed)
    ckpt_dir.mkdir(exist_ok=True)

    torch.manual_seed(seed)
    train_loader, val_loader = make_dataset(cfg, TRAIN_FRAC, seed)
    model = GrokkingTransformer(cfg).to(cfg.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=WEIGHT_DECAY,
        betas=cfg.betas,
    )

    def lr_lambda(s):
        return min(1.0, s / 500)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_set = set(CHECKPOINT_STEPS)
    max_step = max(CHECKPOINT_STEPS)

    print(f"\n[Seed {seed}] Training bis Step {max_step} | device={cfg.device}")
    print(f"{'step':>8}  {'train_acc':>10}  {'val_acc':>10}")
    print("─" * 33)

    def infinite(loader):
        while True:
            yield from loader

    step = 0
    for x, y in infinite(train_loader):
        if step >= max_step:
            break
        model.train()
        x, y = x.to(cfg.device), y.to(cfg.device)
        loss = F.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        step += 1

        if step in ckpt_set:
            t_acc = eval_acc(model, train_loader, cfg.device)
            v_acc = eval_acc(model, val_loader,   cfg.device)
            print(f"{step:8d}  {t_acc*100:9.2f}%  {v_acc*100:9.2f}%")
            torch.save({
                'step':        step,
                'seed':        seed,
                'model_state': model.state_dict(),
                'train_acc':   t_acc,
                'val_acc':     v_acc,
            }, ckpt_dir / f"step_{step:06d}.pt")

    print(f"[Seed {seed}] Checkpoints gespeichert in: {ckpt_dir}/")


# ══════════════════════════════════════════════════════════════
# Fourier-Hilfsfunktionen
# ══════════════════════════════════════════════════════════════

def find_dominant_frequency(emb: np.ndarray, p: int) -> int:
    """
    Mittlere FFT-Power über alle Embedding-Dimensionen,
    dann argmax über k=1..p//2 (DC ignorieren).
    """
    fft_power  = np.abs(np.fft.fft(emb, axis=0)) ** 2  # [p, d_model]
    mean_power = fft_power.mean(axis=1)                  # [p]
    half = p // 2
    return int(np.argmax(mean_power[1:half]) + 1)


def load_checkpoint(path: Path, cfg: Config):
    ckpt  = torch.load(path, map_location="cpu")
    model = GrokkingTransformer(cfg)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["train_acc"], ckpt["val_acc"]


# ══════════════════════════════════════════════════════════════
# Frequenz-Extraktion aller Seeds
# ══════════════════════════════════════════════════════════════

def extract_all_seeds(cfg: Config) -> dict:
    """
    Returns:
        {seed: [(step, dominant_k, val_acc, train_acc), ...]}
    """
    results = {}
    for seed in SEEDS:
        ckpt_dir = checkpoint_dir(seed)
        if not ckpt_dir.exists():
            print(f"[Seed {seed}] Kein Checkpoint-Ordner — überspringe.")
            continue

        records = []
        for step in CHECKPOINT_STEPS:
            path = ckpt_dir / f"step_{step:06d}.pt"
            if not path.exists():
                print(f"  [Seed {seed}] Step {step} fehlt — überspringe.")
                continue
            model, t_acc, v_acc = load_checkpoint(path, cfg)
            emb = model.tok_emb.weight[:cfg.p].detach().numpy()
            k   = find_dominant_frequency(emb, cfg.p)
            records.append((step, k, v_acc, t_acc))

        if records:
            results[seed] = records
            print(f"[Seed {seed}] {len(records)} Checkpoints geladen.")
        else:
            print(f"[Seed {seed}] Keine Checkpoints gefunden.")

    return results


# ══════════════════════════════════════════════════════════════
# Plot: alle Seeds auf einem Dual-Axis-Graph
# ══════════════════════════════════════════════════════════════

def plot_robustness(results: dict, cfg: Config):
    if not results:
        print("Keine Daten zum Plotten.")
        return

    fig, ax1 = plt.subplots(figsize=(13, 6))
    fig.suptitle(
        f'Dominante Fourier-Frequenz über Seeds — Robustness-Check\n'
        f'(p={cfg.p}, train_frac={TRAIN_FRAC}, weight_decay={WEIGHT_DECAY})',
        fontsize=13, fontweight='bold'
    )

    ax2 = ax1.twinx()

    # ── Linke Achse: dominante Frequenz ────────────────────────
    ax1.set_ylabel('Dominante Frequenz k', fontsize=11)
    ax1.set_xlabel('Training Step',        fontsize=11)
    ax1.set_ylim(0, cfg.p // 2)
    ax1.grid(True, alpha=0.18, linestyle='--')
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(5))

    # ── Rechte Achse: Val Acc ───────────────────────────────────
    ax2.set_ylabel('Val Accuracy (%)', fontsize=11, color='#888888')
    ax2.set_ylim(-3, 108)
    ax2.tick_params(axis='y', labelcolor='#888888')
    ax2.axhline(95, color='#888888', linewidth=1,
                linestyle=':', alpha=0.6, label='95 % Schwelle')

    for seed, records in sorted(results.items()):
        steps      = [r[0] for r in records]
        dom_freqs  = [r[1] for r in records]
        val_accs   = [r[2] * 100 for r in records]
        color      = SEED_COLORS.get(seed, 'gray')

        # Frequenz — durchgehende Linie mit Schritten (step-Darstellung)
        ax1.step(steps, dom_freqs, where='post',
                 color=color, linewidth=2.3, zorder=3,
                 label=f'Seed {seed}')
        ax1.plot(steps, dom_freqs, 'o',
                 color=color, markersize=5, zorder=4)

        # Val Acc — gestrichelt
        ax2.plot(steps, val_accs, '--',
                 color=color, linewidth=1.3, alpha=0.55,
                 markersize=3)

    # Konsensus-Bereich hervorheben: Spaltenweise prüfen ob alle Seeds
    # an diesem Step einen Sprung zeigen
    all_steps = sorted({r[0] for recs in results.values() for r in recs})
    freq_by_step = {}
    for seed, records in results.items():
        for step, k, *_ in records:
            freq_by_step.setdefault(step, {})[seed] = k

    prev_freqs = {}
    for step in all_steps:
        curr = freq_by_step.get(step, {})
        changed_seeds = {s for s in curr if prev_freqs.get(s) is not None
                         and curr[s] != prev_freqs[s]}
        if len(changed_seeds) >= math.ceil(len(results) * 0.6):
            # Mehrheit der Seeds ändert hier — hervorheben
            step_idx = all_steps.index(step)
            prev_step = all_steps[step_idx - 1] if step_idx > 0 else step
            ax1.axvspan(prev_step, step, alpha=0.12,
                        color='#f7e76a', zorder=1)
        prev_freqs.update(curr)

    # Frequenz-Konsens annotieren (finale stabile Frequenz)
    final_freqs = [records[-1][1] for records in results.values()]
    if len(set(final_freqs)) == 1:
        ax1.axhline(final_freqs[0], color='gray', linewidth=1,
                    linestyle=':', alpha=0.5)
        ax1.text(all_steps[-1], final_freqs[0] + 1,
                 f' k={final_freqs[0]} (Konsens)',
                 fontsize=8, color='gray', va='bottom')

    # Legenden
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper left', fontsize=9, framealpha=0.92,
               ncol=2)

    plt.tight_layout()
    out = 'robustness_frequency.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Gespeichert: {out}")

    # ── Tabellarische Ausgabe ───────────────────────────────────
    print(f"\n{'Step':>6}", end='')
    for seed in sorted(results):
        print(f"  S{seed}-k  S{seed}-val", end='')
    print()
    print('─' * (6 + len(results) * 14))

    for step in all_steps:
        print(f"{step:6d}", end='')
        for seed in sorted(results):
            rec = next((r for r in results[seed] if r[0] == step), None)
            if rec:
                print(f"  {rec[1]:5d}  {rec[2]*100:6.1f}%", end='')
            else:
                print(f"  {'—':>5}  {'—':>6} ", end='')
        print()


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Grokking Robustness-Analyse über mehrere Seeds'
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--seed', type=int, choices=SEEDS, metavar='{' + ','.join(map(str, SEEDS)) + '}',
        help='Seed für diesen Training-Run'
    )
    group.add_argument(
        '--plot', action='store_true',
        help='Alle vorhandenen Checkpoints einlesen und Plot erstellen'
    )
    args = parser.parse_args()

    cfg = Config(num_steps=max(CHECKPOINT_STEPS))

    if args.seed is not None:
        # ── Einzelner Training-Run ──────────────────────────────
        seed = args.seed
        ckpt_dir = checkpoint_dir(seed)

        missing = [s for s in CHECKPOINT_STEPS
                   if not (ckpt_dir / f"step_{s:06d}.pt").exists()]

        if not missing:
            print(f"[Seed {seed}] Alle Checkpoints vorhanden — überspringe Training.")
        else:
            print(f"[Seed {seed}] Fehlende Steps: {missing} — starte Training...")
            train_with_checkpoints(cfg, seed)

    else:
        # ── Plot aller Seeds ────────────────────────────────────
        print("Lese Checkpoints aller Seeds...")
        results = extract_all_seeds(cfg)
        print(f"\n{len(results)} Seeds geladen: {sorted(results.keys())}")
        plot_robustness(results, cfg)


if __name__ == "__main__":
    main()
