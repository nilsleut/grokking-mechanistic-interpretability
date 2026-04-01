# Grokking — Mechanistic Interpretability & Multi-Modulus Fourier Analysis

Reproduction and extension of [Nanda et al. (2023)](https://arxiv.org/abs/2301.05217) *"Progress measures for grokking via mechanistic interpretability"* (ICLR 2023, Oral).

A 1-layer Transformer is trained on modular addition `(a + b) mod p`. After grokking, the model encodes a **Fourier multiplication algorithm**: it represents inputs as rotations on a circle using a sparse set of key frequencies, then combines them via trigonometric identities to compute the sum. This project (1) replicates Nanda's four core findings on `p = 97` and (2) extends the analysis across six prime moduli to test whether the chosen frequencies are structurally bound to `p`.

---

## Results

### Part 1 — Replication on mod 97

Four lines of evidence that the trained model uses the Fourier multiplication algorithm:

#### 1. Sparse Fourier Spectrum in W_E

The embedding matrix `W_E ∈ ℝ^{97 × 128}` is sparse in the Fourier basis. Five key frequencies dominate: **k = {3, 7, 36, 44, 45}**. The top-10 Fourier components account for the majority of total norm (Gini coefficient = 0.605).

<img width="2083" height="742" alt="fourier_01_embedding_spectrum" src="https://github.com/user-attachments/assets/4affbbfc-5f59-487f-b97d-25a7b1311df4" />

#### 2. Grid Structure in W_E · W_Lᵀ

The 2D FFT of the linear prediction matrix `W_E · W_Lᵀ` shows a clear grid pattern — bright lines at exactly the key frequencies in both row and column dimensions. This is the signature of the algorithm's structure: predictions take the form `cos(k·a) · cos(k·c)`, encoding modular addition as rotation.

<img width="2040" height="887" alt="fourier_02_we_wl_heatmap" src="https://github.com/user-attachments/assets/ee6d1b35-fb67-4785-9398-aa230f2f6eac" />

#### 3. Ablation in Fourier Space

Removing key frequencies from `W_E` collapses performance to chance level. Retaining only the key frequencies preserves most performance. Each individual key frequency is necessary.

| Condition | Accuracy | Δ vs. Baseline |
|---|---|---|
| Baseline (original) | 100.0% | — |
| Only key frequencies | 82.8% | −17.2 pp |
| **Key frequencies removed** | **1.7%** | **−98.3 pp** |
| Without k=3 | 40.3% | −59.7 pp |
| Without k=7 | 38.5% | −61.5 pp |
| Without k=36 | 31.8% | −68.2 pp |
| Without k=44 | 79.8% | −20.2 pp |
| Without k=45 | 71.3% | −28.7 pp |

Removing all key frequencies reduces accuracy to 1.7% — consistent with random guessing over 97 classes (chance = 1/97 ≈ 1.03%). `k=36` is the single most critical frequency.

<img width="2085" height="740" alt="fourier_03_ablation" src="https://github.com/user-attachments/assets/18087642-53f0-4511-9aae-098050e1ab74" />

#### 4. Low Effective Rank of W_L

The neuron-logit matrix `W_L ∈ ℝ^{97 × 128}` has effective rank ~10. The top-10 singular values capture **95.9% of total energy** — consistent with Nanda's prediction of 5 key frequencies × {cos, sin} = 10 dominant directions. Projecting `W_L` onto the 10-dimensional Fourier basis of the key frequencies explains **95.4% of variance**.

<img width="2085" height="742" alt="fourier_04_neuron_logit" src="https://github.com/user-attachments/assets/2c928d4c-5e0f-4a6c-8c97-ad20f60e4c5c" />

---

### Part 2 — Multi-Modulus Extension

**Research question:** Are the key frequencies structurally bound to `p`, or are they chosen arbitrarily?

Transformers were trained on `p ∈ {41, 59, 71, 83, 97, 113}` with 5 seeds each (25,000 steps, train fraction 0.25, weight decay 1.0). `p = 41` and `p = 59` did not reliably grok under these hyperparameters and are excluded from the main analysis. Results for the four fully-grokked moduli:

| p | Grokked | val_acc | Gini | Consensus key frequencies |
|---|---|---|---|---|
| 71 | 4/5 | 0.850 | 0.218 | k=7 (k/p=0.099), k=8 (0.113), k=29 (0.408), k=33 (0.465) |
| 83 | 5/5 | 1.000 | 0.470 | k=3 (k/p=0.036), k=18 (0.217), k=22 (0.265), k=23 (0.277) |
| 97 | 5/5 | 0.996 | 0.334 | k=3 (k/p=0.031), k=20 (0.206), k=35 (0.361), k=36 (0.371) |
| 113 | 5/5 | 0.999 | 0.243 | k=26 (k/p=0.230), k=36 (0.319) |

#### Key Frequencies per Modulus

<img width="2234" height="3642" alt="multimod_01_frequencies" src="https://github.com/user-attachments/assets/48a69514-7af5-45d7-983b-4286bf269565" />

#### k/p Ratios — Are They Structurally Bound?

<img width="2231" height="889" alt="multimod_02_ratios" src="https://github.com/user-attachments/assets/18e20dfe-1bcb-4c4b-a84b-66519d258ed1" />

#### Fourier Power Heatmap Across Moduli

<img width="1900" height="742" alt="multimod_03_heatmap" src="https://github.com/user-attachments/assets/80848891-7f2b-41eb-a18d-055eb9d5e54c" />

#### Finding

**Key frequencies are within-p stable but not universally structured across p.**

- Within a given `p`, the same key frequencies are selected consistently across all 5 seeds — the Fourier algorithm is a deterministic outcome of training, not an accident.
- Across different `p`, the k/p ratios do not cluster at simple fractions (1/4, 1/3, 1/2). The heatmap shows no consistent vertical stripes across moduli.
- Two frequencies recur across moduli: **k=3** appears as a key frequency for both p=83 and p=97; **k=36** for both p=97 and p=113. These exceptions may reflect coincidence given the small sample size.
- The histogram of k/p values (all grokked models pooled) shows a slight peak near k/p ≈ 0.20, but the distribution is broadly uniform without strong structure.

**Conclusion:** Frequency selection is p-specific. Each prime induces its own Fourier basis; there is no universal harmonic structure governing which frequencies the model prefers.

---

## Methods

**Architecture** — 1-layer Transformer with multi-head attention (d_model=128, 4 heads), trained on the full set of p² addition pairs.

**Dataset** — All pairs `(a, b, a+b mod p)` for `a, b ∈ {0, ..., p−1}`. Train fraction 0.25–0.30 depending on experiment.

**Training** — AdamW (lr=1e-3, β=(0.9, 0.98)), weight decay=1.0, linear warmup over 500 steps.

**Fourier analysis** — Key frequencies identified by L2-norm of FFT components along the token dimension of `W_E`. Ablations performed by zeroing non-selected Fourier components in `W_E` and evaluating on the full dataset.

**Multi-modulus experiment** — 6 prime moduli, 5 seeds each, 25,000 steps. Consensus key frequencies defined as those appearing in ≥3/5 seeds.

---

## Relation to Nanda et al. (2023)

Nanda et al. trained on `p = 113` and fully reverse-engineered the Fourier multiplication algorithm. This project:

1. **Replicates** their four core findings on `p = 97` with an independent implementation.
2. **Extends** the analysis to ask whether the specific frequencies chosen by the model are structurally determined by `p`. This question is raised but not answered in the original paper. The finding here is that no universal harmonic structure governs frequency selection across primes.

---

## Repository Structure

```
├── grokking_correct.py               # Original model definition and training
├── grokking_robustness.py            # Seed robustness analysis (frequency dynamics)
├── fourier_analysis.py               # Part 1: mechanistic interpretability on mod 97
├── modulus_frequency_analysis.py     # Part 2: multi-modulus extension
├── checkpoints/
│   └── step_020000.pt                # Trained model checkpoint (mod 97)
├── multimod_results/                 # Multi-modulus checkpoints and outputs
└── outputs/                          # Figures and summary files
```

---

## References

- Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). *Progress measures for grokking via mechanistic interpretability*. ICLR 2023. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Power, A., Katarzyna, Y., Edwards, H., Abu-El-Haija, S., & Dieterle, D. (2022). *Grokking: Generalization beyond overfitting on small algorithmic datasets*. [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- Elhage, N., et al. (2021). *A mathematical framework for transformer circuits*. Anthropic. [link](https://transformer-circuits.pub/2021/framework/index.html)
