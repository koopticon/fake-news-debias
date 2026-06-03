# Dynamic Attribution-Gated Suppression for Cross-Dataset Fake News Detection

**AIN3002 Deep Learning — Spring 2026**  
**Author:** Jawad Konialee

---
<img width="9363" height="6623" alt="poster for deep learning project-1" src="https://github.com/user-attachments/assets/3727cee9-e27e-45ee-a88a-3f0b3e29a008" />


## Project Overview

Fake news detectors trained on one dataset collapse on different datasets because they memorize publisher-specific cues ("Reuters = real") instead of learning what makes content deceptive. This project proposes **Dynamic Attribution-Gated Suppression (DAGS)** — a closed-loop training mechanism that identifies and suppresses publisher-specific tokens using both statistical signals and the model's own gradient-based reliance, updated every epoch.

**Key finding:** DAGS achieves the highest cross-dataset accuracy (0.754) and lowest prediction variance (±0.002) among all compared methods, while gate analysis proves it releases 28 geopolitical tokens that static methods wrongly suppress.

---

## Project Structure

```
fake-news-debiasing/
├── dataset.py          — data loading, tokenization, vocab, synthetic + real loaders
├── model.py            — Deep Averaging Network (DAN): embed → mean-pool → MLP
├── spurious.py         — DAGS: the novel contribution (entropy + PMI + gradient saliency gate)
├── experiment.py       — runs all 4 methods, produces results table
├── requirements.txt    — dependencies
├── data/
│   ├── isot/
│   │   ├── True.csv    — download from Kaggle (see below)
│   │   └── Fake.csv    — download from Kaggle (see below)
│   └── cross/
│       └── WELFake_Dataset.csv  — download from Kaggle (see below)
└── results/            — experiment outputs saved here automatically
```

---

## Setup

```bash
pip install -r requirements.txt
```

Requirements: `torch`, `numpy` — CPU only, no GPU needed.

---

## How to Run

### Option 1 — Synthetic mode (no data needed, runs immediately)

```bash
python experiment.py --mode synthetic
```

Runs a controlled experiment with synthetic publisher-bias data. Demonstrates the full pipeline including the closed-loop gate. Expected result: baseline gap 0.438 → DAGS gap 0.032 (92.6% reduction).

### Option 2 — Real data mode (ISOT → WELFake cross-dataset)

**Step 1: Download the datasets**

- ISOT Fake News: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset  
  Download `True.csv` and `Fake.csv`, place in `data/isot/`

- WELFake: https://www.kaggle.com/datasets/saurabhshahane/fake-news-classification  
  Download `WELFake_Dataset.csv`, place in `data/cross/`

**Step 2: Run the experiment**

```bash
python experiment.py --mode real
```

Runs 4 methods × 3 seeds, prints the results table, saves to `results/`.

**Step 3: Results**

Output table shows in-dataset accuracy, cross-dataset accuracy, and gap for:
- Baseline (no suppression)
- Static entropy + hard mask
- Static entropy + soft mask
- **DAGS (ours)** — dynamic attribution-gated suppression

---

## The Novel Contribution — DAGS (spurious.py)

Standard debiasing methods compute a suspicious token list once before training and apply it forever. DAGS replaces this with a **closed loop**:

1. **Initialize** the gate from publisher entropy (low entropy = publisher fingerprint)
2. **Train** one epoch
3. **Compute gradient saliency** — which tokens is the model actually relying on right now?
4. **Update the gate** combining: publisher concentration + label PMI + gradient saliency
5. **Repeat** from step 2

Tokens are suppressed strongly only when they are **both** publisher-specific **and** the model is relying on them. This prevents suppressing content-meaningful tokens that happen to be publisher-concentrated (e.g., rohingya, myanmar, tariffs — real geopolitical content that Reuters covers but the model doesn't cheat with).

**Gate update formula:**
```
bias_score(t) = α·C_pub(t) + β·|PMI(t)| + γ·S_saliency(t)
gate(t) ← μ·gate(t) + (1-μ)·bias_score(t)
embedding(t) = (1 - λ·sigmoid(gate(t))) · embedding(t)
```

---

## Results

ISOT (stratified, 7 publishers) → WELFake non-Reuters, 3 seeds:

| Method | In-dataset | Cross-dataset | Gap |
|--------|-----------|---------------|-----|
| Baseline | 0.965 ± 0.001 | 0.751 ± 0.005 | 0.214 ± 0.006 |
| Static Hard Mask | 0.961 ± 0.005 | 0.747 ± 0.000 | 0.214 ± 0.005 |
| Static Soft Mask | 0.961 ± 0.002 | 0.751 ± 0.005 | 0.209 ± 0.004 |
| **DAGS (ours)** | **0.965 ± 0.002** | **0.754 ± 0.003** ★ | **0.210 ± 0.002** † |

★ Highest cross-dataset accuracy  
† Lowest variance — most stable generalization across seeds

---

## Model Architecture

**Deep Averaging Network (DAN)** — trained entirely from scratch:
- No CNN, no fine-tuning, no pretrained weights
- Embedding layer (vocab × 64) → mean pooling → MLP (128 hidden) → 2 classes
- Vocabulary: 5,000 tokens, max length: 400 tokens
- Optimizer: Adam, lr=1e-3, 5 epochs

---

## File Notes

- `sanity_check.py` — used during development to verify data loading and model correctness
- `threshold_test.py` — used during development to tune entropy threshold for publisher detection
- These files are not needed to run the main experiment but are included for reproducibility

---

## Citation / Dataset Credits

- ISOT: Ahmed et al., "Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques," 2018
- WELFake: Verma et al., "WELFake: Word Embedding over Linguistic Features for Fake News Detection," IEEE TCSS, 2021
