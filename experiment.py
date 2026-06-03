"""
experiment.py -- full experiment runner.

Synthetic mode (default, no files required):
    python experiment.py

Real ISOT + WELFake cross-dataset (with dateline stripped -- realistic):
    python experiment.py --real

Real ISOT without dateline stripping (dramatic demo of the bias problem):
    python experiment.py --real --no-strip-dateline

Override cross-dataset path or column names:
    python experiment.py --real --cross-path data/cross/other.csv \\
        --cross-text-col text --cross-label-col label --cross-true-label 1

Produces a 4-row comparison table:
  1. Baseline             -- no suppression
  2. Static hard mask     -- entropy percentile, binary 0/1 (original method, ablation)
  3. Static soft mask     -- entropy-derived weights, no gradient feedback (ablation)
  4. Dynamic gate         -- main method: soft weights updated each epoch via gradient saliency

Real mode runs 3 seeds and reports mean +/- std.
Synthetic mode runs a single seed (for speed).
"""

import argparse
import math
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset import (
    simple_tokenize, build_vocab, encode, pad_batch,
    make_synthetic_dataset, make_cross_dataset,
    load_isot, load_generic_csv,
    load_isot_real, load_isot_by_subject, load_isot_stratified,
    load_cross_dataset, load_welfake_nonreuters,
)
from model   import DeepAveragingNetwork
from spurious import (
    find_spurious_tokens, print_flagged_tokens, make_suppress_mask,
    compute_publisher_concentration, compute_label_pmi,
    make_soft_suppress_mask, compute_gradient_saliency,
    DynamicSuppressionGate, print_gate_analysis,
)


# ── hyper-parameters ───────────────────────────────────────────────────────────

SEED        = 42
EMBED_DIM   = 64
HIDDEN_DIM  = 128
EPOCHS      = 5
BATCH_SIZE  = 64
LR          = 1e-3
ENTROPY_PCT = 20      # bottom 20% by publisher-entropy -> hard mask

# Soft / dynamic suppression
LAMBDA           = 0.7
ALPHA            = 0.5    # publisher concentration weight in bias score
BETA             = 0.2    # label PMI weight
GAMMA            = 0.3    # gradient saliency weight (the dynamic part)
EMA_DECAY        = 0.85
SALIENCY_BATCHES = 15

# Synthetic mode sizes
MAX_LEN_SYN   = 200
MAX_VOCAB_SYN = 15_000
MIN_FREQ_SYN  = 2

# Real-data sizes
MAX_LEN_REAL   = 400      # truncate at 400 tokens to keep CPU runtime reasonable
MAX_VOCAB_REAL = 20_000
MIN_FREQ_REAL  = 3
MIN_TOKENS     = 20       # drop articles shorter than this (noise)
MAX_PER_CLASS  = 3000     # ISOT articles per class; reduce if too slow
CROSS_MAX_ROWS = 5000     # WELFake rows used for cross-dataset eval

# Multi-seed
DEFAULT_SEEDS = [42, 123, 456]

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ── Dataset + collate ──────────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, encoded_ids, labels):
        self.ids    = encoded_ids
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.ids[idx], self.labels[idx]


def collate_fn(batch):
    ids_list, labels = zip(*batch)
    padded, masks = pad_batch(list(ids_list))
    return (
        torch.tensor(padded,  dtype=torch.long),
        torch.tensor(masks,   dtype=torch.float),
        torch.tensor(labels,  dtype=torch.long),
    )


# ── training / evaluation ──────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, suppress_mask):
    model.train()
    total_loss = 0.0
    for token_ids, pad_mask, labels in loader:
        optimizer.zero_grad()
        logits = model(token_ids, padding_mask=pad_mask, suppress_mask=suppress_mask)
        loss   = criterion(logits, labels)
        if math.isnan(loss.item()):
            return float('nan')
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model, loader, suppress_mask):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for token_ids, pad_mask, labels in loader:
            preds   = model(token_ids, padding_mask=pad_mask,
                            suppress_mask=suppress_mask).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += len(labels)
    return correct / total if total > 0 else 0.0


# ── per-method training runners ────────────────────────────────────────────────

def train_and_eval(vocab_size, train_loader, test_a_loader, test_b_loader,
                   suppress_mask, run_name, verbose=True):
    """Train a fresh DAN with a fixed suppress_mask. Covers variants 1-3."""
    if verbose:
        print(f"\n  -- {run_name} --")
    model     = DeepAveragingNetwork(vocab_size, EMBED_DIM, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, suppress_mask)
        if math.isnan(loss):
            print(f"    WARNING: NaN loss at epoch {epoch} for [{run_name}] -- aborting run")
            return {"acc_a": float('nan'), "acc_b": float('nan'), "gap": float('nan')}
        if verbose:
            acc_a = evaluate(model, test_a_loader, suppress_mask)
            print(f"    epoch {epoch}/{EPOCHS}  loss={loss:.4f}  in-dataset acc={acc_a:.3f}")

    acc_a = evaluate(model, test_a_loader, suppress_mask)
    acc_b = evaluate(model, test_b_loader, suppress_mask)
    gap   = acc_a - acc_b
    if verbose:
        print(f"    In-dataset (A): {acc_a:.3f}  |  Cross-dataset (B): {acc_b:.3f}"
              f"  |  Gap: {gap:.3f}")
    return {"acc_a": acc_a, "acc_b": acc_b, "gap": gap}


def train_and_eval_dynamic(vocab_size, train_loader, test_a_loader, test_b_loader,
                           pub_conc, label_pmi, idx_to_token, run_name, verbose=True):
    """
    Train a DAN with the dynamic attribution gate (variant 4).
    Each epoch: train -> measure gradient saliency -> update gate.
    Returns (results_dict, trained_gate).
    """
    if verbose:
        print(f"\n  -- {run_name} --")
    model     = DeepAveragingNetwork(vocab_size, EMBED_DIM, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    gate = DynamicSuppressionGate(vocab_size, pub_conc, label_pmi)

    for epoch in range(1, EPOCHS + 1):
        suppress_mask = gate.get_soft_mask(lam=LAMBDA)
        loss = train_one_epoch(model, train_loader, optimizer, criterion, suppress_mask)
        if math.isnan(loss):
            print(f"    WARNING: NaN loss at epoch {epoch} for [{run_name}] -- aborting run")
            return {"acc_a": float('nan'), "acc_b": float('nan'), "gap": float('nan')}, gate

        if verbose:
            acc_a    = evaluate(model, test_a_loader, suppress_mask)
            top5     = gate.get_epoch_top5(idx_to_token)
            top5_str = ", ".join(f"{t}({v:.2f})" for t, v in top5)
            print(f"    epoch {epoch}/{EPOCHS}  loss={loss:.4f}  in-dataset acc={acc_a:.3f}"
                  f"  |  top-suppressed: [{top5_str}]")

        # Closed-loop update: measure what the model relied on, then update the gate
        saliency = compute_gradient_saliency(
            model, train_loader, vocab_size, criterion, max_batches=SALIENCY_BATCHES
        )
        gate.update(saliency, alpha=ALPHA, beta=BETA, gamma=GAMMA, ema_decay=EMA_DECAY)

    final_mask = gate.get_soft_mask(lam=LAMBDA)
    acc_a = evaluate(model, test_a_loader, final_mask)
    acc_b = evaluate(model, test_b_loader, final_mask)
    gap   = acc_a - acc_b
    if verbose:
        print(f"    In-dataset (A): {acc_a:.3f}  |  Cross-dataset (B): {acc_b:.3f}"
              f"  |  Gap: {gap:.3f}")
    return {"acc_a": acc_a, "acc_b": acc_b, "gap": gap}, gate


# ── experiment coordinator ─────────────────────────────────────────────────────

def _build_prepared(train_data, test_a_data, test_b_data, max_len, max_vocab, min_freq,
                    entropy_pct=None):
    """
    Tokenize, build vocab, encode all splits, compute all static signals.
    Called once per experiment (not per seed) -- everything here is deterministic.
    Returns a 'prepared' dict with everything needed to run training.
    entropy_pct overrides the global ENTROPY_PCT when provided (used for the
    stratified 7-publisher setup where pct=44 is needed to catch 'reuters').
    """
    if entropy_pct is None:
        entropy_pct = ENTROPY_PCT
    train_tok  = [simple_tokenize(d["text"], max_len) for d in train_data]
    test_a_tok = [simple_tokenize(d["text"], max_len) for d in test_a_data]
    test_b_tok = [simple_tokenize(d["text"], max_len) for d in test_b_data]

    train_labels  = [d["label"]     for d in train_data]
    test_a_labels = [d["label"]     for d in test_a_data]
    test_b_labels = [d["label"]     for d in test_b_data]
    train_pubs    = [d["publisher"] for d in train_data]

    # Build vocab from training data only -- no leakage from test or cross-dataset
    flat_tokens = [tok for doc in train_tok for tok in doc]
    vocab       = build_vocab(flat_tokens, max_vocab=max_vocab, min_freq=min_freq)
    vocab_size  = len(vocab)

    train_ids  = [encode(t, vocab, max_len) for t in train_tok]
    test_a_ids = [encode(t, vocab, max_len) for t in test_a_tok]
    test_b_ids = [encode(t, vocab, max_len) for t in test_b_tok]

    idx_to_token = {v: k for k, v in vocab.items()}

    # Static signals -- computed from training data once
    spurious_ids, token_entropy = find_spurious_tokens(
        train_tok, train_pubs, vocab, entropy_pct=entropy_pct
    )
    pub_conc  = compute_publisher_concentration(train_tok, train_pubs, vocab)
    label_pmi = compute_label_pmi(train_tok, train_labels, vocab)

    hard_mask = make_suppress_mask(vocab_size, spurious_ids)
    soft_mask = make_soft_suppress_mask(pub_conc, lam=LAMBDA)

    return {
        "vocab_size":    vocab_size,
        "vocab":         vocab,
        "idx_to_token":  idx_to_token,
        "train_ids":     train_ids,  "train_labels":  train_labels,
        "test_a_ids":    test_a_ids, "test_a_labels": test_a_labels,
        "test_b_ids":    test_b_ids, "test_b_labels": test_b_labels,
        "hard_mask":     hard_mask,
        "soft_mask":     soft_mask,
        "pub_conc":      pub_conc,
        "label_pmi":     label_pmi,
        "spurious_ids":  spurious_ids,
        "token_entropy": token_entropy,
    }


def _run_all_methods(prepared, seed, verbose=True):
    """
    Run all 4 suppression methods with the given random seed.
    Creates fresh DataLoaders (shuffle affected by seed) and fresh model weights.
    Returns ({method -> result_dict}, trained_gate_from_dynamic_method).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    p = prepared
    train_loader  = DataLoader(
        TextDataset(p["train_ids"],  p["train_labels"]),
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn,
    )
    test_a_loader = DataLoader(
        TextDataset(p["test_a_ids"], p["test_a_labels"]),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
    )
    test_b_loader = DataLoader(
        TextDataset(p["test_b_ids"], p["test_b_labels"]),
        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
    )

    results = {}

    results["Baseline (no suppression)"] = train_and_eval(
        p["vocab_size"], train_loader, test_a_loader, test_b_loader,
        suppress_mask=None, run_name="Baseline (no suppression)", verbose=verbose,
    )

    results["Static entropy + hard mask"] = train_and_eval(
        p["vocab_size"], train_loader, test_a_loader, test_b_loader,
        suppress_mask=p["hard_mask"],
        run_name="Static entropy + hard mask  [ablation]", verbose=verbose,
    )

    results["Static entropy + soft mask"] = train_and_eval(
        p["vocab_size"], train_loader, test_a_loader, test_b_loader,
        suppress_mask=p["soft_mask"],
        run_name="Static entropy + soft mask  [ablation]", verbose=verbose,
    )

    r_dyn, gate = train_and_eval_dynamic(
        p["vocab_size"], train_loader, test_a_loader, test_b_loader,
        p["pub_conc"], p["label_pmi"], p["idx_to_token"],
        run_name="Dynamic attribution gate  [main method]", verbose=verbose,
    )
    results["Dynamic attribution gate"] = r_dyn

    return results, gate


# ── single-seed experiment (used by synthetic mode) ───────────────────────────

def run_experiment(train_data, test_a_data, test_b_data, experiment_name,
                   max_len=MAX_LEN_SYN, max_vocab=MAX_VOCAB_SYN, min_freq=MIN_FREQ_SYN):
    """
    Single-seed experiment. Used by synthetic mode and as the inner loop
    for the multi-seed real-data run.
    """
    print(f"\n{'='*58}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Train: {len(train_data)}  |  Test-A: {len(test_a_data)}"
          f"  |  Test-B: {len(test_b_data)}")
    print(f"{'='*58}")

    prepared = _build_prepared(train_data, test_a_data, test_b_data,
                                max_len, max_vocab, min_freq)
    p = prepared
    print(f"\n  Vocabulary size: {p['vocab_size']:,}")
    print(f"  Publisher categories in training data: "
          f"{len(set(d['publisher'] for d in train_data))}")
    print(f"  Flagged {len(p['spurious_ids']):,} spurious token IDs "
          f"(bottom {ENTROPY_PCT}th percentile by publisher-entropy)")
    print_flagged_tokens(p["token_entropy"], p["spurious_ids"], p["vocab"], top_n=15)

    results, gate = _run_all_methods(prepared, seed=SEED, verbose=True)
    print_gate_analysis(gate, p["idx_to_token"], top_n=30)
    return results


# ── multi-seed experiment (used by real-data mode) ────────────────────────────

def run_experiment_multi_seed(train_data, test_a_data, test_b_data,
                               seeds, experiment_name,
                               max_len=MAX_LEN_REAL,
                               max_vocab=MAX_VOCAB_REAL,
                               min_freq=MIN_FREQ_REAL,
                               entropy_pct=None):
    """
    Run all 4 methods over multiple seeds.
    Vocab and static signals are built once; only training is repeated per seed.
    Returns (aggregated_results_dict, gate_from_last_seed).
    """
    print(f"\n{'='*62}")
    print(f"  Experiment: {experiment_name}  ({len(seeds)} seeds: {seeds})")
    print(f"  Train: {len(train_data):,}  |  Test-A: {len(test_a_data):,}"
          f"  |  Test-B: {len(test_b_data):,}")
    print(f"{'='*62}")

    eff_pct = entropy_pct if entropy_pct is not None else ENTROPY_PCT
    # Build vocab and signals once -- they're deterministic, no seed needed
    print("\n  Building vocabulary and computing static signals ...")
    prepared = _build_prepared(train_data, test_a_data, test_b_data,
                                max_len, max_vocab, min_freq, entropy_pct=eff_pct)
    p = prepared
    print(f"  Vocabulary size: {p['vocab_size']:,}")
    print(f"  Publisher categories: "
          f"{sorted(set(d['publisher'] for d in train_data))}")
    print(f"  Flagged {len(p['spurious_ids']):,} spurious token IDs "
          f"(bottom {eff_pct}th percentile by publisher-entropy)")
    print_flagged_tokens(p["token_entropy"], p["spurious_ids"], p["vocab"], top_n=20)

    # Collect per-seed results
    METHOD_NAMES = [
        "Baseline (no suppression)",
        "Static entropy + hard mask",
        "Static entropy + soft mask",
        "Dynamic attribution gate",
    ]
    per_seed   = {m: [] for m in METHOD_NAMES}
    last_gate  = None

    for i, seed in enumerate(seeds):
        print(f"\n  {'='*50}")
        print(f"  Seed {seed}  ({i+1}/{len(seeds)})")
        print(f"  {'='*50}")
        # For multi-seed runs, suppress per-epoch noise; just show final per-method line
        results, gate = _run_all_methods(prepared, seed=seed, verbose=False)
        last_gate = gate

        for name in METHOD_NAMES:
            r = results[name]
            per_seed[name].append(r)
            a = r["acc_a"]; b = r["acc_b"]; g = r["gap"]
            a_str = f"{a:.3f}" if not math.isnan(a) else " NaN "
            b_str = f"{b:.3f}" if not math.isnan(b) else " NaN "
            g_str = f"{g:.3f}" if not math.isnan(g) else " NaN "
            print(f"    {name:<36} in={a_str}  cross={b_str}  gap={g_str}")

    # Gate analysis from the last seed -- shows what the closed loop did on real data
    print_gate_analysis(last_gate, p["idx_to_token"], top_n=30)

    # Aggregate mean +/- std across seeds
    def _ms(values):
        valid = [v for v in values if not math.isnan(v)]
        if not valid:
            return float('nan'), float('nan')
        m = sum(valid) / len(valid)
        std = (sum((v - m)**2 for v in valid) / max(len(valid) - 1, 1)) ** 0.5
        return m, std

    agg = {}
    for name in METHOD_NAMES:
        runs = per_seed[name]
        m_a, s_a = _ms([r["acc_a"] for r in runs])
        m_b, s_b = _ms([r["acc_b"] for r in runs])
        m_g, s_g = _ms([r["gap"]   for r in runs])
        agg[name] = {
            "acc_a_mean": m_a, "acc_a_std": s_a,
            "acc_b_mean": m_b, "acc_b_std": s_b,
            "gap_mean":   m_g, "gap_std":   s_g,
        }

    return agg, last_gate


# ── results printing ───────────────────────────────────────────────────────────

def print_results_table(results):
    """Single-seed results table (used by synthetic mode)."""
    print(f"\n{'='*70}")
    print(f"  RESULTS TABLE")
    print(f"{'='*70}")
    print(f"  {'Method':<36} {'In-dataset':>11} {'Cross-dset':>12} {'Gap':>7}")
    print(f"  {'-'*66}")
    for name, r in results.items():
        def _f(v): return f"{v:.3f}" if not math.isnan(v) else " NaN "
        print(f"  {name:<36} {_f(r['acc_a']):>11} {_f(r['acc_b']):>12} {_f(r['gap']):>7}")
    print(f"{'='*70}")
    print("  Gap = In-dataset acc - Cross-dataset acc.  Smaller gap = less publisher-leakage.")
    print()

    b_gap = results["Baseline (no suppression)"]["gap"]
    d_gap = results["Dynamic attribution gate"]["gap"]
    if not (math.isnan(b_gap) or math.isnan(d_gap)):
        reduction = (b_gap - d_gap) / max(abs(b_gap), 1e-6) * 100
        print(f"  Dynamic gate reduced the gap by {reduction:.1f}% vs baseline.")

    s_gap = results["Static entropy + soft mask"]["gap"]
    if not (math.isnan(d_gap) or math.isnan(s_gap)):
        if d_gap < s_gap:
            print(f"  Closed-loop feedback beat the static soft mask "
                  f"({d_gap:.3f} vs {s_gap:.3f} gap).")
        else:
            print(f"  Static soft mask matched or beat the dynamic gate "
                  f"({s_gap:.3f} vs {d_gap:.3f} gap).")
    print()


def print_multiseed_table(agg, n_seeds):
    """Multi-seed results table with mean +/- std. Used by real-data mode."""

    def _fmt(mean, std):
        if math.isnan(mean):
            return "  NaN   "
        return f"{mean:.3f} +/- {std:.3f}"

    col_w = 20   # width per mean+/-std cell

    print(f"\n{'='*82}")
    print(f"  RESULTS TABLE  (mean +/- std over {n_seeds} seeds)")
    print(f"{'='*82}")
    print(f"  {'Method':<36}  {'In-dataset':<{col_w}}  {'Cross-dataset':<{col_w}}  Gap")
    print(f"  {'-'*78}")
    for name, r in agg.items():
        in_str    = _fmt(r["acc_a_mean"], r["acc_a_std"])
        cross_str = _fmt(r["acc_b_mean"], r["acc_b_std"])
        gap_str   = _fmt(r["gap_mean"],   r["gap_std"])
        print(f"  {name:<36}  {in_str:<{col_w}}  {cross_str:<{col_w}}  {gap_str}")
    print(f"{'='*82}")
    print("  Gap = In-dataset - Cross-dataset acc.  Smaller gap = less publisher-leakage.")
    print()

    # Verdict
    b  = agg["Baseline (no suppression)"]
    d  = agg["Dynamic attribution gate"]
    sm = agg["Static entropy + soft mask"]
    hm = agg["Static entropy + hard mask"]

    b_gap  = b["gap_mean"]
    d_gap  = d["gap_mean"]
    sm_gap = sm["gap_mean"]
    hm_gap = hm["gap_mean"]

    if not math.isnan(b_gap) and not math.isnan(d_gap):
        reduction = (b_gap - d_gap) / max(abs(b_gap), 1e-6) * 100
        print(f"  Dynamic gate vs baseline gap: {b_gap:.3f} -> {d_gap:.3f} "
              f"({reduction:+.1f}%)")

    best_name = min(agg.items(), key=lambda kv: kv[1]["gap_mean"] if not math.isnan(kv[1]["gap_mean"]) else 999)[0]
    best_gap  = agg[best_name]["gap_mean"]
    print(f"  Best gap: {best_name} ({best_gap:.3f})")

    if not math.isnan(d_gap):
        if d_gap <= min(sm_gap, hm_gap):
            print(f"  Dynamic gate beat both static methods on cross-dataset gap.")
        elif d_gap <= sm_gap:
            print(f"  Dynamic gate beat static soft mask on gap "
                  f"({d_gap:.3f} vs {sm_gap:.3f}) but not hard mask ({hm_gap:.3f}).")
        else:
            print(f"  Static methods outperformed the dynamic gate on gap.")
            print(f"  (This can happen when publisher tokens are very concentrated "
                  f"and a hard zero is the optimal suppression.)")
    print()

    # Cross-dataset accuracy verdict
    b_cross  = b["acc_b_mean"]
    d_cross  = d["acc_b_mean"]
    if not (math.isnan(b_cross) or math.isnan(d_cross)):
        cross_delta = d_cross - b_cross
        if cross_delta > 0:
            print(f"  Dynamic gate cross-dataset accuracy: {d_cross:.3f} "
                  f"(+{cross_delta:.3f} vs baseline {b_cross:.3f}).")
        else:
            print(f"  Dynamic gate cross-dataset accuracy: {d_cross:.3f} "
                  f"({cross_delta:.3f} vs baseline {b_cross:.3f}).")
    print()


def save_results_to_file(path, agg, n_seeds, experiment_name, dateline_stripped):
    """Save the multi-seed results table to a text file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _fmt(mean, std):
        if math.isnan(mean):
            return "NaN"
        return f"{mean:.4f} +/- {std:.4f}"

    lines = []
    lines.append(f"Experiment: {experiment_name}")
    lines.append(f"Seeds: {n_seeds}  |  Dateline stripped: {dateline_stripped}")
    lines.append(f"EPOCHS={EPOCHS}  BATCH_SIZE={BATCH_SIZE}  LR={LR}"
                 f"  LAMBDA={LAMBDA}  ALPHA={ALPHA}  BETA={BETA}  GAMMA={GAMMA}"
                 f"  EMA_DECAY={EMA_DECAY}  MAX_PER_CLASS={MAX_PER_CLASS}")
    lines.append("")
    lines.append(f"{'Method':<36}  {'In-dataset':<22}  {'Cross-dataset':<22}  Gap")
    lines.append("-" * 90)
    for name, r in agg.items():
        lines.append(
            f"{name:<36}  {_fmt(r['acc_a_mean'], r['acc_a_std']):<22}"
            f"  {_fmt(r['acc_b_mean'], r['acc_b_std']):<22}"
            f"  {_fmt(r['gap_mean'], r['gap_std'])}"
        )
    lines.append("")
    lines.append("Gap = In-dataset acc - Cross-dataset acc.  Smaller = less publisher-leakage.")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Results saved to {path}")


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fake-news publisher-debiasing experiment")
    parser.add_argument("--real", action="store_true",
                        help="Use real ISOT + WELFake data")
    parser.add_argument("--no-strip-dateline", action="store_true",
                        help="Keep the Reuters dateline in text (dramatic demo of bias)")
    parser.add_argument("--isot-path", default="data/isot",
                        help="Directory containing True.csv and Fake.csv (default: data/isot)")
    parser.add_argument("--cross-path", default="data/cross/WELFake_Dataset.csv",
                        help="Path to cross-dataset CSV (default: data/cross/WELFake_Dataset.csv)")
    parser.add_argument("--cross-option",
                        choices=['welfake', 'welfake-nonreuters', 'isot-worldnews-leftnews'],
                        default='welfake',
                        help=(
                            "welfake: full WELFake (default). "
                            "welfake-nonreuters: WELFake with Reuters-sourced real articles removed "
                            "(~62%% of WELFake real contains reuters -- filtering those creates a "
                            "harder test for a Reuters-trained model). "
                            "isot-worldnews-leftnews: use ISOT worldnews real + left-news fake "
                            "as the cross-dataset target."
                        ))
    parser.add_argument("--cross-text-col",   default=None,
                        help="Text column name in cross CSV (auto-detected if omitted)")
    parser.add_argument("--cross-label-col",  default=None,
                        help="Label column name in cross CSV (auto-detected if omitted)")
    parser.add_argument("--cross-true-label", default="0",
                        help="Label value meaning 'real news' in cross CSV (WELFake default: 0)")
    parser.add_argument("--max-per-class", type=int, default=MAX_PER_CLASS,
                        help=f"Max ISOT articles per class (default: {MAX_PER_CLASS})")
    parser.add_argument("--entropy-pct", type=int, default=None,
                        help=(
                            "Entropy percentile cutoff for the static hard mask "
                            "(default: 20 for binary publishers, 44 for --stratified). "
                            "pct=44 is the level where 'reuters' first gets flagged "
                            "with 7 stratified publisher categories."
                        ))
    parser.add_argument("--stratified", action="store_true",
                        help=(
                            "Use stratified fake-subject sampling (500 per subject across "
                            "all 6 ISOT fake categories). Gives 7 publisher bins instead of 2, "
                            "making reuters entropy low enough for the entropy method to flag it."
                        ))
    parser.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                        help="Comma-separated random seeds (default: 42,123,456)")
    # Legacy flags kept for backward compatibility
    parser.add_argument("--cross", default=None, metavar="PATH",
                        help="(legacy) path to cross-dataset CSV; use --cross-path instead")
    parser.add_argument("--text-col",    default="text")
    parser.add_argument("--label-col",   default="label")
    parser.add_argument("--true-label",  default="0")
    args = parser.parse_args()

    if args.real:
        strip_dl    = not args.no_strip_dateline
        seeds       = [int(s) for s in args.seeds.split(",") if s.strip()]
        isot_true   = os.path.join(args.isot_path, "True.csv")
        isot_fake   = os.path.join(args.isot_path, "Fake.csv")
        cross_path  = args.cross or args.cross_path

        dateline_label = "stripped" if strip_dl else "NOT stripped"
        print(f"\n{'='*62}")
        print(f"  REAL DATA MODE")
        print(f"  ISOT path:    {args.isot_path}/")
        print(f"  Cross-dataset: {cross_path}")
        print(f"  Dateline:     {dateline_label}")
        print(f"  Seeds:        {seeds}")
        print(f"  Max per class: {args.max_per_class}")
        print(f"{'='*62}")

        print("\nLoading ISOT ...")
        if args.stratified:
            # Stratified: 3000 real + 500 per fake subject (6 subjects = 3000 fake).
            # Gives 7 publisher categories so the entropy method can discriminate
            # 'reuters' from the fake-specific vocabulary properly.
            all_isot = load_isot_stratified(
                isot_true, isot_fake,
                strip_dl=strip_dl,
                real_per_class=args.max_per_class,
                fake_per_subject=max(1, args.max_per_class // 6),
            )
            print(f"  Stratified ISOT: {len(all_isot):,} articles across 7 publisher categories")
        else:
            all_isot = load_isot_real(
                isot_true, isot_fake,
                strip_dl=strip_dl,
                max_per_class=args.max_per_class,
            )
        split       = int(0.8 * len(all_isot))
        train_data  = all_isot[:split]
        test_a_data = all_isot[split:]
        print(f"  ISOT loaded: {len(all_isot):,} articles  "
              f"(train={len(train_data):,}, test={len(test_a_data):,})")

        # Load cross-dataset based on selected option
        cross_option = args.cross_option
        print(f"\nLoading cross-dataset (option: {cross_option}) ...")

        if cross_option == 'welfake-nonreuters':
            # Remove WELFake real articles containing 'reuters' in text.
            # ~62% of WELFake real is Reuters-sourced, so the standard full-WELFake
            # cross-test is too easy for a Reuters-trained model. Filtering those out
            # leaves only non-Reuters mainstream real news -- a harder cross-domain test.
            test_b_data = load_welfake_nonreuters(cross_path, max_rows=CROSS_MAX_ROWS)

        elif cross_option == 'isot-worldnews-leftnews':
            # ISOT internal topic shift:
            #   real  = worldnews Reuters (international coverage, rows 11272+)
            #   fake  = left-news articles (partisan echo-chamber style)
            # The training real articles are all politicsNews (first 3000 rows),
            # so worldnews articles are genuinely unseen territory -- no overlap.
            test_b_data = load_isot_by_subject(
                isot_true, isot_fake,
                true_subjects=['worldnews'],
                fake_subjects=['left-news', 'Government News', 'Middle-east', 'US_News'],
                strip_dl=strip_dl,
                max_per_class=CROSS_MAX_ROWS // 2,
            )
            print(f"  Loaded {len(test_b_data):,} ISOT-shift cross-dataset articles")

        else:   # default: full WELFake
            test_b_data = load_cross_dataset(
                cross_path,
                text_col=args.cross_text_col,
                label_col=args.cross_label_col,
                true_label=args.cross_true_label,
                max_rows=CROSS_MAX_ROWS,
            )

        dl_tag          = "no_dateline" if strip_dl else "with_dateline"
        strat_tag       = "_stratified" if args.stratified else ""
        tag             = f"{dl_tag}{strat_tag}_{cross_option}"
        experiment_name = f"ISOT vs {cross_option}  [{dl_tag}{strat_tag}]"

        # Resolve entropy_pct: explicit flag > stratified default (44) > global default (20)
        if args.entropy_pct is not None:
            eff_entropy_pct = args.entropy_pct
        elif args.stratified:
            eff_entropy_pct = 44   # pct=44 is where 'reuters' first gets flagged with 7 publishers
        else:
            eff_entropy_pct = ENTROPY_PCT

        print(f"  Entropy percentile: {eff_entropy_pct}  "
              f"({'stratified default' if args.stratified and args.entropy_pct is None else 'explicit'})")

        agg, last_gate = run_experiment_multi_seed(
            train_data, test_a_data, test_b_data,
            seeds=seeds,
            experiment_name=experiment_name,
            max_len=MAX_LEN_REAL,
            max_vocab=MAX_VOCAB_REAL,
            min_freq=MIN_FREQ_REAL,
            entropy_pct=eff_entropy_pct,
        )

        print_multiseed_table(agg, n_seeds=len(seeds))

        out_path = f"results/real_data_results_{tag}.txt"
        out_path = out_path.replace('/', os.sep)  # Windows path safety
        save_results_to_file(out_path, agg, n_seeds=len(seeds),
                             experiment_name=experiment_name,
                             dateline_stripped=strip_dl)

    else:
        print("=" * 58)
        print("  SYNTHETIC MODE -- no real data files required")
        print("  Publisher tokens are a perfect shortcut in this data.")
        print("  Expect a large baseline gap that shrinks with debiasing.")
        print("=" * 58)

        all_data    = make_synthetic_dataset(n_samples=2000, seed=SEED)
        split       = int(0.8 * len(all_data))
        train_data  = all_data[:split]
        test_a_data = all_data[split:]
        test_b_data = make_cross_dataset(n_samples=400, seed=SEED + 1)

        results = run_experiment(
            train_data, test_a_data, test_b_data,
            experiment_name="Synthetic Mode",
            max_len=MAX_LEN_SYN, max_vocab=MAX_VOCAB_SYN, min_freq=MIN_FREQ_SYN,
        )
        print_results_table(results)


if __name__ == "__main__":
    main()
