"""
sanity_check.py -- Quick 1-seed baseline check across 3 cross-dataset options.

Trains ONE model (3 epochs, seed 42) on the standard ISOT setup and evaluates
it against three different cross-dataset targets. Prints the in-dataset and
cross-dataset accuracy for each, then recommends which to use for the full
4-method experiment.

Usage:
    python sanity_check.py

Options evaluated:
    A  worldnews real + left-news fake
       Real domain shift: US politics Reuters -> international Reuters (Kuwait, Nigeria, ...)
       Fake domain shift: generic conspiracy ('News') -> partisan echo-chamber ('left-news')

    B  worldnews real + left-news + Government News + Middle-east + US_News fake
       Same real domain shift as A; broader fake subject shift.

    C  WELFake with Reuters-sourced real articles removed
       Removes the ~62% of WELFake real articles that contain 'reuters' in text,
       leaving only non-Reuters mainstream sources (harder for a Reuters-trained model).

The option with in-dataset - cross-dataset gap >= 0.25 goes to the full experiment.
If none reaches 0.25, the largest gap wins.
"""

import random
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from dataset import (
    simple_tokenize, build_vocab, encode, pad_batch,
    load_isot_real, load_isot_by_subject, load_welfake_nonreuters,
)
from model import DeepAveragingNetwork

# ── Config (kept small for a quick sanity pass) ────────────────────────────────
SEED        = 42
EPOCHS      = 3        # fast; full experiment uses 5
BATCH_SIZE  = 64
LR          = 1e-3
EMBED_DIM   = 64
HIDDEN_DIM  = 128
MAX_LEN     = 400
MAX_VOCAB   = 20_000
MIN_FREQ    = 3
MIN_TOKENS  = 20
MAX_PER_CLASS_TRAIN = 2500   # smaller for speed
MAX_PER_CLASS_CROSS = 2000

ISOT_TRUE = 'data/isot/True.csv'
ISOT_FAKE = 'data/isot/Fake.csv'
WELFAKE   = 'data/cross/WELFake_Dataset.csv'


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ── Minimal training loop ──────────────────────────────────────────────────────

class TextDataset(Dataset):
    def __init__(self, ids, labels):
        self.ids = ids; self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, i): return self.ids[i], self.labels[i]


def collate_fn(batch):
    ids_list, labels = zip(*batch)
    padded, masks = pad_batch(list(ids_list))
    return (torch.tensor(padded, dtype=torch.long),
            torch.tensor(masks,  dtype=torch.float),
            torch.tensor(labels, dtype=torch.long))


def train_eval(train_ids, train_labels, test_a_ids, test_a_labels,
               test_b_ids, test_b_labels, vocab_size):
    """Train a baseline DAN (no suppression) and return (in_acc, cross_acc)."""
    set_seed(SEED)
    train_loader  = DataLoader(TextDataset(train_ids,  train_labels),
                               batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_a_loader = DataLoader(TextDataset(test_a_ids, test_a_labels),
                               batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_b_loader = DataLoader(TextDataset(test_b_ids, test_b_labels),
                               batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model     = DeepAveragingNetwork(vocab_size, EMBED_DIM, HIDDEN_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for token_ids, pad_mask, labels in train_loader:
            optimizer.zero_grad()
            logits = model(token_ids, padding_mask=pad_mask, suppress_mask=None)
            loss   = criterion(logits, labels)
            if math.isnan(loss.item()):
                print(f"  NaN loss at epoch {epoch}")
                return float('nan'), float('nan')
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"    epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}")

    def acc(loader):
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for ids, mask, lbs in loader:
                preds = model(ids, padding_mask=mask, suppress_mask=None).argmax(1)
                correct += (preds == lbs).sum().item()
                total   += len(lbs)
        return correct / max(total, 1)

    return acc(test_a_loader), acc(test_b_loader)


def encode_split(data, vocab, max_len):
    ids    = [encode(simple_tokenize(d['text'], max_len), vocab, max_len) for d in data]
    labels = [d['label'] for d in data]
    return ids, labels


# ── Main sanity check ──────────────────────────────────────────────────────────

def main():
    print("=" * 62)
    print("  SANITY CHECK: finding the best cross-dataset target")
    print("=" * 62)

    # ── Standard training data (same for all three options) ────────────────────
    # politicsNews real (first 11 272 True.csv rows) + 'News' fake (first 9 050 Fake.csv rows)
    # This is exactly the training setup used in the main experiment.
    print("\nLoading standard ISOT training data ...")
    print("  Real : True.csv subject='politicsNews'")
    print("  Fake : Fake.csv subject='News'")

    all_train = load_isot_by_subject(
        ISOT_TRUE, ISOT_FAKE,
        true_subjects=['politicsNews'],
        fake_subjects=['News'],
        strip_dl=True,
        max_per_class=MAX_PER_CLASS_TRAIN,
    )
    split       = int(0.8 * len(all_train))
    train_data  = all_train[:split]
    test_a_data = all_train[split:]
    print(f"  Train: {len(train_data):,}  In-dataset test: {len(test_a_data):,}")

    # Build vocab from training data only (no cross-dataset leakage)
    train_tok   = [simple_tokenize(d['text'], MAX_LEN) for d in train_data]
    flat        = [t for doc in train_tok for t in doc]
    vocab       = build_vocab(flat, max_vocab=MAX_VOCAB, min_freq=MIN_FREQ)
    vocab_size  = len(vocab)
    print(f"  Vocab size: {vocab_size:,}")

    train_ids,  train_labels  = encode_split(train_data,  vocab, MAX_LEN)
    test_a_ids, test_a_labels = encode_split(test_a_data, vocab, MAX_LEN)

    results = {}

    # ── Option A: worldnews real + left-news fake ──────────────────────────────
    # Real domain shift: US politics Reuters -> international Reuters (Kuwait, Myanmar, etc.)
    # Fake domain shift: generic conspiracy -> partisan echo-chamber
    # Key hypothesis: the politicsNews-trained model learned US political vocabulary as
    # 'real' markers. International Reuters articles don't have that vocabulary, so the
    # model may confidently misclassify them as 'fake'.
    print("\n--- Option A: worldnews real + left-news fake ---")
    cross_a = load_isot_by_subject(
        ISOT_TRUE, ISOT_FAKE,
        true_subjects=['worldnews'],
        fake_subjects=['left-news'],
        strip_dl=True,
        max_per_class=MAX_PER_CLASS_CROSS,
    )
    print(f"  Cross-test size: {len(cross_a):,}")
    ids_a, labels_a = encode_split(cross_a, vocab, MAX_LEN)
    in_acc, cross_acc = train_eval(train_ids, train_labels,
                                   test_a_ids, test_a_labels,
                                   ids_a, labels_a, vocab_size)
    gap = in_acc - cross_acc
    results['A'] = {'label': 'worldnews real + left-news fake',
                    'in': in_acc, 'cross': cross_acc, 'gap': gap}
    print(f"  In-dataset: {in_acc:.3f}  Cross: {cross_acc:.3f}  Gap: {gap:.3f}")

    # ── Option B: worldnews real + left-news+Government+Middle-east+US_News fake ──
    # Same real domain shift as A; uses a broader range of rare fake subjects.
    print("\n--- Option B: worldnews real + left-news+Govt+ME+US_News fake ---")
    cross_b = load_isot_by_subject(
        ISOT_TRUE, ISOT_FAKE,
        true_subjects=['worldnews'],
        fake_subjects=['left-news', 'Government News', 'Middle-east', 'US_News'],
        strip_dl=True,
        max_per_class=MAX_PER_CLASS_CROSS,
    )
    print(f"  Cross-test size: {len(cross_b):,}")
    ids_b, labels_b = encode_split(cross_b, vocab, MAX_LEN)
    # Reuse trained model -- same training, only cross changes
    _, cross_acc_b = train_eval(train_ids, train_labels,
                                test_a_ids, test_a_labels,
                                ids_b, labels_b, vocab_size)
    gap_b = in_acc - cross_acc_b
    results['B'] = {'label': 'worldnews real + left-news+Govt+ME+US_News fake',
                    'in': in_acc, 'cross': cross_acc_b, 'gap': gap_b}
    print(f"  In-dataset: {in_acc:.3f}  Cross: {cross_acc_b:.3f}  Gap: {gap_b:.3f}")

    # ── Option C: WELFake non-Reuters ─────────────────────────────────────────
    # Remove WELFake real articles containing 'reuters' in text (~62% of real).
    # Leaves non-Reuters mainstream news -- harder for a Reuters-trained model.
    print("\n--- Option C: WELFake without Reuters real articles ---")
    cross_c = load_welfake_nonreuters(WELFAKE, max_rows=MAX_PER_CLASS_CROSS * 2)
    ids_c, labels_c = encode_split(cross_c, vocab, MAX_LEN)
    _, cross_acc_c = train_eval(train_ids, train_labels,
                                test_a_ids, test_a_labels,
                                ids_c, labels_c, vocab_size)
    gap_c = in_acc - cross_acc_c
    results['C'] = {'label': 'WELFake non-Reuters real',
                    'in': in_acc, 'cross': cross_acc_c, 'gap': gap_c}
    print(f"  In-dataset: {in_acc:.3f}  Cross: {cross_acc_c:.3f}  Gap: {gap_c:.3f}")

    # ── Summary ────────────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  SANITY CHECK SUMMARY")
    print("=" * 62)
    print(f"  {'Option':<5} {'Description':<42} {'In':>6} {'Cross':>7} {'Gap':>6}")
    print(f"  {'-'*64}")
    for k, v in results.items():
        flag = " <-- BEST" if v['gap'] == max(r['gap'] for r in results.values()) else ""
        print(f"  {k:<5} {v['label']:<42} {v['in']:>6.3f} {v['cross']:>7.3f} {v['gap']:>6.3f}{flag}")
    print()

    best_key = max(results, key=lambda k: results[k]['gap'])
    best_gap = results[best_key]['gap']

    if best_gap >= 0.25:
        print(f"  Option {best_key} meets the 0.25+ gap target.")
        print(f"  Proceeding with: {results[best_key]['label']}")
    else:
        print(f"  No option reached 0.25 gap. Largest gap is Option {best_key} ({best_gap:.3f}).")
        print(f"  Proceeding with Option {best_key} as best available.")

    print()
    print(f"  To run the full 4-method experiment:")
    print(f"    python experiment.py --real --cross-option {best_key}")


if __name__ == "__main__":
    main()
