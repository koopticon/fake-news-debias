# Fake-News Publisher Debiasing

Demonstrates that fake-news classifiers trained on ISOT cheat by memorising
publisher fingerprints ("reuters", datelines) rather than learning content.
Fixes this with an entropy-based spurious-token detector.

## Project structure

```
fake-news-debiasing/
├── data/            ← put True.csv and Fake.csv here for real-data mode
├── dataset.py       — tokenisation, vocab, data loaders (synthetic + real)
├── model.py         — Deep Averaging Network (DAN) trained from scratch
├── spurious.py      — find_spurious_tokens(): the novel contribution
├── experiment.py    — runs both baseline and debiased, prints results table
└── requirements.txt
```

## Install

```
cd fake-news-debiasing
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run — synthetic mode (no download needed)

```
python experiment.py
```

Expected output:
- Vocabulary size ~100 tokens (toy data)
- Flagged tokens list will contain: `reuters`, `(reuters)`, `reuters.com`, `tabloid`, `tabloidnews`
- Baseline: in-dataset acc ~0.95+, cross-dataset ~0.50, gap ~0.45
- Debiased:  in-dataset acc ~0.65-0.75, cross-dataset ~0.65-0.70, gap ~0.05-0.10
- The gap shrinking is the result.

## Run — real ISOT data

1. Download from Kaggle: `clmentbisaillon/fake-and-real-news-dataset`
2. Put `True.csv` and `Fake.csv` in `data/`
3. Run:
   ```
   python experiment.py --real
   ```

With a real cross-dataset (e.g. WELFake from `saurabhshahane/fake-news-classification`):
```
python experiment.py --real --cross data/WELFake_Dataset.csv --text-col text --label-col label --true-label 1
```

Faster ISOT run (2000 per class instead of 5000):
```
python experiment.py --real --max-per-class 2000
```

## How to interpret results

| Method                    | In-dataset | Cross-dataset | Gap  |
|---------------------------|-----------|---------------|------|
| Baseline (no suppression) | high      | low           | big  |
| Debiased (entropy)        | modest    | similar       | small|

**Gap = In-dataset − Cross-dataset.** A big gap means the model relied on
publisher shortcuts that don't exist in the second dataset. Entropy suppression
shrinks the gap by forcing the model to use content words instead.

## The method (for your defense)

`find_spurious_tokens()` in `spurious.py`:

1. For every token, count how many articles from each publisher contain it
   (article-level presence, not raw frequency).
2. Compute Shannon entropy H(t) = −Σ p_i log₂(p_i) over the publisher distribution.
3. Flag the bottom 20% by entropy as "publisher fingerprints".
4. Build a `suppress_mask`: 0.0 for flagged tokens, 1.0 for everything else.
5. In `model.py`, multiply every token embedding by `suppress_mask[token_id]`
   before mean-pooling → flagged tokens are invisible to the classifier.

Entropy = 0 means the token appeared in only one publisher's articles.
Entropy = log₂(n_publishers) means it appeared equally in all publishers.

## Novelty vs prior work

| Prior approach               | Requires                        | This method |
|------------------------------|---------------------------------|-------------|
| Adversarial training         | Second gradient stream          | No          |
| Counterfactual augmentation  | New synthetic sentences         | No          |
| Domain-adaptive pretraining  | Large LM + target-domain corpus | No          |
| **Entropy suppression**      | Token counts from training data | ← this work |

The key advantage: computed once before training, fully interpretable
(you can print the flagged tokens), and zero inference-time overhead.
