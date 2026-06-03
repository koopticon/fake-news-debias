"""
spurious.py — static and dynamic spurious-token suppression.

DEFENSE FRAMING
───────────────
Existing publisher-debiasing methods use fixed, statistically-determined token
lists. This method uses the model's own gradient signal to identify which
suspicious tokens it is actually relying on, and updates suppression each epoch
based on that feedback — closed-loop, not preprocessing. The specific combination
(token-level, publisher-entropy + label-PMI + gradient saliency, soft per-token
gate, in a from-scratch DAN for fake-news cross-dataset generalization) is my
own design within the broader direction of model-guided spurious correlation
mitigation.

CORE IDEA
─────────
A publisher-fingerprint token (e.g. "reuters", "(reuters)") appears almost
exclusively in articles from one publisher, so its distribution over publishers
has low Shannon entropy. A genuine content word ("economy") appears across many
publishers in roughly equal proportion — high entropy. Suppressing low-entropy
tokens forces the model to rely on cross-publisher content instead of shortcuts.

LIMITATIONS (self-awareness helps at defense)
──────────────────────────────────────────────
1. Entropy threshold is a fixed percentile — adaptive elbow detection is better.
2. We count article-level presence, not TF-IDF, so a rarely-occurring but
   accidentally balanced token can get high entropy by chance (small-count noise).
3. No multi-word publisher fingerprints — "new york times" wouldn't be caught.
4. Gradient saliency is measured without suppression applied (raw model reliance).
   Measuring under the current gate would create different dynamics; the open-loop
   measurement is a deliberate choice for interpretability.
"""

import numpy as np
from collections import defaultdict
import torch


# ---------------------------------------------------------------------------
# ORIGINAL STATIC METHOD — kept verbatim as ablation baseline
# ---------------------------------------------------------------------------

def find_spurious_tokens(texts, publishers, vocab, entropy_pct=20):
    """
    For each vocabulary token, compute the Shannon entropy of its publisher
    distribution. Flag the bottom entropy_pct% as spurious.

    Args:
        texts       : list of token-lists (already tokenised)
        publishers  : list of publisher strings, one per article
        vocab       : dict  token_str -> token_id
        entropy_pct : percentile cutoff; lower = stricter (fewer tokens flagged)

    Returns:
        spurious_ids  : set of integer token IDs flagged as spurious
        token_entropy : dict  token_str -> float entropy

    For each token t:
      count how many *articles* from each publisher contain t (article-level
      presence, not raw frequency — so a long Reuters article doesn't unfairly
      inflate the Reuters count for "the").
      Normalise -> P(publisher | token).
      H(t) = -sum p_i * log2(p_i).  H=0 means token only ever appears in one
      publisher; H=log2(n_publishers) means perfectly even spread.
    Flag tokens at or below the entropy_pct-th percentile.
    """
    pub_list = sorted(set(publishers))
    pub_idx  = {p: i for i, p in enumerate(pub_list)}
    n_pubs   = len(pub_list)

    counts = defaultdict(lambda: np.zeros(n_pubs, dtype=np.float32))

    for token_list, pub in zip(texts, publishers):
        p = pub_idx[pub]
        for tok in set(token_list):          # set() -> presence per article, not count
            counts[tok][p] += 1

    token_entropy = {}
    for tok, cnt in counts.items():
        total = cnt.sum()
        if total == 0:
            continue
        probs   = cnt / total
        nonzero = probs[probs > 0]
        token_entropy[tok] = float(-np.sum(nonzero * np.log2(nonzero)))

    if not token_entropy:
        return set(), {}

    threshold    = float(np.percentile(list(token_entropy.values()), entropy_pct))
    spurious_ids = {vocab[tok] for tok, h in token_entropy.items()
                    if h <= threshold and tok in vocab}

    return spurious_ids, token_entropy


def print_flagged_tokens(token_entropy, spurious_ids, vocab, top_n=30):
    """
    Print the most spurious tokens (lowest entropy) for inspection.
    The list should visibly contain publisher fingerprints, not content words.
    """
    id_set  = set(spurious_ids)
    flagged = [(tok, h) for tok, h in token_entropy.items()
               if vocab.get(tok) in id_set]
    flagged.sort(key=lambda x: x[1])

    print(f"\n{'='*48}")
    print(f"  Top {top_n} spurious tokens  (lowest publisher-entropy)")
    print(f"  (entropy=0 = exclusively one publisher)")
    print(f"{'='*48}")
    print(f"  {'Token':<24} {'Entropy':>10}")
    print(f"  {'-'*36}")
    for tok, h in flagged[:top_n]:
        print(f"  {tok:<24} {h:>10.4f}")
    print()


def make_suppress_mask(vocab_size, spurious_ids, device="cpu"):
    """
    Hard binary mask: 1.0 for normal tokens, 0.0 for spurious tokens.
    Multiplying an embedding by 0 makes the token invisible to mean-pool.
    """
    mask = torch.ones(vocab_size, dtype=torch.float32, device=device)
    for sid in spurious_ids:
        if 0 <= sid < vocab_size:
            mask[sid] = 0.0
    return mask


# ---------------------------------------------------------------------------
# STATIC COMPONENTS FOR THE DYNAMIC METHOD
# Computed once from training data, never updated during training.
# ---------------------------------------------------------------------------

def compute_publisher_concentration(texts, publishers, vocab):
    """
    Continuous version of the entropy signal: concentration[i] = 1 - H_norm[i],
    where H_norm is entropy normalised by log2(n_publishers).

    High concentration (near 1): token mostly in one publisher — likely spurious.
    Low  concentration (near 0): token spread evenly — more like genuine content.

    Returns a float tensor of shape (vocab_size,).
    """
    pub_list = sorted(set(publishers))
    pub_idx  = {p: i for i, p in enumerate(pub_list)}
    n_pubs   = len(pub_list)
    max_ent  = float(np.log2(n_pubs)) if n_pubs > 1 else 1.0

    counts = defaultdict(lambda: np.zeros(n_pubs, dtype=np.float32))
    for token_list, pub in zip(texts, publishers):
        p = pub_idx[pub]
        for tok in set(token_list):
            counts[tok][p] += 1

    concentration = torch.zeros(len(vocab))
    for tok, tok_id in vocab.items():
        if tok not in counts:
            continue
        cnt   = counts[tok]
        total = cnt.sum()
        if total == 0:
            continue
        probs   = cnt / total
        nonzero = probs[probs > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        concentration[tok_id] = 1.0 - (entropy / max_ent)

    return concentration


def compute_label_pmi(texts, labels, vocab):
    """
    For each vocabulary token, compute max |PMI| across label classes.

    PMI(token, label) = log( P(token|label) / P(token) )

    Taking the absolute value and the max over labels gives a single score
    that is high when a token is strongly associated with *any* class — whether
    it's a fake-news marker or a real-news marker.

    Returns a normalised float tensor of shape (vocab_size,) in [0, 1].
    """
    label_set = sorted(set(labels))
    n_docs    = len(texts)

    tok_count = defaultdict(int)
    lbl_count = defaultdict(int)
    co_count  = defaultdict(lambda: defaultdict(int))

    for token_list, lbl in zip(texts, labels):
        lbl_count[lbl] += 1
        for tok in set(token_list):
            tok_count[tok] += 1
            co_count[tok][lbl] += 1

    pmi_scores = torch.zeros(len(vocab))

    for tok, tok_id in vocab.items():
        if tok_id <= 1:      # skip <PAD> and <UNK>
            continue
        if tok not in tok_count or tok_count[tok] == 0:
            continue
        p_tok = tok_count[tok] / n_docs
        best  = 0.0
        for lbl in label_set:
            p_lbl = lbl_count[lbl] / n_docs
            co    = co_count[tok][lbl]
            if co == 0 or p_lbl == 0:
                continue
            p_tok_given_lbl = co / lbl_count[lbl]
            pmi = abs(float(np.log((p_tok_given_lbl / (p_tok + 1e-10)) + 1e-10)))
            if pmi > best:
                best = pmi
        pmi_scores[tok_id] = best

    max_val = pmi_scores.max()
    if max_val > 1e-10:
        pmi_scores = pmi_scores / max_val

    return pmi_scores


def make_soft_suppress_mask(publisher_concentration, lam=0.7):
    """
    Static soft suppression using publisher concentration only.
    suppress_weight[i] = 1 - lam * concentration[i]

    Values range from (1 - lam) for maximally concentrated tokens to 1.0
    for tokens spread evenly across publishers.

    Note: since sigmoid(logit(x)) = x, this equals 1 - lam * sigmoid(logit(conc)),
    which is the DynamicSuppressionGate output at epoch 0. The two methods start
    at the same point; only the dynamic gate changes from there.
    """
    return 1.0 - lam * publisher_concentration


# ---------------------------------------------------------------------------
# DYNAMIC COMPONENT — recomputed every epoch to close the feedback loop
# ---------------------------------------------------------------------------

def compute_gradient_saliency(model, loader, vocab_size, criterion, max_batches=15):
    """
    Estimate per-token gradient saliency on a sample of training batches.

    For each vocabulary token i, saliency[i] is the average L2 norm of
    d(loss)/d(embedding[i]), averaged over batches where the token appeared.

    Why this measures reliance: the embedding gradient norm tells us how much
    the loss wants to change token i's vector. If the model is using "reuters"
    to predict label=real, the gradient through "reuters" will be large.

    Saliency is computed without any suppression mask so we get the model's raw
    reliance, not its reliance-after-suppression. This avoids a confounding
    cycle where already-suppressed tokens get low saliency just because they're
    suppressed, and therefore never get re-evaluated.

    Returns a normalised float tensor of shape (vocab_size,) in [0, 1].
    """
    model.eval()   # no dropout noise in the gradient estimates

    grad_sum   = torch.zeros(vocab_size)
    appear_cnt = torch.zeros(vocab_size, dtype=torch.long)

    for batch_idx, (token_ids, pad_mask, labels) in enumerate(loader):
        if batch_idx >= max_batches:
            break

        model.zero_grad()
        logits = model(token_ids, padding_mask=pad_mask, suppress_mask=None)
        loss   = criterion(logits, labels)
        loss.backward()

        with torch.no_grad():
            if model.embedding.weight.grad is not None:
                # embedding.weight.grad: (vocab_size, embed_dim)
                # norm(dim=1) -> one scalar per token row
                grad_norms  = model.embedding.weight.grad.norm(dim=1)
                appeared    = grad_norms > 1e-12
                grad_sum   += grad_norms
                appear_cnt += appeared.long()

    # Average over batches where the token appeared (not total batches),
    # so rare tokens aren't penalised just for being rare.
    saliency  = torch.zeros(vocab_size)
    seen      = appear_cnt > 0
    saliency[seen] = grad_sum[seen] / appear_cnt[seen].float()

    max_val = saliency.max()
    if max_val > 1e-10:
        saliency = saliency / max_val

    model.zero_grad()
    model.train()

    return saliency


# ---------------------------------------------------------------------------
# DYNAMIC SUPPRESSION GATE — the main contribution
# ---------------------------------------------------------------------------

class DynamicSuppressionGate:
    """
    Per-token suppression gate: a vector g of shape (vocab_size,).

    The weight applied to each token embedding during the forward pass is:
        suppress_weight[i] = 1 - lambda * sigmoid(g[i])

    g >> 0  ->  sigmoid near 1  ->  weight near (1 - lambda)  (strong suppression)
    g near 0  ->  sigmoid = 0.5  ->  weight near (1 - lambda/2)  (moderate)
    g << 0  ->  sigmoid near 0  ->  weight near 1              (no suppression)

    Initialisation: g is set so sigmoid(g[i]) = publisher_concentration[i].
    This means at epoch 0 the gate produces the same weights as make_soft_suppress_mask.
    They diverge once gradient saliency starts driving the EMA updates.

    Update rule (called once per epoch after computing gradient saliency):
        bias[i] = alpha * pub_conc[i] + beta * label_pmi[i] + gamma * saliency[i]
        g[i] = ema_decay * g[i] + (1 - ema_decay) * logit(bias[i])

    The EMA smooths out single-epoch spikes. Higher ema_decay = slower adaptation.
    """

    def __init__(self, vocab_size, publisher_concentration, label_pmi):
        self.vocab_size = vocab_size
        self.publisher_concentration = publisher_concentration
        self.label_pmi               = label_pmi

        # Initialise g so that sigmoid(g[i]) = concentration[i].
        # Clamp before logit to keep values finite at the edges.
        conc = publisher_concentration.clamp(0.01, 0.99)
        self.g = torch.log(conc / (1.0 - conc))   # logit(concentration)

        # PAD and UNK should never be suppressed
        self.g[0] = -10.0
        self.g[1] = -10.0

        self.g_initial = self.g.clone()   # snapshot before any training

    def get_soft_mask(self, lam=0.7):
        """Return suppress_weight = 1 - lam * sigmoid(g) as a (vocab_size,) tensor."""
        with torch.no_grad():
            return 1.0 - lam * torch.sigmoid(self.g)

    def update(self, saliency, alpha=0.5, beta=0.2, gamma=0.3, ema_decay=0.85):
        """
        Closed-loop update: blend the new gradient saliency into g via EMA.

        The three components of the bias score:
          alpha * publisher_concentration  — static, measures publisher bias
          beta  * label_pmi               — static, measures class specificity
          gamma * saliency                — dynamic, what the model actually uses

        Without the gamma term this would be static. The saliency term is what
        makes this different from the static soft mask.
        """
        with torch.no_grad():
            bias  = (alpha * self.publisher_concentration
                   + beta  * self.label_pmi
                   + gamma * saliency)
            b     = bias.clamp(0.01, 0.99)
            logit = torch.log(b / (1.0 - b))
            self.g = ema_decay * self.g + (1.0 - ema_decay) * logit
            self.g[0] = -10.0
            self.g[1] = -10.0

    def get_epoch_top5(self, idx_to_token):
        """Return the 5 most-suppressed tokens right now (for per-epoch logging)."""
        top_vals, top_ids = self.g.topk(min(10, self.vocab_size))
        out = []
        for val, idx in zip(top_vals, top_ids):
            tok = idx_to_token.get(int(idx), f"<id:{int(idx)}>")
            if tok not in ("<PAD>", "<UNK>"):
                out.append((tok, float(val)))
            if len(out) == 5:
                break
        return out

    def get_top_tokens(self, idx_to_token, top_n=30, use_initial=False):
        """Return [(token, g_value), ...] for the top_n tokens by g."""
        g_src = self.g_initial if use_initial else self.g
        top_vals, top_ids = g_src.topk(min(top_n + 5, self.vocab_size))
        out = []
        for val, idx in zip(top_vals, top_ids):
            tok = idx_to_token.get(int(idx), f"<id:{int(idx)}>")
            if tok not in ("<PAD>", "<UNK>"):
                out.append((tok, float(val)))
            if len(out) == top_n:
                break
        return out


def print_gate_analysis(gate, idx_to_token, top_n=30):
    """
    Compare the top-suppressed tokens at initialisation vs. end of training.

    Tokens that appear in the final list but not the initial list were added
    because the model relied on them heavily during training — that is the
    closed-loop signal. Tokens that dropped out were initially flagged by the
    statistics but turned out not to be heavily used by the model.

    A non-empty 'added by saliency' list is the key evidence that the gradient
    feedback loop did something beyond pure statistical preprocessing.
    """
    initial   = gate.get_top_tokens(idx_to_token, top_n, use_initial=True)
    final     = gate.get_top_tokens(idx_to_token, top_n, use_initial=False)

    init_set  = {t for t, _ in initial}
    final_set = {t for t, _ in final}

    print(f"\n{'='*64}")
    print(f"  GATE ANALYSIS: did the closed loop reshape suppression?")
    print(f"{'='*64}")

    print(f"\n  Initialisation (publisher concentration only, before training):")
    print(f"  {'Token':<22} {'g':>8}")
    print(f"  {'-'*32}")
    for tok, val in initial:
        print(f"  {tok:<22} {val:>8.3f}")

    print(f"\n  End of training (after {gate.vocab_size} tokens, gradient saliency feedback):")
    print(f"  {'Token':<22} {'g':>8}  note")
    print(f"  {'-'*52}")
    for tok, val in final:
        note = "  <- added by saliency" if tok not in init_set else ""
        print(f"  {tok:<22} {val:>8.3f}{note}")

    newly_added = sorted(final_set - init_set)
    released    = sorted(init_set  - final_set)

    print(f"\n  Added by gradient saliency ({len(newly_added)}): "
          f"{', '.join(newly_added) if newly_added else '(none)'}")
    print(f"  Released by the loop       ({len(released)}):  "
          f"{', '.join(released) if released else '(none)'}")
    print()
