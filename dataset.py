"""
dataset.py — data loading for the fake-news debiasing experiment.

Provides:
  - simple_tokenize / build_vocab / encode / pad_batch  (shared utilities)
  - make_synthetic_dataset  (runs with NO real files — fast sanity check)
  - make_cross_dataset      (simulated second-domain data, no publisher tokens)
  - load_isot               (original loader — loads from a flat path)
  - load_generic_csv        (generic loader for a second real dataset)
  - strip_dateline          (removes "(Reuters) -" patterns from ISOT real text)
  - load_isot_real          (updated ISOT loader with publisher proxy + dateline toggle)
  - load_cross_dataset      (flexible cross-dataset loader with column auto-detection)
"""

import re
import csv
import random
from collections import Counter


# ── tokenization utilities ─────────────────────────────────────────────────────

def simple_tokenize(text, max_len=200):
    """Lowercase, strip non-alphanumeric, split on whitespace, cap length."""
    text   = text.lower()
    text   = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()[:max_len]
    return tokens


def build_vocab(all_tokens, max_vocab=15_000, min_freq=2):
    """
    Build token→id mapping from a flat list of tokens.
    Reserve 0 for <PAD> (padding) and 1 for <UNK> (out-of-vocab words).
    Only keep tokens that appear at least min_freq times — removes noise.
    """
    counts = Counter(all_tokens)
    vocab  = {"<PAD>": 0, "<UNK>": 1}
    for tok, cnt in counts.most_common(max_vocab - 2):
        if cnt >= min_freq:
            vocab[tok] = len(vocab)
    return vocab


def encode(tokens, vocab, max_len=200):
    """Map token strings → integer IDs. Unknown tokens get ID 1 (<UNK>)."""
    return [vocab.get(tok, 1) for tok in tokens[:max_len]]


def pad_batch(batch_ids, pad_id=0):
    """
    Pad a list of id-lists to the same length (length of the longest one).
    Returns padded ids and a binary mask (1 = real token, 0 = padding).
    """
    max_len = max(len(ids) for ids in batch_ids)
    padded  = [ids + [pad_id] * (max_len - len(ids)) for ids in batch_ids]
    masks   = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in batch_ids]
    return padded, masks


# ── synthetic data ─────────────────────────────────────────────────────────────

def make_synthetic_dataset(n_samples=2000, seed=42):
    """
    Generate a toy dataset that deliberately bakes in publisher bias.

    Design intent:
      - Real articles come from publisher "reuters".
        They contain publisher fingerprints: "reuters", "(reuters)", "reuters.com".
        These tokens perfectly predict the label → a shortcut the model can cheat with.
      - Fake articles come from publisher "tabloid".
        Fingerprints: "tabloid", "tabloidnews".
      - Content words carry a WEAK signal (65% chance of drawing from the correct
        pool). They represent the actual information the model should learn from.

    In-dataset test:   publisher tokens are present → baseline can cheat → high acc.
    Cross-dataset test: publisher tokens are absent  → cheater collapses → low acc.
    Debiased model:     publisher tokens suppressed  → must use content → stable acc.
    """
    random.seed(seed)

    real_content  = ["election", "government", "official", "minister", "report",
                     "parliament", "senate", "economy", "trade", "statement"]
    fake_content  = ["hoax", "conspiracy", "secret", "exposed", "shocking",
                     "chemtrail", "deepstate", "crisis", "fabricated", "coverup"]
    neutral_words = ["the", "and", "of", "in", "to", "a", "is", "was", "that",
                     "with", "said", "on", "for", "at", "from", "by", "an"]

    data = []
    for i in range(n_samples):
        is_real   = (i % 2 == 0)
        label     = 0 if is_real else 1         # 0 = real, 1 = fake
        publisher = "reuters" if is_real else "tabloid"

        # Publisher fingerprints — the spurious shortcut
        if is_real:
            pub_tokens = ["reuters", "(reuters)", "reuters.com"]
        else:
            pub_tokens = ["tabloid", "tabloidnews"]

        # Content words: 65% from the correct pool (noisy but real signal)
        pool_a = real_content if is_real else fake_content
        pool_b = fake_content if is_real else real_content
        content = [
            random.choice(pool_a if random.random() < 0.65 else pool_b)
            for _ in range(random.randint(8, 15))
        ]

        filler = random.choices(neutral_words, k=random.randint(10, 20))
        tokens = pub_tokens + content + filler
        random.shuffle(tokens)

        data.append({"text": " ".join(tokens), "label": label, "publisher": publisher})

    return data


def make_cross_dataset(n_samples=500, seed=99):
    """
    Simulated cross-dataset: same content vocabulary, but NO publisher tokens.

    This mimics evaluating on a different news source where "reuters" is meaningless.
    A baseline model that memorised publisher tokens will perform near chance here.
    The debiased model, forced to rely on content, should remain stable.
    """
    random.seed(seed)

    real_content  = ["election", "government", "official", "minister", "report",
                     "parliament", "senate", "economy", "trade", "statement"]
    fake_content  = ["hoax", "conspiracy", "secret", "exposed", "shocking",
                     "chemtrail", "deepstate", "crisis", "fabricated", "coverup"]
    neutral_words = ["the", "and", "of", "in", "to", "a", "is", "was", "that",
                     "with", "said", "on", "for", "at", "from", "by", "an"]

    data = []
    for i in range(n_samples):
        is_real   = (i % 2 == 0)
        label     = 0 if is_real else 1
        publisher = "unknown"   # no source info — the whole point of cross-dataset

        pool_a = real_content if is_real else fake_content
        pool_b = fake_content if is_real else real_content
        content = [
            random.choice(pool_a if random.random() < 0.65 else pool_b)
            for _ in range(random.randint(8, 15))
        ]

        filler = random.choices(neutral_words, k=random.randint(10, 20))
        tokens = content + filler       # <-- no pub_tokens here
        random.shuffle(tokens)

        data.append({"text": " ".join(tokens), "label": label, "publisher": publisher})

    return data


# ── real data loaders ──────────────────────────────────────────────────────────

def load_isot(true_csv_path, fake_csv_path, max_per_class=5000):
    """
    Load the ISOT fake-news dataset (kaggle: clmentbisaillon/fake-and-real-news-dataset).

    Files:
      True.csv — real news, all sourced from Reuters wire service
      Fake.csv — fake news, scraped from flagged/fact-checked outlets

    Publisher proxy:
      True → publisher = "reuters"   (accurate: the CSV is 100% Reuters)
      Fake → publisher = "other"     (simplification: multiple fake outlets lumped
                                      together; the important bias is real=reuters)

    Why concatenate title + text?
      The title often contains the dateline "(Reuters) -" which is a strong
      publisher fingerprint — exactly what we want our entropy method to catch.

    Note on 'subject' column:
      ISOT also has a 'subject' column (e.g. "politicsNews" vs "News") that
      correlates strongly with label. We ignore it — it's another shortcut,
      not meaningful content signal. Worth mentioning in your defense.
    """
    data = []

    def _read(path, label, publisher, max_rows):
        rows = []
        with open(path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                # Combine title and body — title carries dateline patterns
                text = (row.get("title", "") + " " + row.get("text", "")).strip()
                if text:
                    rows.append({"text": text, "label": label, "publisher": publisher})
        return rows

    data += _read(true_csv_path, label=0, publisher="reuters", max_rows=max_per_class)
    data += _read(fake_csv_path, label=1, publisher="other",   max_rows=max_per_class)
    random.shuffle(data)
    return data


def load_generic_csv(path, text_col="text", label_col="label",
                     true_label="1", max_rows=5000):
    """
    Generic loader for a second fake-news dataset (used for cross-dataset evaluation).

    Suggested datasets (download from Kaggle):
      WELFake — "saurabhshahane/fake-news-classification"
                 columns: Unnamed: 0, title, text, label  (0=fake, 1=real)
                 usage: load_generic_csv("WELFake_Dataset.csv", text_col="text",
                                         label_col="label", true_label="1")

      LIAR     — needs preprocessing to binary; not recommended for first run.

    All cross-dataset articles get publisher="unknown" because we don't have
    source info — and that's fine, they're only used for evaluation, not for
    computing spurious tokens (which is training-time only).
    """
    data = []
    with open(path, encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            text  = row.get(text_col, "").strip()
            label = 0 if str(row.get(label_col, "")).strip() == true_label else 1
            if text:
                data.append({"text": text, "label": label, "publisher": "unknown"})
    random.shuffle(data)
    return data


# ── real ISOT + flexible cross-dataset loaders ────────────────────────────────

def strip_dateline(text):
    """
    Remove the Reuters dateline from the start of ISOT real articles.

    All ISOT True.csv articles open with a city name and "(Reuters) -":
        "WASHINGTON (Reuters) - The head of..."
        "NEW YORK (Reuters) - ..."

    Why this matters: if left in, the token "reuters" (or "(reuters)") is a
    perfect label predictor — even the baseline gets ~100% in-dataset accuracy
    because every real article literally starts with the publisher name.
    Stripping it removes that trivially obvious shortcut and makes the
    cross-dataset generalization test meaningful.

    The caller decides whether to strip (via the --strip-dateline flag).
    Running with strip_dl=False is still useful as a dramatic demo of the
    publisher-bias problem — the gap without debiasing is enormous.
    """
    return re.sub(r'^\s*[A-Z][A-Z ,./]*\(Reuters\)\s*[-–]\s*', '', text.strip())


def load_isot_real(true_csv, fake_csv, strip_dl=True,
                   max_per_class=3000, min_tokens=20, max_tokens=400):
    """
    Load the ISOT dataset with a subject-based publisher proxy.

    Publisher proxy:
      Real  articles: publisher = "reuters"
        Accurate — every article in True.csv is a Reuters wire story.
      Fake  articles: publisher = subject column value (normalised)
        The subject column contains values like "News", "politics",
        "left-news", "Government News", "US_News", "Middle-east".
        These represent different fake-news ecosystems, giving the entropy
        calculation meaningful variation across fake sources.
        This is imperfect (it's a topic tag, not a real publisher name),
        but it is strictly better than lumping all fake articles into a
        single "other" bucket: 6 fake "publishers" vs. 1 means the entropy
        method can distinguish tokens that are specific to one fake source.

    strip_dl: strip the "(Reuters) -" dateline from the text field before
        concatenating with the title.  See strip_dateline() for details.
    """
    csv.field_size_limit(10 * 1024 * 1024)
    data = []

    def _read(path, label, publisher_fn, max_rows):
        rows = []
        with open(path, encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                title = row.get('title', '').strip()
                text  = row.get('text',  '').strip()
                if strip_dl:
                    text = strip_dateline(text)
                combined = (title + ' ' + text).strip()
                # Drop very short articles — they're usually parse noise
                if len(simple_tokenize(combined, max_tokens)) < min_tokens:
                    continue
                rows.append({
                    'text':      combined,
                    'label':     label,
                    'publisher': publisher_fn(row),
                })
        return rows

    data += _read(true_csv, label=0,
                  publisher_fn=lambda row: 'reuters',
                  max_rows=max_per_class)
    data += _read(fake_csv, label=1,
                  publisher_fn=lambda row: (
                      row.get('subject', 'unknown').strip().lower().replace(' ', '_')
                  ),
                  max_rows=max_per_class)

    random.shuffle(data)
    return data


def load_cross_dataset(path, text_col=None, label_col=None, true_label=None,
                       max_rows=5000, max_tokens=400, min_tokens=20):
    """
    Flexible loader for a second fake-news dataset used only for cross-dataset eval.

    Auto-detects text and label columns if not specified.  Pass overrides via
    --cross-text-col, --cross-label-col, --cross-true-label if auto-detection fails.

    true_label: the string value in label_col that means 'real news' in the
        source dataset's convention.  Our internal convention is always 0=real /
        1=fake, so we flip as needed.

    WELFake (data/cross/WELFake_Dataset.csv) defaults:
        columns: (unnamed index), title, text, label
        label: 0 = real, 1 = fake  (confirmed by spot-checking content)
        -> true_label = '0'   (this is the default)
    """
    csv.field_size_limit(10 * 1024 * 1024)

    COMMON_TEXT_COLS  = ['text', 'content', 'article', 'body', 'statement', 'Text']
    COMMON_LABEL_COLS = ['label', 'Label', 'class', 'fake', 'target', 'veracity']

    data = []
    with open(path, encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []

        if text_col is None:
            for c in COMMON_TEXT_COLS:
                if c in fields:
                    text_col = c
                    break
        if text_col is None:
            raise ValueError(
                f"Could not auto-detect text column in {path}.\n"
                f"Available columns: {fields}\n"
                f"Pass --cross-text-col to specify one."
            )

        if label_col is None:
            for c in COMMON_LABEL_COLS:
                if c in fields:
                    label_col = c
                    break
        if label_col is None:
            raise ValueError(
                f"Could not auto-detect label column in {path}.\n"
                f"Available columns: {fields}\n"
                f"Pass --cross-label-col to specify one."
            )

        if true_label is None:
            true_label = '0'   # WELFake default: 0=real, 1=fake

        print(f"  Cross-dataset columns: text={repr(text_col)}, "
              f"label={repr(label_col)}, true_label={repr(true_label)}")

        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            text = row.get(text_col, '').strip()
            if not text:
                continue
            if len(simple_tokenize(text, max_tokens)) < min_tokens:
                continue
            # Normalise to 0=real, 1=fake regardless of source convention
            our_label = 0 if str(row.get(label_col, '')).strip() == str(true_label) else 1
            data.append({'text': text, 'label': our_label, 'publisher': 'unknown'})

    random.shuffle(data)
    print(f"  Loaded {len(data):,} cross-dataset articles from {path}")
    return data


def load_isot_stratified(true_csv, fake_csv, strip_dl=True,
                          real_per_class=3000, fake_per_subject=500,
                          min_tokens=20, max_tokens=400):
    """
    Load ISOT with STRATIFIED fake-subject sampling.

    Why this matters for publisher-entropy:
      The default load_isot_real takes the first N fake articles, which are
      all subject='News' (9 050 rows). That gives only 2 publisher categories
      (reuters vs news), and the entropy threshold lands at H=0 — too coarse
      to catch 'reuters', which has H≈0.09 from appearing in a few fake articles.

      With 6 fake subjects as separate publisher proxies we get 7 categories
      (reuters + news + politics + left_news + government_news + us_news +
      middle_east). The maximum entropy grows to log2(7)≈2.81, and 'reuters'
      — which appears almost exclusively in the reuters publisher bin — now
      sits in the bottom ~20–30% of the entropy distribution, within the
      flag threshold.

    Parameters:
      real_per_class   : number of real articles to load (from True.csv)
      fake_per_subject : number of fake articles from EACH subject category
                         (total fake = 6 × fake_per_subject)
    """
    csv.field_size_limit(10 * 1024 * 1024)
    FAKE_SUBJECTS = ['News', 'politics', 'left-news',
                     'Government News', 'US_News', 'Middle-east']
    data = []

    # Real articles
    with open(true_csv, encoding='utf-8', errors='replace') as f:
        n = 0
        for row in csv.DictReader(f):
            if n >= real_per_class:
                break
            title = row.get('title', '').strip()
            text  = row.get('text',  '').strip()
            if strip_dl:
                text = strip_dateline(text)
            combined = (title + ' ' + text).strip()
            if len(simple_tokenize(combined, max_tokens)) < min_tokens:
                continue
            data.append({'text': combined, 'label': 0, 'publisher': 'reuters'})
            n += 1

    # Fake articles: fake_per_subject rows from each subject
    for subj in FAKE_SUBJECTS:
        pub_tag = subj.lower().replace(' ', '_')
        n = 0
        with open(fake_csv, encoding='utf-8', errors='replace') as f:
            for row in csv.DictReader(f):
                if row.get('subject', '').strip() != subj:
                    continue
                if n >= fake_per_subject:
                    break
                title = row.get('title', '').strip()
                text  = row.get('text',  '').strip()
                combined = (title + ' ' + text).strip()
                if len(simple_tokenize(combined, max_tokens)) < min_tokens:
                    continue
                data.append({'text': combined, 'label': 1, 'publisher': pub_tag})
                n += 1

    random.shuffle(data)
    return data


def load_isot_by_subject(true_csv, fake_csv,
                          true_subjects=None, fake_subjects=None,
                          strip_dl=True, max_per_class=3000,
                          min_tokens=20, max_tokens=400):
    """
    Load ISOT articles filtered to specific subject values.

    true_subjects : list of True.csv subject values to keep, e.g. ['worldnews']
                    None means all subjects.
    fake_subjects : list of Fake.csv subject values to keep, e.g. ['left-news']
                    None means all subjects.

    ISOT True.csv subjects  : 'politicsNews' (11 272), 'worldnews' (10 145)
    ISOT Fake.csv subjects  : 'News' (9 050), 'politics' (6 841), 'left-news' (4 459),
                               'Government News' (1 570), 'US_News' (783), 'Middle-east' (778)

    Used to create topic-shifted cross-dataset splits:
      Train  : politicsNews real + News fake
      Cross  : worldnews real  + left-news fake
    The two real-article pools share no rows (different subject tags), so there
    is no data leakage between the train and cross evaluation sets.
    """
    csv.field_size_limit(10 * 1024 * 1024)
    data = []

    def _read(path, label, publisher_fn, subject_filter, max_rows):
        rows = []
        with open(path, encoding='utf-8', errors='replace') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if subject_filter and row.get('subject', '').strip() not in subject_filter:
                    continue
                if len(rows) >= max_rows:
                    break
                title = row.get('title', '').strip()
                text  = row.get('text',  '').strip()
                if strip_dl:
                    text = strip_dateline(text)
                combined = (title + ' ' + text).strip()
                if len(simple_tokenize(combined, max_tokens)) < min_tokens:
                    continue
                rows.append({
                    'text':      combined,
                    'label':     label,
                    'publisher': publisher_fn(row),
                })
        return rows

    data += _read(true_csv, label=0,
                  publisher_fn=lambda row: 'reuters_' + row.get('subject', 'unknown').strip(),
                  subject_filter=set(true_subjects) if true_subjects else None,
                  max_rows=max_per_class)

    data += _read(fake_csv, label=1,
                  publisher_fn=lambda row: row.get('subject', 'unknown').strip().lower().replace(' ', '_'),
                  subject_filter=set(fake_subjects) if fake_subjects else None,
                  max_rows=max_per_class)

    random.shuffle(data)
    return data


def load_welfake_nonreuters(path, max_rows=4000, max_tokens=400, min_tokens=20):
    """
    Load WELFake but remove real articles that contain 'reuters' in the text.

    Why: 61.8% of WELFake real articles are sourced from Reuters. A model
    trained on ISOT real news (all Reuters) trivially classifies them correctly
    by matching Reuters vocabulary — it's the same publisher. Removing them
    leaves only non-Reuters mainstream news (Washington Post, NY Times, etc.),
    which is a genuinely harder cross-domain test for a Reuters-trained model.

    WELFake label convention: 0 = real, 1 = fake.
    """
    csv.field_size_limit(10 * 1024 * 1024)
    data = []
    with open(path, encoding='utf-8', errors='replace') as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= max_rows * 4:   # over-scan since we'll discard ~62% of real articles
                break
            text = row.get('text', '').strip()
            if not text:
                continue
            our_label = 0 if row.get('label', '') == '0' else 1
            # Skip real articles that look like Reuters -- those are too easy
            if our_label == 0 and 'reuters' in text.lower():
                continue
            if len(simple_tokenize(text, max_tokens)) < min_tokens:
                continue
            data.append({'text': text, 'label': our_label, 'publisher': 'unknown'})
            if len(data) >= max_rows:
                break
    random.shuffle(data)
    print(f"  WELFake (non-Reuters real): {len(data):,} articles loaded from {path}")
    return data
