"""
threshold_test.py -- confirm pct=44 (stratified, 7 publishers) catches reuters and helps.
"""
import random, numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from dataset import (simple_tokenize, build_vocab, encode, pad_batch,
                     load_isot_stratified, load_welfake_nonreuters)
from model import DeepAveragingNetwork
from spurious import find_spurious_tokens, make_suppress_mask

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

data = load_isot_stratified('data/isot/True.csv','data/isot/Fake.csv',
                             strip_dl=False, real_per_class=3000, fake_per_subject=500)
random.shuffle(data)
split = int(0.8*len(data))
train_data, test_a = data[:split], data[split:]
test_b = load_welfake_nonreuters('data/cross/WELFake_Dataset.csv', max_rows=4000)

train_tok  = [simple_tokenize(d['text'],400) for d in train_data]
test_a_tok = [simple_tokenize(d['text'],400) for d in test_a]
test_b_tok = [simple_tokenize(d['text'],400) for d in test_b]
train_pubs = [d['publisher'] for d in train_data]

vocab = build_vocab([t for doc in train_tok for t in doc], max_vocab=20000, min_freq=3)
vs    = len(vocab)
rid   = vocab.get('reuters')

tid = [encode(t,vocab,400) for t in train_tok]; tla = [d['label'] for d in train_data]
aid = [encode(t,vocab,400) for t in test_a_tok]; ala = [d['label'] for d in test_a]
bid = [encode(t,vocab,400) for t in test_b_tok]; bla = [d['label'] for d in test_b]

sp20, _ = find_spurious_tokens(train_tok, train_pubs, vocab, 20)
sp44, _ = find_spurious_tokens(train_tok, train_pubs, vocab, 44)
print(f"pct=20: {len(sp20)} tokens, reuters={rid in sp20}")
print(f"pct=44: {len(sp44)} tokens, reuters={rid in sp44}\n")

class TD(Dataset):
    def __init__(self, i, l): self.i=i; self.l=l
    def __len__(self): return len(self.l)
    def __getitem__(self, x): return self.i[x], self.l[x]

def cf(batch):
    ids, labs = zip(*batch)
    p, m = pad_batch(list(ids))
    return (torch.tensor(p,dtype=torch.long), torch.tensor(m,dtype=torch.float), torch.tensor(labs,dtype=torch.long))

def run(lbl, mask):
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    tl = DataLoader(TD(tid,tla),64,shuffle=True, collate_fn=cf)
    al = DataLoader(TD(aid,ala),64,shuffle=False,collate_fn=cf)
    bl = DataLoader(TD(bid,bla),64,shuffle=False,collate_fn=cf)
    model = DeepAveragingNetwork(vs,64,128)
    opt   = torch.optim.Adam(model.parameters(),lr=1e-3)
    crit  = nn.CrossEntropyLoss()
    for ep in range(5):
        model.train(); ls=0
        for ids,pm,labs in tl:
            opt.zero_grad(); loss=crit(model(ids,pm,mask),labs); loss.backward(); opt.step(); ls+=loss.item()
        print(f"  ep{ep+1} loss={ls/len(tl):.4f}")
    model.eval()
    def acc(ldr):
        c=t=0
        with torch.no_grad():
            for ids,pm,labs in ldr:
                p=model(ids,pm,mask).argmax(1); c+=(p==labs).sum().item(); t+=len(labs)
        return c/max(t,1)
    a,b=acc(al),acc(bl); print(f"  {lbl:<35} in={a:.3f}  cross={b:.3f}  gap={a-b:.3f}\n"); return a,b

results={}
print("[1] Baseline"); results['Baseline']=run('Baseline',None)
print("[2] Hard pct=20 (no reuters)"); results['pct20']=run('pct20',make_suppress_mask(vs,sp20))
print("[3] Hard pct=44 (catches reuters)"); results['pct44']=run('pct44',make_suppress_mask(vs,sp44))

print("="*55)
for n,(a,b) in results.items(): print(f"  {n:<30} in={a:.3f}  cross={b:.3f}  gap={a-b:.3f}")
base_gap = results['Baseline'][0]-results['Baseline'][1]
best = min(results.items(), key=lambda kv: kv[1][0]-kv[1][1])
print(f"\n  Baseline gap: {base_gap:.3f}  |  Best: {best[0]} gap={best[1][0]-best[1][1]:.3f}")
imp = best[1][0]-best[1][1]-base_gap
print(f"  Improvement: {imp:+.3f}  |  {'PROCEED with full 3-seed run' if imp < -0.01 else 'MARGINAL -- reassess'}")
