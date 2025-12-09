import os, math, json, urllib.request, collections
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.special import digamma as psi
from math import log, pi
from math import gamma as gamma_fn
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform

# ──────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED = "agnews_embbag_mlp.pth"
C_SLACK = 1e-6
BETA_LIST = []
ALT_EPOCHS = 40
FT_EPOCHS = 6
N_HC_SAMPLES = 10
INNER_EPOCHS = 5
RHO = 1e6
WARM_EPOCHS = 5
LR = 1e-4
WD = 1e-5
MAX_SEQ_LEN = 256
BATCH_SIZE = 512
SEED = 0
MU_MAX = 3.0
LOGSTD_MAX = 1.0
H_X = 9.7109 # Data entropy
torch.manual_seed(SEED); np.random.seed(SEED)
PRETTY_TABLE = True

# ──────────────────────────────────────────────────

def safe_log(msg: str):
    print(msg, flush=True)
    with open("ag_news_startup_check.log", "a") as f:
        f.write(msg + "\n")
def _print_table_header_ag():
    print("\nβ    | Target_Hc   | Achieved_Hc |  PAC_raw | PAC_acc |  Eff_raw | Eff_acc || OPT_raw(cal/full)     | OPT_acc |     α")
    print("-" * 120)
def _print_table_row_ag(beta, target, Hc_ach, pac_mag, pac_acc, Eff_mag, Eff_acc, opt_mag_cal, opt_mag_full, opt_acc, alpha_final):
    print(f"{beta:4.2f} | {target:11.4f} | {Hc_ach:10.4f} | "
          f"{pac_mag:8.1f} | {pac_acc:7.4f} | {Eff_mag:8.1f} | {Eff_acc:7.4f} || "
          f"{opt_mag_cal: .3e}/{opt_mag_full: .3e} | {opt_acc:7.4f} | {alpha_final: .3e}")

# ───────────────────────── Data ─────────────────────────
def clean_logprob(lp: torch.Tensor, floor: float = -50.0) -> torch.Tensor:
    lp = torch.nan_to_num(lp, neginf=floor, posinf=0.0)
    return lp.clamp_min(floor)

def download_if_missing(url, dest):
    if not os.path.exists(dest):
        print(f"Downloading {os.path.basename(dest)} …")
        urllib.request.urlretrieve(url, dest)

def load_ag_news_raw():
    TRAIN_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
    TEST_URL  = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    os.makedirs("./data", exist_ok=True)
    train_path = "./data/ag_news_train.csv"; test_path = "./data/ag_news_test.csv"
    download_if_missing(TRAIN_URL, train_path); download_if_missing(TEST_URL, test_path)
    train_df = pd.read_csv(train_path, header=None); test_df = pd.read_csv(test_path, header=None)
    train_raw = [{"text": f"{str(r[1])} {str(r[2])}", "label": int(r[0]) - 1} for r in train_df.itertuples(index=False, name=None)]
    test_raw  = [{"text": f"{str(r[1])} {str(r[2])}", "label": int(r[0]) - 1} for r in test_df.itertuples(index=False, name=None)]
    return train_raw, test_raw

def build_vocab(train_raw, vocab_size=30000):
    ctr = collections.Counter()
    for item in train_raw:
        ctr.update(item["text"].lower().split())
    most_common = ctr.most_common(vocab_size)
    vocab_tokens = [w for w,_ in most_common]
    vocab = {w: i+1 for i,w in enumerate(vocab_tokens)}
    return vocab, vocab_tokens

class AGNewsEmbBagDataset(Dataset):
    def __init__(self, raw_list, vocab, max_len=MAX_SEQ_LEN, unk_index=0):
        self.samples=[]; self.labels=[]; self.max_len=max_len; self.vocab=vocab; self.unk_index=unk_index
        for item in raw_list:
            toks = item["text"].lower().split()[: self.max_len]
            idxs = [vocab.get(tok, self.unk_index) for tok in toks]
            if len(idxs)==0: idxs=[0]
            self.samples.append(torch.tensor(idxs, dtype=torch.long))
            self.labels.append(item["label"])
    def __len__(self): return len(self.samples)
    def __getitem__(self,i): return self.samples[i], self.labels[i]

def collate_batch(batch):
    token_lists, label_list = zip(*batch)
    offsets=[0]; all_ids=[]
    for toks in token_lists:
        all_ids.append(toks); offsets.append(offsets[-1]+len(toks))
    all_ids=torch.cat(all_ids,0); offsets=torch.tensor(offsets[:-1],dtype=torch.long)
    labels=torch.tensor(label_list,dtype=torch.long)
    return all_ids, offsets, labels

# ──────────────────────────────────────────────────
class EmbBagMLP(nn.Module):
    def __init__(self, vocab_size, emb_dim=300, hidden=256, num_classes=4):
        super().__init__()
        self.embedbag = nn.EmbeddingBag(num_embeddings=vocab_size+1, embedding_dim=emb_dim, mode="mean", sparse=False)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )
    def forward(self, ids, offsets):
        emb = self.embedbag(ids, offsets)
        return self.head(emb)

class ScaledNG(nn.Module):
    def __init__(self, base_gen, alpha: float = 1.0):
        super().__init__()
        self.base = base_gen; self.alpha = float(alpha)
    def forward(self, z):
        B, mu, lv = self.base(z)
        a = self.alpha
        return a*B, a*mu, lv
    def forward_with_eps(self, z, eps):
        B, mu, lv = self.base.forward_with_eps(z, eps)
        a = self.alpha
        return a*B, a*mu, lv

# ─────────────────────────PAC-Efficient─────────────────────────
def pac_eff_closed_form_raw(CovRaw: torch.Tensor, beta: float) -> torch.Tensor:
    lam, U = torch.linalg.eigh(CovRaw)
    lam = lam.clamp_min(C_SLACK)
    s = torch.sqrt(lam)
    c = s.sum() / (2.0 * beta)
    e = c * s
    return U @ torch.diag(e.to(CovRaw.dtype)) @ U.T

# ───────────────────────── SR-PAC ─────────────────────────
ALPHA_CAP = 4.0
K_MEAS = 32
def tamed_logprob(lp: torch.Tensor, floor=-60.0, ceil=None, tau=1.0):
    lp = lp / tau
    lp = torch.nan_to_num(lp, neginf=floor, posinf=floor)
    return lp if ceil is None else lp.clamp(min=floor, max=ceil)
def compute_Hc_train(z_w, noise_gen, posterior, eps_list):
    vals = []
    B = z_w.size(0)
    for eps in eps_list:
        eps_use = eps[:B] if eps.size(0) >= B else torch.randn(B, eps.size(1), device=eps.device)
        Bw,_,_ = noise_gen.forward_with_eps(z_w, eps_use)
        y = z_w + Bw
        vals.append(-posterior.log_prob(z_w, y))
    return torch.stack(vals, 0).mean()

@torch.no_grad()
def compute_Hc_eval(z_w, noise_gen, posterior, eps_list):
    vals = []
    B = z_w.size(0)
    for eps in eps_list:
        if eps.size(0) != B:
            eps_use = eps[:B] if eps.size(0) > B else torch.randn(B, eps.size(1), device=eps.device)
        else:
            eps_use = eps
        Bw,_,_ = noise_gen.forward_with_eps(z_w, eps_use)
        y = z_w + Bw
        vals.append(-posterior.log_prob(z_w, y))
    return torch.stack(vals, 0).mean()

@torch.no_grad()
def calibrate_opt_scale(posterior, Zw_cal, gen, target_Hc, eps_list, alpha_cap=ALPHA_CAP, tol=1e-3, max_bisect=30):
    assert eps_list is not None and len(eps_list) > 0, "Pass the same cal-CRNs used for the constraint."

    def Hc_of(alpha: float) -> float:
        vals = []
        for eps in eps_list:
            B,_,_ = gen.forward_with_eps(Zw_cal, eps)
            y = Zw_cal + float(alpha) * B
            vals.append(-posterior.log_prob(Zw_cal, y).mean())
        return torch.stack(vals).mean().item()

    lo, hi = 0.0, 1.0
    while Hc_of(hi) < target_Hc and hi < alpha_cap:
        hi *= 2.0
    if hi >= alpha_cap and Hc_of(hi) < target_Hc:
        return alpha_cap

    for _ in range(max_bisect):
        mid = 0.5*(lo+hi)
        if Hc_of(mid) >= target_Hc:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(1.0, hi): break
    return hi

def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return (logits.argmax(1) == labels).float().mean().item()

def knn_entropy_bootstrap(samples: torch.Tensor, k=5, bs=5, max_pts=2000):

    def one(xs):
        n,d = xs.shape; m=min(n, max_pts); idx=torch.randperm(n)[:m]
        pts = xs[idx].cpu().numpy()
        dists=np.sqrt(((pts[:,None,:]-pts[None,:,:])**2).sum(axis=2)+1e-12)
        np.fill_diagonal(dists, np.inf)
        rk = np.partition(dists, k, axis=1)[:,k]
        vol = pi**(d/2) / gamma_fn(d/2+1)
        base = d*np.mean(np.log(rk)) + log(vol)
        return base + psi(torch.tensor(n, dtype=torch.float64)).item() - psi(torch.tensor(k, dtype=torch.float64)).item()
    vals=[one(samples) for _ in range(bs)]
    return float(np.mean(vals)), float(np.std(vals))

def solve_lambda_diag_kkt(sigma: torch.Tensor, beta: float, tol: float = 1e-6, max_iter: int = 80):
    target = 2.0 * float(beta)
    if beta <= 0: return torch.zeros_like(sigma)
    def F(lmbd: float) -> float:
        lam = torch.tensor(lmbd, dtype=sigma.dtype, device=sigma.device)
        t = torch.sqrt(sigma*sigma + 4.0*lam*sigma)
        e = 0.5*(t - sigma).clamp_min(1e-12)
        return float(torch.log1p(sigma / e).sum().item())
    lam_lo, lam_hi = 1e-12, 1.0
    while F(lam_hi) > target and lam_hi < 1e12: lam_hi *= 2.0
    for _ in range(max_iter):
        lam_mid = 0.5*(lam_lo+lam_hi); val=F(lam_mid)
        if val > target: lam_lo = lam_mid
        else: lam_hi = lam_mid
        if lam_hi - lam_lo <= tol*max(1.0, lam_hi): break
    lam = torch.tensor(lam_hi, dtype=sigma.dtype, device=sigma.device)
    t = torch.sqrt(sigma*sigma + 4.0*lam*sigma)
    e = 0.5*(t - sigma).clamp_min(1e-12)
    return e

class FlowPosterior(nn.Module):
    def __init__(self, dim, context_dim, depth=3, hidden=128):
        super().__init__()
        base = StandardNormal([dim])
        transforms = []
        for _ in range(depth):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=dim, hidden_features=hidden, context_features=context_dim,
                )
            )
        self.flow = Flow(CompositeTransform(transforms), base)
    def log_prob(self, z, y):
        return self.flow.log_prob(inputs=z, context=y)

def posterior_warmup_identity(Zw_tr, posterior, epochs=5, sigmas=(0.0, 0.005, 0.01, 0.02)):
    opt = optim.AdamW(posterior.parameters(), lr=LR, weight_decay=WD)
    ds = DataLoader(TensorDataset(Zw_tr), batch_size=256, shuffle=True)
    posterior.train()
    for _ in range(epochs):
        for (z,) in ds:
            z = z.to(DEVICE)
            loss_id = -tamed_logprob(posterior.log_prob(z, z)).mean()
            sigma = float(np.random.choice(sigmas))
            y_small = z if sigma==0.0 else z + sigma*torch.randn_like(z)
            loss_sm = -tamed_logprob(posterior.log_prob(z, y_small)).mean()
            loss = 0.5*loss_id + 0.5*loss_sm
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(posterior.parameters(), 1.0)
            opt.step()
    posterior.eval()
    return posterior

class NoiseGenerator(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_layer = nn.Linear(hidden, dim)
        self.lv_layer = nn.Linear(hidden, dim)
    def _stats(self, z):
        h  = self.net(z)
        mu = MU_MAX * torch.tanh(self.mu_layer(h))
        lv = self.lv_layer(h).clamp(min=-7.0, max=LOGSTD_MAX)
        return mu, lv

    def forward(self, z):
        h = self.net(z)
        lv = self.lv_layer(h).clamp(min=-7.0, max=LOGSTD_MAX)
        std = torch.exp(0.5 * lv)
        eps = torch.randn_like(std)
        B = std * eps
        return B, torch.zeros_like(z), lv

    def forward_with_eps(self, z, eps):
        h = self.net(z)
        lv = self.lv_layer(h).clamp(min=-7.0, max=LOGSTD_MAX)
        std = torch.exp(0.5 * lv)
        B = std * eps
        return B, torch.zeros_like(z), lv

def compute_Hc_raw(z_w, noise_gen, posterior, n: int = N_HC_SAMPLES):
    lp_vals = []
    for _ in range(n):
        B, _, _ = noise_gen(z_w)
        y = z_w + B
        lp = posterior.log_prob(z_w, y)
        lp_vals.append(lp)
    return -torch.stack(lp_vals, 0).mean()

class ZeroNG(nn.Module):
    def forward(self, z):
        z0 = torch.zeros_like(z)
        return z0, z0, z0
    def forward_with_eps(self, z, eps):
        z0 = torch.zeros_like(z)
        return z0, z0, z0

@torch.no_grad()
def expected_raw_power(NG_eval, Zw, Lz, eps_bank):
    mags = []
    for eps in eps_bank:
        Bw, _, _ = NG_eval.forward_with_eps(Zw, eps)
        Braw = (Lz @ Bw.T).T
        mags.append(Braw.pow(2).sum(1).mean())
    return float(torch.stack(mags).mean().item())

@torch.no_grad()
def acc_from_bank_pac_imp(Z_raw, chol, y, eps_bank):
    accs = []
    for eps in eps_bank:
        B = eps @ chol.T
        accs.append((Z_raw + B).argmax(1).eq(y).float().mean().item())
    return float(np.mean(accs)), float(np.std(accs))

@torch.no_grad()
def acc_from_bank_opt(NG_eval, Zw, Lz, Z_raw, y, eps_bank):
    accs = []
    for eps in eps_bank:
        Bw, _, _ = NG_eval.forward_with_eps(Zw, eps)
        noise_raw = (Lz @ Bw.T).T
        accs.append((Z_raw + noise_raw).argmax(1).eq(y).float().mean().item())
    return float(np.mean(accs)), float(np.std(accs))

def run_opt(Zw_tr, Zw_cal, target, Lz, eps_cal, posterior_meter, epochs=ALT_EPOCHS):
    dim = Zw_tr.size(1)
    NG  = NoiseGenerator(dim).to(DEVICE)
    opt_noise = optim.AdamW(NG.parameters(), lr=LR, weight_decay=WD)
    lam = torch.tensor(0.0, device=DEVICE)
    e_int = torch.tensor(0.0, device=DEVICE)
    ds = DataLoader(TensorDataset(Zw_tr), batch_size=256, shuffle=True)

    for ep in range(1, epochs+1):
        alpha = calibrate_opt_scale(posterior_meter, Zw_cal, NG, target_Hc=target, eps_list=eps_cal)
        NGs = ScaledNG(NG, alpha)

        for (z,) in ds:
            z = z.to(DEVICE)
            Hc_meas = compute_Hc_train(z, NGs, posterior_meter, eps_cal)
            short = (target - Hc_meas).clamp_min(0.0)
            aug = lam*short + 0.5*RHO*short.pow(2)

            Bw,_,_ = NGs(z)
            Br = (Lz @ Bw.T).T
            mag = Br.pow(2).sum(1).mean()

            loss = mag + aug
            opt_noise.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(NG.parameters(), 1.0)
            opt_noise.step()

        with torch.no_grad():
            Hc_meas_full = compute_Hc_eval(Zw_cal, NGs, posterior_meter, eps_cal)
            err = target - Hc_meas_full
            e_int = 0.9*e_int + err
            lam = torch.clamp(lam + 0.05*float(err.item()) + 0.01*float(e_int.item()), min=0.0)

        if ep % 5 == 0:
            print(f"[OPT] epoch {ep:02d} | α={alpha:.4f} | λ={lam.item():.4f} | "
                  f"Hc_meas={Hc_meas_full.item():.4f} | target={target:.4f}")

    return NG

# ──────────────────────────────────────────────────
def load_checkpoint_components(ckpt):
    vocab = None; emb_dim = None; hidden = None; emb_state = None; head_state = None
    if isinstance(ckpt, dict):
        if "vocab" in ckpt and isinstance(ckpt["vocab"], dict): vocab = ckpt["vocab"]
        if "emb_dim" in ckpt and isinstance(ckpt["emb_dim"], int): emb_dim = ckpt["emb_dim"]
        if "hidden" in ckpt and isinstance(ckpt["hidden"], int): hidden = ckpt["hidden"]
        if "embeddings_state" in ckpt and isinstance(ckpt["embeddings_state"], dict): emb_state = ckpt["embeddings_state"]
        if "head_state" in ckpt and isinstance(ckpt["head_state"], dict): head_state = ckpt["head_state"]
        for key in ["model","state_dict","model_state_dict","weights"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                sd = ckpt[key]
                emb_keys = {k: v for k, v in sd.items() if k.startswith("embedbag.")}
                head_keys= {k.replace("head.",""): v for k, v in sd.items() if k.startswith("head.")}
                if emb_keys: emb_state = {k.split("embedbag.",1)[1]: v for k,v in emb_keys.items()}
                if head_keys: head_state = {k: v for k,v in head_keys.items()}
                if emb_state and "weight" in emb_state and emb_dim is None: emb_dim = int(emb_state["weight"].shape[1])
                if head_state and "0.weight" in head_state and hidden is None: hidden = int(head_state["0.weight"].shape[0])
                break
    return vocab, emb_dim, hidden, emb_state, head_state

# ───────────────────────── Main ─────────────────────────
def main():
    os.makedirs("./data", exist_ok=True)
    ckpt = torch.load(PRETRAINED, map_location=DEVICE)
    vocab_ckpt, emb_dim_ckpt, hidden_ckpt, emb_state, head_state = load_checkpoint_components(ckpt)
    train_raw, test_raw = load_ag_news_raw()
    if vocab_ckpt is not None:
        vocab = vocab_ckpt;
        safe_log(f"Loaded vocab from checkpoint (size={len(vocab)})")
    else:
        vocab, vocab_tokens = build_vocab(train_raw, vocab_size=30000)
        safe_log(f"Built vocab from training data (size={len(vocab)})")

    UNK_INDEX = ckpt["unk_index"] if isinstance(ckpt, dict) and isinstance(ckpt.get("unk_index", None), int) \
                else (0 if 0 in set(vocab.values()) else min(vocab.values()))
    safe_log(f"Using UNK_INDEX={UNK_INDEX}")

    vocab_size_effective = max(vocab.values()) if len(vocab) > 0 else 0
    num_embeddings = vocab_size_effective + 1

    all_raw = train_raw + test_raw
    full_ds = AGNewsEmbBagDataset(all_raw, vocab, max_len=MAX_SEQ_LEN, unk_index=UNK_INDEX)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch,
                             pin_memory=torch.cuda.is_available())
    emb_dim = emb_dim_ckpt if emb_dim_ckpt is not None else 300
    hidden  = hidden_ckpt  if hidden_ckpt  is not None else 256
    model = EmbBagMLP(vocab_size=num_embeddings-1, emb_dim=emb_dim, hidden=hidden, num_classes=4).to(DEVICE)

    loaded = False; missing_info = ""
    def has_full_keys(d):
        return isinstance(d, dict) and "embedbag.weight" in d and "head.0.weight" in d and "head.3.weight" in d

    if emb_state is not None:
        try:
            model.embedbag.load_state_dict(emb_state, strict=True); loaded=True
            safe_log("Loaded embeddings_state into model.embedbag (strict=True).")
        except Exception as e: missing_info += f"[embeddings_state fail: {e}] "
    if head_state is not None:
        try:
            model.head.load_state_dict(head_state, strict=True); loaded=True
            safe_log("Loaded head_state into model.head (strict=True).")
        except Exception as e: missing_info += f"[head_state fail: {e}] "
    if not loaded and isinstance(ckpt, dict) and "state_dict" in ckpt and has_full_keys(ckpt["state_dict"]):
        try:
            model.load_state_dict(ckpt["state_dict"], strict=True); loaded=True
            safe_log("Loaded state_dict (strict=True).")
        except Exception as e: missing_info += f"[state_dict fail: {e}] "
    if not loaded and has_full_keys(ckpt):
        try:
            model.load_state_dict(ckpt, strict=True); loaded=True
            safe_log("Loaded direct module keys (strict=True).")
        except Exception as e: missing_info += f"[direct keys fail: {e}] "
    if not loaded:
        raise RuntimeError("Checkpoint format not recognized or incomplete. " + missing_info)

    model.eval()

    Z_list, y_list = [], []
    with torch.no_grad():
        for all_ids, offsets, labels in full_loader:
            all_ids=all_ids.to(DEVICE); offsets=offsets.to(DEVICE); labels=labels.to(DEVICE)
            logits = model(all_ids, offsets)
            Z_list.append(logits.cpu()); y_list.append(labels.cpu())
    Z_raw = torch.cat(Z_list, 0).to(DEVICE)
    y     = torch.cat(y_list, 0).to(DEVICE)

    base_acc = accuracy_from_logits(Z_raw, y)
    safe_log(f"AG-News base accuracy (no noise): {base_acc:.4f}")
    if base_acc < 0.4:
        safe_log("WARNING: Base accuracy is low (<0.4). This usually indicates a vocab/index or weight mismatch.")

    mu = Z_raw.mean(0, keepdim=True)
    Xc = Z_raw - mu
    d = Z_raw.size(1); N = Z_raw.size(0)
    CovRaw = (Xc.T @ Xc) / N + C_SLACK * torch.eye(d, device=DEVICE)

    Lz = torch.linalg.cholesky(CovRaw)
    try:
        Zw = torch.linalg.solve_triangular(Lz, Xc.T, upper=False).T
    except AttributeError:
        Zw = torch.triangular_solve(Xc.T, Lz, upper=False)[0].T

    safe_log(f"H_X = {H_X:.4f} (fixed)")

    split_idx = Zw.size(0) // 2
    Zw_tr = Zw[:split_idx].contiguous()
    Zw_cal = Zw[split_idx:].contiguous()

    posterior = FlowPosterior(Zw.size(1), Zw.size(1)).to(DEVICE)
    posterior = posterior_warmup_identity(Zw_tr, posterior, epochs=15)
    posterior.eval()
    safe_log("Using FlowPosterior meter (warmed on identity+tiny Gaussians).")

    g_meas = torch.Generator(device=DEVICE); g_meas.manual_seed(4242)
    EPS_CAL_GLOBAL = [torch.randn(Zw_cal.size(0), Zw_cal.size(1), device=DEVICE, generator=g_meas)
                      for _ in range(K_MEAS)]

    g_rep = torch.Generator(device=DEVICE); g_rep.manual_seed(2026)
    EPS_REPORT_BANK = [torch.randn(Zw.size(0), Zw.size(1), device=DEVICE, generator=g_rep)
                       for _ in range(K_MEAS)]

    gen = torch.Generator(device=DEVICE); gen.manual_seed(12345)
    Eps_fixed = torch.randn(Z_raw.size(0), d, device=DEVICE, generator=gen)

    zero_ng = ZeroNG().to(DEVICE)
    Hc0 = compute_Hc_eval(Zw_cal, zero_ng, posterior, EPS_CAL_GLOBAL).item()
    beta_star = H_X - Hc0
    safe_log(f"Hc(0) = {Hc0:.4f}  →  β* = {beta_star:.4f}")

    if PRETTY_TABLE:
        _print_table_header_ag()
    else:
        print("\nβ    | Target_Hc   | Achieved_Hc | PAC_raw | PAC_acc | Eff_raw | Eff_acc || OPT_raw(cal/full) | OPT_acc | α")
        print("-" * 132)

    first = True
    for beta in BETA_LIST:
        target = H_X - beta

        skip_opt_train = target <= Hc0 + 1e-6
        if skip_opt_train:
            NG_eval = ZeroNG().to(DEVICE)
            alpha_final = 0.0
            Hc_ach = Hc0

        # ────────────────── Auto-PAC ────────────────────────────────
        c = 1.0 / (math.exp(2.0 * beta / d) - 1.0)
        Sigma_PAC = c * CovRaw
        chol_pac = torch.linalg.cholesky(Sigma_PAC + C_SLACK * torch.eye(d, device=DEVICE))
        B_pac = Eps_fixed @ chol_pac.T
        pac_mag = (B_pac.pow(2).sum(1).mean()).item()
        pac_acc, pac_acc_sd = acc_from_bank_pac_imp(Z_raw, chol_pac, y, EPS_REPORT_BANK)

        # ────────────────── Efficient-PAC (IMP/Eff) ────────────────────────────────
        Sigma_Eff = pac_eff_closed_form_raw(CovRaw, beta)
        chol_Eff = torch.linalg.cholesky(Sigma_Eff + C_SLACK * torch.eye(d, device=DEVICE))
        B_Eff = Eps_fixed @ chol_Eff.T
        Eff_mag = (B_Eff.pow(2).sum(1).mean()).item()
        Eff_acc, Eff_acc_sd = acc_from_bank_pac_imp(Z_raw, chol_Eff, y, EPS_REPORT_BANK)

        with torch.no_grad():
            lam, U = torch.linalg.eigh(CovRaw)
            lam = lam.clamp_min(C_SLACK)
            e = torch.diagonal(U.T @ Sigma_Eff @ U).clamp_min(C_SLACK)
            lhs_exact = torch.log1p(lam / e).sum().item()
            lhs_bound = (lam / e).sum().item()
            safe_log(f"[PAC-EFF] Σlog(1+σ/e)={lhs_exact:.4f} ≤ Σσ/e={lhs_bound:.4f} ≈ 2β={2 * beta:.4f}")

        # ────────────────── SR-PAC ───────────────────────────────
        if not skip_opt_train:
            epochs = ALT_EPOCHS if first else FT_EPOCHS
            NG = run_opt(Zw_tr, Zw_cal, target, Lz, EPS_CAL_GLOBAL, posterior, epochs=epochs)
            first = False
            alpha_final = calibrate_opt_scale(posterior, Zw_cal, NG, target_Hc=target, eps_list=EPS_CAL_GLOBAL)
            NG_eval = ScaledNG(NG, alpha_final)
            Hc_ach = compute_Hc_eval(Zw_cal, NG_eval, posterior, EPS_CAL_GLOBAL).item()
        if alpha_final > 1.2:
            print(f"[WARN] alpha_final={alpha_final:.3f} (>1.2). "
                  f"This usually means π trained adversarially or α wasn't applied inside the loop.")

        if abs(Hc_ach - target) > 0.05 * max(1.0, target):
            print(f"[WARN] Hc miss: achieved={Hc_ach:.4f} vs target={target:.4f} (>|5%|).")

        if not skip_opt_train:
            with torch.no_grad():
                a_small = max(1e-3, 0.5 * alpha_final)
                a_big = 1.5 * max(1e-3, alpha_final)
                Hc_small = compute_Hc_eval(Zw_cal, ScaledNG(NG, a_small), posterior, EPS_CAL_GLOBAL).item()
                Hc_big = compute_Hc_eval(Zw_cal, ScaledNG(NG, a_big), posterior, EPS_CAL_GLOBAL).item()
                if Hc_big < Hc_small - 1e-3:
                    print("[WARN] Hc not monotone in α on cal-CRNs. Check CRN alignment.")

        opt_mag_cal = expected_raw_power(NG_eval, Zw_cal, Lz, EPS_CAL_GLOBAL)
        opt_mag_full = expected_raw_power(NG_eval, Zw,     Lz, EPS_REPORT_BANK)

        opt_acc, opt_acc_sd = acc_from_bank_opt(NG_eval, Zw, Lz, Z_raw, y, EPS_REPORT_BANK)

        eps_dbg = EPS_REPORT_BANK[0]
        with torch.no_grad():
            Bw_dbg, _, _ = NG_eval.forward_with_eps(Zw, eps_dbg)
            noise_raw = (Lz @ Bw_dbg.T).T
            OPT_logits = Z_raw + noise_raw

            d_dbg = noise_raw.size(1)
            cm = noise_raw.mean(1, keepdim=True)
            B_cm = cm.expand(-1, d_dbg)
            B_dec = noise_raw - B_cm

            pow_total = (noise_raw ** 2).sum(1).mean().item()
            pow_cm = (B_cm ** 2).sum(1).mean().item()
            pow_dec = (B_dec ** 2).sum(1).mean().item()
            frac_cm = pow_cm / (pow_total + 1e-12)

            top2_base, _ = Z_raw.topk(2, dim=1)
            top2_opt, _ = OPT_logits.topk(2, dim=1)
            margin_base = (top2_base[:, 0] - top2_base[:, 1]).mean().item()
            margin_opt = (top2_opt[:, 0]  - top2_opt[:, 1]).mean().item()

        print(
            f"    OPT power: total={pow_total:.6f} | decision={pow_dec:.6f} | common={pow_cm:.6f} "
            f"(frac_cm={frac_cm:.3f}) | margins base/opt={margin_base:.3f}/{margin_opt:.3f}"
        )

        if PRETTY_TABLE:
            _print_table_row_ag(beta, target, Hc_ach,
                                pac_mag, pac_acc, Eff_mag, Eff_acc,
                                opt_mag_cal, opt_mag_full, opt_acc, alpha_final)
        else:
            print(f"{beta:4.2f} | {target:11.4f} | {Hc_ach:10.4f} | "
                  f"{pac_mag:7.1f}/{pac_acc:.4f} | {Eff_mag:7.1f}/{Eff_acc:.4f} || "
                  f"{opt_mag_cal:.3e}/{opt_mag_full:.3e} | {opt_acc:.4f} | {alpha_final:.3e}")

    summary = {
        "BETA_LIST": BETA_LIST, "ALT_EPOCHS": ALT_EPOCHS, "FT_EPOCHS": FT_EPOCHS,
        "N_HC_SAMPLES": N_HC_SAMPLES, "H_X_est": H_X, "Hc0": Hc0, "beta_star": beta_star,
        "MODEL": "EmbeddingBag+MLP", "MAX_SEQ_LEN": MAX_SEQ_LEN,
    }
    with open("agnews_run_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
