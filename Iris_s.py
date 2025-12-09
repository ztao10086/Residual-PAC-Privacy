import math, os, random, copy, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from lira_eval import lira_posteriors_and_success, theoretical_psr


# ──────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 0
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
BETAS = [1/128, 1/64, 1/32, 1/16, 1/8, 1/4, 1/2, 1.0, 2.0, 4.0]
POOL_SIZE = 30
MEMB_PROB = 0.5
DELTA_DP = 1e-6
VAR_FLOOR = 1e-6
AUTO_c = 1e-3
SLACK_RATIO = 0.5
EFFICIENT_k = 5
SR_OUTER = 60
SR_INNER = 100
LR_LEADER = 2e-3
LR_FOLL = 1e-3
CAL_TOL_MIN = 5e-3
CAL_TOL_SCALE = 0.05
CAL_BATCH = 768
REPORT_BATCH = 768
safe = lambda msg: print(msg, flush=True)

# ──────────────────────────────────────────────────
def _h(p): return -p*math.log(max(p,1e-12)) - (1-p)*math.log(max(1-p,1e-12))
H_M = POOL_SIZE * _h(MEMB_PROB)

def load_iris_data():
    fname = "iris.data"
    if not os.path.exists(fname):
        raise FileNotFoundError("expected 'iris.data' in cwd")
    species = ['Iris-setosa','Iris-versicolor','Iris-virginica']
    X, y = [], []
    with open(fname) as f:
        for line in f:
            p = line.strip().split(',')
            if len(p) != 5: continue
            X.append([float(x) for x in p[:4]])
            y.append(species.index(p[4]))

    return np.asarray(X, np.float32), np.asarray(y, np.int64)

def get_data():
    X, y = load_iris_data()
    n, d = X.shape

    train_idx, val_idx = [], []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0].tolist()
        random.shuffle(idx)
        cut = int(0.8*len(idx))
        train_idx += idx[:cut]
        val_idx += idx[cut:]

    train_idx = np.array(train_idx, np.int64)
    val_idx = np.array(val_idx,   np.int64)
    Xt, yt = X[train_idx], y[train_idx]
    Xv, yv = X[val_idx],   y[val_idx]
    mu = Xt.mean(0, keepdims=True)
    sig = Xt.std(0, keepdims=True) + 1e-8

    return ((Xt-mu)/sig, yt, (Xv-mu)/sig, yv, d)

def trimmed_mean(X):
    return X.mean(axis=0).astype(np.float32)

def mechanism(pool_X: np.ndarray, pool_m: np.ndarray) -> np.ndarray:
    return trimmed_mean(pool_X[pool_m==1]).astype(np.float32)

def sample_pool(X):
    n = len(X)
    replace = n < POOL_SIZE
    idx = np.random.choice(n, POOL_SIZE, replace=replace)
    Xp = X[idx]
    m = (np.random.rand(POOL_SIZE) < MEMB_PROB).astype(np.float32)
    if m.sum()==0: m[np.random.randint(POOL_SIZE)] = 1
    if m.sum()==POOL_SIZE: m[np.random.randint(POOL_SIZE)] = 0

    return Xp, m, trimmed_mean(Xp[m==1])

# ──────────────────────────────────────────────────
def minibatch(X, batch_size=64):
    Ms, Ys, Xps = [], [], []
    for _ in range(batch_size):
        Xp, m, _ = sample_pool(X)
        Ms.append(m.astype(np.float32))
        Ys.append(mechanism(Xp, m))
        Xps.append(Xp)
    return (
        torch.tensor(np.stack(Ms), dtype=torch.float32, device=DEVICE),
        torch.tensor(np.stack(Ys), dtype=torch.float32, device=DEVICE),
        torch.tensor(np.stack(Xps), dtype=torch.float32, device=DEVICE),
    )

class DecoderAttn(nn.Module):
    def __init__(self, d, hid=128, n_heads=4, n_layers=2):
        super().__init__()
        feat = 8*d
        self.embed = nn.Sequential(nn.LayerNorm(feat), nn.Linear(feat, hid))
        enc = []
        for _ in range(n_layers):
            enc.append(nn.TransformerEncoderLayer(
                d_model=hid, nhead=n_heads, dim_feedforward=hid*2,
                batch_first=True, dropout=0.1, activation="gelu"))
        self.encoder = nn.Sequential(*enc)
        self.head = nn.Sequential(nn.Linear(hid, hid//2), nn.GELU(), nn.Linear(hid//2, 1))
    def _feats(self, Y, Xp):
        B,P,d = Xp.shape
        Yrep = Y[:,None,:].expand(-1,P,-1)
        mu = Xp.mean(1,keepdim=True).expand(-1,P,-1)
        sd = (Xp.std(1,keepdim=True)+1e-8).expand(-1,P,-1)
        feats= torch.cat([Yrep, Xp, Xp-Yrep, (Xp-Yrep).abs(), Xp-mu, (Xp-mu)/sd, mu, sd], -1)
        return feats
    def forward(self, Y, Xp):
        z = self.embed(self._feats(Y, Xp))
        z = self.encoder(z)
        return self.head(z).squeeze(-1)
    def log_prob(self, M, Y, Xp):
        g = self.forward(Y, Xp)
        return (M*(-F.softplus(-g)) + (1-M)*(-F.softplus(g))).sum(-1)

def train_base_decoder(X_train, X_val, d, epochs=80, batch_size=128, patience=8, steps_per_epoch=400, val_batches=100):
    dec = DecoderAttn(d).to(DEVICE)
    opt = torch.optim.AdamW(dec.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = CosineAnnealingLR(opt, T_max=epochs)
    best_val, stalls = float('inf'), 0
    safe(f"Training base decoder up to {epochs} epochs…")
    for ep in range(1, epochs+1):
        dec.train()
        for _ in range(steps_per_epoch):
            M, Y0, Xp = minibatch(X_train, batch_size)
            loss = -dec.log_prob(M, Y0, Xp).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        dec.eval()
        v_losses=[]
        with torch.no_grad():
            for _ in range(val_batches):
                Mv, Yv0, Xpv = minibatch(X_val, batch_size)
                v_loss = -dec.log_prob(Mv, Yv0, Xpv).mean().item()
                v_losses.append(v_loss)
        v_loss = float(np.mean(v_losses))
        sched.step()
        safe(f"Epoch {ep:02d}/{epochs} – val_loss={v_loss:.4f}")
        if v_loss + 1e-6 < best_val:
            best_val, stalls = v_loss, 0
        else:
            stalls += 1
            if stalls >= patience:
                safe(f"Early stopping at epoch {ep}")
                break

    return dec.eval()

def _project_pd(Sigma: np.ndarray, ridge: float = 1e-10) -> np.ndarray:
    S = 0.5*(Sigma+Sigma.T)
    w,V = np.linalg.eigh(S)
    w = np.clip(w, ridge, None)

    return (V @ np.diag(w) @ V.T).astype(np.float32)

@torch.no_grad()
def _chol_or_diag(S: torch.Tensor):
    d = S.size(0)
    eye = torch.eye(d, device=S.device, dtype=S.dtype)
    try:
        return torch.linalg.cholesky(S + 1e-8*eye), True
    except RuntimeError:
        diag = torch.sqrt(torch.clamp(torch.diag(S), min=1e-8))
        return diag, False

def _hc_and_power_on_frozen_torch(decoder, Sigma_fn, frozen):
    M, Y0, Xp, Z = frozen
    S = Sigma_fn()
    L, ok = _chol_or_diag(S)
    eps = Z @ L.T if ok else (Z * L)
    Hc = -(decoder.log_prob(M, Y0 + eps, Xp)).mean()
    pow_raw = (eps.pow(2).sum(dim=1).mean())

    return Hc, pow_raw

@torch.no_grad()
def _freeze_minibatch(X: np.ndarray, batch_count: int, batch_size: int = 32, seed: int = 777):
    Ms, Ys, Xps = [], [], []
    for _ in range(batch_count):
        M, Y0, Xp = minibatch(X, batch_size)
        Ms.append(M); Ys.append(Y0); Xps.append(Xp)
    M = torch.cat(Ms, 0)
    Y0 = torch.cat(Ys, 0)
    Xp = torch.cat(Xps, 0)
    try:
        g = torch.Generator(device=DEVICE)
    except TypeError:
        g = torch.Generator()
    g.manual_seed(seed)
    try:
        Z = torch.randn(Y0.shape, device=DEVICE, dtype=Y0.dtype, generator=g)
    except TypeError:
        torch.manual_seed(seed)
        Z = torch.randn_like(Y0, device=DEVICE)

    return (M, Y0, Xp, Z)

@torch.no_grad()
def estimate_Hc_crn(decoder, frozen, Sigma: np.ndarray) -> float:
    M, Y0, Xp, Z = frozen
    d = Sigma.shape[0]
    try:
        L = np.linalg.cholesky(Sigma + 1e-8*np.eye(d)).astype(np.float32)
        L = torch.tensor(L, device=DEVICE)
        eps = Z @ L.T
    except np.linalg.LinAlgError:
        diag = torch.tensor(np.sqrt(np.clip(np.diag(Sigma), 0.0, None) + 1e-8), device=DEVICE)
        eps = Z * diag

    return float((-decoder.log_prob(M, Y0 + eps, Xp)).mean().item())

def estimate_Hc_report(decoder, X: np.ndarray, Sigma: np.ndarray, *, batch_count=512, batch_size=32, seed=777, frozen=None) -> float:
    if frozen is None:
        frozen = _freeze_minibatch(X, batch_count=batch_count, batch_size=batch_size, seed=seed)

    return estimate_Hc_crn(decoder, frozen, Sigma)

def _calibrate_isotropic(decoder, frozen, d, target, tol):
    lo, hi = 0.0, 1.0
    def Hc_of(s2): return estimate_Hc_crn(decoder, frozen, (s2*np.eye(d, dtype=np.float32)))
    while Hc_of(hi) < target - tol and hi < 1e6: hi *= 2.0
    for _ in range(28):
        mid = 0.5*(lo+hi)
        if Hc_of(mid) >= target - tol:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-3*max(1.0, hi): break

    return (hi*np.eye(d, dtype=np.float32))

def calibrate_pac_cov(decoder, X, beta, Sigma_raw, H_M, tol=None, batch_count=256, max_expand=18, max_bisect=28, seed=777, frozen=None, return_alpha=False):
    d = Sigma_raw.shape[0]
    Sigma_raw = _project_pd(Sigma_raw)
    target = float(H_M - beta)
    tol = max(CAL_TOL_MIN, CAL_TOL_SCALE*float(beta)) if tol is None else float(tol)
    frozen = _freeze_minibatch(X, batch_count=batch_count, batch_size=32, seed=seed) if frozen is None else frozen

    Hc0 = estimate_Hc_crn(decoder, frozen, np.zeros((d,d), np.float32))
    if Hc0 >= target - tol:
        return (np.zeros((d,d), np.float32), 0.0) if return_alpha else np.zeros((d,d), np.float32)

    def Hc_of(alpha: float) -> float:
        return estimate_Hc_crn(decoder, frozen, _project_pd(alpha * Sigma_raw))

    lo, hi = 0.0, 1.0
    H_hi, expand = Hc_of(hi), 0
    while (H_hi < target - tol) and (expand < max_expand):
        hi *= 2.0
        H_hi = Hc_of(hi); expand += 1

    if H_hi < target - tol:
        Sigma_cal = _calibrate_isotropic(decoder, frozen, d, target, tol)
        return (Sigma_cal, None) if return_alpha else Sigma_cal

    for _ in range(max_bisect):
        mid = 0.5*(lo+hi)
        H_mid = Hc_of(mid)
        if H_mid >= target - tol:
            hi = mid
        else:
            lo = mid
        if hi - lo <= 1e-3*max(1.0, hi): break
    Sigma_cal = _project_pd(hi * Sigma_raw)

    return (Sigma_cal, hi) if return_alpha else Sigma_cal

def auto_pac(m, X, beta):
    Ys = [mechanism(*sample_pool(X)[:2]) for _ in range(m)]
    Ys = np.stack(Ys).astype(np.float32)
    Sig_raw = np.cov(Ys.T, bias=True).astype(np.float32)
    d = Sig_raw.shape[0]
    c = 1.0 / (math.exp(2.0 * beta / d) - 1.0)
    Sig_B = c * Sig_raw + 1e-8 * np.eye(d, dtype=np.float32)

    return Sig_B.astype(np.float32)


def pac_eff_closed_form_raw(CovRaw: np.ndarray, beta: float) -> np.ndarray:
    CovRaw = 0.5*(CovRaw + CovRaw.T)
    w, U = np.linalg.eigh(CovRaw.astype(np.float64))
    w = np.clip(w, 1e-12, None)
    s = np.sqrt(w)
    c = s.sum() / (2.0 * float(beta))
    e = (c * s).astype(np.float32)

    return (U @ np.diag(e) @ U.T).astype(np.float32)

def expected_power(Sigma: np.ndarray, mu: np.ndarray = None) -> float:
    p = float(np.trace(Sigma))
    if mu is not None:
        p += float(np.dot(mu, mu))

    return p

def expected_norm(Sigma, n_mc: int = 400) -> float:
    rng = np.random.default_rng(0)
    try:
        L = np.linalg.cholesky(Sigma + 1e-8*np.eye(Sigma.shape[0]))
        z = rng.standard_normal((n_mc, Sigma.shape[0])).astype(np.float32)
        x = z @ L.T
        return float(np.linalg.norm(x, axis=1).mean())
    except np.linalg.LinAlgError:
        return math.sqrt(float(np.trace(Sigma)))

def _mi_from_p(p):
    return p*math.log(2*p) + (1-p)*math.log(2*(1-p)) if p>0.5 else 0.0

def invert_beta(beta, tol=1e-12):
    lo, hi = 0.5, 1.0
    if beta >= math.log(2): return 1.0
    for _ in range(60):
        mid = 0.5*(lo+hi)
        if _mi_from_p(mid) < beta:
            lo = mid
        else:
            hi = mid
        if hi-lo < tol: break

    return 0.5*(lo+hi)

def eps_from_beta(beta):
    p_star = invert_beta(beta)

    return math.inf if p_star >= 1.0-1e-15 else math.log(p_star/(1-p_star))

# DP bits
def invert_amplification_poisson(eps_pop: float, delta_pop: float, q: float):
    q = max(1e-12, min(1.0, float(q)))
    eps0 = math.log(1.0 + (math.exp(eps_pop) - 1.0) / q)
    delta0 = float(delta_pop) / q

    return eps0, delta0

def l2_clip_rows(X: np.ndarray, C: float) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    scale = np.minimum(1.0, C / norms)

    return X * scale

def gaussian_sigma_for_sum(eps0: float, delta0: float, C: float, *, adjacency: str = "add_remove") -> float:
    if not math.isfinite(eps0) or eps0 <= 0.0: return 0.0
    const = math.sqrt(2.0 * math.log(1.25 / max(delta0, 1e-30)))
    Delta2 = (2.0*C) if adjacency=="replace" else C

    return (Delta2 * const) / eps0

def expected_noise_norm_sum_to_mean(d: int, sigma_sum: float, pool_size: int, memb_prob: float) -> float:
    if sigma_sum <= 0.0: return 0.0
    k_d = math.sqrt(2.0) * math.gamma(0.5*(d+1)) / math.gamma(0.5*d)
    P, p = int(pool_size), float(memb_prob)
    p0 = (1.0 - p)**P; Z = 1.0 - p0
    if Z <= 0.0: return k_d * sigma_sum
    exp_inv_s = 0.0
    for s in range(1, P+1):
        pmf = math.comb(P, s) * (p**s) * ((1-p)**(P-s)) / Z
        exp_inv_s += pmf * (1.0/s)

    return k_d * sigma_sum * exp_inv_s

def find_opt_clipping_sum_to_mean(
    X: np.ndarray,
    eps_pop: float,
    delta_pop: float,
    pool_size: int,
    memb_prob: float,
    *,
    adjacency: str = "add_remove",
    C_candidates = None,
    trials: int = 200,
    rng: np.random.Generator = None,
):
    if rng is None:
        rng = np.random.default_rng(0)
    if C_candidates is None:
        C_candidates=[0.5,1.0,1.5,2.0,3.0,4.0,5.0,6.0,8.0,10.0]

    n, d = int(X.shape[0]), int(X.shape[1]); P, p = int(pool_size), float(memb_prob)
    q = (P / max(1,n)) * p
    eps0, delta0 = invert_amplification_poisson(eps_pop, delta_pop, q)
    best_C, best_err = None, float('inf')
    for C in C_candidates:
        sigma_sum = gaussian_sigma_for_sum(eps0, delta0, C, adjacency=adjacency)
        errs=[]
        for _ in range(trials):
            replace = n < P
            idx = rng.choice(n, size=P, replace=replace)
            Xp = X[idx]
            m = (rng.random(P) < p).astype(np.int32)
            s = int(m.sum()); retry=0
            while s==0 and retry<10:
                m = (rng.random(P) < p).astype(np.int32)
                s = int(m.sum()); retry+=1
            if s==0: continue
            Xm = Xp[m==1]
            Xm_clip = l2_clip_rows(Xm, C)
            y_true = Xm_clip.mean(axis=0)
            noise = rng.normal(0.0, sigma_sum, size=d)
            y_dp = (Xm_clip.sum(axis=0) + noise) / s
            errs.append(np.linalg.norm(y_dp - y_true))
        if not errs: continue
        avg_err = float(np.mean(errs))
        if avg_err < best_err:
            best_err, best_C = avg_err, C
    if best_C is None:
        best_C = 3.0
    sigma_sum_best = gaussian_sigma_for_sum(eps0, delta0, best_C, adjacency=adjacency)

    return best_C, sigma_sum_best

# ──────────────────────────────────────────────────
def _sample_noise_np(Sigma: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    d = Sigma.shape[0]
    try:
        L = np.linalg.cholesky(Sigma + 1e-8*np.eye(d)).astype(np.float32)
        z = rng.normal(size=d).astype(np.float32)
        return z @ L.T
    except np.linalg.LinAlgError:
        diag = np.sqrt(np.clip(np.diag(Sigma), 0.0, None).astype(np.float32) + 1e-8)
        return rng.normal(size=d).astype(np.float32) * diag

def _build_pool_for_index(X: np.ndarray, idx: int, y: int, rng: np.random.Generator):
    N, _ = X.shape
    pos = int(rng.integers(POOL_SIZE))
    pool_idx = rng.choice(N, size=POOL_SIZE, replace=True)
    pool_idx[pos] = idx
    Xp = X[pool_idx].astype(np.float32)

    m = (rng.random(POOL_SIZE) < MEMB_PROB).astype(np.int32)
    m[pos] = int(y)
    s = int(m.sum())
    if s == 0:
        j = int(rng.integers(POOL_SIZE));  j = (j + (j == pos)) % POOL_SIZE
        m[j] = 1
    if s == POOL_SIZE:
        j = int(rng.integers(POOL_SIZE));  j = (j + (j == pos)) % POOL_SIZE
        m[j] = 0

    return Xp, m.astype(np.float32), pos

@torch.no_grad()
def _decoder_score(decoder, Y: np.ndarray, Xp: np.ndarray, pos: int) -> float:
    Yt = torch.tensor(Y[None, :], dtype=torch.float32, device=DEVICE)
    Xpt = torch.tensor(Xp[None, :, :], dtype=torch.float32, device=DEVICE)
    logits = decoder(Yt, Xpt)

    return float(logits[0, pos].item())

def collect_lira_inputs_for_sigma(
    decoder, X: np.ndarray, Sigma: np.ndarray,
    *, n_points=80, n_in=40, n_out=40, seed=1337
):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idxs = rng.choice(N, size=min(n_points, N), replace=False)

    scores_in, scores_out = [], []
    victim_scores = np.zeros(len(idxs), dtype=np.float32)
    labels = np.zeros(len(idxs), dtype=np.int32)

    for j, idx in enumerate(idxs):
        y = 1 if rng.random() < MEMB_PROB else 0
        Xp, msk, pos = _build_pool_for_index(X, idx, y, rng)
        Y = mechanism(Xp, msk) + _sample_noise_np(Sigma, rng)
        victim_scores[j] = _decoder_score(decoder, Y, Xp, pos)
        labels[j] = y

        sh_in = []
        for _ in range(n_in):
            Xp, msk, pos = _build_pool_for_index(X, idx, 1, rng)
            Y = mechanism(Xp, msk) + _sample_noise_np(Sigma, rng)
            sh_in.append(_decoder_score(decoder, Y, Xp, pos))
        scores_in.append(np.asarray(sh_in, dtype=np.float32))

        sh_out = []
        for _ in range(n_out):
            Xp, msk, pos = _build_pool_for_index(X, idx, 0, rng)
            Y = mechanism(Xp, msk) + _sample_noise_np(Sigma, rng)
            sh_out.append(_decoder_score(decoder, Y, Xp, pos))
        scores_out.append(np.asarray(sh_out, dtype=np.float32))

    return scores_in, scores_out, victim_scores, labels

def _safe_var_1d(x: np.ndarray, floor: float = 1e-6) -> float:
    v = float(np.var(x.astype(np.float64)))
    return v if (np.isfinite(v) and v >= floor) else floor

def robust_gaussian_psr(scores_in, scores_out, victim_scores, labels, prior=0.5, var_floor=1e-6) -> float:
    correct = 0
    for i in range(len(victim_scores)):
        mu_in = float(np.mean(scores_in[i]).astype(np.float64))
        s2_in = _safe_var_1d(scores_in[i], var_floor)
        mu_out = float(np.mean(scores_out[i]).astype(np.float64))
        s2_out = _safe_var_1d(scores_out[i], var_floor)
        v = float(victim_scores[i])

        logp_in = np.log(prior) - 0.5*((v - mu_in)**2 / s2_in + np.log(2*np.pi*s2_in))
        logp_out = np.log(1-prior) - 0.5*((v - mu_out)**2 / s2_out + np.log(2*np.pi*s2_out))
        m = max(logp_in, logp_out)
        p_in = np.exp(logp_in - m) / (np.exp(logp_in - m) + np.exp(logp_out - m))
        pred = 1 if p_in >= 0.5 else 0
        correct += int(pred == int(labels[i]))

    return correct / max(1, len(victim_scores))

def train_noise_aware_decoder(base_decoder, X, Sigma, *, steps=1200, batch_size=128, lr=1e-4, weight_decay=1e-4, seed=123):
    torch.manual_seed(seed)
    import copy as _copy
    dec = _copy.deepcopy(base_decoder).to(DEVICE)
    dec.train()
    d = Sigma.shape[0]
    Sigma_t = torch.tensor(Sigma, dtype=torch.float32, device=DEVICE)
    try:
        L = torch.linalg.cholesky(Sigma_t + 1e-8 * torch.eye(d, device=DEVICE))
        def sample_eps(B):
            z = torch.randn(B, d, device=DEVICE);  return z @ L.T
    except RuntimeError:
        diag = torch.sqrt(torch.clamp(torch.diag(Sigma_t), min=1e-8))
        def sample_eps(B):
            return torch.randn(B, d, device=DEVICE) * diag

    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        M, Y0, Xp = minibatch(X, batch_size=batch_size)
        eps = sample_eps(M.size(0))
        loss = -dec.log_prob(M, Y0 + eps, Xp).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()

    return dec.eval()

def _dp_mechanism_output(Xp: np.ndarray, m: np.ndarray, C: float, sigma_sum: float, rng: np.random.Generator) -> np.ndarray:
    Xm = Xp[m == 1]
    s = max(1, Xm.shape[0])
    Xm_clip = l2_clip_rows(Xm.astype(np.float32), C)
    d = Xp.shape[1]
    noise = rng.normal(0.0, sigma_sum, size=d).astype(np.float32)
    return (Xm_clip.sum(axis=0) + noise) / s

def collect_lira_inputs_for_dp(
    decoder, X: np.ndarray, C: float, sigma_sum: float,
    *, n_points=80, n_in=40, n_out=40, seed=2024
):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idxs = rng.choice(N, size=min(n_points, N), replace=False)

    scores_in, scores_out = [], []
    victim_scores = np.zeros(len(idxs), dtype=np.float32)
    labels = np.zeros(len(idxs), dtype=np.int32)

    for j, idx in enumerate(idxs):
        y = 1 if rng.random() < MEMB_PROB else 0
        Xp, msk, pos = _build_pool_for_index(X, idx, y, rng)
        Y = _dp_mechanism_output(Xp, msk, C, sigma_sum, rng)
        victim_scores[j] = _decoder_score(decoder, Y, Xp, pos)
        labels[j] = y

        sh_in = []
        for _ in range(n_in):
            Xp, msk, pos = _build_pool_for_index(X, idx, 1, rng)
            Y = _dp_mechanism_output(Xp, msk, C, sigma_sum, rng)
            sh_in.append(_decoder_score(decoder, Y, Xp, pos))
        scores_in.append(np.asarray(sh_in, dtype=np.float32))

        sh_out = []
        for _ in range(n_out):
            Xp, msk, pos = _build_pool_for_index(X, idx, 0, rng)
            Y = _dp_mechanism_output(Xp, msk, C, sigma_sum, rng)
            sh_out.append(_decoder_score(decoder, Y, Xp, pos))
        scores_out.append(np.asarray(sh_out, dtype=np.float32))

    return scores_in, scores_out, victim_scores, labels

def train_noise_aware_decoder_dp(
    base_decoder, X: np.ndarray, C: float, sigma_sum: float,
    *, steps=1200, batch_size=128, lr=1e-4, weight_decay=1e-4, seed=135
):
    torch.manual_seed(seed)
    import copy as _copy
    dec = _copy.deepcopy(base_decoder).to(DEVICE)
    dec.train()
    opt = torch.optim.AdamW(dec.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    for _ in range(steps):
        Ms, Ys, Xps = [], [], []
        for _b in range(batch_size):
            Xp, m, _ = sample_pool(X)
            Y_dp = _dp_mechanism_output(Xp, m, C, sigma_sum, rng)
            Ms.append(m.astype(np.float32)); Ys.append(Y_dp.astype(np.float32)); Xps.append(Xp.astype(np.float32))
        M_t = torch.tensor(np.stack(Ms), dtype=torch.float32, device=DEVICE)
        Y_t = torch.tensor(np.stack(Ys), dtype=torch.float32, device=DEVICE)
        Xp_t = torch.tensor(np.stack(Xps), dtype=torch.float32, device=DEVICE)

        loss = -dec.log_prob(M_t, Y_t, Xp_t).mean()
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
    return dec.eval()

# ──────────────────────────────────────────────────
class WhitenedNoise(nn.Module):
    def __init__(self, std_base: np.ndarray):
        super().__init__()
        self.std = torch.tensor(3*std_base, dtype=torch.float32, device=DEVICE)
        d = len(std_base)
        self.raw = nn.Parameter(torch.randn(d, d, dtype=torch.float32) * 0.3)
    def Sigma(self) -> torch.Tensor:
        L = torch.tril(self.raw, -1)
        L = L + torch.diag(F.softplus(torch.diag(self.raw)) + 1e-6)
        D = torch.diag(self.std)
        return D @ (L @ L.T) @ D
    def sample(self, B: int) -> torch.Tensor:
        S = self.Sigma()
        try:
            L = torch.linalg.cholesky(S)
            z = torch.randn(B, self.std.numel(), device=DEVICE)
            return z @ L.T
        except RuntimeError:
            diag_std = torch.sqrt(torch.diag(S)+1e-6)
            return torch.randn(B, self.std.numel(), device=DEVICE) * diag_std

def sr_pac(base_decoder, X, beta, frozen_cal):
    target = H_M - beta

    cov_samples = [sample_pool(X)[2] for _ in range(400)]
    cov = np.cov(np.stack(cov_samples).T).astype(np.float32)
    w, V = np.linalg.eigh(cov); w = np.clip(w, VAR_FLOOR, None)
    std_base = np.sqrt(np.diag(V @ np.diag(w) @ V.T))

    class WhitenedNoise(nn.Module):
        def __init__(self, std_base_np: np.ndarray):
            super().__init__()
            self.std = torch.tensor(3*std_base_np, dtype=torch.float32, device=DEVICE)
            d = len(std_base_np)
            self.raw = nn.Parameter(torch.randn(d, d, dtype=torch.float32) * 0.3)
        def Sigma(self) -> torch.Tensor:
            L = torch.tril(self.raw, -1)
            L = L + torch.diag(F.softplus(torch.diag(self.raw)) + 1e-6)
            D = torch.diag(self.std)
            return D @ (L @ L.T) @ D

    leader = WhitenedNoise(std_base).to(DEVICE)
    follower = copy.deepcopy(base_decoder).to(DEVICE)

    opt_phi = torch.optim.AdamW(follower.parameters(), lr=LR_FOLL, weight_decay=1e-4)
    opt_lam = torch.optim.AdamW(leader.parameters(),   lr=LR_LEADER, weight_decay=1e-3)
    sched_lam = CosineAnnealingLR(opt_lam, T_max=SR_OUTER)
    lam = torch.tensor(0.0, device=DEVICE)
    rho = 1000.0
    tol = max(1e-2, 0.10*beta)

    best_power, best_S = float('inf'), None

    safe(f"[SR-PAC] target H_c ≥ {target:.3f} (β={beta:.3f}) — fixed CRNs")

    for t in range(1, SR_OUTER+1):
        follower.train()
        for _ in range(SR_INNER):
            def Sigma_now(): return leader.Sigma().detach()
            Hc_f, _ = _hc_and_power_on_frozen_torch(follower, Sigma_now, frozen_cal)
            loss_phi = Hc_f
            opt_phi.zero_grad(); loss_phi.backward()
            torch.nn.utils.clip_grad_norm_(follower.parameters(), 1.0)
            opt_phi.step()
        follower.eval()

        def Sigma_now(): return leader.Sigma()
        Hc_meas, pow_meas = _hc_and_power_on_frozen_torch(follower, Sigma_now, frozen_cal)
        gap = Hc_meas - target

        util_energy = torch.trace(leader.Sigma())
        hinge = F.relu(-gap)
        loss_leader = util_energy + lam*hinge + 0.5*rho*(hinge**2)

        opt_lam.zero_grad(); loss_leader.backward()
        torch.nn.utils.clip_grad_norm_(leader.parameters(), 1.0)
        opt_lam.step(); sched_lam.step()
        with torch.no_grad():
            lam = torch.clamp(lam + rho*hinge, min=0.0, max=1e6)

        if gap.item() >= -tol and pow_meas.item() < best_power:
            best_power, best_S = pow_meas.item(), leader.Sigma().detach().cpu().numpy()
            safe("        ↑ new BEST (feasible on frozen CRNs)        ")

        safe(f"[SR] {t:02d}/{SR_OUTER} power≈{float(pow_meas):.4f} H_c≈{float(Hc_meas):.4f} gap={float(gap):+.4f} λ={float(lam):.2f}")

    if best_S is None:
        best_S = leader.Sigma().detach().cpu().numpy()
        safe("        (no feasible Σ found on frozen CRNs; returning last iterate)       ")
    return best_power, best_S


# ──────────────────────────────────────────────────
def main():
    Xt, yt, Xv, yv, d = get_data()
    safe(f"Dataset: {len(Xt)} × {d}, device={DEVICE}")

    base_decoder = train_base_decoder(Xt, Xv, d, epochs=80, batch_size=128, patience=10, steps_per_epoch=400, val_batches=100)
    frozen_global = _freeze_minibatch(Xt, batch_count=REPORT_BATCH, batch_size=32, seed=777)
    H0 = estimate_Hc_report(base_decoder, Xt, np.zeros((d,d), np.float32), batch_count=REPORT_BATCH, batch_size=32, seed=777, frozen=frozen_global)
    intrinsic_mi = max(0.0, H_M - H0)
    safe(f"Intrinsic MI ≈ {intrinsic_mi:.3f} nats")

    k = min(EFFICIENT_k, d)
    rng = np.random.default_rng(SEED)
    G = rng.standard_normal((d, k)).astype(np.float32)

    results = []
    safe("\n" + "=" * 70)
    safe(f"{'β':>8} {'power-SR':>10} {'power-DP':>10} {'power-Auto':>12} {'power-Eff':>12}")
    safe("=" * 70)
    emp_psr = {}

    for beta in BETAS:
        tol_beta  = max(CAL_TOL_MIN, CAL_TOL_SCALE*beta)
        Hc_target = H_M - beta
        # ────────────── SR-PAC ────────────────────────────────────
        sr_util, Sig_SR_shape = sr_pac(base_decoder, Xt, beta, frozen_cal=frozen_global)
        Sig_SR = calibrate_pac_cov(base_decoder, Xt, beta, Sig_SR_shape, H_M, tol=tol_beta, batch_count=CAL_BATCH, frozen=frozen_global)
        sr_pow = expected_power(Sig_SR)
        Hc_SR = estimate_Hc_report(base_decoder, Xt, Sig_SR, batch_count=REPORT_BATCH, batch_size=32, seed=777, frozen=frozen_global)
        safe(f"[SR-PAC β={beta:.3f}] target H_c={Hc_target:.3f}, achieved H_c={Hc_SR:.3f}")

        # ──────────── DP baseline──────────────────────────────────────
        eps_pop = eps_from_beta(beta)
        C_star, sigma_sum = find_opt_clipping_sum_to_mean(
            Xt, eps_pop=eps_pop, delta_pop=DELTA_DP,
            pool_size=POOL_SIZE, memb_prob=MEMB_PROB,
            adjacency="add_remove", trials=200
        )
        dp_norm = expected_noise_norm_sum_to_mean(d, sigma_sum, POOL_SIZE, MEMB_PROB)
        s_eff = max(1, int(round(POOL_SIZE * MEMB_PROB)))
        sigma_mean_eff = sigma_sum / s_eff
        Sig_DP_diag = (sigma_mean_eff ** 2) * np.eye(d, dtype=np.float32)
        dp_pow = expected_power(Sig_DP_diag)
        Hc_DP = estimate_Hc_report(base_decoder, Xt, Sig_DP_diag,
                                   batch_count=REPORT_BATCH, batch_size=32,
                                   seed=777, frozen=frozen_global)
        safe(f"[DP β={beta:.3f}] ε_pop≈{eps_pop:.3f}, C*≈{C_star:.3f}, σ_sum≈{sigma_sum:.4f}, "
             f"‖B‖_mean≈{dp_norm:.3f}; diag H_c={Hc_DP:.3f}")

        # ──────────── Auto-PAC ──────────────────────────────────────
        Sig_Auto_raw = auto_pac(300, Xt, beta)
        d_local = Sig_Auto_raw.shape[0]
        c_auto = 1.0 / (math.exp(2.0 * beta / d_local) - 1.0)
        CovRaw_est = Sig_Auto_raw / max(c_auto, 1e-12)
        Sig_Auto = calibrate_pac_cov(base_decoder, Xt, beta, Sig_Auto_raw, H_M, tol=tol_beta, batch_count=CAL_BATCH, frozen=frozen_global)
        auto_pow = expected_power(Sig_Auto)
        Hc_Auto = estimate_Hc_report(base_decoder, Xt, Sig_Auto, batch_count=REPORT_BATCH, batch_size=32, seed=777, frozen=frozen_global)
        safe(f"[PAC-Auto β={beta:.3f}] achieved H_c={Hc_Auto:.3f}")

        # ───────────── Efficient-PAC (IMP/Eff)─────────────────────────────────────
        Sig_Eff_raw = pac_eff_closed_form_raw(CovRaw_est, beta)
        Sig_Eff = calibrate_pac_cov(base_decoder, Xt, beta, Sig_Eff_raw, H_M,
                                    tol=tol_beta, batch_count=CAL_BATCH, frozen=frozen_global)
        eff_pow = expected_power(Sig_Eff)
        Hc_Eff = estimate_Hc_report(base_decoder, Xt, Sig_Eff,
                                    batch_count=REPORT_BATCH, batch_size=32,
                                    seed=777, frozen=frozen_global)
        safe(f"[PAC-Eff β={beta:.3f}] achieved H_c={Hc_Eff:.3f}")

        results.append([beta, sr_pow, dp_pow, auto_pow, eff_pow])
        safe(f"{beta:8.3f} {sr_pow:10.6f} {dp_pow:10.6f} {auto_pow:12.6f} {eff_pow:12.6f}")

        # ──────────── Empirical LiRA──────────────────────────────────────
        # DP
        attack_dec_dp = train_noise_aware_decoder_dp(base_decoder, Xt, C_star, sigma_sum, steps=1200)
        s_in_dp, s_out_dp, v_dp, y_dp = collect_lira_inputs_for_dp(
            attack_dec_dp, Xt, C_star, sigma_sum, n_points=120, n_in=80, n_out=80, seed=555 + int(1e6 * beta)
        )
        _, psr_emp_dp, _ = lira_posteriors_and_success(s_in_dp, s_out_dp, v_dp, y_dp, prior=MEMB_PROB)
        if not np.isfinite(psr_emp_dp):
            psr_emp_dp = robust_gaussian_psr(s_in_dp, s_out_dp, v_dp, y_dp, prior=MEMB_PROB, var_floor=1e-6)
        emp_psr[("DP", beta)] = psr_emp_dp
        safe(f"[LiRA] Empirical PSR (DP, β={beta:.3g}) = {psr_emp_dp:.4f}")

        # SR-PAC
        attack_dec_sr = train_noise_aware_decoder(base_decoder, Xt, Sig_SR, steps=1200)
        s_in_sr, s_out_sr, v_sr, y_sr = collect_lira_inputs_for_sigma(
            attack_dec_sr, Xt, Sig_SR, n_points=120, n_in=80, n_out=80, seed=777 + int(1e6 * beta)
        )
        _, psr_emp_sr, _ = lira_posteriors_and_success(s_in_sr, s_out_sr, v_sr, y_sr, prior=MEMB_PROB)
        if not np.isfinite(psr_emp_sr):
            psr_emp_sr = robust_gaussian_psr(s_in_sr, s_out_sr, v_sr, y_sr, prior=MEMB_PROB, var_floor=1e-6)
        emp_psr[("SR", beta)] = psr_emp_sr
        safe(f"[LiRA] Empirical PSR (SR, β={beta:.3g}) = {psr_emp_sr:.4f}")

        # Auto-PAC
        attack_dec_auto = train_noise_aware_decoder(base_decoder, Xt, Sig_Auto, steps=1200)
        s_in_auto, s_out_auto, v_auto, y_auto = collect_lira_inputs_for_sigma(
            attack_dec_auto, Xt, Sig_Auto, n_points=120, n_in=80, n_out=80, seed=879 + int(1e6 * beta)
        )
        _, psr_emp_auto, _ = lira_posteriors_and_success(s_in_auto, s_out_auto, v_auto, y_auto, prior=MEMB_PROB)
        if not np.isfinite(psr_emp_auto):
            psr_emp_auto = robust_gaussian_psr(s_in_auto, s_out_auto, v_auto, y_auto, prior=MEMB_PROB, var_floor=1e-6)
        emp_psr[("PAC-Auto", beta)] = psr_emp_auto
        safe(f"[LiRA] Empirical PSR (PAC-Auto, β={beta:.3g}) = {psr_emp_auto:.4f}")

        # Efficient-PAC
        attack_dec_eff = train_noise_aware_decoder(base_decoder, Xt, Sig_Eff, steps=1200)
        s_in_eff, s_out_eff, v_eff, y_eff = collect_lira_inputs_for_sigma(
            attack_dec_eff, Xt, Sig_Eff, n_points=120, n_in=80, n_out=80, seed=991 + int(1e6 * beta)
        )
        _, psr_emp_eff, _ = lira_posteriors_and_success(s_in_eff, s_out_eff, v_eff, y_eff, prior=MEMB_PROB)
        if not np.isfinite(psr_emp_eff):
            psr_emp_eff = robust_gaussian_psr(s_in_eff, s_out_eff, v_eff, y_eff, prior=MEMB_PROB, var_floor=1e-6)
        emp_psr[("PAC-Eff", beta)] = psr_emp_eff
        safe(f"[LiRA] Empirical PSR (PAC-Eff, β={beta:.3g}) = {psr_emp_eff:.4f}")

    # ──────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"{'beta':>8s} {'power-SR':>10s} {'power-DP':>10s} {'power-PAC-Auto':>14s} {'power-PAC-Eff':>14s}")

    print("-"*64)
    for beta, sr, dp, pac_auto, pac_eff in results:
        print(f"{beta:8.3f} {sr:10.3f} {dp:10.3f} {pac_auto:14.3f} {pac_eff:14.3f}")
    print("="*64)

    # ──────────────────────────────────────────────────
    eps_by_beta = {b: eps_from_beta(b) for b in BETAS}
    rows = []
    for (method, b), pe in emp_psr.items():
        pt = theoretical_psr(eps_by_beta[b], prior=MEMB_PROB)
        rows.append((b, method, pe, pt, abs(pe - pt)))
    rows.sort(key=lambda r: (r[0], r[1]))
    print("\nLiRA: empirical vs theoretical PSR")
    print("beta       method    PSR_emp   PSR_theory   |diff|")
    print("--------------------------------------------------")
    for b, m, pe, pt, dff in rows:
        print(f"{b:<10g} {m:<9s}  {pe:7.4f}     {pt:8.4f}   {dff:7.4f}")

    sr_avg = np.mean([r[1] for r in results])
    dp_avg = np.mean([r[2] for r in results])
    auto_avg = np.mean([r[3] for r in results])
    eff_avg = np.mean([r[4] for r in results])
    safe("\nAverage noise power (E||B||^2):")
    safe(f"  SR-PAC:        {sr_avg:.4f}")
    safe(f"  Gaussian DP:   {dp_avg:.4f}")
    safe(f"  Auto-PAC:      {auto_avg:.4f}")
    safe(f"  Efficient-PAC: {eff_avg:.4f}")
    if sr_avg > 0:
        safe("\nSR-PAC improvement (× less noise):")
        safe(f"  vs DP:   {dp_avg/sr_avg:.2f}x")
        safe(f"  vs Auto: {auto_avg/sr_avg:.2f}x")
        safe(f"  vs Eff:  {eff_avg/sr_avg:.2f}x")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()