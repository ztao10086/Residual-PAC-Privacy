
import os, math, json, csv, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import os, torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform
from math import log, pi
from math import gamma as gamma_fn
from torch.special import digamma as psi


# ───────────────────────── Config (faithful defaults) ─────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0
torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
PRETRAINED = "mnist_cnn_non_gaussian.pth"  # if missing, we quick-train fallback
BETA_LIST = []
C_SLACK = 1e-6
ALT_EPOCHS = 40
FEW_EPOCHS = 6
N_HC_SAMPLES = 10
RHO = 1e6
LR = 1e-4
WD = 1e-5
WARM_EPOCHS = 5
SMALL_SIGMAS = [0.0, 0.005, 0.01, 0.02]
K_MEAS = 8
CRN_MEAS_SEED = 4242
CRN_REPORT_SEED = 777
CRN_DIAG_SEED = 2025
ROTATION_SEED = 31415
PERMUTE_SEED = 27182
PRETTY_TABLE = True
OUT_CSV = "mnist_sr_pac_faithfulness_summary.csv"

# ──────────────────────────────────────────────────
class MNIST_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7, 128), nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.fc(self.conv(x))

class DiagPosterior(nn.Module):
    def __init__(self, dim, sigma=0.25):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.full((dim,), math.log(sigma)))
        self.dim = dim
        self.const = 0.5*dim*math.log(2*math.pi)
    def log_prob(self, z, y):
        diff = z - y
        inv_var = torch.exp(-2.0*self.log_sigma)
        quad = 0.5*(diff.pow(2)*inv_var).sum(dim=1)
        logdet = self.log_sigma.sum()*2.0
        return -(self.const + 0.5*logdet + quad)

class FlowPosterior(nn.Module):
    def __init__(self, dim, context_dim, depth=3, hidden=128):
        super().__init__()
        base = StandardNormal([dim])
        trs = []
        for _ in range(depth):
            trs.append(MaskedAffineAutoregressiveTransform(
                features=dim, hidden_features=hidden, context_features=context_dim))
        self.flow = Flow(CompositeTransform(trs), base)
    def log_prob(self, z, y):
        return self.flow.log_prob(inputs=z, context=y)

class NoiseGenerator(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, dim)
        self.lv = nn.Linear(hidden, dim)
    def _stats(self, z):
        h = self.net(z)
        mu = self.mu(h)
        lv = self.lv(h).clamp(min=-5)
        return mu, lv
    def forward(self, z):
        mu, lv = self._stats(z)
        std = torch.exp(0.5*lv)
        eps = torch.randn_like(std)
        B = mu + std*eps
        return B, mu, lv
    def forward_with_eps(self, z, eps):
        mu, lv = self._stats(z)
        std = torch.exp(0.5*lv)
        B = mu + std*eps
        return B, mu, lv

class ScaledNG(nn.Module):
    def __init__(self, base: NoiseGenerator, alpha: float = 1.0):
        super().__init__()
        self.base = base; self.alpha = float(alpha)
    def forward(self, z):
        B, mu, lv = self.base(z)
        a = self.alpha
        return a*B, a*mu, lv
    def forward_with_eps(self, z, eps):
        B, mu, lv = self.base.forward_with_eps(z, eps)
        a = self.alpha
        return a*B, a*mu, lv

# ──────────────────────────────────────────────────
def make_posterior(dim, use_flow=True):
    if use_flow:
        return FlowPosterior(dim, dim).to(DEVICE), True
    return DiagPosterior(dim, sigma=0.25).to(DEVICE), False

def tamed_logprob(lp: torch.Tensor, floor=-50.0, ceil=8.0, tau=2.0):
    lp = lp / tau
    lp = torch.nan_to_num(lp, neginf=floor, posinf=ceil)
    return lp.clamp(min=floor, max=ceil)

@torch.no_grad()
def accuracy_from_logits(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()

def knn_entropy_bootstrap(samples: torch.Tensor, k=5, bs=5, max_pts=2000):
    def one(xs):
        n, d = xs.shape
        m = min(n, max_pts)
        idx = torch.randperm(n, device=xs.device)[:m]
        pts = xs[idx].cpu().numpy()
        dists = np.sqrt(((pts[:,None,:]-pts[None,:,:])**2).sum(2) + 1e-12)
        np.fill_diagonal(dists, np.inf)
        rk = np.partition(dists, k, axis=1)[:, k]
        vol = pi**(d/2)/gamma_fn(d/2+1)
        base = d*np.mean(np.log(rk+1e-12)) + log(vol)
        return base + psi(torch.tensor(m, dtype=torch.float64)).item() - psi(torch.tensor(k, dtype=torch.float64)).item()
    vals = [one(samples) for _ in range(bs)]

    return float(np.mean(vals)), float(np.std(vals))

def random_orthogonal(d, seed):
    g = torch.Generator(device=DEVICE); g.manual_seed(seed)
    M = torch.randn(d, d, device=DEVICE, generator=g)
    Q, R = torch.linalg.qr(M)
    s = torch.sign(torch.diag(R))
    Q = Q @ torch.diag(s)
    return Q

def compute_Hc_raw(z_w, noise_gen, posterior, n=N_HC_SAMPLES):
    vals = []
    for _ in range(n):
        B,_,_ = noise_gen(z_w)
        y = z_w + B
        vals.append(tamed_logprob(posterior.log_prob(z_w, y)))
    return -torch.stack(vals, 0).mean()

@torch.no_grad()
def compute_Hc_eval(z_w, noise_gen, posterior, eps_list):
    vals = []
    for eps in eps_list:
        B,_,_ = noise_gen.forward_with_eps(z_w, eps)
        y = z_w + B
        vals.append(tamed_logprob(posterior.log_prob(z_w, y)))
    return -torch.stack(vals, 0).mean()

def calibrate_opt_scale(posterior, Zw, gen, target_Hc,
                        K=K_MEAS, eps_list=None, tol=1e-3, max_bisect=30):
    torch.manual_seed(777)
    B_list = []
    if eps_list is None:
        for _ in range(K):
            B_list.append(gen(Zw)[0])
    else:
        for eps in eps_list:
            B_list.append(gen.forward_with_eps(Zw, eps)[0])
    def Hc_of_alpha(a):
        vals = [-tamed_logprob(posterior.log_prob(Zw, Zw + float(a)*B)).mean().item() for B in B_list]
        return float(np.mean(vals))

    lo, hi = 0.0, 1.0
    while Hc_of_alpha(hi) < target_Hc and hi < 1e6:
        hi *= 2.0

    for _ in range(max_bisect):
        mid = 0.5*(lo+hi)
        if Hc_of_alpha(mid) >= target_Hc:
            hi = mid
        else:
            lo = mid
        if hi-lo <= tol*max(1.0,hi): break
    return hi

def _print_table_header():
    print("\nβ | Target_Hc  | Achieved_Hc | PAC_raw | PAC_acc | Eff_raw | Eff_acc || OPT_raw | OPT_acc")
    print("-"*100)

def _print_table_row(beta, target, Hc_ach, pac_mag, pac_acc, Eff_mag, Eff_acc, opt_mag, opt_acc):
    print(f"{beta:4.2f} | {target:10.4f} | {Hc_ach:11.4f} | "
          f"{pac_mag:7.3f} | {pac_acc:7.3f} | {Eff_mag:7.3f} | {Eff_acc:7.3f} || "
          f"{opt_mag:7.3f} | {opt_acc:7.3f}")

def pac_cov_raw(CovRaw: torch.Tensor, beta: float):
    d = CovRaw.size(0)
    c = 1.0 / (math.exp(2.0*beta/d) - 1.0)
    return c * CovRaw

def pac_eff_closed_form_raw(CovRaw: torch.Tensor, beta: float) -> torch.Tensor:
    lam, U = torch.linalg.eigh(CovRaw)
    lam = lam.clamp_min(C_SLACK)
    s = torch.sqrt(lam)
    c = s.sum() / (2.0 * beta)
    e = c * s
    return U @ torch.diag(e.to(CovRaw.dtype)) @ U.T

def imp_diag_kkt(CovRaw: torch.Tensor, beta: float, tol=1e-6, max_iter=80):
    sigma = CovRaw.diag().clamp_min(C_SLACK)
    target = 2.0*beta
    def F(lam):
        t = torch.sqrt(sigma*sigma + 4.0*lam*sigma)
        e = 0.5*(t - sigma).clamp_min(1e-12)
        return float(torch.log1p(sigma/e).sum().item())
    lo, hi = 1e-12, 1.0
    while F(hi) > target and hi < 1e12: hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        if F(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi-lo <= tol*max(1.0,hi): break
    lam = hi
    t = torch.sqrt(sigma*sigma + 4.0*lam*sigma)
    e = 0.5*(t - sigma).clamp_min(1e-12)
    return torch.diag(e)

@torch.no_grad()
def estimate_opt_cov_w(NG_eval, Zw_all, nsamples=4096, seed=123):
    g = torch.Generator(device=Zw_all.device); g.manual_seed(seed)
    idx = torch.randint(0, Zw_all.size(0), (nsamples,), generator=g, device=Zw_all.device)
    eps = torch.randn(nsamples, Zw_all.size(1), device=Zw_all.device, generator=g)
    Bw,_,_ = NG_eval.forward_with_eps(Zw_all[idx], eps)
    Bw = Bw - Bw.mean(0, keepdim=True)
    Sw = (Bw.T @ Bw)/nsamples + 1e-8*torch.eye(Bw.size(1), device=Bw.device)
    return Sw

def effective_rank(Sw: torch.Tensor):
    vals = torch.linalg.eigvalsh(Sw).clamp_min(1e-12)
    w = vals / vals.sum()
    H = -(w * (w+1e-12).log()).sum()
    return float(torch.exp(H).item()), vals.flip(0)

@torch.no_grad()
def margin_power(noise_raw, z_raw, ylab):
    top2 = torch.topk(z_raw, k=2, dim=1).indices
    yhat = top2[:,0]
    runner = torch.where(yhat==ylab, top2[:,1], top2[:,0])
    g_raw = F.one_hot(ylab, num_classes=z_raw.size(1)).float() - F.one_hot(runner, num_classes=z_raw.size(1)).float()
    g_raw = g_raw / math.sqrt(2.0)
    proj = (noise_raw * g_raw).sum(1)
    return float((proj.pow(2)).mean().item())

def run_opt_faithful(Zw_tr, Zw_cal, Zraw_tr, y_tr, target_Hc, Lz, posterior, epochs=ALT_EPOCHS):
    dim = Zw_tr.size(1)
    NG = NoiseGenerator(dim).to(DEVICE)
    opt_noise = optim.AdamW(NG.parameters(), lr=LR, weight_decay=WD)

    with torch.no_grad():
        Hc0 = float((-tamed_logprob(posterior.log_prob(Zw_cal, Zw_cal))).mean().item())
    if target_Hc <= Hc0 + 1e-4:
        return NG

    g_meas = torch.Generator(device=DEVICE); g_meas.manual_seed(CRN_MEAS_SEED)
    EPS_cal = [
        torch.randn(
            Zw_cal.size(0), Zw_cal.size(1),
            dtype=Zw_cal.dtype, device=Zw_cal.device, generator=g_meas
        )
        for _ in range(K_MEAS)
    ]

    lam = torch.tensor(0.0, device=DEVICE); e_int = torch.tensor(0.0, device=DEVICE)
    ds = DataLoader(TensorDataset(Zw_tr, Zraw_tr, y_tr), batch_size=256, shuffle=True)

    for ep in range(1, epochs+1):
        with torch.no_grad():
            alpha = calibrate_opt_scale(posterior, Zw_cal, NG, target_Hc, eps_list=EPS_cal)
        NGs = ScaledNG(NG, alpha)

        for (z_w, z_raw, ylab) in ds:
            z_w = z_w.to(DEVICE); z_raw = z_raw.to(DEVICE); ylab = ylab.to(DEVICE)
            Hc_g = compute_Hc_raw(z_w, NGs, posterior)
            short = (target_Hc - Hc_g).clamp_min(0.0)
            aug = lam*short + 0.5*RHO*short.pow(2)
            Bw,_,_ = NGs(z_w)
            Braw = (Lz @ Bw.T).T
            mag = Braw.pow(2).sum(1).mean()
            loss = mag + aug
            opt_noise.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(NG.parameters(), 1.0)

            opt_noise.step()

        with torch.no_grad():
            Hc_meas = compute_Hc_eval(Zw_cal, NGs, posterior, EPS_cal)
            err = target_Hc - Hc_meas
            e_int = 0.9*e_int + err
            lam = torch.clamp(lam + 0.05*float(err.item()) + 0.01*float(e_int.item()), min=0.0)

        if ep % 5 == 0:
            print(f"[OPT] epoch {ep:02d} | α={alpha:.4f} | λ={lam.item():.4f} | Hc_meas={Hc_meas.item():.4f} | target={target_Hc:.4f}")

    return NG

# ──────────────────────────────────────────────────
def _maybe_remap_keys_to_this_model(state, model):
    want = set(model.state_dict().keys())
    have = set(state.keys())
    if want & have:
        return state
    remapped = {}
    for k, v in state.items():
        if k.startswith("conv_layers."):
            remapped["conv." + k[len("conv_layers."):]] = v
            continue
        if k.startswith("fc_layers."):
            remapped["fc." + k[len("fc_layers."):]] = v
            continue
        if k.startswith("conv."):
            remapped["conv_layers." + k[len("conv."):]] = v
            continue
        if k.startswith("fc."):
            remapped["fc_layers." + k[len("fc."):]] = v
            continue
        remapped[k] = v
    return remapped

def _quick_train_and_save(pretrained_path, device, epochs=2):
    print(f"[INFO] Pretrained weights not found or incompatible. "
          f"Quick-training MNIST CNN for {epochs} epoch(s)…")
    tf_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    trainset = datasets.MNIST(root="./data", train=True, download=True, transform=tf_train)
    loader = DataLoader(trainset, batch_size=512, shuffle=True,
                          num_workers=2, pin_memory=torch.cuda.is_available())

    model = MNIST_CNN().to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()

    os.makedirs(os.path.dirname(pretrained_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), pretrained_path)
    abs_path = os.path.abspath(pretrained_path)
    print(f"[OK] Saved new parameters to {abs_path}")
    return model

def load_or_train_cnn(pretrained_path=PRETRAINED, device=DEVICE):
    model = MNIST_CNN().to(device)
    if os.path.exists(pretrained_path):
        try:
            raw = torch.load(pretrained_path, map_location=device)
            state = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
            try:
                model.load_state_dict(state, strict=True)
                print(f"[OK] Loaded parameters from {os.path.abspath(pretrained_path)}")
            except RuntimeError:
                print("[WARN] Key mismatch. Trying a safe key remap…")
                state2 = _maybe_remap_keys_to_this_model(state, model)
                missing, unexpected = model.load_state_dict(state2, strict=False)
                if missing or unexpected:
                    print(f"[WARN] Missing keys: {missing}\n[WARN] Unexpected: {unexpected}")
                    model = _quick_train_and_save(pretrained_path, device, epochs=2)
                else:
                    print(f"[OK] Loaded after remap from {os.path.abspath(pretrained_path)}")
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint ({e}). Rebuilding weights…")
            model = _quick_train_and_save(pretrained_path, device, epochs=2)
    else:
        model = _quick_train_and_save(pretrained_path, device, epochs=2)
    model.eval()
    return model

def get_test_logits(model):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    testset = datasets.MNIST("./data", train=False, download=True, transform=tf)
    loader = DataLoader(testset, batch_size=512, shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
    Z_list, y_list = [], []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE); logits = model(x)
            Z_list.append(logits.cpu()); y_list.append(y)
    Z_raw = torch.cat(Z_list, 0).to(DEVICE); y = torch.cat(y_list, 0).to(DEVICE)
    return Z_raw, y

# ──────────────────────────────────────────────────
def posterior_warmup_identity(Zw_tr, posterior):
    opt = optim.AdamW(posterior.parameters(), lr=LR, weight_decay=WD)
    ds = DataLoader(TensorDataset(Zw_tr), batch_size=256, shuffle=True)
    posterior.train()
    for _ in range(WARM_EPOCHS):
        for (z,) in ds:
            z = z.to(DEVICE)
            loss_id = -tamed_logprob(posterior.log_prob(z, z)).mean()
            sigma = float(np.random.choice(SMALL_SIGMAS))
            y_small = z if sigma==0.0 else z + sigma*torch.randn_like(z)
            loss_sm = -tamed_logprob(posterior.log_prob(z, y_small)).mean()
            loss = 0.5*loss_id + 0.5*loss_sm
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(posterior.parameters(), 1.0)
            opt.step()
    posterior.eval()
    return posterior

def sweep_one_curve(Z_raw, y, tag, rotate_Q=None, permute_labels=False, independent_meter=False):
    mu = Z_raw.mean(0, keepdim=True)
    Xc = Z_raw - mu
    CovRaw = (Xc.T @ Xc)/Z_raw.size(0) + C_SLACK*torch.eye(Z_raw.size(1), device=DEVICE)
    Lz = torch.linalg.cholesky(CovRaw)
    Zw = torch.linalg.solve_triangular(Lz, Xc.T, upper=False).T

    if rotate_Q is not None:
        Zw = Zw @ rotate_Q.T

    if permute_labels:
        rng = torch.Generator(device=DEVICE); rng.manual_seed(PERMUTE_SEED)
        perm = torch.randperm(y.size(0), generator=rng, device=DEVICE)
        y = y[perm]

    H_X, H_X_std = knn_entropy_bootstrap(Zw.detach().cpu(), k=max(3, int(Zw.size(0)**0.5)-1), bs=5)
    print(f"H_X = {H_X:.4f} (fixed)")
    split_idx = Zw.size(0)//2
    Zw_tr, Zw_cal = Zw[:split_idx].contiguous(), Zw[split_idx:].contiguous()
    Zraw_tr, y_tr = Z_raw[:split_idx].contiguous(), y[:split_idx].contiguous()

    posterior, used_flow = make_posterior(Zw.size(1), use_flow=True)
    posterior = posterior_warmup_identity(Zw_tr, posterior)

    if independent_meter:
        meter, _ = make_posterior(Zw.size(1), use_flow=True)
        meter = posterior_warmup_identity(Zw[split_idx:], meter)
    else:
        meter = posterior

    base_acc = accuracy_from_logits(Z_raw, y)
    print(f"[{tag}] Baseline test accuracy (no noise): {base_acc:.4f}")
    g_eval_w   = torch.Generator(device=DEVICE); g_eval_w.manual_seed(CRN_REPORT_SEED)
    g_eval_raw = torch.Generator(device=DEVICE); g_eval_raw.manual_seed(CRN_REPORT_SEED)
    Eps_opt_w  = torch.randn(Zw.size(0), Zw.size(1), device=DEVICE, generator=g_eval_w)
    Eps_raw    = torch.randn(Z_raw.size(0), Z_raw.size(1), device=DEVICE, generator=g_eval_raw)

    g_meas = torch.Generator(device=DEVICE); g_meas.manual_seed(CRN_MEAS_SEED)
    EPS_CAL_GLOBAL = [torch.randn(Zw_cal.size(0), Zw_cal.size(1), device=DEVICE, generator=g_meas) for _ in range(K_MEAS)]

    results = []
    first = True
    for beta in BETA_LIST:
        target = H_X - beta

        Sigma_PAC = pac_cov_raw(CovRaw, beta)
        I_pac = torch.eye(Sigma_PAC.size(0), device=Sigma_PAC.device, dtype=Sigma_PAC.dtype)
        chol_pac = torch.linalg.cholesky(Sigma_PAC + C_SLACK * I_pac)
        B_pac = Eps_raw @ chol_pac.T

        PAC_logits = Z_raw + B_pac
        pac_acc = accuracy_from_logits(PAC_logits, y)
        pac_mag = float(B_pac.pow(2).sum(1).mean().item())

        Sigma_Eff = pac_eff_closed_form_raw(CovRaw, beta)
        I_Eff = torch.eye(Sigma_Eff.size(0), device=Sigma_Eff.device, dtype=Sigma_Eff.dtype)
        chol_Eff = torch.linalg.cholesky(Sigma_Eff + C_SLACK * I_Eff)

        B_Eff = Eps_raw @ chol_Eff.T
        Eff_logits = Z_raw + B_Eff
        Eff_acc = accuracy_from_logits(Eff_logits, y)
        Eff_mag = float(B_Eff.pow(2).sum(1).mean().item())

        epochs = ALT_EPOCHS if first else FEW_EPOCHS
        NG = run_opt_faithful(Zw_tr, Zw_cal, Zraw_tr, y_tr, target, Lz, posterior, epochs=epochs)
        first = False

        alpha_final = calibrate_opt_scale(meter, Zw_cal, NG, target, eps_list=EPS_CAL_GLOBAL)
        NG_eval = ScaledNG(NG, alpha_final)
        Hc_ach = compute_Hc_eval(Zw_cal, NG_eval, meter, EPS_CAL_GLOBAL).item()

        Bw_eval,_,_ = NG_eval.forward_with_eps(Zw, Eps_opt_w)
        Braw_eval = (Lz @ Bw_eval.T).T
        OPT_logits = Z_raw + Braw_eval
        opt_acc = accuracy_from_logits(OPT_logits, y)
        opt_mag = float(Braw_eval.pow(2).sum(1).mean().item())

        Sw = estimate_opt_cov_w(NG_eval, Zw, nsamples=4096, seed=CRN_DIAG_SEED)
        erank, evals_desc = effective_rank(Sw)
        top3_mass = float((evals_desc[:3].sum()/evals_desc.sum()).item())

        g_diag = torch.Generator(device=DEVICE); g_diag.manual_seed(CRN_DIAG_SEED)
        idx = torch.randint(0, Zw.size(0), (4096,), generator=g_diag, device=DEVICE)
        eps = torch.randn(4096, Zw.size(1), device=DEVICE, generator=g_diag)
        Bw_s,_,_ = NG_eval.forward_with_eps(Zw[idx], eps)
        Br_s = (Lz @ Bw_s.T).T
        mp = margin_power(Br_s, Z_raw[idx], y[idx])
        raw_energy = float(Br_s.pow(2).sum(1).mean().item())
        mp_frac = mp / max(raw_energy, 1e-12)

        if PRETTY_TABLE:
            _print_table_row(beta, target, Hc_ach, pac_mag, pac_acc, Eff_mag, Eff_acc, opt_mag, opt_acc)
        else:
            print(f"[{tag}] β={beta:4.2f} | target={target:7.4f} | Hc={Hc_ach:7.4f} || "
                  f"PAC {pac_mag:7.1f}/{pac_acc:.4f} | Eff {Eff_mag:7.1f}/{Eff_acc:.4f} || "
                  f"OPT {opt_mag:7.1f}/{opt_acc:.4f} | erank={erank:.2f} | top3={top3_mass:.3f} | mp%={mp_frac:.4f}")

        results.append({
            "tag": tag, "beta": beta, "target_Hc": target, "Hc": Hc_ach,
            "PAC_raw": pac_mag, "PAC_acc": pac_acc,
            "Eff_raw": Eff_mag, "Eff_acc": Eff_acc,
            "OPT_raw": opt_mag, "OPT_acc": opt_acc,
            "OPT_erank": erank, "OPT_top3mass": top3_mass, "OPT_margin_frac": mp_frac
        })
    return results

def main():
    os.makedirs("./data", exist_ok=True)

    model = load_or_train_cnn()
    Z_raw, y = get_test_logits(model)
    base_acc = accuracy_from_logits(Z_raw, y)
    print(f"Clean acc: {base_acc:.4f}")
    res1 = sweep_one_curve(Z_raw, y, tag="baseline", rotate_Q=None, permute_labels=False, independent_meter=False)
    res2 = sweep_one_curve(Z_raw, y, tag="indep_meter", rotate_Q=None, permute_labels=False, independent_meter=True)
    Q = random_orthogonal(d=Z_raw.size(1), seed=ROTATION_SEED)
    res3 = sweep_one_curve(Z_raw, y, tag="rot_Q", rotate_Q=Q, permute_labels=False, independent_meter=False)
    res4 = sweep_one_curve(Z_raw, y, tag="perm_labels", rotate_Q=None, permute_labels=True, independent_meter=False)
    fieldnames = ["tag","beta","target_Hc","Hc",
                  "PAC_raw","PAC_acc","Eff_raw","Eff_acc","OPT_raw","OPT_acc",
                  "OPT_erank","OPT_top3mass","OPT_margin_frac"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in (res1+res2+res3+res4):
            w.writerow(row)
    print(f"[DONE] Wrote {OUT_CSV} with all curves.")

if __name__ == "__main__":
    main()