import os, math
import torch, torch.nn as nn, torch.optim as optim
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.special import digamma as psi
import numpy as np
from math import log, pi, gamma
import numpy as np
import math
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform


# ──────────────────────────────────────────────────
def safe_log(msg):
    print(msg, flush=True)
    with open("startup_check.log","a") as f:
        f.write(msg+"\n")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED = "cnn_cifar10.pth"
C_SLACK = 1e-6
BETA_LIST = []
ALT_EPOCHS = 40
INNER_EPOCHS = 5
N_HC_SAMPLES = 10
RHO = 1e6
WARM_EPOCHS = 5
LR = 1e-4
WD = 1e-5

# ──────────────────────────────────────────────────
class SmallCNN(nn.Module):
    def __init__(self,num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64*8*8,128), nn.ReLU(),
            nn.Linear(128,num_classes)
        )
    def forward(self,x):
        return self.classifier(self.features(x))

class NoiseGenerator(nn.Module):
    def __init__(self,dim,hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hidden), nn.ReLU(),
            nn.Linear(hidden,hidden), nn.ReLU()
        )
        self.mu_layer = nn.Linear(hidden,dim)
        self.lv_layer = nn.Linear(hidden,dim)
    def forward(self,z):
        h = self.net(z)
        mu = self.mu_layer(h)
        lv = self.lv_layer(h).clamp(min=-7)
        std = torch.exp(0.5*lv)
        eps = torch.randn_like(std)
        B = mu + std*eps
        return B, mu, lv

class FlowPosterior(nn.Module):
    def __init__(self,dim,context_dim,depth=3,hidden=128):
        super().__init__()
        base = StandardNormal([dim])
        transforms = []
        for _ in range(depth):
            transforms.append(
                MaskedAffineAutoregressiveTransform(
                    features=dim,
                    hidden_features=hidden,
                    context_features=context_dim
                )
            )
        self.flow = Flow(CompositeTransform(transforms), base)
    def log_prob(self,z,y):
        return self.flow.log_prob(inputs=z, context=y)

# ──────────────────────────────────────────────────
def compute_Hc(z_w, noise_gen, posterior):
    logp, Bs = [], []
    for _ in range(N_HC_SAMPLES):
        B,_,_ = noise_gen(z_w)
        y = z_w + B
        logp.append(posterior.log_prob(z_w, y))
        Bs.append(B)
    return -torch.stack(logp,0).mean(0).mean(), Bs[0]

def analytic_covariance(Zw,beta):
    dim = Zw.size(1)
    Sigma_ = Zw.T@Zw/Zw.size(0) + C_SLACK*torch.eye(dim,device=DEVICE)
    lam,U = torch.linalg.eigh(Sigma_)
    c = 1.0/(math.exp(2*beta/dim)-1.0)
    return (U*(c*lam).unsqueeze(0))@U.T

def run_opt(Zw_tr, target, Lz):
    dim = Zw_tr.size(1)
    noise_gen = NoiseGenerator(dim).to(DEVICE)
    posterior = FlowPosterior(dim, dim).to(DEVICE)
    for p in posterior.flow.parameters():
        p.requires_grad_(True)

    noise_gen.train()
    posterior.train()
    opt_post = optim.AdamW(posterior.parameters(), lr=LR, weight_decay=WD)
    opt_noise = optim.AdamW(noise_gen.parameters(), lr=LR, weight_decay=WD)
    ds = DataLoader(Zw_tr, batch_size=256, shuffle=True)
    lambda_dual = torch.tensor(0.0, device=DEVICE)
    ETA_DUAL = 1e-2

    for _ in range(WARM_EPOCHS):
        for z in ds:
            z0 = z.to(DEVICE)
            y0 = z0 + 0.01 * torch.randn_like(z0)
            loss_warm = -posterior.log_prob(z0, y0).mean()
            opt_post.zero_grad()
            loss_warm.backward()
            torch.nn.utils.clip_grad_norm_(posterior.parameters(), 1.0)
            opt_post.step()

    for _ in range(ALT_EPOCHS):
        for _ in range(INNER_EPOCHS):
            for z in ds:
                zc = z.to(DEVICE)
                Hc, _ = compute_Hc(zc, noise_gen, posterior)
                loss_pi = Hc
                opt_post.zero_grad()
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(posterior.parameters(), 1.0)
                opt_post.step()

        for z in ds:
            zc = z.to(DEVICE)
            Hc, B = compute_Hc(zc, noise_gen, posterior)
            noise_raw = (Lz @ B.T).T
            noise_m = noise_raw.pow(2).sum(1).mean()
            viol = target - Hc
            hinge = torch.relu(viol)
            loss_g = noise_m + lambda_dual * hinge + 0.5 * RHO * hinge.pow(2)
            opt_noise.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(noise_gen.parameters(), 1.0)
            opt_noise.step()

        with torch.no_grad():
            try:
                zc = next(iter(ds)).to(DEVICE)
            except StopIteration:
                zc = Zw_tr[:256].to(DEVICE)
            Hc_tmp, _ = compute_Hc(zc, noise_gen, posterior)
            lambda_dual = torch.clamp(lambda_dual + ETA_DUAL * (target - Hc_tmp), min=0.0)
    return noise_gen, posterior

def analytic_covariance_raw(Cov_raw: torch.Tensor, beta: float) -> torch.Tensor:
    d = Cov_raw.size(0)
    c = 1.0 / (math.exp(2.0 * beta / d) - 1.0)
    return c * Cov_raw

def pac_eff_closed_form_raw(Cov_raw: torch.Tensor, beta: float) -> torch.Tensor:
    lam, U = torch.linalg.eigh(Cov_raw)
    lam = lam.clamp_min(C_SLACK)
    s = torch.sqrt(lam)
    c = s.sum() / (2.0 * beta)
    e = c * s
    return U @ torch.diag(e.to(Cov_raw.dtype)) @ U.T

def pac_imp_covariance_raw_kkt(Cov_raw: torch.Tensor, beta: float, tol: float = 1e-6, max_iter: int = 80) -> torch.Tensor:
    sigma = torch.diag(Cov_raw).clamp_min(C_SLACK).to(dtype=torch.float64)

    def e_of_lambda(lam: float) -> torch.Tensor:
        lam = torch.tensor(lam, dtype=torch.float64, device=sigma.device)
        return 0.5 * (torch.sqrt(sigma*sigma + 4.0*lam*sigma) - sigma)

    def g(lam: float) -> float:
        e = e_of_lambda(lam).clamp_min(C_SLACK)
        return float(torch.log1p(sigma / e).sum().item())

    target = 2.0 * float(beta)
    lo, hi = 1e-12, 1.0
    while g(hi) > target and hi < 1e12:
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5*(lo+hi)
        if g(mid) > target: lo = mid
        else:               hi = mid
        if hi - lo <= tol * max(1.0, hi): break

    e = e_of_lambda(hi).to(dtype=Cov_raw.dtype).clamp_min(C_SLACK)
    return torch.diag(e)

@torch.no_grad()
def calibrate_opt_scale(posterior, Zw, gen, target_Hc,
                        K: int = 3, tol: float = 1e-3, max_bisect: int = 30, seed: int = 777) -> float:
    torch.manual_seed(seed)
    Bs = [gen(Zw)[0] for _ in range(K)]

    def Hc_of(alpha: float) -> float:
        vals = []
        for Bk in Bs:
            y = Zw + alpha * Bk
            vals.append(-posterior.log_prob(Zw, y).mean())
        return float(torch.stack(vals).mean().item())

    lo, hi = 0.0, 1.0
    while Hc_of(hi) < target_Hc and hi < 1e6:
        hi *= 2.0
    for _ in range(max_bisect):
        mid = 0.5 * (lo + hi)
        if Hc_of(mid) >= target_Hc:
            hi = mid
        else:
            lo = mid
        if hi - lo <= tol * max(1.0, hi): break
    return hi


# ──────────────────────────────────────────────────

def main():
    os.makedirs("./data",exist_ok=True)
    tf = Compose([ToTensor(),Normalize((0.5,)*3,(0.5,)*3)])
    ds = CIFAR10("./data",train=False,download=True,transform=tf)
    loader = DataLoader(ds,batch_size=512,shuffle=False)
    cnn = SmallCNN().to(DEVICE)
    cnn.load_state_dict(torch.load(PRETRAINED,map_location=DEVICE))
    cnn.eval()

    Zs,ys=[],[]
    with torch.no_grad():
        for xb,yb in loader:
            Zs.append(cnn(xb.to(DEVICE)).cpu()); ys.append(yb)
    Z_raw=torch.cat(Zs).to(DEVICE)
    y = torch.cat(ys).to(DEVICE)
    safe_log(f"Clean acc: {(Z_raw.argmax(1)==y).float().mean():.4f}")
    mu = Z_raw.mean(0,keepdim=True)
    Xc = Z_raw - mu
    cov = Xc.T@Xc/Z_raw.size(0) + C_SLACK*torch.eye(Xc.size(1),device=DEVICE)
    Lz = torch.linalg.cholesky(cov)
    Zw = (torch.linalg.inv(Lz)@Xc.T).T
    N = Zw.size(0)
    H_X = 13
    safe_log(f"H_X = {H_X:.4f} (fixed)")
    Zw_tr,Zw_val = Zw[:N//2], Zw[N//2:]

    print("\nβ | Target_Hc  | Achieved_Hc | PAC_raw | PAC_acc | Eff_raw | Eff_acc || OPT_raw | OPT_acc")
    print("-"*100)

    g = torch.Generator(device=DEVICE)
    g.manual_seed(1234)
    Eps_fixed = torch.randn(N, Zw.size(1), device=DEVICE, generator=g)  # reused for all β & baselines

    for beta in BETA_LIST:
        target = H_X - beta
        # ───────────── Auto-PAC ─────────────────────────────────────
        Sigma_B_o = analytic_covariance_raw(cov, beta)
        chol_o = torch.linalg.cholesky(Sigma_B_o + C_SLACK * torch.eye(Zw.size(1), device=DEVICE))
        B_o = Eps_fixed @ chol_o.T
        acc_o = (Z_raw + B_o).argmax(1).eq(y).float().mean().item()
        mag_o = B_o.pow(2).sum(1).mean().item()

        # ──────────────── Efficient-PAC (IMP/Eff) ──────────────────────────────────
        Sigma_B_i = pac_imp_covariance_raw_kkt(cov, beta)
        chol_i = torch.linalg.cholesky(Sigma_B_i + C_SLACK * torch.eye(Zw.size(1), device=DEVICE))
        B_i = Eps_fixed @ chol_i.T
        acc_i = (Z_raw + B_i).argmax(1).eq(y).float().mean().item()
        mag_i = B_i.pow(2).sum(1).mean().item()
        lhs = torch.log1p(torch.diag(cov).clamp_min(C_SLACK) /
                          torch.diag(Sigma_B_i).clamp_min(C_SLACK)).sum().item()
        print(f"Eff check: sum log(1+σ/e) = {lhs:.4f} vs 2β = {2 * beta:.4f}")

        # ───────────── SR-PAC ─────────────────────────────────────
        gen, post = run_opt(Zw_tr, target, Lz)
        with torch.no_grad():
            alpha = calibrate_opt_scale(post, Zw.to(DEVICE), gen, target_Hc=target, K=3)
            B_eval = gen(Zw.to(DEVICE))[0]
            B_eval = alpha * B_eval
            noise_e = (Lz @ B_eval.T).T
            Hc_full = -post.log_prob(Zw.to(DEVICE), Zw.to(DEVICE) + alpha * gen(Zw.to(DEVICE))[0]).mean().item()
            mag_opt = noise_e.pow(2).sum(1).mean().item()
            acc_opt = (Z_raw + noise_e).argmax(1).eq(y).float().mean().item()
        print(f"{beta:4.2f} | {target:10.4f} | {Hc_full:11.4f} | "
              f"{mag_o:7.3f} | {acc_o:7.3f} | {mag_i:7.3f} | {acc_i:7.3f} || "
              f"{mag_opt:7.3f} | {acc_opt:7.3f}")

if __name__=="__main__":
    main()