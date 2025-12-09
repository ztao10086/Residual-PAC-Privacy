import os, math
import torch, torch.nn as nn, torch.optim as optim
from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from torch.special import digamma as psi
import numpy as np
from math import log, pi, gamma
from nflows.flows import Flow
from nflows.distributions import StandardNormal
from nflows.transforms import CompositeTransform, MaskedAffineAutoregressiveTransform

# ──────────────────────────────────────────────────
def safe_log(msg):
    print(msg, flush=True)
    with open("startup_check.log","a") as f:
        f.write(msg+"\n")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRETRAINED = "cnn_cifar100_deep.pth"
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
class DeepWideCNN(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.features = nn.Sequential(

            nn.Conv2d(3, 64, 3, padding=1),   nn.BatchNorm2d(64),   nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),  nn.BatchNorm2d(64),   nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),  nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),nn.BatchNorm2d(128),  nn.ReLU(),
            nn.MaxPool2d(2),


            nn.Conv2d(128, 256, 3, padding=1),nn.BatchNorm2d(256),  nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),nn.BatchNorm2d(256),  nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256),     nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))


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

class GlobalWN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.raw = nn.Parameter(0.05*torch.randn(dim, dim))
    def _L(self):
        L = torch.tril(self.raw, -1)
        L = L + torch.diag(torch.nn.functional.softplus(torch.diag(self.raw)) + 1e-6)
        return L
    def forward(self, z):
        eps = torch.randn_like(z)
        return eps @ self._L().T, torch.zeros_like(z), None
    def forward_with_eps(self, z, eps):
        return eps @ self._L().T, torch.zeros_like(z), None

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
    def forward_with_eps(self, z, eps):
        h = self.net(z)
        mu = self.mu_layer(h)
        lv = self.lv_layer(h).clamp(min=-7)
        std = torch.exp(0.5*lv)
        B = mu + std*eps
        return B, mu, lv


# ──────────────────────────────────────────────────
@torch.no_grad()
def compute_Hc_eval(z_w, noise_gen, posterior, eps_list, alpha: float = 1.0):
    vals = []
    for eps in eps_list:
        B,_,_ = noise_gen.forward_with_eps(z_w, eps)
        y = z_w + float(alpha) * B
        vals.append(-posterior.log_prob(z_w, y))
    return torch.stack(vals, 0).mean()

def run_opt(Zw_tr, Zw_cal, target, Lz, eps_cal, epochs=ALT_EPOCHS):
    dim = Zw_tr.size(1)
    NG  = GlobalWN(dim).to(DEVICE)
    posterior = FlowPosterior(dim, dim).to(DEVICE)

    if hasattr(NG, "mu_layer"):
        for p in NG.mu_layer.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            NG.mu_layer.weight.zero_()
            NG.mu_layer.bias.zero_()
    #
    for p in posterior.flow.parameters():
        p.requires_grad_(True)

    opt_post = optim.AdamW(posterior.parameters(), lr=LR, weight_decay=WD)
    opt_noise = optim.AdamW(NG.parameters(),        lr=LR, weight_decay=WD)

    ds = DataLoader(Zw_tr, batch_size=256, shuffle=True)

    for _ in range(WARM_EPOCHS):
        for z in ds:
            z0 = z.to(DEVICE)
            y0 = z0 + 0.01 * torch.randn_like(z0)
            loss = -posterior.log_prob(z0, y0).mean()
            opt_post.zero_grad(); loss.backward(); opt_post.step()

    for _ in range(epochs):
        for _ in range(INNER_EPOCHS):
            for z in ds:
                zc = z.to(DEVICE)
                Hc_vals = []
                for eps in eps_cal[:min(4, len(eps_cal))]:
                    Bc, _, _ = NG.forward_with_eps(zc, eps[:zc.size(0)])
                    yc = zc + Bc
                    Hc_vals.append(-posterior.log_prob(zc, yc).mean())
                loss_pi = torch.stack(Hc_vals).mean()
                opt_post.zero_grad(); loss_pi.backward(); opt_post.step()

        for z in ds:
            zc = z.to(DEVICE)

            Hc_vals = []
            for eps in eps_cal:
                Bc, _, _ = NG.forward_with_eps(zc, eps[:zc.size(0)])
                yc = zc + Bc
                Hc_vals.append(-posterior.log_prob(zc, yc).mean())
            Hc_g = torch.stack(Hc_vals).mean()

            short = (target - Hc_g).clamp_min(0.0)
            aug = 0.5 * RHO * short.pow(2)
            B, _, _ = NG(zc)
            Br = (Lz @ B.T).T
            mag = Br.pow(2).sum(1).mean()

            loss = mag + aug
            opt_noise.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(NG.parameters(), 1.0)
            opt_noise.step()

    return NG, posterior

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

@torch.no_grad()
def calibrate_opt_scale(posterior, Zw_cal, gen, target_Hc, eps_list, tol: float = 1e-3, max_bisect: int = 30) -> float:
    def Hc_of(alpha: float) -> float:
        vals = []
        for eps in eps_list:
            B,_,_ = gen.forward_with_eps(Zw_cal, eps)
            y = Zw_cal + float(alpha) * B
            vals.append(-posterior.log_prob(Zw_cal, y).mean())
        return torch.stack(vals).mean().item()

    lo, hi = 0.0, 1.0
    while Hc_of(hi) < target_Hc and hi < 1e6:
        hi *= 2.0
    for _ in range(max_bisect):
        mid = 0.5*(lo+hi)
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
    ds = CIFAR100("./data",train=False,download=True,transform=tf)
    loader = DataLoader(ds,batch_size=512,shuffle=False)
    cnn = DeepWideCNN(num_classes=100).to(DEVICE)
    cnn.load_state_dict(torch.load(PRETRAINED, map_location=DEVICE))
    cnn.eval()

    Zs,ys=[],[]
    with torch.no_grad():
        for xb,yb in loader:
            Zs.append(cnn(xb.to(DEVICE)).cpu()); ys.append(yb)
    Z_raw=torch.cat(Zs).to(DEVICE)
    y=torch.cat(ys).to(DEVICE)

    clean_acc = (Z_raw.argmax(1)==y).float().mean().item()
    safe_log(f"Clean acc: {clean_acc:.4f}")

    mu = Z_raw.mean(0,keepdim=True)
    Xc = Z_raw - mu
    Cov_raw = (Xc.T @ Xc) / Z_raw.size(0)
    Cov_raw = 0.5*(Cov_raw + Cov_raw.T) + C_SLACK*torch.eye(Xc.size(1),device=DEVICE)

    Lz = torch.linalg.cholesky(Cov_raw)
    Zw = torch.linalg.solve_triangular(Lz, Xc.T, upper=False).T

    H_X = 110
    safe_log(f"H_X = {H_X:.4f} (fixed)")

    N = Zw.size(0)
    Zw_tr,Zw_cal = Zw[:N//2], Zw[N//2:]
    d = Z_raw.size(1)

    print("\nβ | Target_Hc  | Achieved_Hc | PAC_raw | PAC_acc | Eff_raw | Eff_acc || OPT_raw | OPT_acc")
    print("-"*100)

    g_raw = torch.Generator(device=DEVICE); g_raw.manual_seed(1234)
    Eps_fixed = torch.randn(Z_raw.size(0), d, device=DEVICE, generator=g_raw)

    K_MEAS = 64
    g_meas = torch.Generator(device=DEVICE); g_meas.manual_seed(4242)
    EPS_CAL = [torch.randn(Zw_cal.size(0), d, device=DEVICE, generator=g_meas)
               for _ in range(K_MEAS)]

    g_acc = torch.Generator(device=DEVICE); g_acc.manual_seed(777)
    Eps_opt_w = torch.randn(Zw.size(0), d, device=DEVICE, generator=g_acc)

    for beta in BETA_LIST:
        target = H_X - beta

        # ───────────── Auto-PAC ─────────────────────────────────────
        SigB_o = analytic_covariance_raw(Cov_raw, beta)
        eye_o = torch.eye(SigB_o.size(0), device=SigB_o.device, dtype=SigB_o.dtype)
        chol_o = torch.linalg.cholesky(SigB_o + C_SLACK * eye_o)
        B_o = Eps_fixed @ chol_o.T
        acc_o = (Z_raw + B_o).argmax(1).eq(y).float().mean().item()
        mag_o = B_o.pow(2).sum(1).mean().item()

        # ───────────── Efficient-PAC (IMP/Eff) ─────────────────────────────────────
        Sigma_EFF = pac_eff_closed_form_raw(Cov_raw, beta)
        eye_i = torch.eye(Sigma_EFF.size(0), device=Sigma_EFF.device, dtype=Sigma_EFF.dtype)
        chol_i = torch.linalg.cholesky(Sigma_EFF + C_SLACK * eye_i)
        B_i = Eps_fixed @ chol_i.T
        acc_i = (Z_raw + B_i).argmax(1).eq(y).float().mean().item()
        mag_i = B_i.pow(2).sum(1).mean().item()

        with torch.no_grad():
            lam, U = torch.linalg.eigh(Cov_raw)
            lam = lam.clamp_min(C_SLACK)
            e = torch.diagonal(U.T @ Sigma_EFF @ U).clamp_min(C_SLACK)
            lhs_exact = torch.log1p(lam / e).sum().item()
            lhs_bound = (lam / e).sum().item()
            safe_log(f"[PAC-EFF β={beta}] Σlog(1+λ/e)={lhs_exact:.4f} ≤ Σλ/e={lhs_bound:.4f} ≈ 2β={2 * beta:.4f}")

        # ───────────── SR-PAC ─────────────────────────────────────
        NG, post = run_opt(Zw_tr, Zw_cal, target, Lz, EPS_CAL, epochs=ALT_EPOCHS)
        Hc_floor = compute_Hc_eval(Zw_cal, NG, post, EPS_CAL, alpha=0.0).item()
        target_eff = max(target, Hc_floor + 1e-3)
        alpha = calibrate_opt_scale(post, Zw_cal, NG, target_Hc=target_eff, eps_list=EPS_CAL)
        Hc_ach = compute_Hc_eval(Zw_cal, NG, post, EPS_CAL, alpha=alpha).item()
        safe_log(f"Hc_floor={Hc_floor:.4f}  target={target:.4f}  target_eff={target_eff:.4f}  achieved={Hc_ach:.4f}")
        Bw_eval, _, _ = NG.forward_with_eps(Zw, Eps_opt_w)
        noise_raw = (Lz @ (alpha * Bw_eval).T).T
        acc_opt = (Z_raw + noise_raw).argmax(1).eq(y).float().mean().item()
        g_pow = torch.Generator(device=DEVICE); g_pow.manual_seed(2026)
        EPS_REPORT = [torch.randn(Zw.size(0), d, device=DEVICE, generator=g_pow) for _ in range(4)]
        mags = []

        with torch.no_grad():
            for eps in EPS_REPORT:
                Bw, _, _ = NG.forward_with_eps(Zw, eps)
                Br = (Lz @ (alpha * Bw).T).T
                mags.append(Br.pow(2).sum(1).mean())
        mag_opt = float(torch.stack(mags).mean().item())

        print(f"{beta:4.2f} | {target:10.4f} | {Hc_ach:11.4f} | "
              f"{mag_o:7.3f} | {acc_o:7.3f} | {mag_i:7.3f} | {acc_i:7.3f} || "
              f"{mag_opt:7.3f} | {acc_opt:7.3f}")

if __name__=="__main__":
    main()