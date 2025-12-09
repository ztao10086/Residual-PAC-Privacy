import numpy as np

# ──────────────────────────────────────────────────
def _fit_gaussian(xs, eps=1e-6):
    xs = np.asarray(xs, float)
    mu = float(xs.mean()) if xs.size else 0.0
    sd = float(xs.std(ddof=1)) if xs.size > 1 else 1.0
    return mu, max(sd, eps)

def _logpdf_normal(x, mu, sd):
    var = sd * sd
    return -0.5 * (np.log(2.0 * np.pi * var) + ((x - mu) ** 2) / var)

def _tpr_at_fpr(pos_scores, neg_scores, fpr_target=1e-3):
    pos_scores = np.asarray(pos_scores, float)
    neg_scores = np.asarray(neg_scores, float)
    if neg_scores.size == 0 or pos_scores.size == 0:
        return np.nan, np.nan, np.nan
    k = int(np.ceil((1.0 - float(fpr_target)) * neg_scores.size)) - 1
    k = np.clip(k, 0, neg_scores.size - 1)
    tau = np.sort(neg_scores)[k]
    tpr = np.mean(pos_scores > tau)
    fpr = np.mean(neg_scores > tau)
    return tpr, fpr, tau

# ──────────────────────────────────────────────────
def lira_posteriors_and_success(scores_in, scores_out, victim_scores, labels, prior=0.5):
    scores_in, scores_out = list(scores_in), list(scores_out)
    victim_scores = np.asarray(victim_scores, float)
    labels = np.asarray(labels, int)
    N = len(victim_scores)
    assert len(scores_in) == len(scores_out) == N
    llr = np.zeros(N, dtype=float)
    for i in range(N):
        mu_in, sd_in = _fit_gaussian(scores_in[i])
        mu_out, sd_out = _fit_gaussian(scores_out[i])
        s = victim_scores[i]
        ll_in = _logpdf_normal(s, mu_in,  sd_in)
        ll_out = _logpdf_normal(s, mu_out, sd_out)
        llr[i] = ll_in - ll_out
    log_prior_odds = np.log(prior) - np.log(1.0 - prior)
    post = 1.0 / (1.0 + np.exp(-(llr + log_prior_odds)))
    psr_emp = np.where(labels == 1, post, 1.0 - post).mean()
    pos_llr = llr[labels == 1]
    neg_llr = llr[labels == 0]
    def tpr_at(fpr_target=1e-3):
        tpr, fpr, tau = _tpr_at_fpr(pos_llr, neg_llr, fpr_target=fpr_target)
        return tpr
    return post, psr_emp, tpr_at

# ──────────────────────────────────────────────────

def theoretical_psr(eps, prior=0.5):
    if np.isinf(eps):
        return 1.0
    logit_prior = np.log(prior) - np.log(1.0 - prior)
    x = float(eps) + logit_prior
    return 1.0 / (1.0 + np.exp(-x))

def psr_comparison_table(empirical_psr_by_method_beta, eps_by_beta, prior=0.5):
    rows = []
    for (method, beta), psr_emp in empirical_psr_by_method_beta.items():
        eps = eps_by_beta[beta]
        psr_th = theoretical_psr(eps, prior=prior)
        rows.append((beta, method, psr_emp, psr_th, abs(psr_emp - psr_th)))
    rows.sort(key=lambda r: (r[0], r[1]))

    beta_col = max(6, max(len(f"{b:g}") for b, *_ in rows))
    meth_col = max(6, max(len(m) for _, m, *_ in rows))
    print(f"{'beta'.ljust(beta_col)}  {'method'.ljust(meth_col)}  PSR_emp   PSR_theory   |diff|")
    print("-"*(beta_col + meth_col + 33))
    for b, m, pe, pt, d in rows:
        print(f"{str(b).ljust(beta_col)}  {m.ljust(meth_col)}  {pe:7.4f}   {pt:10.4f}   {d:7.4f}")