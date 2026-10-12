"""EnzymeAdvisor engine.

Ranks single-point mutations of a protein for a small experimental budget.
 - fewer than MIN_MEASURED measured variants: ESM-2 zero-shot ranking
 - MIN_MEASURED or more: Gaussian-process model + uncertainty bonus (beta) + near-duplicate rule
Measured values must follow "higher = better" (enter negative values if your assay is lower = better).
Every number in an explanation comes from the same quantities that were used to choose the panel.
"""
import re
import numpy as np
import torch
import gpytorch
import esm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

AAS = 'ACDEFGHIKLMNPQRSTVWY'
MIN_MEASURED = 20      # the method was only tested with at least 20 measured variants
BETA = 2.0             # exploration weight (tuned on 10 datasets, tested on 12 others)
MAX_LEN = 1000

_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_esm = {}


def _load_esm():
    if not _esm:
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        _esm['model'] = model.eval().to(_device)
        _esm['alphabet'] = alphabet
    return _esm['model'], _esm['alphabet']


def clean_sequence(text):
    lines = [l for l in str(text).splitlines() if not l.strip().startswith('>')]
    seq = re.sub(r'[\s\d]+', '', ''.join(lines)).upper()
    bad = sorted(set(seq) - set(AAS))
    if bad:
        raise ValueError(f'Unsupported characters in the sequence: {bad}. Use the 20 standard amino acids.')
    if not 20 <= len(seq) <= MAX_LEN:
        raise ValueError(f'Sequence length {len(seq)} is outside the supported range 20-{MAX_LEN}.')
    return seq


def mutation_table(seq):
    """All single mutations with their features: ESM-2 embedding at the position + ESM-2 zero-shot score."""
    model, alphabet = _load_esm()
    converter = alphabet.get_batch_converter()
    _, _, tokens = converter([('wt', seq)])
    with torch.no_grad():
        out = model(tokens.to(_device), repr_layers=[12], return_contacts=False)
    reps = out['representations'][12][0, 1:len(seq) + 1].cpu().numpy()
    logp = out['logits'][0, 1:len(seq) + 1].log_softmax(dim=-1).cpu().numpy()
    idx = {a: i for i, a in enumerate(alphabet.all_toks)}
    names, rows = [], []
    for pos, wt in enumerate(seq, 1):
        for mt in AAS:
            if mt == wt:
                continue
            score = logp[pos - 1, idx[mt]] - logp[pos - 1, idx[wt]]
            names.append(f'{wt}{pos}{mt}')
            rows.append(np.concatenate([reps[pos - 1], [score]]))
    return names, np.array(rows, dtype=np.float64)


def parse_measured(seq, measured):
    """measured: list of (mutation, value) pairs. Returns (valid dict, list of problems)."""
    ok, problems = {}, []
    for m, v in measured:
        m = str(m).strip().upper()
        mm = re.match(r'^([A-Z])(\d+)([A-Z])$', m)
        if not mm:
            problems.append(f'{m}: not a single mutation like A148D')
            continue
        wt, pos, mt = mm.group(1), int(mm.group(2)), mm.group(3)
        if not (1 <= pos <= len(seq)) or seq[pos - 1] != wt:
            problems.append(f'{m}: position or original residue does not match the sequence')
            continue
        if mt not in AAS or mt == wt:
            problems.append(f'{m}: invalid substitution')
            continue
        try:
            ok[m] = float(v)
        except (TypeError, ValueError):
            problems.append(f'{m}: the measured value "{v}" is not a number')
    return ok, problems


class _GP(gpytorch.models.ExactGP):
    def __init__(self, x, y, lik):
        super().__init__(x, y, lik)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


def _gp_predict(X, known_idx, y_known, pool_idx, n_iter=50, n_components=10):
    scaler = StandardScaler().fit(X[known_idx])
    Xs = scaler.transform(X[known_idx])
    pca = PCA(n_components=min(n_components, len(known_idx) - 1, Xs.shape[1])).fit(Xs)
    xtr = torch.tensor(pca.transform(Xs), dtype=torch.float32).to(_device)
    y_mean, y_std = y_known.mean(), y_known.std() + 1e-8
    ytr = torch.tensor((y_known - y_mean) / y_std, dtype=torch.float32).to(_device)
    lik = gpytorch.likelihoods.GaussianLikelihood().to(_device)
    gp = _GP(xtr, ytr, lik).to(_device)
    gp.train(); lik.train()
    opt = torch.optim.Adam(gp.parameters(), lr=0.05)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, gp)
    for _ in range(n_iter):
        opt.zero_grad()
        loss = -mll(gp(xtr), ytr)
        loss.backward()
        opt.step()
    gp.eval(); lik.eval()
    xp = pca.transform(scaler.transform(X[pool_idx]))
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred = lik(gp(torch.tensor(xp, dtype=torch.float32).to(_device)))
        mean = pred.mean.cpu().numpy() * y_std + y_mean
        std = pred.stddev.cpu().numpy() * y_std
    return mean.astype(np.float64), std.astype(np.float64), xp, gp.covar_module.base_kernel.lengthscale.item()


def _greedy_select(mean, std, Xp, ls, beta, budget):
    base = mean + beta * std
    utility = base.copy()
    remaining = list(range(len(mean)))
    chosen, disc_at_pick, util_at_pick = [], [], []
    discount = np.zeros(len(mean))
    top_cut = np.zeros(len(mean))
    top_who = np.full(len(mean), -1)
    for _ in range(budget):
        b = max(remaining, key=lambda i: utility[i])
        chosen.append(b)
        remaining.remove(b)
        disc_at_pick.append(discount[b])
        util_at_pick.append(utility[b])
        d = np.linalg.norm(Xp[remaining] - Xp[b], axis=1)
        sim = np.exp(-(d ** 2) / (2 * ls ** 2))
        for j, r in enumerate(remaining):
            cut = beta * std[r] * sim[j]
            utility[r] -= cut
            discount[r] += cut
            if cut > top_cut[r]:
                top_cut[r] = cut
                top_who[r] = b
    return dict(chosen=chosen, base=base, discount=discount, top_who=top_who,
                disc_at_pick=disc_at_pick, util_at_pick=util_at_pick)


def recommend(sequence, budget=5, measured=None, beta=BETA):
    seq = clean_sequence(sequence)
    budget = int(budget)
    if not 1 <= budget <= 50:
        raise ValueError('The budget must be between 1 and 50.')
    if isinstance(measured, dict):
        measured = list(measured.items())
    names, X = mutation_table(seq)
    index = {n: i for i, n in enumerate(names)}
    ok, problems = parse_measured(seq, measured or [])
    res = dict(sequence=seq, budget=budget, beta=beta, names=names, X=X, problems=problems,
               n_measured=len(ok), measured=ok)
    pool = np.array([i for i, n in enumerate(names) if n not in ok])
    if len(ok) < MIN_MEASURED:
        order = pool[np.argsort(-X[pool, -1])]
        res.update(mode='esm_only', _pool=pool, _order=order,
                   panel=[dict(mutation=names[i], esm_score=float(X[i, -1]), rank=r + 1)
                          for r, i in enumerate(order[:budget])])
        return res
    known_idx = np.array([index[n] for n in ok])
    y_known = np.array([ok[n] for n in ok], dtype=np.float32)
    if np.ptp(y_known) == 0:
        raise ValueError('All measured values are identical, so the model has nothing to learn from.')
    mean, std, Xp, ls = _gp_predict(X, known_idx, y_known, pool)
    G = _greedy_select(mean, std, Xp, ls, beta, budget)
    pool_names = [names[i] for i in pool]
    esm_pool = X[pool, -1]
    panel = [dict(mutation=pool_names[i], predicted=float(mean[i]), std=float(std[i]),
                  esm_score=float(esm_pool[i]), score=float(G['base'][i]),
                  discount_at_pick=float(G['disc_at_pick'][k]), adjusted=float(G['util_at_pick'][k]))
             for k, i in enumerate(G['chosen'])]
    res.update(mode='gp_diverse', pool=pool, pool_names=pool_names, mean=mean, std=std,
               Xp=Xp, ls=ls, G=G, panel=panel)
    return res


def _pos(m):
    return int(re.findall(r'\d+', m)[0])


def format_recommendation(res):
    L, b = len(res['sequence']), res['budget']
    lines = [f"Protein of {L} residues, {len(res['names'])} possible single mutations, budget {b}, "
             f"{res['n_measured']} measured variants entered."]
    if res['problems']:
        lines.append(f"Ignored {len(res['problems'])} entries: " + '; '.join(res['problems'][:5])
                     + ('...' if len(res['problems']) > 5 else ''))
    if res['mode'] == 'esm_only':
        lines += ['', f"RECOMMENDED PANEL (ESM-2 ranking only; the learning method needs at least {MIN_MEASURED} measured variants)"]
        for p in res['panel']:
            lines.append(f"{p['rank']}. {p['mutation']}: ESM-2 score {p['esm_score']:.2f}")
        npos = len({_pos(p['mutation']) for p in res['panel']})
        lines.append(f"These {len(res['panel'])} picks cover {npos} different position(s).")
        if npos < len(res['panel']):
            lines.append("Note: picks at the same position test nearly the same idea. If your budget is small, "
                         "consider taking only the best variant at each position (this rule has not been benchmarked).")
        lines += ['', 'This panel ranks mutations by ESM-2 alone. Several picks may sit at the same position, and nothing '
                      'has been learned from your lab results yet. After measuring about 20 variants, enter them to switch '
                      'to the learning method.']
        return '\n'.join(lines)
    G, names, mean, std, beta = res['G'], res['pool_names'], res['mean'], res['std'], res['beta']
    base, disc, who, chosen = G['base'], G['discount'], G['top_who'], G['chosen']
    thr = min(G['util_at_pick'])
    esm = res['X'][:, -1][res['pool']]
    lines += ['', f'RECOMMENDED PANEL (learning method, exploration weight beta = {beta:g})']
    for k, i in enumerate(chosen, 1):
        d = G['disc_at_pick'][k - 1]
        s = (f"{k}. {names[i]}: predicted {mean[i]:.2f} (+/- {std[i]:.2f}); ESM-2 score {esm[i]:.2f}; "
             f"score {base[i]:.2f} = {mean[i]:.2f} + {beta:g} x {std[i]:.2f}")
        s += (f"; minus {d:.2f} for similarity to earlier picks = {G['util_at_pick'][k - 1]:.2f}"
              if d >= 0.05 else '; not similar to earlier picks')
        lines.append(s)
    lines.append(f"Lowest adjusted score among the picks: {thr:.2f}. "
                 f"The panel covers {len({_pos(names[i]) for i in chosen})} different positions.")
    chosen_set = set(chosen)
    comp = [r for r in range(len(base)) if r not in chosen_set and disc[r] >= 0.05 and base[r] >= thr]
    lines += ['', 'SKIPPED AS NEAR-DUPLICATES']
    found = False
    for p in chosen:
        mem = sorted((r for r in comp if who[r] == p), key=lambda r: -base[r])
        if mem:
            ex = ', '.join(names[r] for r in mem[:3])
            lines.append(f"- {len(mem)} other variants that scored at least {thr:.2f} on their own (e.g. {ex}) "
                         f"were skipped because they are very similar to {names[p]}")
            found = True
    if not found:
        lines.append('- none: no strong candidate was a near-duplicate of a pick')
    lines += ['', 'These are model estimates, not measurements. In benchmark tests on 22 enzymes, this method found about '
                  'twice as many high-fitness variants as random picking and performed about the same as ESM-2 ranking '
                  'on the best variant found.']
    return '\n'.join(lines)


def why_not(res, mutation):
    m = str(mutation).strip().upper()
    if m not in set(res['names']):
        return f'{m} is not a single mutation of this sequence (check the position and the original residue).'
    if m in res['measured']:
        return f"{m} was already measured ({res['measured'][m]:g}), so it is not a candidate."
    b = res['budget']
    if res['mode'] == 'esm_only':
        order_names = [res['names'][i] for i in res['_order']]
        rank = order_names.index(m) + 1
        if rank <= b:
            return f'{m} is in the panel: rank {rank} by ESM-2 score.'
        return f'{m} ranks {rank} of {len(order_names)} by ESM-2 score; the panel takes the top {b}.'
    G, names = res['G'], res['pool_names']
    j = names.index(m)
    if j in G['chosen']:
        return f"{m} is in the panel (pick {G['chosen'].index(j) + 1})."
    thr = min(G['util_at_pick'])
    raw, d = G['base'][j], G['discount'][j]
    adj = raw - d
    if d >= 0.05:
        return (f"{m}: raw score {raw:.2f}, minus {d:.2f} because it is very similar to {names[G['top_who'][j]]} "
                f"(already picked) = {adj:.2f}, which does not beat the weakest pick ({thr:.2f}).")
    return f"{m}: raw score {raw:.2f}, adjusted {adj:.2f}, which does not beat the weakest pick ({thr:.2f})."


def verify(res):
    """Recompute the explanation numbers independently of the selection loop."""
    if res['mode'] == 'esm_only':
        esm, pool = res['X'][:, -1], res['_pool']
        expect = [res['names'][i] for i in pool[np.argsort(-esm[pool])][:res['budget']]]
        return {'panel_equals_top_esm_scores': expect == [p['mutation'] for p in res['panel']]}
    G = res['G']
    chosen = set(G['chosen'])
    thr = min(G['util_at_pick'])
    adj = G['base'] - G['discount']
    others = np.array([i for i in range(len(adj)) if i not in chosen])
    violations = int((adj[others] > thr + 1e-9).sum())
    P = np.array(G['chosen'])
    D2 = ((res['Xp'][others][:, None, :] - res['Xp'][P][None, :, :]) ** 2).sum(-1)
    terms = res['beta'] * res['std'][others][:, None] * np.exp(-D2 / (2 * res['ls'] ** 2))
    err = float(np.abs(terms.sum(1) - G['discount'][others]).max())
    m = terms.sum(1) > 0.05
    label_ok = float((P[terms.argmax(1)] == G['top_who'][others])[m].mean()) if m.any() else 1.0
    return dict(non_selected_checked=len(others), violations=violations,
                max_discount_error=err, nearest_pick_label_correct=label_ok)
