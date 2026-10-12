"""Statistics, tables and figures for the ratio sweep. Reads results/runs.jsonl, results/model_outputs/
and results/diversity.json; writes results/analysis.json, results/tables.md and figures/*.png.

Performance metric per task ("perf", higher is better):
    mbpp  : pass@1 (%)            gsm8k : accuracy (%)            roc : -NLL of human test stories (nats/token)
A metric shared by all tasks is the per-token NLL of the real reference answers ("nll", lower is better).
"""
import gzip
import json
import re
import warnings
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from scipy.optimize import curve_fit

import common as C

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)
N_BOOT = 4000
FIG = C.ROOT / "figures"
# categorical slots 1-3 of the validated reference palette (pass all-pairs CVD checks); neutral ink for the rest
COL = {"mbpp": "#2a78d6", "gsm8k": "#eb6834", "roc": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#6b6a66", "#e4e3df"
MARK = {"mbpp": "o", "gsm8k": "s", "roc": "^"}
PERF_LABEL = {"mbpp": "pass@1 (%)", "gsm8k": "accuracy (%)", "roc": "−NLL on human stories (nats/token)"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "figure.facecolor": "#fcfcfb",
                     "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 7, "axes.titlesize": 11, "axes.titleweight": "bold"})


# --------------------------------------------------------------------------- loading
def load_runs():
    runs = {}
    for l in open(C.RUNS_FILE):
        r = json.loads(l)
        runs[r["key"]] = r
    return runs


def load_items(key):
    with gzip.open(C.OUT_DIR / (key.replace("|", "_") + ".json.gz"), "rt") as f:
        return json.load(f)


def item_arrays(task, key, metric):
    """(numerator, denominator) per test item such that the metric = sum(num)/sum(den)."""
    d = load_items(key)
    if metric == "nll" or (metric == "perf" and task == "roc"):
        sign = -1.0 if metric == "perf" else 1.0
        return sign * np.array(d["nll_sum"]), np.array(d["n_tok"])
    if task == "mbpp":
        v = np.array([100.0 * (it["status"] == "pass") for it in d["items"]])
    else:
        v = np.array([100.0 * it["correct"] for it in d["items"]])
    return v, np.ones_like(v)


class Grid:
    """Item-level results for one task / condition / metric: arr[seed][ratio] = (num, den)."""

    def __init__(self, runs, task, cond, metric):
        self.task, self.cond, self.metric = task, cond, metric
        self.arr = defaultdict(dict)
        for k, r in runs.items():
            if r["task"] != task:
                continue
            if r["cond"] == cond or (r["cond"] == "synth" and r["ratio"] == 0):   # r=0 is shared by all conds
                self.arr[r["seed"]][float(r["ratio"])] = item_arrays(task, k, metric)
        # use only seeds whose block is complete enough (the sweep may be analysed while still running)
        need = {0.0, 1.0, 3.0, 10.0, 30.0} if cond == "synth" else set()
        n_max = max((len(v) for v in self.arr.values()), default=0)
        for s in list(self.arr):
            if not need <= set(self.arr[s]) or (cond != "synth" and len(self.arr[s]) < n_max):
                del self.arr[s]
        self.seeds = sorted(self.arr)
        self.ratios = sorted(set.intersection(*[set(self.arr[s]) for s in self.seeds])) if self.seeds else []
        self.n_items = len(next(iter(self.arr[self.seeds[0]].values()))[0]) if self.seeds else 0

    def value(self, seed, ratio, idx=None):
        num, den = self.arr[seed][ratio]
        return num.sum() / den.sum() if idx is None else num[idx].sum() / den[idx].sum()

    def seed_table(self):
        return np.array([[self.value(s, r) for r in self.ratios] for s in self.seeds])   # seeds x ratios

    def boot(self, stat, n=N_BOOT):
        """Two-level bootstrap (seeds, then test items; item indices shared across ratios = paired).
        `stat` maps a vector of seed-mean performances (one per ratio) to a scalar or vector."""
        out = []
        for _ in range(n):
            ss = RNG.choice(self.seeds, len(self.seeds))
            idx = RNG.integers(0, self.n_items, self.n_items)
            out.append(stat(np.mean([[self.value(s, r, idx) for r in self.ratios] for s in ss], axis=0)))
        return np.array(out)


def ci(x, lo=2.5, hi=97.5):
    return [float(np.nanpercentile(x, lo)), float(np.nanpercentile(x, hi))]


# --------------------------------------------------------------------------- curve models (gain vs ratio)
MODELS = {
    "linear": (lambda r, a: a * r, [0.1]),
    "log": (lambda r, a: a * np.log1p(r), [1.0]),
    "power": (lambda r, a, b: a * np.power(r, b), [1.0, 0.5]),
    "hyperbolic": (lambda r, a, k: a * r / (r + k), [1.0, 1.0]),
    "exp_sat": (lambda r, a, k: a * (1 - np.exp(-r / k)), [1.0, 1.0]),
}
BOUNDS = {"power": ([-np.inf, 0.01], [np.inf, 3]), "hyperbolic": ([-np.inf, 1e-3], [np.inf, 1e4]),
          "exp_sat": ([-np.inf, 1e-3], [np.inf, 1e4])}


def fit_models(r, g):
    """Least-squares fit of each gain model g(r) (through the origin). Returns per-model params, R2, AICc."""
    res, n = {}, len(r)
    sst = ((g - g.mean()) ** 2).sum()
    for name, (f, p0) in MODELS.items():
        try:
            p0s = [np.sign(g.mean()) * abs(g).max() if i == 0 else p for i, p in enumerate(p0)]
            popt, _ = curve_fit(f, r, g, p0=p0s, bounds=BOUNDS.get(name, (-np.inf, np.inf)), maxfev=20000)
            rss = float(((g - f(r, *popt)) ** 2).sum())
            k = len(popt) + 1   # + noise variance
            res[name] = {"params": [float(x) for x in popt], "r2": 1 - rss / sst,
                         "aicc": n * np.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)}
        except Exception as e:
            res[name] = {"error": str(e)}
    best = min((v["aicc"] for v in res.values() if "aicc" in v))
    for v in res.values():
        if "aicc" in v:
            v["delta_aicc"] = v["aicc"] - best
    return res


# --------------------------------------------------------------------------- per-task analysis
def analyse_task(runs, task, out, md):
    g = Grid(runs, task, "synth", "perf")
    if not g.seeds or len(g.ratios) < 3:
        return None
    R = np.array(g.ratios)
    core = [i for i, r in enumerate(R) if r in C.RATIOS]          # ratios available for every task
    tab = g.seed_table()
    mean, sd = tab.mean(0), tab.std(0, ddof=1) if len(g.seeds) > 1 else np.zeros(len(R))
    bs = g.boot(lambda v: v)                                       # replicates x ratios
    res = {"seeds": g.seeds, "ratios": g.ratios, "mean": mean.tolist(), "sd": sd.tolist(),
           "ci": [ci(bs[:, i]) for i in range(len(R))], "per_seed": tab.tolist()}

    i0, i3, i10 = (int(np.where(R == x)[0][0]) for x in (0, 3, 10))
    imax_core = core[-1]
    gain = mean - mean[i0]
    res["gain"] = gain.tolist()
    res["gain_ci"] = [ci(bs[:, i] - bs[:, i0]) for i in range(len(R))]

    # H1b: share of the maximum benefit reached at 3:1 (max over the observed grid)
    def frac(v, i=i3):
        gg = v - v[i0]
        return gg[i] / gg.max() if gg.max() > 0 else np.nan
    res["B3"] = {"point": float(frac(mean)), "ci": ci(g.boot(frac)), "per_seed": [float(frac(t)) for t in tab],
                 "p_ge_0.8": float(np.nanmean(g.boot(frac) >= 0.8))}
    res["B_curve"] = [float(frac(mean, i)) for i in range(len(R))]
    res["r_peak"] = float(R[int(np.argmax(mean))])
    pk = g.boot(lambda v: R[int(np.argmax(v))])
    res["r_peak_boot"] = {str(k): float(v) / len(pk) for k, v in sorted(Counter(pk.tolist()).items())}
    ok = [r for r, b in zip(R, res["B_curve"]) if r > 0 and b >= 0.95]
    res["r95"] = float(min(ok)) if ok else None

    # H2a: beyond 10:1
    def contrast(a, b):
        d = bs[:, a] - bs[:, b]
        return {"diff": float(mean[a] - mean[b]), "ci": ci(d), "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
                "per_seed": (tab[:, a] - tab[:, b]).tolist()}
    res["contrasts"] = {f"{R[imax_core]:g}_vs_10": contrast(imax_core, i10), "10_vs_3": contrast(i10, i3),
                        "3_vs_0": contrast(i3, i0), f"{R[-1]:g}_vs_10": contrast(len(R) - 1, i10),
                        "peak_vs_0": contrast(int(np.argmax(mean)), i0)}
    hi = [i for i, r in enumerate(R) if r >= 10]
    slope = lambda v: np.polyfit(np.log(R[hi]), v[hi], 1)[0]
    lo = [i for i, r in enumerate(R) if 1 <= r <= 10]
    slope_lo = lambda v: np.polyfit(np.log(R[lo]), v[lo], 1)[0]
    res["slope_ge10_per_ln_r"] = {"point": float(slope(mean)), "ci": ci(g.boot(slope)),
                                  "per_seed": [float(slope(t)) for t in tab]}
    res["slope_1to10_per_ln_r"] = {"point": float(slope_lo(mean)), "ci": ci(g.boot(slope_lo))}

    # H1a: functional form, fitted to seed-level gains over all ratios > 0
    rr = np.tile(R[1:], len(g.seeds))
    gg = (tab[:, 1:] - tab[:, [i0]]).ravel()
    res["fits"] = fit_models(rr, gg)
    res["fits_seed_mean"] = fit_models(R[1:], gain[1:])

    # real-data control at matched size
    gr = Grid(runs, task, "real", "perf")
    if gr.seeds and len(gr.ratios) > 1:
        rt = gr.seed_table()
        res["real"] = {"ratios": gr.ratios, "mean": rt.mean(0).tolist(), "per_seed": rt.tolist(),
                       "sd": (rt.std(0, ddof=1) if len(gr.seeds) > 1 else np.zeros(len(gr.ratios))).tolist()}
        common_s = [x for x in g.seeds if x in gr.seeds]
        vs = {}
        for r in [x for x in gr.ratios if x > 0 and x in g.ratios]:
            d = []
            for _ in range(N_BOOT):
                ss = RNG.choice(common_s, len(common_s))
                idx = RNG.integers(0, g.n_items, g.n_items)
                d.append(np.mean([g.value(x, r, idx) - gr.value(x, r, idx) for x in ss]))
            d = np.array(d)
            vs[f"{r:g}"] = {"diff": float(np.mean([g.value(x, r) - gr.value(x, r) for x in common_s])), "ci": ci(d),
                            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean()))}
        res["synth_minus_real"] = vs
        # real-equivalent size of the synthetic data: how many extra REAL examples buy the same gain?
        rm, rx = rt.mean(0), np.array(gr.ratios)
        eq = {}
        for r, m in zip(R[1:], mean[1:]):
            if m <= rm.min():
                eq[f"{r:g}"] = 0.0
            elif m >= rm.max():
                eq[f"{r:g}"] = f">{rx[int(np.argmax(rm))]:g}"
            else:
                eq[f"{r:g}"] = float(np.expm1(np.interp(m, rm, np.log1p(rx)))) if np.all(np.diff(rm) > 0) else None
        res["real_equivalent_ratio"] = eq
    for cond in ("synth_unfiltered",):
        gu = Grid(runs, task, cond, "perf")
        if gu.seeds and len(gu.ratios) > 1:
            res[cond] = {"ratios": gu.ratios, "mean": gu.seed_table().mean(0).tolist(), "per_seed": gu.seed_table().tolist()}

    # shared metric: NLL of real references
    gn = Grid(runs, task, "synth", "nll")
    nt = gn.seed_table()
    res["nll"] = {"mean": nt.mean(0).tolist(), "per_seed": nt.tolist()}
    R_n = np.array(gn.ratios)
    bn = gn.boot(lambda v: v)
    nm = nt.mean(0)
    res["nll"]["ratios"] = gn.ratios
    res["nll"]["ci"] = [ci(bn[:, i]) for i in range(len(R_n))]
    res["nll"]["best_ratio"] = float(R_n[int(np.argmin(nm))])
    bb = Counter(R_n[np.argmin(bn, axis=1)].tolist())
    res["nll"]["best_ratio_boot"] = {f"{k:g}": v / len(bn) for k, v in sorted(bb.items())}
    ix = lambda x: int(np.where(R_n == x)[0][0])
    def ncontrast(a, b):
        d = bn[:, ix(a)] - bn[:, ix(b)]
        return {"diff": float(nm[ix(a)] - nm[ix(b)]), "ci": ci(d), "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
                "per_seed": (nt[:, ix(a)] - nt[:, ix(b)]).tolist()}
    res["nll"]["contrasts"] = {"3_vs_0": ncontrast(3, 0), "10_vs_3": ncontrast(10, 3), "30_vs_10": ncontrast(30, 10),
                               "30_vs_best": ncontrast(30, res["nll"]["best_ratio"]), "30_vs_0": ncontrast(30, 0)}
    gnr = Grid(runs, task, "real", "nll")
    if gnr.seeds and len(gnr.ratios) > 1:
        res["nll_real"] = {"ratios": gnr.ratios, "mean": gnr.seed_table().mean(0).tolist()}
    zs = runs.get(f"{task}|zeroshot|0|0")
    if zs:
        res["zeroshot"] = {"perf": (-zs["test_nll"] if task == "roc" else 100 * zs["score"]), "nll": zs["test_nll"],
                           "lenient": 100 * zs["score_lenient"] if "score_lenient" in zs else None}
    out[task] = res

    md.append(f"\n### {C.TASK_LABEL[task]} — {PERF_LABEL[task]} ({len(g.seeds)} seeds, {g.n_items} test items)\n")
    md.append("| ratio | n synth | mean | sd (seeds) | 95% CI | gain vs 0 | share of max gain | real control | real NLL |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    rc = dict(zip(res.get("real", {}).get("ratios", []), res.get("real", {}).get("mean", [])))
    for i, r in enumerate(R):
        md.append(f"| {r:g}:1 | {int(r * C.N_REAL)} | {mean[i]:.3f} | {sd[i]:.3f} | [{res['ci'][i][0]:.3f}, {res['ci'][i][1]:.3f}] | "
                  f"{gain[i]:+.3f} | {res['B_curve'][i]:.2f} | {rc[r]:.3f} |".replace("| nan |", "| – |")
                  if r in rc else
                  f"| {r:g}:1 | {int(r * C.N_REAL)} | {mean[i]:.3f} | {sd[i]:.3f} | [{res['ci'][i][0]:.3f}, {res['ci'][i][1]:.3f}] | "
                  f"{gain[i]:+.3f} | {res['B_curve'][i]:.2f} | – |".replace("| nan |", "| – |"))
        md[-1] += f" {res['nll']['mean'][i]:.4f} |"
    if "real" in res:
        extra = [f"{r:g}:1 → {m:.3f}" for r, m in zip(res["real"]["ratios"], res["real"]["mean"]) if r not in R]
        if extra:
            md.append("\nReal control at other sizes: " + "; ".join(extra))
    return res


# --------------------------------------------------------------------------- creative generation metrics
def roc_generation_metrics(runs, out, md):
    keys = [k for k, r in runs.items() if r["task"] == "roc"]
    if not keys:
        return
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    emb = lambda xs: enc.encode(xs, batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    test = C.load_real("roc")["test"]
    human = [e["story"] for e in test[:C.N_GEN_TEST["roc"]]]
    ref = emb([e["story"] for e in test])                       # 1000 human stories define the reference space
    mu = ref.mean(0)
    _, _, vt = np.linalg.svd(ref - mu, full_matrices=False)
    P = vt[:32].T                                               # 32-d PCA: stable covariances with 500 samples
    proj = lambda E: (E - mu) @ P

    def frechet(A, B):
        from scipy.linalg import sqrtm
        m1, m2, c1, c2 = A.mean(0), B.mean(0), np.cov(A.T), np.cov(B.T)
        return float(((m1 - m2) ** 2).sum() + np.trace(c1 + c2 - 2 * sqrtm(c1 @ c2).real))

    def metrics(stories, E):
        toks = [re.findall(r"[a-z']+", s.lower()) for s in stories]
        d = lambda n: len({tuple(t[i:i + n]) for t in toks for i in range(len(t) - n + 1)}) / max(1, sum(max(0, len(t) - n + 1) for t in toks))
        S = E @ E.T
        first = Counter(t[0] if t else "" for t in toks)
        return {"distinct2": d(2), "distinct3": d(3), "mean_words": float(np.mean([len(t) for t in toks])),
                "five_sentences": float(np.mean([len(re.findall(r"[.!?]+(?:\s|$)", s)) == 5 for s in stories])),
                "pairwise_sim": float((S.sum() - len(E)) / (len(E) * (len(E) - 1))),
                "top_first_word_share": first.most_common(1)[0][1] / len(toks),
                "fd_to_human": frechet(proj(E), proj(ref[C.N_GEN_TEST["roc"]:]))}   # vs the 500 held-out human stories

    res = {"human": metrics(human, ref[:C.N_GEN_TEST["roc"]])}
    per = defaultdict(list)
    for k in keys:
        r = runs[k]
        stories = [it["gen"] for it in load_items(k)["items"]]
        per[(r["cond"], float(r["ratio"]))].append(metrics(stories, emb(stories)))
    res["conditions"] = {f"{c}|{r:g}": {m: float(np.mean([x[m] for x in v])) for m in v[0]} | {"n_seeds": len(v)}
                         for (c, r), v in sorted(per.items())}
    res["per_seed"] = {f"{c}|{r:g}": v for (c, r), v in sorted(per.items())}
    out["roc_generation"] = res
    md.append("\n### Creative: properties of 500 sampled student stories (T=1, top-p 0.95), mean over seeds\n")
    cols = ["distinct2", "distinct3", "pairwise_sim", "fd_to_human", "mean_words", "five_sentences"]
    md.append("| condition | " + " | ".join(cols) + " |")
    md.append("|---|" + "---|" * len(cols))
    md.append("| human reference | " + " | ".join(f"{res['human'][c]:.3f}" for c in cols) + " |")
    for k, v in res["conditions"].items():
        md.append(f"| {k.replace('|', ' ')}:1 | " + " | ".join(f"{v[c]:.3f}" for c in cols) + " |")


# --------------------------------------------------------------------------- error analysis
def error_analysis(runs, out, md):
    res = {}
    for task in ("mbpp", "gsm8k"):
        per = defaultdict(Counter)
        for k, r in runs.items():
            if r["task"] != task or r["cond"] not in ("synth", "zeroshot"):
                continue
            for it in load_items(k)["items"]:
                if task == "mbpp":
                    cat = it["status"] if it["status"] in ("pass", "AssertionError", "SyntaxError", "NameError", "TypeError", "timeout") else "other runtime error"
                else:
                    cat = "correct" if it["correct"] else ("no '####' answer line" if "####" not in it["gen"] else "wrong answer")
                per[(r["cond"], float(r["ratio"]))][cat] += 1
        res[task] = {f"{c}|{r:g}": {k: v / sum(cnt.values()) for k, v in cnt.items()} for (c, r), cnt in sorted(per.items())}
        cats = sorted({c for v in res[task].values() for c in v})
        md.append(f"\n### Error breakdown — {C.TASK_LABEL[task]} (share of test items, pooled over seeds)\n")
        md.append("| condition | " + " | ".join(cats) + " |")
        md.append("|---|" + "---|" * len(cats))
        for k, v in res[task].items():
            md.append(f"| {k.replace('|', ' ')}:1 | " + " | ".join(f"{100 * v.get(c, 0):.1f}%" for c in cats) + " |")
    out["errors"] = res


# --------------------------------------------------------------------------- figures
def xt(r):
    """Plot position of a ratio: ln(1+r), so 0:1 is on the axis."""
    return np.log1p(np.asarray(r, dtype=float))


def set_ratio_axis(ax, ratios):
    ax.set_xticks(xt(ratios))
    ax.set_xticklabels([f"{r:g}" for r in ratios])
    ax.set_xlabel("synthetic : real ratio (100 real examples; log(1+r) axis)")


def fig_dose_response(out):
    tasks = [t for t in C.TASKS if t in out]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.3))
    for ax, t in zip(np.atleast_1d(axes), tasks):
        r = out[t]
        R, m = np.array(r["ratios"]), np.array(r["mean"])
        for row in r["per_seed"]:
            ax.plot(xt(R), row, color=COL[t], lw=0.8, alpha=0.35)
        lo, hi = np.array(r["ci"]).T
        ax.fill_between(xt(R), lo, hi, color=COL[t], alpha=0.15, lw=0)
        ax.plot(xt(R), m, color=COL[t], marker=MARK[t], markeredgecolor="#fcfcfb", markeredgewidth=1.5,
                label="100 real + synthetic (mean, 95% CI, seeds thin)")
        if "real" in r:
            ax.plot(xt(r["real"]["ratios"]), r["real"]["mean"], color=INK, ls="--", marker="D", ms=5,
                    markeredgecolor="#fcfcfb", label="all-real control, same size")
        if "synth_unfiltered" in r:
            ax.plot(xt(r["synth_unfiltered"]["ratios"][1:]), r["synth_unfiltered"]["mean"][1:], color=MUTED, ls=":",
                    marker="x", label="synthetic, no execution filter")
        zs = r.get("zeroshot")
        if zs and t != "roc":   # (for stories the zero-shot NLL is far off-scale: see the table)
            z = zs["lenient"] if t == "gsm8k" else zs["perf"]
            ax.axhline(z, color=MUTED, lw=1, ls=(0, (1, 2)))
            ax.annotate("zero-shot student" + (" (lenient parsing)" if t == "gsm8k" else ""), (xt(R.max()), z),
                        xytext=(-4, -11 if t == "gsm8k" else 4), textcoords="offset points", ha="right", fontsize=8, color=MUTED)
        set_ratio_axis(ax, R)
        ax.set_ylabel(PERF_LABEL[t])
        ax.set_title(C.TASK_LABEL[t])
        ax.legend(fontsize=8, loc="best")
    fig.suptitle("Held-out performance on real data vs. synthetic:real ratio", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "dose_response.png", dpi=160)
    plt.close(fig)


def fig_normalised(out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for t in C.TASKS:
        if t not in out:
            continue
        r = out[t]
        R = np.array(r["ratios"])
        axes[0].plot(xt(R), r["B_curve"], color=COL[t], marker=MARK[t], markeredgecolor="#fcfcfb", label=C.TASK_LABEL[t])
        nll = np.array(r["nll"]["mean"])
        axes[1].plot(xt(R), nll - nll[0], color=COL[t], marker=MARK[t], markeredgecolor="#fcfcfb", label=C.TASK_LABEL[t])
    axes[0].axhline(0.8, color=MUTED, lw=0.8, ls=":")
    axes[0].axvline(xt(3), color=MUTED, lw=0.8, ls=":")
    axes[0].axhline(0, color=MUTED, lw=0.8)
    axes[0].annotate("hypothesis: ≥0.8 at 3:1", (xt(3), 0.8), xytext=(8, 5), textcoords="offset points", fontsize=8, color=MUTED)
    axes[0].set_ylabel("share of maximum observed gain over real-only")
    axes[0].set_title("Normalised benefit by task (task metric)")
    axes[1].axhline(0, color=MUTED, lw=0.8)
    axes[1].set_ylabel("Δ NLL of real test references vs. real-only (nats/token)")
    axes[1].set_title("Shared metric: fit to real held-out data (lower = better)")
    allr = sorted({x for t in out if t in C.TASKS for x in out[t]["ratios"]})
    for ax in axes:
        set_ratio_axis(ax, allr)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "normalised_benefit.png", dpi=160)
    plt.close(fig)


def fig_fits(out):
    tasks = [t for t in C.TASKS if t in out]
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.2 * len(tasks), 4.2))
    style = {"log": ("-", INK), "hyperbolic": ("--", MUTED), "power": (":", MUTED), "linear": ("-.", "#b5b4ae")}
    for ax, t in zip(np.atleast_1d(axes), tasks):
        r = out[t]
        R = np.array(r["ratios"])
        tab = np.array(r["per_seed"])
        for row in tab:
            ax.scatter(xt(R[1:]), row[1:] - row[0], color=COL[t], s=22, alpha=0.6, marker=MARK[t], linewidths=0)
        xs = np.linspace(0, R.max(), 400)
        for name, (ls, c) in style.items():
            f = r["fits"].get(name, {})
            if "params" in f:
                ax.plot(xt(xs), MODELS[name][0](xs, *f["params"]), ls=ls, color=c, lw=1.5,
                        label=f"{name} (ΔAICc {f['delta_aicc']:.1f}, R² {f['r2']:.2f})")
        ax.axhline(0, color=MUTED, lw=0.8)
        set_ratio_axis(ax, R)
        ax.set_ylabel("gain over real-only, " + PERF_LABEL[t])
        ax.set_title(C.TASK_LABEL[t])
        ax.legend(fontsize=8)
    fig.suptitle("Which curve describes the gain? (points = seeds)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "curve_fits.png", dpi=160)
    plt.close(fig)


def fig_diversity():
    p = C.ROOT / "results" / "diversity.json"
    if not p.exists():
        return
    st = json.load(open(p))["stats"]
    panels = [("vendi", "effective number of distinct examples (Vendi score)"),
              ("near_dup_rate", "share with a near-duplicate earlier in the set (cos > 0.9)"),
              ("test_coverage", "mean similarity of real test items to nearest training item")]
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for ax, (m, lab) in zip(axes, panels):
        for t in C.TASKS:
            for src, ls in (("synth", "-"), ("real", "--")):
                rows = [st[k] for k in st if k.startswith(f"{t}|{src}|")]
                if not rows:
                    continue
                ns = sorted({int(n) for row in rows for n in row}, key=int)
                ns = [n for n in ns if all(str(n) in row for row in rows)]
                ys = [np.mean([row[str(n)][m] for row in rows]) for n in ns]
                ax.plot(ns, ys, ls=ls, color=COL[t], marker=MARK[t], markeredgecolor="#fcfcfb",
                        label=f"{C.TASK_LABEL[t]} — {'synthetic' if src == 'synth' else 'real'}")
        ax.set_xscale("log")
        if m == "vendi":
            ax.set_yscale("log")
            ax.set_yticks([15, 20, 30, 40, 60, 90])
            ax.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
            ax.minorticks_off()
        ax.set_xlabel("number of examples in the set")
        ax.set_ylabel(lab)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, ncol=6, fontsize=8.5, loc="lower center", handlelength=3.2, title="solid = synthetic pool, dashed = real data")
    axes[0].set_title("Synthetic sets hold fewer distinct examples")
    axes[1].set_title("Near-duplicates accumulate")
    axes[2].set_title("Coverage of the real test distribution")
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    fig.savefig(FIG / "pool_diversity.png", dpi=160)
    plt.close(fig)


def fig_creative(out):
    if "roc_generation" not in out:
        return
    g = out["roc_generation"]
    panels = [("distinct2", "distinct-2 of 500 sampled stories"), ("pairwise_sim", "mean pairwise embedding similarity"),
              ("fd_to_human", "Fréchet distance to human stories")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (m, lab) in zip(axes, panels):
        for cond, ls, c, mk, name in (("synth", "-", COL["roc"], "^", "100 real + synthetic"), ("real", "--", INK, "D", "all-real control")):
            pts = sorted((float(k.split("|")[1]), v[m]) for k, v in g["conditions"].items() if k.startswith(cond + "|") or k == "synth|0")
            if pts:
                ax.plot(xt([p[0] for p in pts]), [p[1] for p in pts], ls=ls, color=c, marker=mk, markeredgecolor="#fcfcfb", label=name)
        if m != "fd_to_human":
            ax.axhline(g["human"][m], color=MUTED, lw=1, ls=":")
            ax.annotate("human stories", (0, g["human"][m]), xytext=(4, 4), textcoords="offset points", fontsize=8, color=MUTED)
        set_ratio_axis(ax, sorted({float(k.split("|")[1]) for k in g["conditions"] if k.startswith("synth|")}))
        ax.set_ylabel(lab)
        ax.legend(fontsize=8)
    axes[0].set_title("Lexical diversity of student stories")
    axes[1].set_title("Stories become more alike")
    axes[2].set_title("Distance from the human distribution")
    fig.tight_layout()
    fig.savefig(FIG / "creative_generation.png", dpi=160)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    FIG.mkdir(exist_ok=True)
    runs = load_runs()
    out, md = {"n_runs": len(runs)}, ["# Result tables (generated by src/analyze.py)"]
    for t in C.TASKS:
        try:
            analyse_task(runs, t, out, md)
        except Exception as e:
            import traceback; traceback.print_exc()
            print("task analysis failed:", t, e)
    md.append("\n## Curve fits to seed-level gains (ΔAICc: 0 = best; R²)\n")
    md.append("| task | " + " | ".join(MODELS) + " |")
    md.append("|---|" + "---|" * len(MODELS))
    for t in C.TASKS:
        if t in out:
            md.append(f"| {C.TASK_LABEL[t]} | " + " | ".join(
                f"{out[t]['fits'][m]['delta_aicc']:.1f} ({out[t]['fits'][m]['r2']:.2f})" if "aicc" in out[t]["fits"][m] else "n/a"
                for m in MODELS) + " |")
    md.append("\n## Hypothesis statistics\n")
    md.append("| task | share of max gain at 3:1 [95% CI] | peak ratio | r95 | perf(30)−perf(10) [95% CI], p | slope per ln r, r≥10 [95% CI] | slope per ln r, 1≤r≤10 |")
    md.append("|---|---|---|---|---|---|---|")
    for t in C.TASKS:
        if t in out:
            r = out[t]
            c = r["contrasts"].get("30_vs_10") or list(r["contrasts"].values())[0]
            b3 = (f"{r['B3']['point']:.2f} [{r['B3']['ci'][0]:.2f}, {r['B3']['ci'][1]:.2f}]" if np.isfinite(r['B3']['point'])
                  else "undefined (no ratio beats real-only)")
            md.append(f"| {C.TASK_LABEL[t]} | {b3} | {r['r_peak']:g}:1 | "
                      f"{r['r95']} | {c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}], p={c['p_two_sided']:.3f} | "
                      f"{r['slope_ge10_per_ln_r']['point']:+.3f} [{r['slope_ge10_per_ln_r']['ci'][0]:+.3f}, {r['slope_ge10_per_ln_r']['ci'][1]:+.3f}] | "
                      f"{r['slope_1to10_per_ln_r']['point']:+.3f} [{r['slope_1to10_per_ln_r']['ci'][0]:+.3f}, {r['slope_1to10_per_ln_r']['ci'][1]:+.3f}] |")
    def holm(ps):
        order = np.argsort(ps)
        adj, run = np.empty(len(ps)), 0.0
        for rank, i in enumerate(order):
            run = max(run, (len(ps) - rank) * ps[i])
            adj[i] = min(1.0, run)
        return adj
    ts = [t for t in C.TASKS if t in out]
    for name in ("30_vs_10", "10_vs_3", "3_vs_0"):
        ps = [out[t]["contrasts"][name]["p_two_sided"] for t in ts if name in out[t]["contrasts"]]
        if len(ps) == len(ts):
            for t, a in zip(ts, holm(np.array(ps))):
                out[t]["contrasts"][name]["p_holm"] = float(a)
    md.append("\n## Planned contrasts on the task metric (paired bootstrap over seeds and items; Holm-adjusted across tasks)\n")
    md.append("| task | 3:1 − 0 | 10:1 − 3:1 | 30:1 − 10:1 |")
    md.append("|---|---|---|---|")
    for t in ts:
        cells = []
        for name in ("3_vs_0", "10_vs_3", "30_vs_10"):
            c = out[t]["contrasts"].get(name)
            cells.append(f"{c['diff']:+.3f} [{c['ci'][0]:+.3f}, {c['ci'][1]:+.3f}], p_holm={c.get('p_holm', float('nan')):.3f}" if c else "–")
        md.append(f"| {C.TASK_LABEL[t]} | " + " | ".join(cells) + " |")
    md.append("\n## Shared metric: NLL of real held-out references (nats/token, lower = better); differences with 95% CI\n")
    md.append("| task | best ratio (bootstrap share) | 3:1 − 0 | 10:1 − 3:1 | 30:1 − 10:1 | 30:1 − best | 30:1 − 0 |")
    md.append("|---|---|---|---|---|---|---|")
    for name in ("3_vs_0", "10_vs_3", "30_vs_10", "30_vs_0"):
        for t, a in zip(ts, holm(np.array([out[t]["nll"]["contrasts"][name]["p_two_sided"] for t in ts]))):
            out[t]["nll"]["contrasts"][name]["p_holm"] = float(a)
    for t in ts:
        n = out[t]["nll"]
        fmt = lambda c: f"{c['diff']:+.4f} [{c['ci'][0]:+.4f}, {c['ci'][1]:+.4f}]" + (f", p_holm={c['p_holm']:.3f}" if "p_holm" in c else "")
        top = max(n["best_ratio_boot"].items(), key=lambda kv: kv[1])
        md.append(f"| {C.TASK_LABEL[t]} | {n['best_ratio']:g}:1 ({top[0]}:1 in {100 * top[1]:.0f}% of resamples) | " + " | ".join(
            fmt(n["contrasts"][k]) for k in ("3_vs_0", "10_vs_3", "30_vs_10", "30_vs_best", "30_vs_0")) + " |")
    md.append("\n## Synthetic mix minus all-real control at the same training-set size (task metric)\n")
    md.append("| task | ratio | difference [95% CI] | p |")
    md.append("|---|---|---|---|")
    for t in ts:
        for r, v in out[t].get("synth_minus_real", {}).items():
            md.append(f"| {C.TASK_LABEL[t]} | {r}:1 | {v['diff']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}] | {v['p_two_sided']:.3f} |")
    md.append("\n## Per-seed task metric (rows = seeds, columns = ratios)\n")
    for t in ts:
        md.append(f"- {C.TASK_LABEL[t]} ratios {out[t]['ratios']}: " + "; ".join("[" + ", ".join(f"{x:.2f}" for x in row) + "]" for row in out[t]["per_seed"]))
    for fn in (roc_generation_metrics, error_analysis):
        try:
            fn(runs, out, md)
        except Exception as e:
            import traceback; traceback.print_exc()
    for fn in (fig_dose_response, fig_normalised, fig_fits, fig_creative):
        try:
            fn(out)
        except Exception as e:
            import traceback; traceback.print_exc()
    fig_diversity()
    json.dump(out, open(C.ROOT / "results" / "analysis.json", "w"), indent=1, default=float)
    open(C.ROOT / "results" / "tables.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
