#!/usr/bin/env python3
"""
Compute correlation between human annotator scores and Gemini scores
for the nanobanana model image adaptation evaluation.

Data:
  - Human scores: results/Yousha.json and results/simran.json (100 pairs each)
  - Gemini scores: code/outputs/gemini_nanobanana_app_pairs.jsonl (100 pairs)

Matching key: image_id (human) == id (gemini)

Metrics computed per dimension (A-E + overall_success) and full score vector:
  - Pearson correlation
  - Spearman correlation (ordinal scores)
  - Cohen's kappa (treating scores as categories)
  - MAE (mean absolute error)
  - Inter-annotator agreement (Yousha vs simran)

Full score vector: all 6 dimension scores concatenated across all 100 pairs (600 values),
capturing overall human-gemini alignment across the entire evaluation rubric.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# Paths
# ============================================================================

BASE = Path(__file__).parent
HUMAN_FILES = {
    "Yousha": BASE / "data" / "Yousha.json",
    "simran": BASE / "data" / "simran.json",
}
GEMINI_FILE = BASE / "data" / "gemini_nanobanana_app_pairs.jsonl"
OUTPUT_DIR = BASE / "results" / "nanobanana_human_vs_gemini"

DIMENSIONS = [
    "A_source_cultural_appropriateness",
    "B_adapted_cultural_appropriateness",
    "C_semantic_preservation",
    "D_visual_coherence",
    "E_structural_similarity",
    "overall_success",
]

# ============================================================================
# Data Loading
# ============================================================================

def load_human_data(path: Path, annotator: str) -> pd.DataFrame:
    with open(path) as f:
        records = json.load(f)
    rows = []
    for rec in records:
        row = {"image_id": rec["image_id"], "category": rec["category"]}
        for dim in DIMENSIONS:
            row[f"{annotator}_{dim}"] = rec["ratings"][dim]
        rows.append(row)
    return pd.DataFrame(rows).set_index("image_id")


def load_gemini_data(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            row = {"image_id": rec["id"]}
            for dim in DIMENSIONS:
                row[f"gemini_{dim}"] = rec["evaluation"][dim]["score"]
            rows.append(row)
    return pd.DataFrame(rows).set_index("image_id")


# ============================================================================
# Metrics
# ============================================================================

def pearson(x, y):
    r, p = stats.pearsonr(x, y)
    return r, p


def spearman(x, y):
    r, p = stats.spearmanr(x, y)
    return r, p


def mae(x, y):
    return np.mean(np.abs(np.array(x) - np.array(y)))


def kappa(x, y):
    """Cohen's kappa for ordinal labels (0-5)."""
    try:
        x, y = np.array(x), np.array(y)
        labels = np.arange(0, 6)
        n = len(x)
        # Observed agreement
        po = np.mean(x == y)
        # Expected agreement
        pe = sum(
            (np.sum(x == k) / n) * (np.sum(y == k) / n)
            for k in labels
        )
        if pe == 1.0:
            return float("nan")
        return (po - pe) / (1.0 - pe)
    except Exception:
        return float("nan")


def bootstrap_ci(x, y, corr_func, n=1000, seed=42, ci=0.95):
    """Bootstrap CI for any correlation function returning (r, p)."""
    rng = np.random.RandomState(seed)
    n_samples = len(x)
    boot = []
    for _ in range(n):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        try:
            r, _ = corr_func(x[idx], y[idx])
            if not np.isnan(r):
                boot.append(r)
        except Exception:
            pass
    if len(boot) < 10:
        return np.nan, np.nan
    boot = np.array(boot)
    alpha = 1 - ci
    return np.percentile(boot, alpha / 2 * 100), np.percentile(boot, (1 - alpha / 2) * 100)


def compute_metrics(scores_a, scores_b, label_a, label_b):
    """Return a dict of Pearson, Spearman, kappa, and MAE for two score arrays."""
    x = np.array(scores_a, dtype=float)
    y = np.array(scores_b, dtype=float)

    pr, pp = pearson(x, y) if (x.std() > 0 and y.std() > 0) else (np.nan, np.nan)
    sr, sp = spearman(x, y)

    pr_lo, pr_hi = bootstrap_ci(x, y, pearson) if not np.isnan(pr) else (np.nan, np.nan)
    sr_lo, sr_hi = bootstrap_ci(x, y, spearman)

    return {
        f"pearson_{label_a}_vs_{label_b}": pr,
        "pearson_p": pp,
        "pearson_ci_lower": pr_lo,
        "pearson_ci_upper": pr_hi,
        f"spearman_{label_a}_vs_{label_b}": sr,
        "spearman_p": sp,
        "spearman_ci_lower": sr_lo,
        "spearman_ci_upper": sr_hi,
        f"kappa_{label_a}_vs_{label_b}": kappa(x.astype(int), y.astype(int)),
        f"mae_{label_a}_vs_{label_b}": mae(x, y),
        f"mean_{label_a}": x.mean(),
        f"mean_{label_b}": y.mean(),
        "n": len(x),
    }


# ============================================================================
# Main
# ============================================================================

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...")
    yousha_df = load_human_data(HUMAN_FILES["Yousha"], "Yousha")
    simran_df = load_human_data(HUMAN_FILES["simran"], "simran")
    gemini_df = load_gemini_data(GEMINI_FILE)

    # Merge all on image_id
    merged = yousha_df.join(simran_df, how="inner", lsuffix="", rsuffix="_dup").join(gemini_df, how="inner")
    # Drop duplicate category column if present
    if "category_dup" in merged.columns:
        merged = merged.drop(columns=["category_dup"])
    n_total = len(merged)
    print(f"  Matched image pairs: {n_total}")

    categories = merged["category"]

    # Compute average human score across annotators
    avg_human = {}
    for dim in DIMENSIONS:
        avg_human[dim] = (merged[f"Yousha_{dim}"] + merged[f"simran_{dim}"]) / 2

    print(f"\n{'='*70}")
    print("HUMAN vs GEMINI CORRELATION (per dimension)")
    print(f"{'='*70}")

    results_per_dim = []

    for dim in DIMENSIONS:
        yousha_scores = merged[f"Yousha_{dim}"].values
        simran_scores = merged[f"simran_{dim}"].values
        gemini_scores = merged[f"gemini_{dim}"].values
        avg_scores = avg_human[dim].values

        # Yousha vs Gemini
        m_yousha = compute_metrics(yousha_scores, gemini_scores, "Yousha", "gemini")
        # Simran vs Gemini
        m_simran = compute_metrics(simran_scores, gemini_scores, "simran", "gemini")
        # Avg human vs Gemini
        m_avg = compute_metrics(avg_scores, gemini_scores, "avg_human", "gemini")
        # Inter-annotator (Yousha vs Simran)
        m_iaa = compute_metrics(yousha_scores, simran_scores, "Yousha", "simran")

        print(f"\n[{dim}]")
        print(f"  {'Comparison':<30} {'Pearson':>8} {'95% CI':>16}  {'Spearman':>9} {'95% CI':>16} {'Kappa':>8} {'MAE':>6}")
        print(f"  {'-'*93}")

        for label, m, k_key, mae_key in [
            ("Yousha vs Gemini",   m_yousha, "kappa_Yousha_vs_gemini",    "mae_Yousha_vs_gemini"),
            ("Simran vs Gemini",   m_simran, "kappa_simran_vs_gemini",    "mae_simran_vs_gemini"),
            ("AvgHuman vs Gemini", m_avg,    "kappa_avg_human_vs_gemini", "mae_avg_human_vs_gemini"),
            ("IAA (Yousha/Simran)",m_iaa,    "kappa_Yousha_vs_simran",    "mae_Yousha_vs_simran"),
        ]:
            pr_key = [k for k in m if k.startswith("pearson_") and not k.endswith(("_p", "_ci_lower", "_ci_upper"))][0]
            sr_key = [k for k in m if k.startswith("spearman_") and not k.endswith(("_p", "_ci_lower", "_ci_upper"))][0]
            pr = m[pr_key]; pr_lo = m["pearson_ci_lower"]; pr_hi = m["pearson_ci_upper"]
            sr = m[sr_key]; sr_lo = m["spearman_ci_lower"]; sr_hi = m["spearman_ci_upper"]
            k_val = m[k_key]; e = m[mae_key]
            def _f(v): return f"{v:+.3f}" if not (isinstance(v, float) and np.isnan(v)) else "   nan"
            print(f"  {label:<30} {_f(pr):>8} [{_f(pr_lo)}, {_f(pr_hi)}]  {_f(sr):>9} [{_f(sr_lo)}, {_f(sr_hi)}] {_f(k_val):>8} {e:>6.3f}")

        results_per_dim.append({
            "dimension": dim,
            "n": n_total,
            "pearson_yousha_vs_gemini": m_yousha["pearson_Yousha_vs_gemini"],
            "pearson_yousha_ci_lower": m_yousha["pearson_ci_lower"],
            "pearson_yousha_ci_upper": m_yousha["pearson_ci_upper"],
            "pearson_simran_vs_gemini": m_simran["pearson_simran_vs_gemini"],
            "pearson_simran_ci_lower": m_simran["pearson_ci_lower"],
            "pearson_simran_ci_upper": m_simran["pearson_ci_upper"],
            "pearson_avg_human_vs_gemini": m_avg["pearson_avg_human_vs_gemini"],
            "pearson_avg_ci_lower": m_avg["pearson_ci_lower"],
            "pearson_avg_ci_upper": m_avg["pearson_ci_upper"],
            "pearson_iaa": m_iaa["pearson_Yousha_vs_simran"],
            "pearson_iaa_ci_lower": m_iaa["pearson_ci_lower"],
            "pearson_iaa_ci_upper": m_iaa["pearson_ci_upper"],
            "spearman_yousha_vs_gemini": m_yousha["spearman_Yousha_vs_gemini"],
            "spearman_yousha_ci_lower": m_yousha["spearman_ci_lower"],
            "spearman_yousha_ci_upper": m_yousha["spearman_ci_upper"],
            "spearman_simran_vs_gemini": m_simran["spearman_simran_vs_gemini"],
            "spearman_simran_ci_lower": m_simran["spearman_ci_lower"],
            "spearman_simran_ci_upper": m_simran["spearman_ci_upper"],
            "spearman_avg_human_vs_gemini": m_avg["spearman_avg_human_vs_gemini"],
            "spearman_avg_ci_lower": m_avg["spearman_ci_lower"],
            "spearman_avg_ci_upper": m_avg["spearman_ci_upper"],
            "spearman_iaa": m_iaa["spearman_Yousha_vs_simran"],
            "spearman_iaa_ci_lower": m_iaa["spearman_ci_lower"],
            "spearman_iaa_ci_upper": m_iaa["spearman_ci_upper"],
            "kappa_yousha_vs_gemini": m_yousha["kappa_Yousha_vs_gemini"],
            "kappa_simran_vs_gemini": m_simran["kappa_simran_vs_gemini"],
            "kappa_avg_human_vs_gemini": m_avg["kappa_avg_human_vs_gemini"],
            "kappa_iaa": m_iaa["kappa_Yousha_vs_simran"],
            "mae_yousha_vs_gemini": m_yousha["mae_Yousha_vs_gemini"],
            "mae_simran_vs_gemini": m_simran["mae_simran_vs_gemini"],
            "mae_avg_human_vs_gemini": m_avg["mae_avg_human_vs_gemini"],
            "mae_iaa": m_iaa["mae_Yousha_vs_simran"],
            "mean_yousha": yousha_scores.mean(),
            "mean_simran": simran_scores.mean(),
            "mean_gemini": gemini_scores.mean(),
        })

    # ---- Full score vector (all dims concatenated) ----
    print(f"\n{'='*70}")
    print("FULL SCORE VECTOR (all 6 dims × 100 pairs = 600 values)")
    print(f"{'='*70}")
    print(f"  {'Comparison':<30} {'Pearson':>8} {'95% CI':>16}  {'Spearman':>9} {'95% CI':>16} {'MAE':>6}")
    print(f"  {'-'*90}")

    full_vec_results = []
    for label_a, label_b, h_getter in [
        ("Yousha",     "gemini", lambda d: merged[f"Yousha_{d}"].values),
        ("simran",     "gemini", lambda d: merged[f"simran_{d}"].values),
        ("avg_human",  "gemini", lambda d: avg_human[d].values),
        ("Yousha",     "simran", lambda d: merged[f"Yousha_{d}"].values),
    ]:
        if label_b == "simran":
            g_getter = lambda d: merged[f"simran_{d}"].values
        else:
            g_getter = lambda d: merged[f"gemini_{d}"].values

        h_vec = np.concatenate([h_getter(d) for d in DIMENSIONS]).astype(float)
        g_vec = np.concatenate([g_getter(d) for d in DIMENSIONS]).astype(float)

        m = compute_metrics(h_vec, g_vec, label_a, label_b)
        pr_key = f"pearson_{label_a}_vs_{label_b}"
        sr_key = f"spearman_{label_a}_vs_{label_b}"
        mae_key = f"mae_{label_a}_vs_{label_b}"
        pr = m[pr_key]; pr_lo = m["pearson_ci_lower"]; pr_hi = m["pearson_ci_upper"]
        sr = m[sr_key]; sr_lo = m["spearman_ci_lower"]; sr_hi = m["spearman_ci_upper"]
        e = m[mae_key]
        display = f"{label_a} vs {label_b}" if label_a != "Yousha" or label_b != "simran" else "IAA (Yousha/Simran)"
        def _f(v): return f"{v:+.3f}" if not (isinstance(v, float) and np.isnan(v)) else "   nan"
        print(f"  {display:<30} {_f(pr):>8} [{_f(pr_lo)}, {_f(pr_hi)}]  {_f(sr):>9} [{_f(sr_lo)}, {_f(sr_hi)}] {e:>6.3f}")
        full_vec_results.append({
            "comparison": display, "n": len(h_vec),
            "pearson": pr, "pearson_ci_lower": pr_lo, "pearson_ci_upper": pr_hi,
            "spearman": sr, "spearman_ci_lower": sr_lo, "spearman_ci_upper": sr_hi,
            "mae": e,
        })

    pd.DataFrame(full_vec_results).to_csv(OUTPUT_DIR / "full_vector.csv", index=False)

    # ---- Per-category breakdown (avg human vs gemini, overall_success) ----
    print(f"\n{'='*70}")
    print("PER-CATEGORY: AvgHuman vs Gemini (overall_success)")
    print(f"{'='*70}")
    print(f"  {'Category':<25} {'n':>4} {'Spearman':>9} {'Kappa':>8} {'MAE':>6} {'HumanMean':>10} {'GeminiMean':>11}")
    print(f"  {'-'*76}")

    cat_results = []
    for cat in sorted(categories.unique()):
        mask = (categories == cat).values
        if mask.sum() < 3:
            continue
        h = avg_human["overall_success"].values[mask]
        g = merged["gemini_overall_success"].values[mask]
        if len(set(h)) < 2 or len(set(g)) < 2:
            r, k_val = float("nan"), float("nan")
        else:
            r, _ = spearman(h, g)
            k_val = kappa(h.astype(int), g.astype(int))
        e = mae(h, g)
        print(f"  {cat:<25} {mask.sum():>4} {r:>+9.3f} {k_val:>+8.3f} {e:>6.3f} {h.mean():>10.2f} {g.mean():>11.2f}")
        cat_results.append({"category": cat, "n": mask.sum(), "spearman": r, "kappa": k_val, "mae": e,
                             "mean_avg_human": h.mean(), "mean_gemini": g.mean()})

    # ---- Save outputs ----
    dim_df = pd.DataFrame(results_per_dim)
    dim_df.to_csv(OUTPUT_DIR / "per_dimension.csv", index=False)

    cat_df = pd.DataFrame(cat_results)
    cat_df.to_csv(OUTPUT_DIR / "per_category_overall_success.csv", index=False)

    # Save merged raw data
    flat = pd.DataFrame({
        "image_id": merged.index,
        "category": categories.values,
    })
    for dim in DIMENSIONS:
        flat[f"yousha_{dim}"] = merged[f"Yousha_{dim}"].values
        flat[f"simran_{dim}"] = merged[f"simran_{dim}"].values
        flat[f"avg_human_{dim}"] = avg_human[dim].values
        flat[f"gemini_{dim}"] = merged[f"gemini_{dim}"].values
    flat.to_csv(OUTPUT_DIR / "merged_scores.csv", index=False)

    print(f"\n{'='*70}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"  per_dimension.csv              - Pearson/Spearman/kappa/MAE per dimension")
    print(f"  full_vector.csv                - correlation on all dims concatenated (600 values)")
    print(f"  per_category_overall_success.csv - per-category breakdown")
    print(f"  merged_scores.csv              - raw merged scores for further analysis")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
