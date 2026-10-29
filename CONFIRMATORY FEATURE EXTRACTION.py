# === CONFIRMATORY FEATURE EXTRACTION ===
# Strict prefix-only causal mask. No token at or after T_claim enters any feature.

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import bootstrap

# ---------- 1. Strict prefix-only feature extraction ----------

def extract_prefix_features(per_token_vars, early_layers, late_layers, T_claim, min_prefix=5):
    """
    Extract Δ features on tokens [0, T_claim-1] ONLY.
    
    Args:
        per_token_vars: [n_tokens, n_layers] array of per-layer variances
        early_layers: list of early layer indices
        late_layers: list of late layer indices
        T_claim: index of the first token of the claim (0-indexed)
        min_prefix: minimum number of prefix tokens required
    
    Returns:
        dict of features, or None if prefix is too short
    """
    per_token = np.array(per_token_vars)  # [n_tokens, n_layers]
    
    # HARD CAUSAL MASK: only tokens strictly before T_claim
    prefix = per_token[:T_claim]  # tokens [0, T_claim-1]
    
    if len(prefix) < min_prefix:
        return None  # too short to compute reliable features
    
    early_var = prefix[:, early_layers].mean(axis=1)  # [T_claim]
    late_var = prefix[:, late_layers].mean(axis=1)
    delta_traj = early_var - late_var  # [T_claim]
    
    # --- Static features (instantaneous Δ state) ---
    features = {
        "delta_mean": float(delta_traj.mean()),
        "delta_min": float(delta_traj.min()),
        "delta_max": float(delta_traj.max()),
        "delta_std": float(delta_traj.std()),
        "delta_last": float(delta_traj[-1]),  # Δ at T_claim - 1
    }
    
    # --- Dynamic features (trajectory shape) ---
    x = np.arange(len(delta_traj))
    if len(delta_traj) >= 2:
        features["delta_slope"] = float(np.polyfit(x, delta_traj, 1)[0])
    else:
        features["delta_slope"] = 0.0
    
    # Flatline count: tokens where |Δ| < threshold (threshold from tuning split, frozen)
    # This will be set externally; for now use median of this trajectory
    threshold = np.median(np.abs(delta_traj)) if len(delta_traj) > 0 else 0.0
    features["flatline_count"] = float(np.sum(np.abs(delta_traj) < threshold))
    
    # --- Metadata for confound control ---
    features["n_prefix_tokens"] = len(prefix)
    
    return features


# ---------- 2. Multivariate confound analysis (H2: Specificity) ----------

def multivariate_confound_analysis(features_list, labels, confound_names):
    """
    Test whether delta_slope (or any Δ feature) adds predictive power
    beyond confounds (length, entropy, semantic_entropy).
    
    Uses likelihood-ratio test: full model (confounds + Δ) vs reduced (confounds only).
    
    Args:
        features_list: list of dicts with all features + confounds
        labels: 0/1 array
        confound_names: list of confound variable names (e.g., ["n_prefix_tokens", "entropy", "semantic_entropy"])
    
    Returns:
        dict with AUROC of each model and likelihood-ratio test results
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from scipy.stats import chi2
    import numpy as np
    
    labels = np.array(labels)
    n = len(labels)
    
    # Build feature matrices
    delta_features = ["delta_mean", "delta_min", "delta_max", "delta_std",
                     "delta_slope", "delta_last", "flatline_count"]
    
    # Confound matrix
    X_confound = np.array([[f[c] for c in confound_names] for f in features_list])
    
    results = {}
    
    for d_feat in delta_features:
        X_delta = np.array([f[d_feat] for f in features_list]).reshape(-1, 1)
        
        # Reduced model: confounds only
        model_reduced = LogisticRegression(max_iter=1000, C=1.0)
        model_reduced.fit(X_confound, labels)
        pred_reduced = model_reduced.predict_proba(X_confound)[:, 1]
        auc_reduced = roc_auc_score(labels, pred_reduced)
        loglik_reduced = -np.sum(labels * np.log(pred_reduced + 1e-10) + 
                                (1 - labels) * np.log(1 - pred_reduced + 1e-10))
        
        # Full model: confounds + delta feature
        X_full = np.hstack([X_confound, X_delta])
        model_full = LogisticRegression(max_iter=1000, C=1.0)
        model_full.fit(X_full, labels)
        pred_full = model_full.predict_proba(X_full)[:, 1]
        auc_full = roc_auc_score(labels, pred_full)
        loglik_full = -np.sum(labels * np.log(pred_full + 1e-10) + 
                             (1 - labels) * np.log(1 - pred_full + 1e-10))
        
        # Likelihood ratio test (1 df added)
        lr_stat = 2 * (loglik_reduced - loglik_full)
        lr_p = chi2.sf(lr_stat, df=1)
        
        results[d_feat] = {
            "auc_confound_only": auc_reduced,
            "auc_with_delta": auc_full,
            "auc_gain": auc_full - auc_reduced,
            "lr_statistic": lr_stat,
            "lr_p_value": lr_p,
        }
    
    return results


# ---------- 3. Paired bootstrap of AUROC gain ----------

def paired_bootstrap_auroc_gain(features_list, labels, delta_feat, confound_names, n_boot=10000):
    """
    Bootstrap the AUROC gain of adding delta_feat to the confound model.
    
    Returns:
        point estimate and 95% CI of the AUROC gain
    """
    labels = np.array(labels)
    n = len(labels)
    X_confound = np.array([[f[c] for c in confound_names] for f in features_list])
    X_delta = np.array([f[delta_feat] for f in features_list]).reshape(-1, 1)
    
    def compute_gain(idx):
        X_c = X_confound[idx]
        X_d = X_delta[idx]
        y = labels[idx]
        
        if len(set(y)) < 2:
            return 0.0
        
        m_r = LogisticRegression(max_iter=1000, C=1.0).fit(X_c, y)
        p_r = m_r.predict_proba(X_c)[:, 1]
        auc_r = roc_auc_score(y, p_r)
        
        X_f = np.hstack([X_c, X_d])
        m_f = LogisticRegression(max_iter=1000, C=1.0).fit(X_f, y)
        p_f = m_f.predict_proba(X_f)[:, 1]
        auc_f = roc_auc_score(y, p_f)
        
        return auc_f - auc_r
    
    # Point estimate
    point = compute_gain(np.arange(n))
    
    # Bootstrap
    rng = np.random.default_rng(42)
    gains = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        gains.append(compute_gain(idx))
    
    gains = np.array(gains)
    ci_low = np.percentile(gains, 2.5)
    ci_high = np.percentile(gains, 97.5)
    
    return {
        "point_estimate": point,
        "ci_95": [ci_low, ci_high],
        "n_bootstrap": n_boot,
    }


# ---------- 4. Full confirmatory pipeline ----------

def run_confirmatory_analysis(records, early_layers, late_layers, span_boundaries, 
                              confound_data=None, n_boot=10000):
    """
    Full confirmatory analysis:
    1. Extract prefix-only features (strict ex-ante)
    2. Univariate AUROC per feature
    3. Multivariate confound analysis (H2)
    4. Paired bootstrap of AUROC gain
    
    Args:
        records: dict of records with per_token_vars
        early_layers, late_layers: layer band indices
        span_boundaries: dict {record_idx: T_claim} — first token of the claim
        confound_data: dict {record_idx: {"entropy": float, "semantic_entropy": float}}
        n_boot: bootstrap iterations
    """
    features_list = []
    labels = []
    
    print("=== STEP 1: Prefix-only feature extraction ===")
    for idx, T_claim in span_boundaries.items():
        r = records[idx]
        if r.get("hallucinated") is None:
            continue
        
        feats = extract_prefix_features(
            r["per_token_vars"], early_layers, late_layers, T_claim
        )
        if feats is None:
            continue
        
        # Add confounds
        if confound_data and idx in confound_data:
            feats.update(confound_data[idx])
        else:
            # Length is always available
            pass  # n_prefix_tokens already in feats
        
        features_list.append(feats)
        labels.append(1 if r["hallucinated"] else 0)
    
    labels = np.array(labels)
    n_pos = int(labels.sum())
    n_total = len(labels)
    
    print(f"  Valid records: {n_total} ({n_pos} halluc, {n_total - n_pos} correct)")
    print(f"  Excluded (prefix too short): {len(span_boundaries) - n_total}")
    
    if n_total < 20 or n_pos < 5:
        print("  INSUFFICIENT DATA for analysis")
        return None
    
    # --- Univariate AUROC ---
    print(f"\n=== STEP 2: Univariate AUROC (prefix-only, ex-ante) ===")
    delta_features = ["delta_mean", "delta_min", "delta_max", "delta_std",
                     "delta_slope", "delta_last", "flatline_count"]
    
    print(f"{'Feature':<20} {'AUROC':>8} {'Direction':>20}")
    print("-" * 50)
    
    for fname in delta_features:
        fvals = np.array([f[fname] for f in features_list])
        if np.std(fvals) == 0:
            print(f"{fname:<20} {'n/a':>8} {'constant':>20}")
            continue
        raw_auc = roc_auc_score(labels, fvals)
        auc = max(raw_auc, 1 - raw_auc)
        direction = "low Δ → halluc" if raw_auc < 0.5 else "high Δ → halluc"
        print(f"{fname:<20} {auc:>8.3f} {direction:>20}")
    
    # Length control
    lengths = np.array([f["n_prefix_tokens"] for f in features_list])
    len_auc = roc_auc_score(labels, lengths)
    len_auc = max(len_auc, 1 - len_auc)
    print(f"\n{'n_prefix_tokens':<20} {len_auc:>8.3f} {'length baseline':>20}")
    
    # --- Multivariate confound analysis ---
    print(f"\n=== STEP 3: Multivariate confound analysis (H2: Specificity) ===")
    
    # Determine available confounds
    available_confounds = ["n_prefix_tokens"]
    if confound_data:
        sample = next(iter(confound_data.values()))
        available_confounds += [k for k in sample.keys() if k != "n_prefix_tokens"]
    
    print(f"  Confounds: {available_confounds}")
    print(f"\n  {'Feature':<20} {'AUC(conf)':>10} {'AUC(+Δ)':>10} {'Gain':>8} {'LR χ²':>8} {'p-value':>10}")
    print("  " + "-" * 70)
    
    mv_results = multivariate_confound_analysis(
        features_list, labels, available_confounds
    )
    
    for d_feat, res in mv_results.items():
        print(f"  {d_feat:<20} {res['auc_confound_only']:>10.3f} {res['auc_with_delta']:>10.3f} "
              f"{res['auc_gain']:>+8.3f} {res['lr_statistic']:>8.3f} {res['lr_p_value']:>10.4f}")
    
    # --- Paired bootstrap for best feature ---
    print(f"\n=== STEP 4: Paired bootstrap of AUROC gain (H2) ===")
    
    # Find best feature by AUROC gain
    best_feat = max(mv_results.keys(), key=lambda k: mv_results[k]["auc_gain"])
    best_gain = mv_results[best_feat]["auc_gain"]
    
    print(f"  Best feature by AUROC gain: {best_feat} (gain = {best_gain:.3f})")
    
    boot_result = paired_bootstrap_auroc_gain(
        features_list, labels, best_feat, available_confounds, n_boot=n_boot
    )
    
    print(f"  AUROC gain: {boot_result['point_estimate']:.3f}")
    print(f"  95% CI: [{boot_result['ci_95'][0]:.3f}, {boot_result['ci_95'][1]:.3f}]")
    
    # H2 verdict: lower bound of CI must be > 0.02
    h2_supported = boot_result['ci_95'][0] > 0.02
    print(f"\n  H2 (Specificity): {'SUPPORTED' if h2_supported else 'NOT SUPPORTED'}")
    print(f"  (Lower CI bound {boot_result['ci_95'][0]:.3f} {'>' if h2_supported else '≤'} 0.02)")
    
    return {
        "univariate_auroc": {f: roc_auc_score(labels, np.array([feat[f] for feat in features_list]))
                            for f in delta_features},
        "multivariate": mv_results,
        "bootstrap": boot_result,
        "n_samples": n_total,
        "h2_supported": h2_supported,
    }


# ---------- 5. Example usage ----------

# This is how the confirmatory split would call it:

# 1. Load saved traces
# with open("/path/to/confirmatory_traces.json") as f:
#     records = json.load(f)

# 2. Span boundaries from judge (T_claim = first token of hallucinated span)
#    For response-level labels, use T_claim = n_tokens // 2 (midpoint approximation)
#    For span-level labels, use the actual span start
# span_boundaries = {idx: find_claim_start(record) for idx, record in records.items()}

# 3. Confound data (entropy, semantic_entropy from separate computation)
# confound_data = {idx: {"entropy": compute_entropy(record), 
#                         "semantic_entropy": compute_semantic_entropy(record)}}

# 4. Layer band (frozen from tuning split)
# early_layers = [0, 1, 2, 3, 4, 5, 6, 7]  # from tuning-half-A
# late_layers = [8, 9, 10, 11, 12, 13, 14, 15]

# 5. Run
# results = run_confirmatory_analysis(
#     records, early_layers, late_layers,
#     span_boundaries, confound_data,
#     n_boot=10000
# )

print("Confirmatory analysis pipeline ready.")
print("Usage: load traces, set span_boundaries, set confound_data, call run_confirmatory_analysis()")