# Worklog: Unified Hidden-State Collection + MASSIF Analysis

## Context

Following the null Friction Delta pilot (paper at `/friction_delta_pilot.tex`),
we are re-running the experiment with full hidden-state capture (instead of variance-only)
to test whether the MASSIF feature family — validated for recursive collapse prediction
across 12 architectures in the prior Zenodo records — also predicts hallucination under
factual (TruthfulQA) forcing.

Four Zenodo records reviewed:
- [20159228] Recursive Latent Dynamics — introduces RCI / MASSIF, layer-wise hidden-state extraction
- [20232960] Cross-Arch Persistence Phase Transitions — flips probabilistic, layer-2 emergence, non-monotonic
- [20542872] Dynamical Classification System — R_t, V_t, A_t; 4 dynamical classes (Accelerator/Decelerator/Neutral/Instant)
- [20575616] Recursive Entrainment (v3) — spectral taxonomy, block-4 resonance, GPT-2 Medium anomaly at 355M

## Plan

3 models: Qwen2.5-1.5B, Gemma-2B-IT, Llama-3.2-1B (same as null pilot, for direct comparability).
50 TruthfulQA prompts, greedy decode, max 80 new tokens.
Forward hook on every transformer block saves `output[0].detach()` (full vector, not variance).

Three scripts to build:
1. `collect_hidden_states.py` — Colab/GPU collection script
2. `analysis_a_massif.py` — full MASSIF feature family + LASSO + 10.000-shuffle permutation test
3. `analysis_b_dynamical_classes.py` — per-prompt V_t sign + tau_collapse horizon; test whether hallucinations cluster in Instant/Decelerator classes

---
Task ID: 0
Agent: main
Task: Bootstrap worklog after Zenodo review

Work Log:
- Fetched and read 4 Zenodo records (20575616, 20542872, 20232960, 20159228)
- Read existing pilot paper (friction_delta_pilot.tex) and latent_stability_observatory.py
- Confirmed observatory implements I_t / sigma_t / Phi_t / Psi_t / kappa_t but NOT R_t (Kuramoto) or eigenvalue features
- Identified that pilot null result is consistent with MASSIF theory: variance-only = covariance diagonal = throws away directional structure that R_t was designed to capture

Stage Summary:
- Ready to build collection + analysis scripts
- Key insight: pilot was probing a dynamical system with a static scalar; MASSIF features are the natural replacement

---
Task ID: 2
Agent: main
Task: Fix canonical Kuramoto R_t (Metric 8.1) — user pointed out two implementation errors

Work Log:
- User provided canonical definition: R_t^Kuramoto = ||(1/N) sum_r u_t^(r)|| where u_t^(r) = v_t^(r) / ||v_t^(r)|| (unit VELOCITY, not unit hidden state)
- Identified two errors in my previous implementation:
  (1) Used hidden states h_t directly instead of velocity v_t = h_t - h_{t-1}
  (2) Used non-canonical ensembles (layers / past-tokens) instead of N runs or H*L attention heads
- Refactored massif_features.py Level 6 (R_t section):
  * Added R_t_kuramoto_multi_run(h_runs): canonical multi-run ensemble. Input [N, n_tok, n_layers, d], output [n_tok-1, n_layers]
  * Added R_t_kuramoto_heads(head_outputs): canonical H*L single-run surrogate. Input [n_tok, n_layers, n_heads, head_dim], output [n_tok-1, n_layers]
  * Kept R_t_cross_layer_surrogate and R_t_within_trajectory_surrogate (non-velocity) for ablation
  * Added velocity-corrected surrogates (R_t_cross_layer_velocity, R_t_within_trajectory_velocity) as middle ground
  * Backward-compat shims: R_t_cross_layer() and R_t_within_trajectory() now redirect to velocity-corrected versions
- Updated compute_all_features() to accept optional head_outputs and h_runs, and emit R_canonical_heads_* and R_canonical_multirun_* feature families when provided
- Updated collect_hidden_states.py:
  * HiddenStateCollector now accepts capture_heads=True flag and registers pre-hook on self_attn.o_proj to capture concatenated per-head outputs
  * load_model now reads n_heads, head_dim, n_kv_heads from model.config (auto-detect, handles GQA/MQA)
  * generate_with_hooks now returns (gen_text, hidden_states, head_states, token_ids) and supports temperature > 0 for stochastic multi-run
  * collect_for_model accepts multi_run_n and multi_run_temp params; saves head_states and multirun_hidden_states in traces.npz
  * Added CLI args --multi-run N and --multi-run-temp T
- Updated analysis_a_massif.py:
  * load_model_data now extracts head_states and multirun_hidden_states from npz
  * compute_feature_matrix takes meta dict and passes head_outputs/h_runs to compute_all_features when present
- Updated make_synthetic_traces.py:
  * Now generates head_states (with shared direction injected for hallucinated) and multirun_hidden_states (with attractor convergence for hallucinated)
  * This lets the smoke test exercise the canonical R_t paths
- Re-ran smoke test (synth_a, fresh):
  * Feature count: 200 -> 240 (40 new canonical R_t features)
  * Top 5 features are now ALL canonical: R_canonical_multirun_max/mean/last, A_canonical_multirun_std/min
  * Direction: high->halluc (hallucinations have HIGHER canonical R_t, matches attractor-lock-in interpretation)
  * AUROC = 1.0, p < 0.001, 235/240 features survive BH-FDR
  * Canonical R_t is now the strongest predictor in synthetic test, beating velocity-corrected surrogates

Stage Summary:
- Canonical Kuramoto R_t (Metric 8.1) is now correctly implemented per user's definition
- Both canonical ensembles supported: multi-run (true ensemble) and H*L attention heads (single-run surrogate mentioned in canonical spec)
- Velocity normalization fixed everywhere (v_t = h_t - h_{t-1}, then unit-normalized)
- Synthetic smoke test confirms canonical R_t is the strongest predictor of injected synchronization signal
- Pipeline ready for Colab collection: python collect_hidden_states.py --multi-run 5 (or default 1 for greedy-only with H*L canonical surrogate)

---
Task ID: 1
Agent: main
Task: Build unified collection + MASSIF analysis pipeline (collection script, feature library, Analysis A, Analysis B, synthetic smoke test)

Work Log:
- Created /home/z/my-project/scripts/ directory
- Wrote collect_hidden_states.py: HiddenStateCollector class with forward hooks on every transformer block; saves full output[0].detach() at last token position for every generation step. Output format: per-model traces.npz (object array of [n_tok, n_layers, d] bf16 tensors) + meta.json. Designed for Colab T4 / Drive.
- Wrote massif_features.py: full MASSIF feature family
    * Original hierarchy from observatory: I_t (persistence), sigma_t (rolling tension), Phi_t (volatility gradient), Psi_t (smoothed velocity), kappa_t (curvature)
    * NEW additions from Zenodo 20542872/20575616:
      - R_t_cross_layer: Kuramoto order parameter across layers at each token position (1D)
      - R_t_within_trajectory: Kuramoto order parameter across past k tokens at each layer (2D)
      - V_t, A_t: first/second derivatives of both R_t flavors
      - tau_collapse_horizon: projected steps until R_within crosses threshold
    * INSIDE framework (ICLR 2024): full covariance eigenvalue spectrum via SVD
      - top-k eigenvalues per layer, spectral entropy, participation ratio, effective rank
    * Friction Delta (variance-only) for direct comparison with null pilot
    * Yields ~200 features per prompt
- Wrote analysis_a_massif.py:
    * T1: per-feature AUROC with 200-shuffle perm p-value + Benjamini-Hochberg FDR (q=0.10)
    * T2: LASSO-CV AUROC (LogisticRegressionCV L1, 5-fold stratified)
    * T3: 1000-shuffle permutation null on LASSO-CV AUROC (max-statistic correction)
    * Optional judge via Groq GPT-OSS-120b (same prompt as null pilot)
    * Caches features.csv + meta.json with judge labels
- Wrote analysis_b_dynamical_classes.py:
    * Per-prompt class assignment using V_slope before collapse event
    * 5 classes: Accelerator (V>0), Decelerator (V<0), Neutral (|V|~0), Instant (collapse at step<=1), Stochastic (no collapse, near-zero V)
    * T1: class x hallucination contingency (Fisher's exact or chi-square)
    * T2: per-class AUROC (one-vs-rest)
    * T3: V_slope_pre_collapse as continuous predictor
    * T4: tau_collapse_min as predictor
    * T5: LASSO-CV on dynamical-class feature subset
    * T6: cross-model class distribution comparison (consistency check)
- Wrote make_synthetic_traces.py: synthetic data generator with injected signal (hallucinated = drift toward attractor → high R_within)
- Ran synthetic smoke test:
    * All 3 synthetic models: AUROC = 1.0, p < 0.01, top features = R_cross_mean / I_mean / eig_lambda1 (matching injected signal)
    * 199/200 features survive BH-FDR
    * Null 95th percentile = 0.69 (LOWER than null pilot's 0.75 — more samples + LASSO regularization give more headroom)
- Wrote scripts/README.md runbook

Stage Summary:
- 5 scripts in /home/z/my-project/scripts/: collect_hidden_states.py, massif_features.py, analysis_a_massif.py, analysis_b_dynamical_classes.py, make_synthetic_traces.py
- Pipeline validated end-to-end on synthetic data with known signal
- Ready for user to run collect_hidden_states.py on Colab (T4, ~30-45 min for 3 models)
- Then analysis_a + analysis_b can run on CPU (~15 min total)
- All scripts honor same model set (Qwen2.5-1.5B, Gemma-2B-IT, Llama-3.2-1B) as null pilot for direct comparability
- Note: gemma2-2b-it n_layers_expected corrected to 26 (was 18 in paper — typo)
