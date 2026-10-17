# Pre-registered Evaluation Protocol: Does the Geometric Flatline Predict Hallucination?

Oct 17, 2026 · Daniel Sols

## 1. Status and freeze

This protocol is draft v0.1

## 2. Question and hypotheses

The question: before a hallucinated span is emitted, does the Friction Delta flatline carry predictive information that cheaper signals lack, and does acting on it reduce hallucinations?

| ID 				| Hypothesis 	| Supported if 	| Refuted if 	|
| --- 				| --- 			| --- 			| --- 			|
| H1 Prediction 	| A flatline score computed strictly before a claim predicts that the claim is hallucinated | Lower 95% CI bound of AUROC is above 0.60 | Upper bound is below 0.55 |
| H2 Specificity 	| The score adds information beyond entropy, variance magnitude, and position controls | Lower bound of the AUROC gain over the strongest unsupervised baseline is above 0.02 | Upper bound of the gain is below 0.01 |
| H3 Intervention 	| Detector-triggered grafting lowers the hallucination rate more than no intervention, random-timing grafting, and filler grafting | Absolute reduction of at least 5 points against all three, Holm-adjusted p below 0.05 | Upper bound of the reduction is below 2 points against any one of them |

Results between the two bounds are reported as inconclusive. Study 2 runs and is reported even if H1 is refuted, because it then tests whether the graft helps regardless of timing.

## 3. Operational definitions

Every quantity follows the [CFKR paper](https://doi.org/10.5281/zenodo.23002427) and is fixed here. Only the detector parameters are tuned, and only on the tuning split.

- **Friction Delta:** Δ = V\_early - V\_late, the difference in hidden-state variance between an early and a late layer band. The record page writes the sign both ways (early minus late in the abstract, late minus early in the trilogy summary), so the convention must be fixed before registration.
- **Variance V and layer bands:** taken from the paper's implementation and frozen by commit hash. Open question: the exact V estimator and the early and late band boundaries, to be filled from the code before the freeze.
- **Flatline:** Δ at or below ε for at least w consecutive tokens. ε comes from a fixed grid and w from {1, 2, 4, 8}. The pair with the highest tuning-split AUROC is chosen, and ties go to the smaller w.
- **Score s:** the rolling minimum of Δ over the last w tokens, read at the token just before a span.
- **Operating point:** the threshold on s is set on the tuning split so that 5% of all queries raise an alert, matching the claim that about 95% of queries run unimpeded.
- **Ex-ante rule:** s uses only tokens strictly before the span's first token. Nothing inside or after the span enters any feature.

## 4. Models, data, and labels

Three models are evaluated: MyceliaLM 1.62B and two open-weight decoder transformers of 1-3B parameters. Δ was developed on MyceliaLM, so a result on that model alone cannot separate the metric from the architecture. Names and checkpoints are frozen at registration.

- **Datasets (candidates, final list frozen at registration):** long-form biographies scored claim by claim (FActScore-style), short-form factual QA (SimpleQA-style), and TruthfulQA generation. One out-of-domain set is held back for the generalization check only.
- **Split:** prompts are assigned at the prompt level by a seeded hash, 30% tuning and 70% confirmatory. The confirmatory split is scored once.
- **Labels:** claim-level factual correctness from a frozen judge model of a different family than the evaluated models, checked against a fixed reference-corpus snapshot.
- **Human audit:** 300 claims labeled by two annotators who are blind to detector scores and to arm. Cohen's κ between judge and humans must reach 0.70, otherwise the primary analysis runs on the audited subset.
- **Matched negatives:** each first hallucinated span is paired with a correct span from another response, matched on token position (within 5 tokens), prompt type, and span type (entity, number, date).
- **Decoding:** greedy is primary. Temperature 0.7 with 3 seeds is secondary.

## 5. Study 1: prediction

The primary metric is the AUROC of s for hallucinated spans against matched correct spans, per model, on the confirmatory split, with 95% bootstrap CIs (10,000 resamples, clustered by prompt). Spans come from the no-intervention generations (arm A in section 6).

Baselines, all read at the same token position:

- Mean token log-probability and predictive entropy
- Semantic entropy over 5 samples
- Hidden-state variance magnitude alone, with no early/late contrast
- Negative control: Δ computed from 100 random layer partitions
- A position, length, and span-type logistic model
- A supervised linear probe on hidden states, as an upper reference and not a competitor

Analyses:

1. **H1:** AUROC of s against 0.5, judged by the bounds in section 2.
2. **H2:** paired bootstrap of the AUROC gain over the strongest unsupervised baseline, plus a likelihood-ratio test of a logistic model with and without s, given the baselines.
3. **Operating point:** true-positive rate and precision at the 5% alert threshold fixed on the tuning split. The realized confirmatory alert rate must fall between 3% and 7%.
4. **Generalization:** AUROC by dataset and span type, and transfer of ε from each dataset to the others.
5. **Confound audit:** AUROC after controlling for entity rarity, measured by a frozen corpus frequency count.

If the random-partition Δ matches the real Δ, H2 is refuted regardless of the other results.

## 6. Study 2: intervention

Study 2 asks whether detector-triggered grafting reduces hallucinations beyond what the graft mechanics and the reflection content explain. Every prompt runs under every arm with the same decoding seed.

| Arm 	| What happens | What it isolates |
| --- 	| --- | --- |
| A 	| No intervention | Baseline hallucination rate |
| B 	| Graft the reflection block when the detector fires | The full CFKR claim |
| C 	| Graft at random times, matched to B's alert rate | Detector validity |
| D 	| Graft filler KV states of the same length G at B's firing points | Reflection content vs graft mechanics |
| E 	| Insert the reflection text as ordinary tokens at B's firing points | Grafting vs plain prompting (secondary) |

The reflection block is a fixed, content-free text frozen at registration. It contains no facts and no answers to any dataset question. Its KV states are produced by one forward pass through all layers per intervention, and that pass is counted in the latency figures.

Primary contrasts are B vs A, B vs C, and B vs D. B vs E is secondary.

Primary outcomes:

1. Hallucination rate per response, under intention to treat.
2. Harm rate: the share of baseline-correct claims turned incorrect, with a margin of 2 points.
3. End-to-end latency overhead per intervention, including the reflection forward pass.

## 7. Power, statistics, and multiplicity

Study 1 needs 1,000 hallucinated spans plus 1,000 matched correct spans per model, and Study 2 needs about 1,700 prompts per arm. Both figures are hand calculations and must be replaced by a simulation before registration.

- **Study 1:** at AUROC 0.60 with 1,000 positives and 1,000 negatives, the Hanley-McNeil standard error is about 0.013, so the CI half-width is about 0.025. That resolves the 0.55 and 0.60 decision bounds.
- **Study 2:** detecting a drop from 30% to 25% at 80% power takes about 1,250 prompts per arm at α = 0.05. With Holm adjustment over three contrasts it rises to about 1,700 prompts per arm.
- **Sizing:** the confirmatory split must hold at least 1,800 prompts per model. The expected positive count is estimated from the tuning split before the confirmatory split is opened, and prompts are added then if it falls short.
- **Tests:** prompt-clustered bootstrap for Study 1. Paired McNemar tests for Study 2, with a mixed-effects logistic model as a sensitivity analysis.
- **Multiplicity:** family-wise α = 0.05, with Holm correction within each hypothesis family and across the three models.
- **Headline claim:** requires the result on MyceliaLM and on at least one comparator model.

## 8. Exclusions, deviations, and reporting

Exclusions are fixed now, and results are published whichever way they fall.

- **Exclusions:** responses under 20 tokens, refusals (reported separately), prompts with no checkable claim, and claims the judge cannot score. All exclusions are applied before results are broken out by arm.
- **No post-hoc tuning:** ε, w, layer bands, the judge, and the datasets do not change after the tuning split closes.
- **Deviations log:** every departure from this protocol is timestamped, and any analysis affected is labeled exploratory.
- **Publication:** confirmatory results are released as positive, negative, or inconclusive. Per-item scores, code, configs, and the unsealed prompt set go to Zenodo.
- **Full baseline reporting:** every baseline and negative control is reported, including those that favor the baselines.
- **Independence:** the author is also the developer of the method, so labeling is blind to arm and an external replicator is invited to re-run the pipeline from the frozen commit.

## 9. Threats to validity and interpretation

The largest threat is that Δ tracks ordinary uncertainty or token rarity under a new name. H2 and the random-partition control test this directly.

| H1 					| H2 		| Reading 	|
| --- 					| --- 		| --- 		|
| Supported 			| Supported | Δ carries signal beyond cheap baselines, so Study 2 is interpretable as a test of a valid detector |
| Supported 			| Refuted 	| Δ predicts hallucination but duplicates entropy or variance, so the geometric framing adds no predictive value here |
| Refuted 				| Any 		| No predictive use in this setup, and Study 2 shows whether grafting helps regardless of timing |
| Inconclusive 			| Any 		| Underpowered or unstable, and any extension needs a registered amendment |

Other threats and their handling:

- **Label noise:** judge errors, handled by the human audit and the κ gate.
- **Confounds:** position, length, entity rarity, and span type, handled by matching and logistic controls.
- **Architecture dependence:** the comparator models test transfer beyond MyceliaLM.
- **Training-data leakage:** benchmark items may appear in pretraining data, so n-gram overlap checks are reported and the out-of-domain set is used.
- **Researcher degrees of freedom:** layer bands, ε, and w are tuned only on the tuning split.
- **Reflection leakage:** the reflection block is fixed and content-free.
- **Scope:** results cover 1-3B models and do not extend to frontier scale without replication.
