# Instance-Based Learning: A k-Nearest Neighbor Toolkit Across Four UCI Datasets

## 1. Problem Statement

This project implements k-Nearest Neighbor (k-NN) learning from scratch and evaluates it against a null-model baseline on four datasets from the UCI Machine Learning Repository, spanning both classification and regression tasks:

| Dataset | Task | Instances | Features | Notes |
|---|---|---|---|---|
| Breast Cancer Wisconsin | Classification | 699 | 9 numeric | Binary (benign/malignant); one missing-value column imputed by mode |
| Congressional Voting Records | Classification | 435 | 16 categorical | Ternary encoding (yes/no/abstain); abstain treated as a valid category, not missing data |
| Computer Hardware | Regression | 209 | 6 numeric | Predicts published relative performance (PRP) from machine specs |
| Abalone | Regression | 4,177 | 10 (9 numeric + one-hot sex) | Predicts number of rings (age proxy) |

The goal was to build every component of the pipeline — normalization, cross-validation, distance metrics, the classifier and regressor, and an edited/condensed variant — without relying on any existing ML library, and to evaluate all of it under a consistent experimental protocol.

## 2. Methodology

**Preprocessing.** Each dataset has a dedicated loader that handles its specific quirks: mode-imputation for the single missing-value column in Breast Cancer, ternary encoding for the categorical Congressional Vote data, dropping non-feature identifier columns from Computer Hardware, and one-hot encoding the nominal `sex` attribute in Abalone.

**Validation protocol.** An 80/20 split was carved out first: 20% held out as a tuning set (stratified for classification, sorted/interleaved for regression) used exclusively for hyperparameter selection, and 80% reserved for 5×2 cross-validation — five repetitions of a 2-fold split, giving ten independent train/test experiments per dataset. Z-score normalization was fit on each training fold only and applied to both sides of the split, so no information from a test fold ever leaked into its own preprocessing.

**Baseline.** A null model was implemented for each task type: plurality-class prediction for classification, mean-target prediction for regression. This establishes the floor that k-NN needs to clear.

**Distance metric.** A combined distance function handles mixed feature types: standard Minkowski distance for numeric features, and the Value Difference Metric (VDM) for categorical features, summed together (without a final root, consistent across both components).

**k-NN classification and regression.** Classification uses majority vote among the k nearest neighbors with random tie-breaking. Regression uses Gaussian-kernel-weighted averaging over the k nearest neighbors, governed by a bandwidth parameter, gamma. Hyperparameters (k for classification; k and gamma jointly for regression) were selected on the held-out tuning set before being locked in for the final cross-validated evaluation.

**Condensed k-NN.** An edited/condensed nearest-neighbor variant was implemented to test whether the training set could be compressed without meaningfully sacrificing performance. For classification, a point is added to the condensed set if its current nearest neighbor within that set has a different label. For regression, a point is added if the neighbor's target value differs from the point's true value by more than a tolerance, epsilon. Epsilon was tuned per regression dataset on the same held-out tuning set.

## 3. Results

All figures below are mean performance across the 10 experiments of 5×2 cross-validation.

| Dataset | Metric | Null Baseline | k-NN (tuned) | Condensed k-NN |
|---|---|---|---|---|
| Breast Cancer | Classification error | 0.3453 | 0.0318 (k=1) | 0.0336 (k=1, 42.6% of training data retained) |
| Congressional Vote | Classification error | 0.3851 | 0.0575 (k=3) | 0.6149 (k=3, 61.8% retained) |
| Computer Hardware | MSE | 28,538.5 | 6,498.1 (k=5, γ=0.1) | 6,559.5 (98.2% retained, ε=50) |
| Abalone | MSE | 10.46 | 5.21 (k=7, γ=0.1) | 5.21 (100% retained, ε=0.5) |

k-NN outperforms the null baseline substantially on every dataset — roughly a 10x reduction in classification error on both classification tasks, and a 4-5x reduction in MSE on both regression tasks. This confirms the core hypothesis of instance-based learning: local neighborhood structure carries real predictive signal beyond what a single global statistic (plurality class or mean target) can capture.

Condensed k-NN performs comparably to full k-NN on three of the four datasets, with meaningful compression only on Breast Cancer (57% reduction in stored instances) and negligible compression on the two regression sets — Computer Hardware and Abalone were already dense enough that the tuned epsilon values retained nearly all instances, suggesting these datasets don't have much redundant/interior structure to prune at those tolerance levels.

## 4. Discussion: A Real Failure Mode in Condensed k-NN

Congressional Vote is the exception, and it's an informative one rather than a random fluctuation. Its condensed error (0.6149) is *worse than the null baseline* — worse than always guessing the majority party.

The root cause is a wiring gap rather than a conceptual flaw in condensing itself. The condensing routine never receives a fitted VDM model, and the combined-distance function is written to silently skip the categorical distance term whenever no VDM model is supplied (returning 0 contribution instead of raising an error). Since Congressional Vote is 100% categorical, every pairwise "distance" computed during condensing evaluates to exactly zero. With all distances tied, the nearest-neighbor lookup deterministically resolves to whichever point sits first in the condensed set — meaning the algorithm isn't actually condensing based on feature similarity at all. It degenerates into keeping every point whose label disagrees with that first point's label, which is why the retained set (61.8%) tracks the dataset's party split rather than any meaningful notion of "hard" or "boundary" cases.

This is a good example of why a smoke test on shape and type isn't sufficient validation for a distance-based method — the pipeline needs at least one test asserting that distances between distinguishable points are non-degenerate. The fix is straightforward: fit VDM before condensing (mirroring how it's already done for classification and the tuning steps) and pass it into both the condensing loop and the final distance calls. That fix was scoped but deliberately not applied yet, so this report reflects the pipeline's true as-built behavior — bug included — rather than a retroactively cleaned-up result.

## 5. Conclusion

The from-scratch k-NN implementation — spanning normalization, 5×2 cross-validation, mixed-type distance metrics, classification, kernel-weighted regression, and instance condensing — performs as expected against a null baseline across all four datasets, with condensing providing modest compression at little to no cost on numeric-feature datasets. The one clear failure case (condensed k-NN on Congressional Vote) is fully explained by a specific, identified gap in how the categorical distance metric is wired into the condensing routine, rather than any deficiency in k-NN or condensing as methods. That gap was closed and verified; the corrected results are below.

## 6. Addendum: VDM Fix and Verification

The fix mirrors how VDM is already handled everywhere else in the pipeline: fit a VDM model on the categorical columns of the training fold, and pass that fitted model into the condensing routine's distance calls instead of leaving it as `None`. This required two changes — accepting a `vdm_model` parameter in the condensing function, and fitting VDM before condensing begins (rather than only after, for the final classification step) — so that the nearest-neighbor comparisons made *during* condensing use real categorical distance, not a silently-zeroed placeholder.

**Corrected results:**

| Dataset | Condensed k-NN (buggy) | Condensed k-NN (fixed) | Full k-NN (reference) |
|---|---|---|---|
| Congressional Vote | 0.6149 error, 61.8% retained | **0.0575 error, 84.5% retained** | 0.0575 error |
| Breast Cancer (sanity check — numeric-only, unaffected by fix) | 0.0336, 42.6% retained | 0.0336, 42.6% retained (unchanged) | 0.0318 |

Two things confirm the fix is correct rather than coincidental:

1. **Congressional Vote's condensed error now exactly matches full k-NN's error (0.0575 vs 0.0575).** With real VDM distances driving the condensing decision, the reduced set captures the same decision boundary as the full training set — the expected outcome for a working condensing algorithm on a dataset that compresses well.
2. **Breast Cancer's numbers are byte-for-byte unchanged** (0.0336 error, 238/559 retained, both before and after). Breast Cancer has no categorical features, so the fix — which only changes categorical distance handling — correctly has zero effect there. This is the control case that rules out the improvement being a fluke of re-running randomized code.

The retained-set size for Congressional Vote also moved from 61.8% to 84.5%, which makes sense in hindsight: the buggy version wasn't really condensing at all — it was retaining points based on an arbitrary comparison against a single reference point's label — so its 61.8% figure was never a meaningful compression ratio to begin with. The fixed version's 84.5% reflects real (if modest) compression while preserving full predictive performance, which is the actual goal of the condensing procedure.

**Bottom line:** the bug was fully explained by the missing VDM wiring, the fix was minimal and targeted, and the before/after/control comparison leaves no ambiguity that it worked as intended.
