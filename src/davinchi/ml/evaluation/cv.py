"""5x2 cross-validation runner."""
import numpy as np
from ..preprocessing.splits import make_5x2_folds
from ..preprocessing.normalize import normalize_fold
from .metrics import classification_error, mean_squared_error


def run_cv(datasets, model_fn_clf, model_fn_reg, label="model", random_state=42, verbose=True):
    """
    Run 5x2 CV for all datasets using provided model functions.

    model_fn_clf(X_train, y_train, X_test) -> y_pred  (classification)
    model_fn_reg(X_train, y_train, X_test) -> y_pred  (regression)

    Returns a dict of {dataset_name: {"mean": ..., "std": ..., "metric": ...}}.
    """
    if verbose:
        print(f"\n{'='*55}\n  Results: {label}\n{'='*55}")

    results = {}

    for name, d in datasets.items():
        task = d["task"]
        X_main = d["X_main"]
        y_main = d["y_main"]

        folds = make_5x2_folds(X_main, y_main, task, random_state=random_state)
        scores = []

        for round_folds in folds:
            for fold in round_folds:
                tr_idx = fold["train_idx"]
                te_idx = fold["test_idx"]

                X_tr, X_te, _, _ = normalize_fold(X_main[tr_idx], X_main[te_idx])
                y_tr = y_main[tr_idx]
                y_te = y_main[te_idx]

                if task == "classification":
                    y_pred = model_fn_clf(X_tr, y_tr, X_te)
                    score = classification_error(y_te, y_pred)
                else:
                    y_pred = model_fn_reg(X_tr, y_tr, X_te)
                    score = mean_squared_error(y_te, y_pred)

                scores.append(score)

        metric = "class error" if task == "classification" else "MSE"
        results[name] = {"metric": metric, "mean": np.mean(scores), "std": np.std(scores)}

        if verbose:
            print(f"  {name:20s} | {metric:11s} | mean: {np.mean(scores):.4f} | std: {np.std(scores):.4f}")

    return results
