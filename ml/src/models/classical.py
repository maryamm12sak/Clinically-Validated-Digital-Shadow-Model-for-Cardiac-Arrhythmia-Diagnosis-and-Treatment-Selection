"""
Classical baselines: one independent binary classifier per label, each an sklearn Pipeline
fitted on the training folds only.

SVM is not included yet: an RBF SVC on ~17k records x ~250 features x 12 labels is slow on CPU
and needs probability calibration (Platt scaling with internal CV) to give scores comparable to
the other models. It can be added once the feature set is fixed.
"""
import lightgbm as lgb
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_NAMES = ["logreg", "random_forest", "extra_trees", "lightgbm"]

LGBM_PARAMS = {"n_estimators": 3000, "learning_rate": 0.03, "num_leaves": 31,
               "min_child_samples": 20, "subsample": 0.8, "subsample_freq": 1,
               "colsample_bytree": 0.8, "reg_lambda": 1.0}
LGBM_EARLY_STOPPING = 100
LGBM_HOLDOUT = 0.10


def model_params(name):
    """Hyperparameters recorded with each run (and hashed into the config hash)."""
    if name == "logreg":
        return {"penalty": "l2", "C": 1.0, "class_weight": "balanced", "max_iter": 2000,
                "imputer": "median", "scaler": "standard"}
    if name in ("random_forest", "extra_trees"):
        return {"n_estimators": 500, "class_weight": "balanced_subsample", "min_samples_leaf": 2,
                "imputer": "median"}
    if name == "lightgbm":
        return {**LGBM_PARAMS, "scale_pos_weight": "n_neg/n_pos per label",
                "early_stopping_rounds": LGBM_EARLY_STOPPING,
                "early_stopping_holdout": f"{LGBM_HOLDOUT:.0%} of train, grouped by patient"}
    raise ValueError(f"unknown model {name!r}")


def _make_pipeline(name, seed):
    if name == "logreg":
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("clf", LogisticRegression(C=1.0, class_weight="balanced",
                                                    max_iter=2000, random_state=seed))])
    forest = {"random_forest": RandomForestClassifier, "extra_trees": ExtraTreesClassifier}[name]
    return Pipeline([("impute", SimpleImputer(strategy="median")),
                     ("clf", forest(n_estimators=500, class_weight="balanced_subsample",
                                    min_samples_leaf=2, n_jobs=-1, random_state=seed))])


class ConstantClassifier:
    """Used when a label has no positives (or no negatives) in the training data."""

    def __init__(self, p):
        self.p = float(p)

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


class PerLabelClassifier:
    """Independent binary classifier per label. predict_proba returns (N, n_labels) scores."""

    def __init__(self, name, label_ids, seed=0):
        if name not in MODEL_NAMES:
            raise ValueError(f"unknown model {name!r}")
        self.name = name
        self.label_ids = list(label_ids)
        self.seed = seed
        self.estimators = []
        self.fit_info = {}

    def fit(self, X, Y, groups):
        """X: (N, F) float array. Y: (N, L) 0/1. groups: patient ids (for LightGBM's holdout)."""
        X = np.asarray(X, dtype=np.float32)
        self.estimators = []
        if self.name == "lightgbm":
            splitter = GroupShuffleSplit(n_splits=1, test_size=LGBM_HOLDOUT, random_state=self.seed)
            fit_idx, es_idx = next(splitter.split(X, groups=groups))
        for j, label in enumerate(self.label_ids):
            y = Y[:, j]
            n_pos = int(y.sum())
            if n_pos == 0 or n_pos == len(y):
                self.estimators.append(ConstantClassifier(y.mean()))
                self.fit_info[label] = {"constant": True}
                continue
            if self.name == "lightgbm":
                y_fit, y_es = y[fit_idx], y[es_idx]
                n_pos_fit = max(int(y_fit.sum()), 1)
                clf = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=self.seed, n_jobs=-1,
                                         verbose=-1,
                                         scale_pos_weight=(len(y_fit) - n_pos_fit) / n_pos_fit)
                if 0 < y_es.sum() < len(y_es):
                    clf.fit(X[fit_idx], y_fit, eval_X=(X[es_idx],), eval_y=(y_es,), eval_metric="auc",
                            callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False)])
                    self.fit_info[label] = {"best_iteration": int(clf.best_iteration_)}
                else:
                    clf.set_params(n_estimators=500)
                    clf.fit(X[fit_idx], y_fit)
                    self.fit_info[label] = {"best_iteration": None,
                                            "note": "no positives in holdout; fixed 500 trees"}
            else:
                clf = _make_pipeline(self.name, self.seed).fit(X, y)
            self.estimators.append(clf)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.column_stack([est.predict_proba(X)[:, 1] for est in self.estimators])
