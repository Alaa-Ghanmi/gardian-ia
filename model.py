"""
model.py
========
Two-model architecture for the Guardian Attack Forecasting Engine.

MODEL 1 — Known Attack Classifier (multi-class)
  Input:  WindowFeatures (both L1 and L2 layers)
  Target: attack_class  (one of 16 known classes + "no_attack")
  Algorithm: Random Forest with class_weight="balanced"
  Why Random Forest:
    - Handles the mix of L1 (count-heavy) and L2 (entropy/rate) features
      without needing feature normalization within trees.
    - Feature importances are directly interpretable.
    - Robust to irrelevant features (many L2 features may not fire for
      Layer 1 known attacks).
  What it learns:
    Given the behavioral signature in the past window, what type of attack
    is most likely to materialize next?
  What it CANNOT learn:
    Timing. It knows "brute force is coming" but not when.

MODEL 2 — Time-to-Attack Regressor
  Input:  WindowFeatures (only samples where will_attack=True)
  Target: minutes_to_attack (continuous, bounded to [10, 240])
  Algorithm: Gradient Boosted Regression (GBR)
  Why GBR:
    - Regression on skewed distributions (attack timings are not normally
      distributed) benefits from boosting's bias reduction.
    - GBR naturally provides uncertainty through quantile regression
      (future work: use QuantileRegressor wrapper).
  What it learns:
    Given the current feature signature, how many minutes until the
    attack materialises?
  Limitations:
    - Training data is synthetic; real attack timings vary enormously.
    - Without enough real incident data the regressor will underfit.
      Treat outputs as order-of-magnitude estimates, not precise timers.
    - Confidence intervals are approximated via bootstrap; a proper
      conformal prediction interval would be more calibrated.

ANOMALY LAYER — Layer 2 IsolationForest
  Input:  L2 features only
  Purpose: Detect behavioral anomalies even when no known attack class fires.
  Produces: anomaly_score in [-1, 1] where negative = more anomalous.
  Why IsolationForest:
    - Unsupervised: does not need labeled attack data.
    - Fast at inference time.
    - Naturally handles high-dimensional sparse feature spaces.
  What it CANNOT do:
    It cannot tell you WHAT the anomaly is, only that something unusual
    is happening. The predictor uses this as a supplementary risk signal.

Training strategy
-----------------
1. Validate inputs
2. Balance classes via stratified undersampling (max 4:1 ratio)
3. Stratified 80/20 train/test split (stratified on attack_class)
4. StandardScaler fit on train, applied to test
5. Random Forest classifier trained on all features
6. GBR regressor trained ONLY on attack-positive samples
7. IsolationForest trained on L2 features of NORMAL samples
   (contamination parameter tuned on attack prevalence in dataset)
8. Evaluate: accuracy, per-class precision/recall, confusion matrix
9. Return ModelBundle with all three models + metadata

Persistence
-----------
All models, scalers, and metadata are bundled in a single joblib file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingRegressor,
    IsolationForest,
)
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import resample

from feature_extractor import WindowFeatures
from temporal_builder import TrainingSample

logger = logging.getLogger("guardian.model")

# ---------------------------------------------------------------------------
# Index ranges into the feature vector for L1 vs L2 features
# ---------------------------------------------------------------------------

def _l1_indices(feature_names: list[str]) -> list[int]:
    return [i for i, n in enumerate(feature_names) if n.startswith("L1_")]

def _l2_indices(feature_names: list[str]) -> list[int]:
    return [i for i, n in enumerate(feature_names) if n.startswith("L2_")]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ClassifierEvaluation:
    accuracy:      float
    macro_f1:      float
    per_class:     dict   # class_name → {precision, recall, f1, support}
    confusion_mat: list   # raw confusion matrix as nested list
    n_train:       int
    n_test:        int


@dataclass
class RegressorEvaluation:
    mae_minutes:  float
    r2:           float
    n_samples:    int


@dataclass
class ModelBundle:
    """
    Everything needed to run inference — saved as a single joblib file.

    Attributes
    ----------
    classifier    : predicts attack_class (multi-class RF)
    regressor     : predicts minutes_to_attack (GBR, may be None if
                    insufficient positive samples)
    anomaly_model : IsolationForest for behavioral anomaly scoring
    scaler_clf    : StandardScaler fitted on classifier training data
    scaler_reg    : StandardScaler fitted on regressor training data
    scaler_anom   : StandardScaler for anomaly model (L2 features only)
    feature_names : ordered list of feature names
    class_names   : ordered list of attack class labels
    label_encoder : fitted LabelEncoder for class_names
    clf_eval      : classifier evaluation results
    reg_eval      : regressor evaluation results (None if no regressor)
    l1_indices    : feature indices belonging to Layer 1
    l2_indices    : feature indices belonging to Layer 2
    """
    classifier:    RandomForestClassifier
    regressor:     Optional[GradientBoostingRegressor]
    anomaly_model: IsolationForest
    scaler_clf:    StandardScaler
    scaler_reg:    Optional[StandardScaler]
    scaler_anom:   StandardScaler
    feature_names: list[str]
    class_names:   list[str]
    label_encoder: LabelEncoder
    clf_eval:      ClassifierEvaluation
    reg_eval:      Optional[RegressorEvaluation]
    l1_indices:    list[int]
    l2_indices:    list[int]


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class ForecastModelTrainer:
    """
    Trains the three-component Guardian model from TrainingSample objects.

    Parameters
    ----------
    n_estimators       : RF trees
    max_depth_clf      : RF max depth
    max_imbalance      : majority/minority cap before undersampling
    test_size          : held-out fraction for evaluation
    gbr_n_estimators   : GBR trees for time-to-attack regression
    contamination      : IsolationForest contamination estimate
    """

    def __init__(
        self,
        n_estimators:     int   = 300,
        max_depth_clf:    int   = 15,
        max_imbalance:    float = 4.0,
        test_size:        float = 0.20,
        gbr_n_estimators: int   = 200,
        contamination:    float = 0.1,
    ) -> None:
        self.n_estimators     = n_estimators
        self.max_depth_clf    = max_depth_clf
        self.max_imbalance    = max_imbalance
        self.test_size        = test_size
        self.gbr_n_estimators = gbr_n_estimators
        self.contamination    = contamination

    def train(self, samples: Sequence[TrainingSample]) -> ModelBundle:
        if len(samples) < 20:
            raise ValueError("Need at least 20 training samples.")

        feature_names = WindowFeatures.feature_names()
        l1_idx = _l1_indices(feature_names)
        l2_idx = _l2_indices(feature_names)

        X_all = pd.DataFrame(
            [s.features.to_list() for s in samples],
            columns=feature_names,
        )
        y_class  = pd.Series([s.label.attack_class  for s in samples])
        y_binary = pd.Series([s.label.binary_label   for s in samples])
        y_time   = pd.Series([s.label.minutes_to_attack for s in samples])

        # ── Encode class labels ───────────────────────────────────────────
        le = LabelEncoder()
        le.fit(y_class)
        y_enc = pd.Series(le.transform(y_class))

        # ── Balance classes ───────────────────────────────────────────────
        X_bal, y_enc_bal, y_time_bal, y_bin_bal = self._balance(
            X_all, y_enc, y_time, y_binary
        )

        # ── Train/test split (stratified on class) ────────────────────────
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_bal, y_enc_bal,
            test_size=self.test_size,
            random_state=42,
            stratify=y_enc_bal,
        )

        # ── Scale ─────────────────────────────────────────────────────────
        scaler_clf = StandardScaler()
        X_tr_s     = scaler_clf.fit_transform(X_tr.fillna(0))
        X_te_s     = scaler_clf.transform(X_te.fillna(0))

        # ── Classifier ────────────────────────────────────────────────────
        clf = RandomForestClassifier(
            n_estimators     = self.n_estimators,
            max_depth        = self.max_depth_clf,
            min_samples_leaf = 3,
            class_weight     = "balanced",
            random_state     = 42,
            n_jobs           = -1,
        )
        clf.fit(X_tr_s, y_tr)

        clf_eval = self._eval_classifier(
            clf, X_te_s, y_te, le, len(X_tr), len(X_te)
        )
        logger.info("Classifier: accuracy=%.1f%% macro_f1=%.1f%%",
                    clf_eval.accuracy * 100, clf_eval.macro_f1 * 100)

        # ── Time-to-attack regressor ──────────────────────────────────────
        # Only trained on attack-positive samples
        pos_mask = y_bin_bal == 1
        X_pos    = X_bal[pos_mask].reset_index(drop=True)
        y_pos    = y_time_bal[pos_mask].reset_index(drop=True)

        regressor  = None
        scaler_reg = None
        reg_eval   = None

        if len(X_pos) >= 15:
            X_pos_tr, X_pos_te, y_pos_tr, y_pos_te = train_test_split(
                X_pos, y_pos, test_size=self.test_size, random_state=42
            )
            scaler_reg = StandardScaler()
            X_pos_tr_s = scaler_reg.fit_transform(X_pos_tr.fillna(0))
            X_pos_te_s = scaler_reg.transform(X_pos_te.fillna(0))

            regressor = GradientBoostingRegressor(
                n_estimators  = self.gbr_n_estimators,
                max_depth     = 5,
                learning_rate = 0.05,
                subsample     = 0.8,
                random_state  = 42,
            )
            regressor.fit(X_pos_tr_s, y_pos_tr)
            y_pred_reg = np.clip(regressor.predict(X_pos_te_s), 1, 480)
            reg_eval   = RegressorEvaluation(
                mae_minutes = float(mean_absolute_error(y_pos_te, y_pred_reg)),
                r2          = float(r2_score(y_pos_te, y_pred_reg)),
                n_samples   = len(X_pos_tr),
            )
            logger.info("Regressor: MAE=%.1f min  R²=%.3f  n=%d",
                        reg_eval.mae_minutes, reg_eval.r2, reg_eval.n_samples)
        else:
            logger.warning(
                "Insufficient positive samples (%d) for regression model. "
                "Time-to-attack estimates will be unavailable.", len(X_pos)
            )

        # ── Anomaly model (L2 features, normal samples only) ─────────────
        neg_mask   = y_bin_bal == 0
        X_normal   = X_bal[neg_mask].reset_index(drop=True)
        X_l2_cols  = [feature_names[i] for i in l2_idx]

        scaler_anom = StandardScaler()

        if len(X_normal) >= 10 and X_l2_cols:
            X_l2_normal = scaler_anom.fit_transform(X_normal[X_l2_cols].fillna(0))
        else:
            X_l2_normal = scaler_anom.fit_transform(X_bal[X_l2_cols].fillna(0))

        iso = IsolationForest(
            n_estimators  = 200,
            contamination = self.contamination,
            random_state  = 42,
            n_jobs        = -1,
        )
        iso.fit(X_l2_normal)
        logger.info("Anomaly model trained on %d normal samples.", len(X_l2_normal))

        # ── Top features ──────────────────────────────────────────────────
        top5 = sorted(
            zip(feature_names, clf.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )[:5]
        logger.info("Top-5 predictive features: %s",
                    ", ".join(f"{n}={v:.3f}" for n, v in top5))

        return ModelBundle(
            classifier    = clf,
            regressor     = regressor,
            anomaly_model = iso,
            scaler_clf    = scaler_clf,
            scaler_reg    = scaler_reg,
            scaler_anom   = scaler_anom,
            feature_names = feature_names,
            class_names   = list(le.classes_),
            label_encoder = le,
            clf_eval      = clf_eval,
            reg_eval      = reg_eval,
            l1_indices    = l1_idx,
            l2_indices    = l2_idx,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _balance(self, X, y_enc, y_time, y_bin):
        counts = y_enc.value_counts()
        minority_n = counts.min()
        cap = int(minority_n * self.max_imbalance)

        keep_idx = []
        for cls in counts.index:
            cls_idx = y_enc[y_enc == cls].index.tolist()
            if len(cls_idx) > cap:
                cls_idx = resample(
                    cls_idx, n_samples=cap, random_state=42, replace=False
                )
            keep_idx.extend(cls_idx)

        X_bal      = X.loc[keep_idx].reset_index(drop=True)
        y_enc_bal  = y_enc.loc[keep_idx].reset_index(drop=True)
        y_time_bal = y_time.loc[keep_idx].reset_index(drop=True)
        y_bin_bal  = y_bin.loc[keep_idx].reset_index(drop=True)
        return X_bal, y_enc_bal, y_time_bal, y_bin_bal

    @staticmethod
    def _eval_classifier(clf, X_te, y_te, le, n_tr, n_te) -> ClassifierEvaluation:
        y_pred = clf.predict(X_te)
        report = classification_report(
            y_te, y_pred,
            target_names=le.classes_,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_te, y_pred)
        per_class = {
            name: {
                "precision": round(report[name]["precision"], 3),
                "recall":    round(report[name]["recall"],    3),
                "f1":        round(report[name]["f1-score"],  3),
                "support":   int(report[name]["support"]),
            }
            for name in le.classes_ if name in report
        }
        return ClassifierEvaluation(
            accuracy      = float(report.get("accuracy", 0)),
            macro_f1      = float(report.get("macro avg", {}).get("f1-score", 0)),
            per_class     = per_class,
            confusion_mat = cm.tolist(),
            n_train       = n_tr,
            n_test        = n_te,
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(bundle: ModelBundle, path: str | Path) -> None:
    joblib.dump(bundle, Path(path))
    logger.info("Model bundle saved → %s", path)


def load_model(path: str | Path) -> ModelBundle:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No model bundle at {p}")
    bundle = joblib.load(p)
    logger.info("Model bundle loaded ← %s", path)
    return bundle
