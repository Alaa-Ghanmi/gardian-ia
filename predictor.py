"""
predictor.py
============
Given a trained ModelBundle, a current window of past logs, and an optional
AttackCorrelator context, returns a richly structured forecast.

Output structure
----------------
{
  "known_attack_prediction":   str,    # top predicted attack class
  "known_attack_probability":  float,  # P(that class) from RF
  "all_class_probabilities":   dict,   # full class probability vector
  "unknown_attack_risk":       float,  # IsolationForest anomaly signal [0,1]
  "estimated_time_to_attack":  str,    # e.g. "~42 min ± 18 min" or "N/A"
  "estimated_minutes":         float | None,
  "attack_stage":              str,    # current kill-chain stage
  "next_expected_stage":       str | None,
  "confidence":                float,  # tree agreement [0,1]
  "confidence_label":          str,    # HIGH / MEDIUM / LOW
  "evidence_signals":          list,   # L1/L2 driven human-readable signals
  "behavioral_anomalies":      list,   # anomaly-specific signals
  "matched_historical_patterns": list, # from correlator
  "recommendation":            str,    # actionable SOC instruction
  "feature_values":            dict,   # full audit trail
}

Design decisions
----------------
- The known_attack_probability threshold for "will_attack" is 0.35,
  deliberately low. A SOC analyst would rather investigate a false positive
  than miss a real attack. Make this configurable per deployment.
- Uncertainty is expressed explicitly:
  * confidence_label reflects how much the RF trees agreed.
  * estimated_time_to_attack includes ± uncertainty from the GBR regressor.
  * When confidence is LOW, the recommendation says so explicitly.
- The anomaly score from IsolationForest is NOT used to trigger the main
  attack prediction — it is a SUPPLEMENTARY signal. Mixing supervised and
  unsupervised outputs without calibration produces unreliable probabilities.

What this system genuinely cannot do
-------------------------------------
- It cannot detect zero-day exploits that leave no trace in Wazuh logs.
- It cannot predict the exact minute of an attack; timing estimates are
  order-of-magnitude guidance.
- It cannot infer attacker intent beyond what Wazuh and behavior reveal.
- It is not a substitute for threat intelligence feeds (IOC matching,
  CVE correlation). These should run alongside, not instead of, this system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd

from feature_extractor import WindowFeatureExtractor, WindowFeatures, normalise_log
from model import ModelBundle
from attack_correlator import AttackCorrelator, CorrelationResult

logger = logging.getLogger("guardian.predictor")

# Probability threshold above which we flag an attack
ATTACK_PROB_THRESHOLD = 0.35

# Anomaly score threshold (IsolationForest decision_function output)
# More negative = more anomalous. Typical range [-0.5, 0.3]
ANOMALY_THRESHOLD = -0.05


@dataclass
class ForecastResult:
    # Layer 1 output
    known_attack_prediction:    str
    known_attack_probability:   float
    all_class_probabilities:    dict[str, float]

    # Layer 2 output
    unknown_attack_risk:        float     # 0 (normal) to 1 (highly anomalous)
    anomaly_score_raw:          float     # raw IsolationForest score

    # Timing estimate
    estimated_minutes:          Optional[float]
    estimated_minutes_std:      Optional[float]
    estimated_time_to_attack:   str       # human-readable

    # Kill-chain context
    attack_stage:               str
    next_expected_stage:        Optional[str]
    stages_seen:                list[str]
    campaign_risk:              float

    # Overall risk
    will_attack:                bool
    risk_level:                 str       # CRITICAL / HIGH / ELEVATED / LOW / NORMAL
    confidence:                 float
    confidence_label:           str

    # Evidence
    evidence_signals:           list[str]
    behavioral_anomalies:       list[str]
    matched_historical_patterns: list[str]

    # Recommendation
    recommendation:             str

    # Audit trail
    feature_values:             dict[str, float]
    prediction_timestamp:       str


class AttackForecaster:
    """
    Runs the full two-layer forecast pipeline on a window of past logs.

    Parameters
    ----------
    bundle      : trained ModelBundle from model.load_model()
    correlator  : optional AttackCorrelator for session context.
                  If None, correlation features will be absent from output.
    threshold   : probability above which known_attack_prediction is
                  considered an active threat. Default 0.35.
    """

    def __init__(
        self,
        bundle:     ModelBundle,
        correlator: Optional[AttackCorrelator] = None,
        threshold:  float = ATTACK_PROB_THRESHOLD,
    ) -> None:
        self.bundle     = bundle
        self.correlator = correlator
        self.threshold  = threshold
        self._extractor = WindowFeatureExtractor()

    def predict(self, past_logs: list[dict]) -> ForecastResult:
        """
        Forecast from a list of normalised log dicts covering the
        past observation window (oldest → newest, all BEFORE prediction point).

        Parameters
        ----------
        past_logs : output of normalise_log(), sorted by timestamp.

        Returns
        -------
        ForecastResult — the complete structured forecast.
        """
        if not past_logs:
            return self._empty_result()

        # ── Feature extraction ────────────────────────────────────────────
        features = self._extractor.extract(past_logs)
        fv       = dict(zip(self.bundle.feature_names, features.to_list()))

        X = pd.DataFrame([features.to_list()], columns=self.bundle.feature_names)
        X_s = self.bundle.scaler_clf.transform(X.fillna(0))

        # ── Layer 1: Known attack classification ──────────────────────────
        proba_vec  = self.bundle.classifier.predict_proba(X_s)[0]
        class_proba = {
            cls: round(float(p), 4)
            for cls, p in zip(self.bundle.class_names, proba_vec)
        }

        # Remove "no_attack" from the ranked attack predictions
        attack_proba = {k: v for k, v in class_proba.items() if k != "no_attack"}
        top_class    = max(attack_proba, key=attack_proba.__getitem__, default="no_attack")
        top_prob     = attack_proba.get(top_class, 0.0)

        no_attack_prob = class_proba.get("no_attack", 0.0)
        will_attack    = (1.0 - no_attack_prob) >= self.threshold

        # ── Confidence (RF tree agreement) ────────────────────────────────
        confidence, conf_label = self._tree_confidence(X_s)

        # ── Layer 2: Behavioral anomaly ───────────────────────────────────
        l2_cols = [self.bundle.feature_names[i] for i in self.bundle.l2_indices]
        X_l2    = X[l2_cols].fillna(0)
        X_l2_s  = self.bundle.scaler_anom.transform(X_l2)
        anom_score_raw = float(self.bundle.anomaly_model.decision_function(X_l2_s)[0])
        # Convert to 0–1 risk (more negative raw score → higher risk)
        unknown_risk = min(1.0, max(0.0, (-anom_score_raw + 0.15) / 0.65))

        # ── Time-to-attack estimate ───────────────────────────────────────
        est_min = None
        est_std = None
        if will_attack and self.bundle.regressor and self.bundle.scaler_reg:
            X_reg = self.bundle.scaler_reg.transform(X.fillna(0))
            est_raw = float(self.bundle.regressor.predict(X_reg)[0])
            est_min = float(np.clip(est_raw, 5.0, 480.0))
            # Bootstrap-estimated uncertainty — without a proper quantile regressor
            # we use a heuristic: ±30% of the estimate, min 10 min
            est_std = max(10.0, est_min * 0.30)

        time_str = self._format_time(est_min, est_std)

        # ── Correlation ───────────────────────────────────────────────────
        corr: Optional[CorrelationResult] = None
        if self.correlator:
            self.correlator.ingest_window(past_logs)
            corr = self.correlator.correlate_window(past_logs, features)

        attack_stage      = corr.current_stage       if corr else "unknown"
        next_stage        = corr.next_expected_stage if corr else None
        stages_seen       = corr.stages_seen         if corr else []
        campaign_risk     = corr.campaign_risk       if corr else 0.0
        matched_patterns  = (
            [corr.matched_pattern.description] if (corr and corr.matched_pattern) else []
        )

        # Use correlation timing if regressor isn't available
        if est_min is None and corr and corr.estimated_minutes:
            est_min = corr.estimated_minutes
            est_std = corr.estimated_minutes_std
            time_str = self._format_time(est_min, est_std)

        # ── Evidence signals ──────────────────────────────────────────────
        evidence   = self._l1_signals(fv)
        anomalies  = self._l2_signals(fv, anom_score_raw)
        if corr:
            evidence.extend(corr.evidence)

        # ── Risk level ────────────────────────────────────────────────────
        combined_risk = max(
            top_prob * (1 - no_attack_prob),
            unknown_risk * 0.5,
            campaign_risk * 0.7,
        )
        risk_level = self._risk_level(combined_risk, conf_label)

        # ── Recommendation ────────────────────────────────────────────────
        rec = self._recommendation(
            top_class, combined_risk, fv, attack_stage, time_str, conf_label
        )

        result = ForecastResult(
            known_attack_prediction   = top_class if will_attack else "no_attack",
            known_attack_probability  = round(top_prob, 4),
            all_class_probabilities   = class_proba,
            unknown_attack_risk       = round(unknown_risk, 3),
            anomaly_score_raw         = round(anom_score_raw, 4),
            estimated_minutes         = round(est_min, 1) if est_min else None,
            estimated_minutes_std     = round(est_std, 1) if est_std else None,
            estimated_time_to_attack  = time_str,
            attack_stage              = attack_stage,
            next_expected_stage       = next_stage,
            stages_seen               = stages_seen,
            campaign_risk             = round(campaign_risk, 3),
            will_attack               = will_attack,
            risk_level                = risk_level,
            confidence                = confidence,
            confidence_label          = conf_label,
            evidence_signals          = evidence[:8],
            behavioral_anomalies      = anomalies[:5],
            matched_historical_patterns = matched_patterns,
            recommendation            = rec,
            feature_values            = {k: round(v, 3) for k, v in fv.items()},
            prediction_timestamp      = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )

        lvl = "warning" if will_attack else "debug"
        getattr(logger, lvl)(
            "Forecast: class=%s prob=%.2f anomaly_risk=%.2f stage=%s time=%s",
            result.known_attack_prediction,
            result.known_attack_probability,
            result.unknown_attack_risk,
            result.attack_stage,
            result.estimated_time_to_attack,
        )
        return result

    # ── Internal helpers ──────────────────────────────────────────────────

    def _tree_confidence(self, X_scaled) -> tuple[float, str]:
        tree_probas = np.array([
            t.predict_proba(X_scaled)[0].max()
            for t in self.bundle.classifier.estimators_
        ])
        std   = float(np.std(tree_probas))
        score = round(max(0.0, min(1.0, 1.0 - std * 2.5)), 3)
        label = "HIGH" if score >= 0.75 else ("MEDIUM" if score >= 0.45 else "LOW")
        return score, label

    @staticmethod
    def _format_time(est_min: Optional[float], est_std: Optional[float]) -> str:
        if est_min is None:
            return "N/A (insufficient data)"
        std_str = f" ± {est_std:.0f} min" if est_std else ""
        if est_min < 60:
            return f"~{est_min:.0f} min{std_str}"
        hours = est_min / 60.0
        return f"~{hours:.1f} h{std_str}"

    @staticmethod
    def _risk_level(combined: float, conf: str) -> str:
        if combined >= 0.75:
            return "CRITICAL"
        if combined >= 0.55:
            return "HIGH"
        if combined >= 0.35:
            return "ELEVATED"
        if combined >= 0.15:
            return "LOW"
        return "NORMAL"

    @staticmethod
    def _l1_signals(fv: dict) -> list[str]:
        """Layer 1 — known attack class signals."""
        sigs = []
        checks = [
            ("L1_brute_force_count",      3,  "Brute-force events in window"),
            ("L1_priv_escalation_count",  1,  "Privilege escalation events"),
            ("L1_lateral_movement_count", 1,  "Lateral movement events"),
            ("L1_recon_count",            2,  "Reconnaissance events"),
            ("L1_exfiltration_count",     1,  "Exfiltration events"),
            ("L1_malware_count",          1,  "Malware events"),
            ("L1_ransomware_count",       1,  "Ransomware events"),
            ("L1_web_shell_count",        1,  "Web shell events"),
            ("L1_suspicious_exec_count",  1,  "Suspicious execution events"),
            ("L1_c2_count",              1,   "C2 communication events"),
            ("L1_persistence_count",      1,  "Persistence mechanism events"),
        ]
        for key, threshold, label in checks:
            val = fv.get(key, 0)
            if val >= threshold:
                sigs.append(f"{label}: {int(val)}")

        if fv.get("L1_auth_fail_count", 0) >= 5:
            sigs.append(f"Auth failures: {int(fv['L1_auth_fail_count'])} "
                        f"(rate: {fv.get('L1_auth_fail_rate', 0):.2f}/min)")
        if fv.get("L1_fail_to_success_ratio", 0) >= 10:
            sigs.append(f"Fail-to-success ratio: {fv['L1_fail_to_success_ratio']:.0f}:1 — brute force indicator")
        if fv.get("L1_auth_fail_trend", 0) > 3:
            sigs.append(f"Auth failures ACCELERATING: +{fv['L1_auth_fail_trend']:.0f} in recent period")
        if fv.get("L1_mitre_tactic_diversity", 0) >= 3:
            sigs.append(f"Multi-tactic campaign: {int(fv['L1_mitre_tactic_diversity'])} MITRE ATT&CK tactics observed")
        return sigs

    @staticmethod
    def _l2_signals(fv: dict, anom_score: float) -> list[str]:
        """Layer 2 — behavioral anomaly signals."""
        sigs = []
        if anom_score < ANOMALY_THRESHOLD:
            sigs.append(f"Behavioral anomaly detected (score={anom_score:.3f}) — pattern deviates from baseline")
        if fv.get("L2_burst_score", 0) >= 3.0:
            sigs.append(f"Activity burst: {fv['L2_burst_score']:.1f}× above rolling average")
        if fv.get("L2_ip_entropy", 0) >= 2.5:
            sigs.append(f"High IP diversity (entropy={fv['L2_ip_entropy']:.2f}) — distributed source attack possible")
        if fv.get("L2_external_ip_count", 0) >= 3:
            sigs.append(f"External IP involvement: {int(fv['L2_external_ip_count'])} events from non-RFC-1918 IPs")
        if fv.get("L2_suspicious_cmd_count", 0) >= 1:
            sigs.append(f"Suspicious command signatures: {int(fv['L2_suspicious_cmd_count'])} events")
        if fv.get("L2_is_night"):
            sigs.append("Activity during night hours (22:00–07:00) — statistically anomalous")
        elif fv.get("L2_is_off_hours"):
            sigs.append("Activity outside business hours (before 08:00 or after 19:00)")
        if fv.get("L2_high_risk_port_count", 0) >= 3:
            sigs.append(f"High-risk port access: {int(fv['L2_high_risk_port_count'])} events on ports 22/3389/445/etc")
        if fv.get("L2_volume_trend", 0) >= 50:
            sigs.append(f"Data volume ESCALATING: +{fv['L2_volume_trend']:.0f} MB in recent period")
        return sigs

    @staticmethod
    def _recommendation(
        attack_class: str,
        risk: float,
        fv: dict,
        stage: str,
        time_str: str,
        conf: str,
    ) -> str:
        uncertainty = " [LOW confidence — treat as a weak signal]" if conf == "LOW" else ""

        if risk >= 0.75:
            class_actions = {
                "brute_force":         "Block offending IP range, enforce MFA, check for successful logins.",
                "credential_stuffing": "Force password reset for targeted accounts, enable MFA immediately.",
                "priv_escalation":     "Revoke escalated privileges, audit sudo/su logs, check for new root sessions.",
                "lateral_movement":    "Isolate affected hosts, rotate credentials, audit SMB/RDP connections.",
                "exfiltration":        "Block outbound transfers from affected hosts, preserve logs for forensics.",
                "malware":             "Quarantine affected host, run EDR scan, collect memory dump.",
                "ransomware":          "ISOLATE ALL AFFECTED HOSTS IMMEDIATELY. Engage IR team. Do not pay without authorization.",
                "web_shell":           "Take web server offline, audit web root for suspicious files, patch CMS.",
                "c2":                  "Block C2 domain/IP, isolate beaconing hosts, engage threat intel team.",
                "persistence":         "Audit startup items, scheduled tasks, and registry run keys on affected hosts.",
            }
            action = class_actions.get(attack_class, "Escalate to SOC Tier 2 immediately and begin incident investigation.")
            return f"CRITICAL — {attack_class.upper()} attack predicted in {time_str}. {action}{uncertainty}"

        if risk >= 0.55:
            return (f"HIGH RISK — {attack_class} activity detected (stage: {stage}). "
                    f"Increase monitoring on affected entities. "
                    f"Review auth and file access logs for the past 2 hours.{uncertainty}")

        if risk >= 0.35:
            return (f"ELEVATED RISK — Early {attack_class} signals (stage: {stage}). "
                    f"Monitor closely. Check next cycle in 15 minutes.{uncertainty}")

        if risk >= 0.15:
            return f"LOW RISK — Weak signals, likely benign. Routine monitoring sufficient.{uncertainty}"

        return "NORMAL — No anomalous patterns detected. Continue standard monitoring."

    def _empty_result(self) -> ForecastResult:
        return ForecastResult(
            known_attack_prediction   = "no_attack",
            known_attack_probability  = 0.0,
            all_class_probabilities   = {},
            unknown_attack_risk       = 0.0,
            anomaly_score_raw         = 0.0,
            estimated_minutes         = None,
            estimated_minutes_std     = None,
            estimated_time_to_attack  = "N/A",
            attack_stage              = "none",
            next_expected_stage       = None,
            stages_seen               = [],
            campaign_risk             = 0.0,
            will_attack               = False,
            risk_level                = "NORMAL",
            confidence                = 0.0,
            confidence_label          = "LOW",
            evidence_signals          = ["No logs provided."],
            behavioral_anomalies      = [],
            matched_historical_patterns = [],
            recommendation            = "No logs provided. Cannot forecast.",
            feature_values            = {},
            prediction_timestamp      = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
