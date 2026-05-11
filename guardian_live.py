"""
guardian_live.py
================
The REAL Guardian AI engine — connects to Wazuh and runs forever.

What this file does every 5 minutes:
  1. Reads the last 60 minutes of real Wazuh alerts from alerts.json
  2. Runs the AI prediction (feature extraction + RandomForest + GBR + IsolationForest)
  3. Writes the prediction result back into Wazuh as a custom alert
  4. Wazuh dashboard shows the AI result automatically

How to run:
  python3 guardian_live.py

How to run at startup (systemd handles this — see guardian.service):
  sudo systemctl start guardian
  sudo systemctl status guardian
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Setup logging ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [GUARDIAN]  %(levelname)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/var/ossec/logs/guardian_ai.log"),
    ]
)
logger = logging.getLogger("guardian.live")

# ── Paths ─────────────────────────────────────────────────────────────────────
ALERTS_JSON     = Path("/var/ossec/logs/alerts/alerts.json")
CUSTOM_ALERTS   = Path("/var/ossec/logs/alerts/guardian_predictions.json")
MODEL_PATH      = Path("/guardian/guardian_model.joblib")
GUARDIAN_DIR    = Path("/guardian")
LOOP_INTERVAL   = 300          # run every 5 minutes (300 seconds)
OBS_WINDOW_MIN  = 60           # read last 60 minutes of alerts

# ── Wazuh custom rule IDs for AI predictions ──────────────────────────────────
# These are in the range 100000+ which is reserved for custom local rules
RULE_ID_NORMAL   = 100001
RULE_ID_LOW      = 100002
RULE_ID_ELEVATED = 100003
RULE_ID_HIGH     = 100004
RULE_ID_CRITICAL = 100005

RISK_TO_RULE = {
    "NORMAL":   (RULE_ID_NORMAL,   3),
    "LOW":      (RULE_ID_LOW,      5),
    "ELEVATED": (RULE_ID_ELEVATED, 8),
    "HIGH":     (RULE_ID_HIGH,     11),
    "CRITICAL": (RULE_ID_CRITICAL, 14),
}

RISK_TO_EMOJI = {
    "NORMAL":   "🟢",
    "LOW":      "🔵",
    "ELEVATED": "🟡",
    "HIGH":     "🟠",
    "CRITICAL": "🔴",
}


# ── Load AI modules ───────────────────────────────────────────────────────────

def load_ai():
    """Load the AI model and set up the forecaster."""
    sys.path.insert(0, str(GUARDIAN_DIR))
    from feature_extractor import normalise_log
    from attack_correlator import AttackCorrelator
    from model import load_model
    from predictor import AttackForecaster

    if not MODEL_PATH.exists():
        logger.error("Model not found at %s — run train_guardian.py first", MODEL_PATH)
        sys.exit(1)

    bundle     = load_model(MODEL_PATH)
    correlator = AttackCorrelator(timeline_ttl_hours=12.0)
    forecaster = AttackForecaster(bundle, correlator=correlator, threshold=0.35)
    logger.info("AI model loaded successfully from %s", MODEL_PATH)
    return forecaster, normalise_log


# ── Read Wazuh alerts ─────────────────────────────────────────────────────────

def read_recent_alerts(normalise_log_fn, minutes: int = OBS_WINDOW_MIN) -> list[dict]:
    """
    Read the last `minutes` minutes of alerts from Wazuh alerts.json.
    Returns a list of normalised log dicts, sorted oldest → newest.
    """
    if not ALERTS_JSON.exists():
        logger.warning("alerts.json not found at %s", ALERTS_JSON)
        return []

    cutoff = datetime.now() - timedelta(minutes=minutes)
    logs   = []

    try:
        with open(ALERTS_JSON, "r", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    log = normalise_log_fn(raw)
                    if log["timestamp"] >= cutoff:
                        logs.append(log)
                except (json.JSONDecodeError, Exception):
                    continue
    except PermissionError:
        logger.error("Cannot read %s — run as root or add user to wazuh group", ALERTS_JSON)
        return []

    logs.sort(key=lambda l: l["timestamp"])
    logger.info("Read %d alerts from last %d minutes", len(logs), minutes)
    return logs


# ── Write prediction back to Wazuh ────────────────────────────────────────────

def write_prediction_to_wazuh(result) -> None:
    """
    Write the AI prediction as a JSON alert that Wazuh can index.
    Wazuh watches this file via a custom localfile configuration.
    """
    rule_id, level = RISK_TO_RULE.get(result.risk_level, (RULE_ID_NORMAL, 3))
    emoji = RISK_TO_EMOJI.get(result.risk_level, "🟢")

    # Build the alert payload — everything visible in the dashboard
    alert = {
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+0000"),
        "rule": {
            "id":          str(rule_id),
            "level":       level,
            "description": (
                f"{emoji} GUARDIAN AI — {result.risk_level} : "
                f"{result.known_attack_prediction.upper().replace('_', ' ')}"
            ),
            "groups":      ["guardian_ai", "ml_prediction", result.risk_level.lower()],
        },
        "agent": {
            "id":   "000",
            "name": "guardian-ai",
        },
        "manager": {
            "name": "wazuh-manager",
        },
        "guardian_ai": {
            # ── Core prediction ────────────────────────────────────────
            "risk_level":                result.risk_level,
            "attack_prediction":         result.known_attack_prediction,
            "attack_probability_pct":    f"{result.known_attack_probability * 100:.1f}%",
            "unknown_anomaly_risk_pct":  f"{result.unknown_attack_risk * 100:.1f}%",

            # ── Timing ────────────────────────────────────────────────
            "estimated_time_to_attack":  result.estimated_time_to_attack,
            "estimated_minutes":         result.estimated_minutes,

            # ── Kill-chain context ────────────────────────────────────
            "attack_stage":              result.attack_stage,
            "next_expected_stage":       result.next_expected_stage or "none",
            "stages_seen":               result.stages_seen,
            "campaign_risk_pct":         f"{result.campaign_risk * 100:.1f}%",

            # ── Confidence ────────────────────────────────────────────
            "confidence":                result.confidence_label,
            "confidence_score":          round(result.confidence, 3),

            # ── Evidence ──────────────────────────────────────────────
            "evidence_signals":          result.evidence_signals,
            "behavioral_anomalies":      result.behavioral_anomalies,
            "matched_patterns":          result.matched_historical_patterns,

            # ── Recommendation ────────────────────────────────────────
            "recommendation":            result.recommendation,

            # ── Top class probabilities ───────────────────────────────
            "top_attack_classes": {
                k: f"{v*100:.1f}%"
                for k, v in sorted(
                    result.all_class_probabilities.items(),
                    key=lambda x: -x[1]
                )[:5]
                if k != "no_attack"
            },
        },
        "full_log": result.recommendation,
        "location": "guardian-ai-engine",
    }

    # Append to the predictions log file (one JSON per line)
    try:
        with open(CUSTOM_ALERTS, "a") as f:
            f.write(json.dumps(alert) + "\n")
        logger.info(
            "%s Prediction written — risk=%s  type=%s  timing=%s",
            emoji,
            result.risk_level,
            result.known_attack_prediction,
            result.estimated_time_to_attack,
        )
    except Exception as e:
        logger.error("Failed to write prediction: %s", e)


# ── Print to terminal (human-readable) ───────────────────────────────────────

def print_result(result) -> None:
    emoji = RISK_TO_EMOJI.get(result.risk_level, "🟢")
    print("\n" + "═" * 65)
    print(f"  {emoji}  GUARDIAN AI PREDICTION  —  {datetime.now().strftime('%H:%M:%S')}")
    print("═" * 65)
    print(f"  Risk Level   : {result.risk_level}")
    print(f"  Prediction   : {result.known_attack_prediction.upper()}")
    print(f"  Probability  : {result.known_attack_probability * 100:.1f}%")
    print(f"  Anomaly Risk : {result.unknown_attack_risk * 100:.1f}%")
    print(f"  Timing       : {result.estimated_time_to_attack}")
    print(f"  Stage        : {result.attack_stage}"
          + (f" → {result.next_expected_stage}" if result.next_expected_stage else ""))
    print(f"  Confidence   : {result.confidence_label} ({result.confidence:.2f})")
    if result.evidence_signals:
        print(f"  Evidence     :")
        for s in result.evidence_signals[:4]:
            print(f"     • {s}")
    if result.behavioral_anomalies:
        print(f"  Anomalies    :")
        for a in result.behavioral_anomalies[:2]:
            print(f"     ⚠ {a}")
    print(f"  Action       : {result.recommendation}")
    print("═" * 65 + "\n")


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    logger.info("═" * 55)
    logger.info("  GUARDIAN AI ENGINE STARTING")
    logger.info("  Reading from : %s", ALERTS_JSON)
    logger.info("  Model        : %s", MODEL_PATH)
    logger.info("  Interval     : every %d seconds", LOOP_INTERVAL)
    logger.info("═" * 55)

    forecaster, normalise_log_fn = load_ai()

    cycle = 0
    while True:
        cycle += 1
        logger.info("── Cycle %d ──────────────────────────────────", cycle)

        try:
            # 1. Read real Wazuh alerts
            logs = read_recent_alerts(normalise_log_fn, minutes=OBS_WINDOW_MIN)

            if len(logs) < 3:
                logger.info("Not enough alerts yet (%d) — waiting for more data", len(logs))
            else:
                # 2. Run AI prediction
                result = forecaster.predict(logs)

                # 3. Print to terminal
                print_result(result)

                # 4. Write back to Wazuh
                write_prediction_to_wazuh(result)

        except Exception as e:
            logger.error("Prediction cycle failed: %s", e, exc_info=True)

        logger.info("Next prediction in %d seconds...", LOOP_INTERVAL)
        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main()
