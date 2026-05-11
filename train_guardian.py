"""
train_guardian.py
=================
One-time training script — reads REAL Wazuh alerts and trains the AI model.

Run this ONCE before starting guardian_live.py.
After training, guardian_model.joblib is saved and the live engine uses it.

If you have no attack history yet in your alerts.json, the script falls back
to synthetic data automatically so the model always exists.

Run with:
    sudo python3 train_guardian.py
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [TRAIN]  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("guardian.train")

ALERTS_JSON  = Path("/var/ossec/logs/alerts/alerts.json")
GUARDIAN_DIR = Path("/guardian")
MODEL_PATH   = GUARDIAN_DIR / "guardian_model.joblib"

sys.path.insert(0, str(GUARDIAN_DIR))

from feature_extractor import normalise_log
from temporal_builder import AttackSessionReconstructor, TemporalSampleBuilder
from model import ForecastModelTrainer, save_model


def read_all_alerts() -> list[dict]:
    """Read ALL alerts from alerts.json (not just last hour) for training."""
    if not ALERTS_JSON.exists():
        logger.warning("alerts.json not found — will use synthetic data")
        return []

    logs = []
    errors = 0
    with open(ALERTS_JSON, "r", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                log = normalise_log(raw)
                logs.append(log)
            except Exception:
                errors += 1
                continue

    logs.sort(key=lambda l: l["timestamp"])
    logger.info("Read %d alerts from alerts.json (%d parse errors ignored)", len(logs), errors)
    return logs


def train_on_real_data(logs: list[dict]):
    """Train using real Wazuh alert history."""
    logger.info("Reconstructing attack sessions from real alerts...")
    reconstructor = AttackSessionReconstructor()
    sessions      = reconstructor.reconstruct(logs)
    logger.info("Found %d attack sessions", len(sessions))

    if len(sessions) < 2:
        logger.warning(
            "Only %d attack sessions found in real data. "
            "Need more attack history for good training. "
            "Falling back to synthetic data.", len(sessions)
        )
        return None

    logger.info("Building temporal training samples...")
    builder = TemporalSampleBuilder(
        obs_window_min      = 60,
        min_lead_min        = 10,
        max_lead_min        = 120,
        samples_per_session = 8,
        quiet_buffer_hrs    = 2.0,
    )
    samples = builder.build(logs, sessions)

    if len(samples) < 20:
        logger.warning("Only %d samples — falling back to synthetic data.", len(samples))
        return None

    pos = sum(1 for s in samples if s.label.will_attack)
    logger.info("Training on %d samples (%d positive, %d negative)", len(samples), pos, len(samples) - pos)
    return samples


def train_on_synthetic_data():
    """
    Fallback: generate synthetic logs and train on them.
    This produces a working model even with no real attack history.
    The model improves automatically as real attacks accumulate.
    """
    logger.info("Using synthetic training data as fallback...")
    import random
    from datetime import datetime, timedelta
    from temporal_builder import AttackSessionReconstructor, TemporalSampleBuilder

    random.seed(42)
    rng  = random.Random(42)
    base = datetime(2024, 1, 1, 8, 0, 0)
    logs = []

    attack_offset = 300

    # Brute force sessions
    for i in range(15):
        T = base + timedelta(hours=attack_offset + i * 12)
        ip = f"198.51.{rng.randint(100,200)}.{rng.randint(1,255)}"
        for m in range(28, 25, -1):
            logs.append(normalise_log({
                "timestamp": (T - timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M:%S"),
                "rule_level": 6, "rule_desc": "port scan",
                "rule_groups": ["recon", "port_scan"], "mitre_tactics": ["reconnaissance"],
                "src_ip": ip, "dst_port": 22, "src_country": "ru",
            }))
        for m in range(20, 0, -1):
            logs.append(normalise_log({
                "timestamp": (T - timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M:%S"),
                "rule_level": rng.randint(7,10), "rule_desc": "sshd failed password",
                "rule_groups": ["authentication_failed", "sshd"],
                "mitre_tactics": ["initial-access"], "src_ip": ip, "dst_port": 22,
            }))
        logs.append(normalise_log({
            "timestamp": T.strftime("%Y-%m-%d %H:%M:%S"),
            "rule_level": 12, "rule_desc": "accepted password for root",
            "rule_groups": ["sshd"], "src_ip": ip, "is_login_success": True,
        }))

    # Exfiltration sessions
    for i in range(10):
        T = base + timedelta(hours=attack_offset + 600 + i * 8)
        T = T.replace(hour=rng.choice([22, 23, 0]))
        for m in range(50, 30, -2):
            logs.append(normalise_log({
                "timestamp": (T - timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M:%S"),
                "rule_level": 7, "rule_desc": "large file read from restricted directory",
                "rule_groups": ["syscheck"], "mitre_tactics": ["collection"],
                "src_ip": "10.0.3.55", "file_path": "/finance/q4.csv",
                "file_count": rng.randint(5, 20), "username": "mallory",
            }))
        for m in range(0, 30, 3):
            logs.append(normalise_log({
                "timestamp": (T + timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M:%S"),
                "rule_level": 13, "rule_desc": "mass data transfer",
                "rule_groups": ["data_exfiltration"],
                "mitre_tactics": ["exfiltration"],
                "src_ip": "10.0.3.55", "file_path": "/finance/export_all.csv",
                "file_count": rng.randint(100, 400),
            }))

    # Malware/ransomware sessions
    for i in range(10):
        T = base + timedelta(hours=attack_offset + 900 + i * 8)
        ip = f"192.168.1.{rng.randint(200,250)}"
        for m in range(60, 40, -3):
            logs.append(normalise_log({
                "timestamp": (T - timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M:%S"),
                "rule_level": 9, "rule_desc": "powershell suspicious execution",
                "rule_groups": ["powershell", "suspicious_process"],
                "mitre_tactics": ["execution"], "src_ip": ip,
                "process_name": "powershell.exe", "command_line": "powershell -enc dABlAHMAdAA=",
            }))
        for m in range(0, 15):
            logs.append(normalise_log({
                "timestamp": (T + timedelta(minutes=m)).strftime("%Y-%m-%d %H:%M:%S"),
                "rule_level": 15, "rule_desc": "ransomware file encryption",
                "rule_groups": ["ransomware", "malware"],
                "mitre_tactics": ["impact"], "src_ip": ip,
                "file_count": rng.randint(50, 200),
            }))

    # Normal background
    for hour in range(100):
        T_n = base + timedelta(hours=hour * 2)
        for _ in range(rng.randint(3, 10)):
            logs.append(normalise_log({
                "timestamp": (T_n + timedelta(seconds=rng.randint(0,3600))).strftime("%Y-%m-%d %H:%M:%S"),
                "rule_level": rng.randint(1, 4), "rule_desc": "normal activity",
                "rule_groups": ["audit"], "src_ip": f"10.0.0.{rng.randint(1,5)}",
            }))

    logs.sort(key=lambda l: l["timestamp"])

    reconstructor = AttackSessionReconstructor()
    sessions      = reconstructor.reconstruct(logs)
    builder       = TemporalSampleBuilder(obs_window_min=60, min_lead_min=10,
                                           max_lead_min=120, samples_per_session=8)
    return builder.build(logs, sessions)


def main():
    logger.info("═" * 55)
    logger.info("  GUARDIAN AI — TRAINING")
    logger.info("═" * 55)

    # Try real data first
    logs    = read_all_alerts()
    samples = None

    if len(logs) >= 50:
        samples = train_on_real_data(logs)

    # Fallback to synthetic if real data insufficient
    if samples is None:
        samples = train_on_synthetic_data()

    if not samples:
        logger.error("Could not build training samples. Exiting.")
        sys.exit(1)

    # Train
    logger.info("Training AI models...")
    trainer = ForecastModelTrainer(
        n_estimators  = 200,
        max_depth_clf = 12,
        max_imbalance = 3.0,
        test_size     = 0.20,
        contamination = 0.15,
    )
    bundle = trainer.train(samples)

    e = bundle.clf_eval
    logger.info("Classifier accuracy  : %.1f%%", e.accuracy * 100)
    logger.info("Classifier macro F1  : %.1f%%", e.macro_f1 * 100)
    if bundle.reg_eval:
        logger.info("Regressor MAE        : %.1f min", bundle.reg_eval.mae_minutes)
        logger.info("Regressor R²         : %.3f", bundle.reg_eval.r2)

    save_model(bundle, MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)
    logger.info("Training complete. You can now start: sudo systemctl start guardian")


if __name__ == "__main__":
    main()
