"""
example.py
==========
End-to-end demonstration of the redesigned Guardian Attack Forecasting Engine.

Architecture overview:

  ┌─────────────────────────────────────────────────────────────────────┐
  │                    Two-Layer Intelligence                           │
  │                                                                     │
  │  Layer 1 — KNOWN ATTACKS                                           │
  │    Wazuh rule.groups + MITRE ATT&CK → attack class features        │
  │    RandomForest multi-class classifier (16 attack types)            │
  │                                                                     │
  │  Layer 2 — UNKNOWN / BEHAVIORAL ANOMALIES                          │
  │    Entropy, burst scores, diversity, timing → anomaly features      │
  │    IsolationForest unsupervised anomaly detector                    │
  │                                                                     │
  │  Timing — TIME-TO-ATTACK                                           │
  │    GradientBoosting regressor trained on attack-positive samples    │
  │    Predicts: minutes until attack materialises                      │
  │                                                                     │
  │  Correlation — KILL-CHAIN CONTEXT                                  │
  │    Entity timelines + cosine pattern matching                       │
  │    Identifies campaign stage and historical pattern matches         │
  └─────────────────────────────────────────────────────────────────────┘

Temporal design (corrected from original):

  ─────────────────────────────────────────────────────────────▶ time
  [  ALL historical logs  ]
       ↓  session reconstruction
  [ session 1 ][quiet][ session 2 ][quiet][ session 3 ]
       ↓  for each session, sample multiple observation points
  [obs window @ T-lead]  lead=10..240 min before session.start
       ↓  features from obs window, label = {attack_class, minutes_to_attack}
  Training pairs: features(past 60 min) → (attack_class, minutes_to_attack)

Run with:
    pip install scikit-learn joblib pandas numpy
    python example.py
"""

import logging
import random
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-28s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)

from feature_extractor import normalise_log
from temporal_builder import AttackSessionReconstructor, TemporalSampleBuilder
from attack_correlator import AttackCorrelator
from model import ForecastModelTrainer, save_model, load_model
from predictor import AttackForecaster


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic log generator — produces logs that look like real Wazuh alerts
# ─────────────────────────────────────────────────────────────────────────────

def _log(t, rule_id="5710", level=3, desc="normal event", groups=None,
         mitre=None, src_ip="10.0.0.1", dst_ip="", src_port=0, dst_port=0,
         src_country="", process="", cmd="", path="", file_count=0,
         username="", agent_id="001", agent_name="server01",
         is_login_success=False):
    return normalise_log({
        "timestamp":        t.strftime("%Y-%m-%d %H:%M:%S"),
        "rule_id":          rule_id,
        "rule_level":       level,
        "rule_desc":        desc,
        "rule_groups":      groups or [],
        "mitre_tactics":    mitre or [],
        "src_ip":           src_ip,
        "dst_ip":           dst_ip,
        "src_port":         src_port,
        "dst_port":         dst_port,
        "src_country":      src_country,
        "process_name":     process,
        "command_line":     cmd,
        "file_path":        path,
        "file_count":       file_count,
        "username":         username,
        "agent_id":         agent_id,
        "agent_name":       agent_name,
        "is_login_success": is_login_success,
    })


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Build a synthetic log corpus with realistic attack sessions
# ─────────────────────────────────────────────────────────────────────────────

def build_log_corpus(n_normal_hours=200, n_attacks=40, seed=42):
    """
    Returns a sorted list of normalised log dicts spanning multiple months.
    Includes quiet periods and attack sessions with realistic multi-stage
    kill chains.
    """
    rng  = random.Random(seed)
    base = datetime(2024, 1, 1, 8, 0, 0)
    logs = []

    # ── Background (benign) traffic ───────────────────────────────────────
    print("  Generating background (benign) log traffic...")
    for hour in range(n_normal_hours):
        T = base + timedelta(hours=hour * 2)
        n = rng.randint(5, 20)
        for _ in range(n):
            logs.append(_log(
                t        = T + timedelta(seconds=rng.randint(0, 3600)),
                rule_id  = rng.choice(["5710", "5302", "5501", "0510"]),
                level    = rng.randint(1, 4),
                desc     = rng.choice(["file read", "user login", "cron job", "dns query"]),
                groups   = rng.choice([["audit"], ["syscheck"], []]),
                src_ip   = f"10.0.0.{rng.randint(1, 10)}",
                path     = f"/home/user{rng.randint(1,3)}/doc.txt",
                file_count = rng.randint(1, 5),
                username = rng.choice(["alice", "bob", "carol"]),
                agent_id = rng.choice(["001", "002"]),
            ))

    print(f"  ✓ {len(logs)} benign events generated\n")

    # ── Attack sessions ───────────────────────────────────────────────────
    print("  Generating attack sessions with kill-chain progression...")

    attack_start_offset = n_normal_hours * 2 + 24  # start attacks after quiet period

    # ── Attack type 1: SSH Brute-force → privilege escalation ─────────────
    for i in range(n_attacks // 4):
        T = base + timedelta(hours=attack_start_offset + i * 12)
        attacker_ip = f"198.51.{rng.randint(100,200)}.{rng.randint(1,255)}"

        # Stage 1 — Recon (port scan) at T-30min
        for m in range(28, 25, -1):
            logs.append(_log(
                t=T - timedelta(minutes=m), rule_id="40101", level=6,
                desc="port scan detected",
                groups=["recon", "port_scan"],
                mitre=["reconnaissance"],
                src_ip=attacker_ip, dst_port=22,
                src_country="ru",
            ))

        # Stage 2 — Brute force SSH at T-20min to T
        for m in range(20, 0, -1):
            logs.append(_log(
                t=T - timedelta(minutes=m), rule_id="5710", level=rng.randint(7, 10),
                desc="sshd failed password for root",
                groups=["authentication_failed", "sshd"],
                mitre=["initial-access", "credential-access"],
                src_ip=attacker_ip, dst_port=22,
                username="root",
                src_country="ru",
            ))

        # Stage 3 — Successful login at T (attack materialises)
        logs.append(_log(
            t=T, rule_id="5715", level=12,
            desc="accepted password for root via sshd",
            groups=["authentication_failed", "sshd"],
            mitre=["initial-access"],
            src_ip=attacker_ip, dst_port=22, username="root",
            is_login_success=True, src_country="ru",
        ))

        # Stage 4 — Privilege escalation at T+5min
        for m in range(5, 15):
            logs.append(_log(
                t=T + timedelta(minutes=m), rule_id="5402", level=14,
                desc="sudo: root privilege escalation",
                groups=["privilege_escalation", "sudo"],
                mitre=["privilege-escalation"],
                src_ip=attacker_ip, username="root",
                path=rng.choice(["/etc/passwd", "/etc/shadow", "/root/.ssh/id_rsa"]),
                cmd="sudo su -",
            ))

    # ── Attack type 2: Web → Shell ─────────────────────────────────────────
    for i in range(n_attacks // 4):
        T = base + timedelta(hours=attack_start_offset + 300 + i * 10)
        attacker_ip = f"203.0.{rng.randint(113,120)}.{rng.randint(1,254)}"

        # SQL injection probing
        for m in range(45, 30, -1):
            logs.append(_log(
                t=T - timedelta(minutes=m), rule_id="31120", level=8,
                desc="sql injection attempt detected",
                groups=["web", "sql_injection"],
                mitre=["initial-access"],
                src_ip=attacker_ip, dst_port=80,
                path="/var/www/html/login.php",
                src_country="cn",
            ))

        # Web shell upload at T-10
        for m in range(10, 5, -1):
            logs.append(_log(
                t=T - timedelta(minutes=m), rule_id="31165", level=12,
                desc="web shell upload detected",
                groups=["web", "web_shell"],
                mitre=["execution", "persistence"],
                src_ip=attacker_ip, dst_port=80,
                path="/var/www/html/uploads/shell.php",
            ))

        # Lateral movement via web shell at T
        for m in range(0, 20):
            logs.append(_log(
                t=T + timedelta(minutes=m), rule_id="5906", level=13,
                desc="multiple hosts accessed from single session",
                groups=["lateral_movement"],
                mitre=["lateral-movement"],
                src_ip=attacker_ip,
                path=f"/srv/dept{rng.randint(1,4)}/config",
                cmd="python3 -c 'import socket...'",
            ))

    # ── Attack type 3: Insider exfiltration ───────────────────────────────
    for i in range(n_attacks // 4):
        T = base + timedelta(hours=attack_start_offset + 600 + i * 8)
        T = T.replace(hour=rng.choice([22, 23, 0, 1]))
        insider_ip = "10.0.3.55"

        # Recon — probing sensitive dirs
        for m in range(50, 30, -2):
            logs.append(_log(
                t=T - timedelta(minutes=m), rule_id="5501", level=7,
                desc="large file read from restricted directory",
                groups=["syscheck"],
                mitre=["collection"],
                src_ip=insider_ip,
                path=rng.choice(["/finance/q4.csv", "/rh/employees.xlsx"]),
                file_count=rng.randint(5, 20),
                username="mallory",
            ))

        # Exfiltration burst at T
        for m in range(0, 30, 3):
            logs.append(_log(
                t=T + timedelta(minutes=m), rule_id="5502", level=13,
                desc="mass data transfer from sensitive directory",
                groups=["data_exfiltration", "outbound_transfer"],
                mitre=["exfiltration", "collection"],
                src_ip=insider_ip,
                path="/finance/export_all.csv",
                file_count=rng.randint(100, 400),
                username="mallory",
            ))

    # ── Attack type 4: Malware / ransomware preparation ───────────────────
    for i in range(n_attacks // 4):
        T = base + timedelta(hours=attack_start_offset + 900 + i * 8)
        attacker_ip = f"192.168.1.{rng.randint(200, 250)}"

        # Suspicious powershell execution
        for m in range(60, 40, -3):
            logs.append(_log(
                t=T - timedelta(minutes=m), rule_id="92200", level=9,
                desc="powershell suspicious execution",
                groups=["powershell", "suspicious_process"],
                mitre=["execution", "defense-evasion"],
                src_ip=attacker_ip,
                process="powershell.exe",
                cmd="powershell -enc dABlAHMAdAA=",
            ))

        # Persistence via registry
        for m in range(30, 20, -2):
            logs.append(_log(
                t=T - timedelta(minutes=m), rule_id="92501", level=11,
                desc="registry run key modification",
                groups=["persistence", "registry"],
                mitre=["persistence"],
                src_ip=attacker_ip,
                path="HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run",
            ))

        # Ransomware encryption begins at T
        for m in range(0, 15):
            logs.append(_log(
                t=T + timedelta(minutes=m), rule_id="92900", level=15,
                desc="ransomware file encryption pattern detected",
                groups=["ransomware", "malware"],
                mitre=["impact"],
                src_ip=attacker_ip,
                path=f"/home/user{rng.randint(1,5)}/documents/file{m}.encrypted",
                file_count=rng.randint(50, 200),
            ))

    # Sort by time
    logs.sort(key=lambda l: l["timestamp"])
    n_attack = sum(1 for l in logs if l["attack_class"] not in ("unknown", "audit"))
    print(f"  ✓ {len(logs)} total events ({n_attack} attack-class)")
    return logs


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Reconstruct sessions + build temporal training pairs
# ─────────────────────────────────────────────────────────────────────────────

def build_training_data(logs):
    print("\nStep 2: Reconstructing attack sessions...")
    reconstructor = AttackSessionReconstructor()
    sessions      = reconstructor.reconstruct(logs)

    print(f"  ✓ {len(sessions)} attack sessions found")
    for i, s in enumerate(sessions[:5]):
        print(f"     [{i+1}] {s.dominant_class:22s}  "
              f"stages={s.stage_sequence}  "
              f"events={s.event_count}  "
              f"duration={s.duration_min:.0f}min")
    if len(sessions) > 5:
        print(f"     ... and {len(sessions)-5} more sessions")

    print("\nStep 3: Building temporal training samples...")
    builder = TemporalSampleBuilder(
        obs_window_min      = 60,
        min_lead_min        = 10,
        max_lead_min        = 120,
        samples_per_session = 8,
        quiet_buffer_hrs    = 2.0,
    )
    samples = builder.build(logs, sessions)

    pos = sum(1 for s in samples if s.label.will_attack)
    print(f"  ✓ {len(samples)} training pairs "
          f"({pos} positive, {len(samples)-pos} negative)")

    class_dist = {}
    for s in samples:
        cls = s.label.attack_class
        class_dist[cls] = class_dist.get(cls, 0) + 1
    print("  Attack class distribution:")
    for cls, cnt in sorted(class_dist.items(), key=lambda x: -x[1])[:8]:
        print(f"     {cls:30s}: {cnt}")

    return samples, sessions


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Train and save
# ─────────────────────────────────────────────────────────────────────────────

def train_and_save(samples, path="guardian_model.joblib"):
    print("\nStep 4: Training models...")
    trainer = ForecastModelTrainer(
        n_estimators  = 200,
        max_depth_clf = 12,
        max_imbalance = 3.0,
        test_size     = 0.20,
        contamination = 0.15,
    )
    bundle = trainer.train(samples)
    e = bundle.clf_eval

    print("═" * 66)
    print("  CLASSIFIER — TRAINING RESULTS")
    print("═" * 66)
    print(f"  Accuracy   : {e.accuracy:.1%}")
    print(f"  Macro F1   : {e.macro_f1:.1%}")
    print(f"  Train/Test : {e.n_train} / {e.n_test}")
    print("  Per-class results:")
    for cls, m in sorted(e.per_class.items(), key=lambda x: -x[1]["support"])[:8]:
        if m["support"] > 0:
            print(f"     {cls:30s}  P={m['precision']:.2f}  R={m['recall']:.2f}  "
                  f"F1={m['f1']:.2f}  n={m['support']}")

    if bundle.reg_eval:
        r = bundle.reg_eval
        print(f"\n  TIME-TO-ATTACK REGRESSOR:")
        print(f"     MAE = {r.mae_minutes:.1f} min  (mean absolute error on test set)")
        print(f"     R²  = {r.r2:.3f}          (0 = no skill, 1 = perfect)")
        print(f"     Note: ±30% uncertainty is expected. Treat as order-of-magnitude.")
    else:
        print("  TIME-TO-ATTACK REGRESSOR: not built (insufficient positive samples)")

    print("═" * 66 + "\n")

    save_model(bundle, path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Run live forecasts
# ─────────────────────────────────────────────────────────────────────────────

def run_forecasts(model_path, sessions):
    bundle     = load_model(model_path)
    correlator = AttackCorrelator(timeline_ttl_hours=12.0)
    forecaster = AttackForecaster(bundle, correlator=correlator, threshold=0.35)

    NOW = datetime(2024, 9, 1, 14, 0, 0)
    rng = random.Random(99)

    scenarios = [
        {
            "name": "Quiet afternoon — no threat",
            "logs": [_log(
                t=NOW - timedelta(minutes=m),
                level=rng.randint(1,4), desc="file read",
                src_ip="10.0.0.1", path="/home/alice/doc.txt",
                file_count=1, username="alice",
            ) for m in range(1, 40, 4)],
        },
        {
            "name": "SSH brute force in progress (recon + auth failures)",
            "logs": (
                # Recon first
                [_log(t=NOW - timedelta(minutes=m), rule_id="40101", level=6,
                      desc="port scan detected", groups=["recon", "port_scan"],
                      mitre=["reconnaissance"], src_ip="203.0.113.5",
                      dst_port=22, src_country="ru",
                ) for m in range(55, 50, -1)]
                +
                # Brute force
                [_log(t=NOW - timedelta(minutes=m), rule_id="5710", level=9,
                      desc="sshd failed password for root",
                      groups=["authentication_failed", "sshd"],
                      mitre=["initial-access"],
                      src_ip=f"203.0.113.{rng.randint(1,10)}", dst_port=22,
                      username="root", src_country="ru",
                ) for m in range(45, 1, -2)]
            ),
        },
        {
            "name": "Web shell + lateral movement (exploitation stage)",
            "logs": (
                [_log(t=NOW - timedelta(minutes=m), rule_id="31120", level=8,
                      desc="sql injection attempt", groups=["web", "sql_injection"],
                      mitre=["initial-access"],
                      src_ip="203.0.114.20", dst_port=80, src_country="cn",
                ) for m in range(40, 25, -2)]
                +
                [_log(t=NOW - timedelta(minutes=m), rule_id="31165", level=12,
                      desc="web shell activity detected",
                      groups=["web", "web_shell"],
                      mitre=["execution", "persistence"],
                      src_ip="203.0.114.20", dst_port=80,
                      path="/var/www/html/uploads/shell.php",
                      cmd="python3 reverse_shell.py",
                ) for m in range(15, 1, -2)]
            ),
        },
        {
            "name": "Suspicious PowerShell + registry persistence (malware prep)",
            "logs": [_log(
                t=NOW - timedelta(minutes=m),
                rule_id="92200", level=rng.randint(9,12),
                desc="powershell suspicious execution",
                groups=["powershell", "suspicious_process"],
                mitre=["execution", "defense-evasion"],
                src_ip="192.168.1.210",
                process="powershell.exe",
                cmd="powershell -nop -w hidden -enc dABlAHMAdAA=",
            ) for m in range(50, 1, -3)],
        },
        {
            "name": "Night-time insider data access (exfiltration risk)",
            "logs": [_log(
                t=(NOW.replace(hour=23)) - timedelta(minutes=m),
                rule_id="5501", level=rng.randint(7,10),
                desc="large file read from restricted directory",
                groups=["syscheck"],
                mitre=["collection"],
                src_ip="10.0.3.55",
                path=rng.choice(["/finance/q4.csv", "/rh/employees.xlsx",
                                 "/confidentiel/plan.pdf"]),
                file_count=rng.randint(20, 80),
                username="mallory",
            ) for m in range(50, 1, -3)],
        },
    ]

    print("═" * 70)
    print("  LIVE FORECAST RESULTS")
    print("═" * 70)

    for i, sc in enumerate(scenarios, 1):
        r = forecaster.predict(sc["logs"])

        print(f"\n  [{i}] {sc['name']}")
        print(f"      Risk Level  : {r.risk_level}")
        print(f"      Prediction  : {r.known_attack_prediction.upper()}")
        print(f"      Probability : {r.known_attack_probability:.1%}")
        print(f"      Anomaly Risk: {r.unknown_attack_risk:.1%} (behavioral layer)")
        print(f"      Est. Timing : {r.estimated_time_to_attack}")
        print(f"      Stage       : {r.attack_stage}" +
              (f" → {r.next_expected_stage}" if r.next_expected_stage else ""))
        print(f"      Confidence  : {r.confidence_label} ({r.confidence:.2f})")
        if r.evidence_signals:
            print(f"      Evidence    :")
            for sig in r.evidence_signals[:5]:
                print(f"         • {sig}")
        if r.behavioral_anomalies:
            print(f"      Anomalies   :")
            for a in r.behavioral_anomalies[:3]:
                print(f"         ⚠ {a}")
        print(f"      Recommend   : {r.recommendation}")

    print("\n" + "═" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(42)

    print("\n" + "═" * 70)
    print("  GUARDIAN — Redesigned Attack Forecasting Engine")
    print("  Two layers: Known Attack Intelligence + Behavioral Anomaly")
    print("═" * 70 + "\n")

    print("Step 1: Building synthetic log corpus...")
    logs = build_log_corpus(n_normal_hours=150, n_attacks=40, seed=42)

    samples, sessions = build_training_data(logs)

    model_path = train_and_save(samples)
    run_forecasts(model_path, sessions)

    print("\nUsage with real Wazuh logs:")
    print("  from feature_extractor import normalise_log")
    print("  from predictor import AttackForecaster")
    print("  from model import load_model")
    print("")
    print("  bundle    = load_model('guardian_model.joblib')")
    print("  forecaster = AttackForecaster(bundle)")
    print("  past = [normalise_log(raw) for raw in wazuh_alerts_last_60min]")
    print("  result = forecaster.predict(past)")
    print("  print(result.risk_level, result.known_attack_prediction)")
    print("  print(result.estimated_time_to_attack)")
    print("  print(result.recommendation)\n")
