"""
feature_extractor.py
====================
Extracts a rich, two-layer feature vector from a past observation window.

LAYER 1 — Known Attack Intelligence
  Uses Wazuh's native metadata: rule.id, rule.groups, MITRE ATT&CK mappings,
  decoder names, process names, command lines, and geo information.
  Features here represent SIGNALS of known attack classes.

LAYER 2 — Behavioral / Unknown Attack Intelligence
  Computes anomaly indicators using trends, entropy, baseline deviations,
  temporal patterns, and activity diversity.
  These features fire even when Wazuh has NO matching rule.

Design decisions
----------------
- normalise_log() is the ONLY place raw Wazuh field names are read.
  Everything downstream works on canonical dicts.
- Features are always floats (scikit-learn requirement).
- The dataclass mirrors the two-layer architecture: L1_* vs L2_* prefixes
  make it immediately obvious which layer each feature belongs to.
- All features describe the PAST observation window only. Nothing from
  the future leaks into feature extraction.

Limitations
-----------
- Geo features require ip_country to be present in the raw log; if missing
  they default to 0. Do not treat them as always reliable.
- Entropy-based features are approximations (discrete Shannon entropy over
  binned values). They are useful but not a substitute for proper density
  estimation.
- User/host baselines are computed from the window itself, not from a
  long-term historical store. A production system should maintain rolling
  per-entity baselines in Redis or a time-series DB.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict, fields
from datetime import datetime
from typing import Sequence

# ---------------------------------------------------------------------------
# Wazuh rule groups → attack class mapping
# These are the groups Wazuh actually writes into rule.groups.
# Mapping them here lets Layer 1 features encode real attack types without
# any handcrafted heuristics in the model.
# ---------------------------------------------------------------------------

ATTACK_CLASS_GROUPS: dict[str, str] = {
    # Authentication attacks
    "authentication_failed":      "brute_force",
    "authentication_failures":    "brute_force",
    "invalid_login":              "brute_force",
    "sshd":                       "brute_force",
    "win_authentication_failed":  "brute_force",
    "credential_access":          "credential_stuffing",
    # Privilege / escalation
    "privilege_escalation":       "priv_escalation",
    "sudo":                       "priv_escalation",
    "rootcheck":                  "priv_escalation",
    # Lateral movement
    "lateral_movement":           "lateral_movement",
    "smb":                        "lateral_movement",
    # Web attacks
    "web":                        "web_attack",
    "sql_injection":              "sqli",
    "web_shell":                  "web_shell",
    # Malware / ransomware
    "malware":                    "malware",
    "ransomware":                 "ransomware",
    "virus":                      "malware",
    # Exfiltration
    "data_exfiltration":          "exfiltration",
    "outbound_transfer":          "exfiltration",
    # Reconnaissance
    "recon":                      "recon",
    "nmap":                       "recon",
    "port_scan":                  "recon",
    # Persistence
    "persistence":                "persistence",
    "startup":                    "persistence",
    "registry":                   "persistence",
    # Execution / suspicious processes
    "powershell":                 "suspicious_exec",
    "process_creation":           "suspicious_exec",
    "suspicious_process":         "suspicious_exec",
    # Integrity / file changes
    "syscheck":                   "file_integrity",
    # Audit / compliance (usually benign, but still tracked)
    "audit":                      "audit",
}

# MITRE ATT&CK tactic → our internal attack class
MITRE_TACTIC_MAP: dict[str, str] = {
    "reconnaissance":          "recon",
    "resource-development":    "recon",
    "initial-access":          "brute_force",
    "execution":               "suspicious_exec",
    "persistence":             "persistence",
    "privilege-escalation":    "priv_escalation",
    "defense-evasion":         "evasion",
    "credential-access":       "credential_stuffing",
    "discovery":               "recon",
    "lateral-movement":        "lateral_movement",
    "collection":              "exfiltration",
    "exfiltration":            "exfiltration",
    "command-and-control":     "c2",
    "impact":                  "ransomware",
}

# All known attack classes — used to create a consistent fixed-length vector
ALL_ATTACK_CLASSES = sorted({
    "brute_force", "credential_stuffing", "sqli", "priv_escalation",
    "lateral_movement", "web_attack", "web_shell", "malware",
    "ransomware", "exfiltration", "recon", "persistence",
    "suspicious_exec", "evasion", "c2", "file_integrity", "audit",
})

# Suspicious keywords in command lines / process names
SUSPICIOUS_CMDS = frozenset([
    "mimikatz", "procdump", "bloodhound", "cobaltstrike",
    "mshta", "wscript", "cscript", "regsvr32",
    "certutil", "bitsadmin", "nc ", "netcat", "ncat",
    "whoami", "net user", "net localgroup",
    "powershell -enc", "powershell -e ", "-nop -w hidden",
    "curl|sh", "wget|sh", "|bash", "base64 -d",
])

SENSITIVE_PATH_KW = frozenset([
    "passwd", "shadow", "id_rsa", ".ssh", "credentials",
    "secret", "/admin", "backup", "dump", ".env",
    "rh/", "paie/", "finance/", "confidentiel/", "lsass",
    "/etc/cron", "authorized_keys", ".bashrc", "hosts",
])

_TS_FMTS = (
    "%Y-%m-%dT%H:%M:%S.%f+0000",
    "%Y-%m-%dT%H:%M:%SZ",
    "%Y-%m-%dT%H:%M:%S.%fZ",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


def parse_ts(ts: str | datetime) -> datetime:
    if isinstance(ts, datetime):
        return ts
    for fmt in _TS_FMTS:
        try:
            return datetime.strptime(ts, fmt)
        except (ValueError, TypeError):
            continue
    return datetime.now()


def normalise_log(raw: dict) -> dict:
    """
    Flatten a raw Wazuh alert JSON into a canonical tagged dict.
    This is the ONLY function that reads raw Wazuh field names.

    Supports both flat dicts (from example.py) and full Wazuh JSON alerts.

    Returns a dict with every field the feature extractor needs.
    """
    # Support both nested Wazuh JSON and flat dicts from example.py
    rb  = raw.get("rule", {})
    db  = raw.get("data", {})
    ag  = raw.get("agent", {})
    loc = raw.get("location", "")

    # ── Core fields ──────────────────────────────────────────────────────
    rule_id    = str(raw.get("rule_id") or rb.get("id") or "0")
    rule_level = int(raw.get("rule_level") or rb.get("level") or 0)
    rule_desc  = (raw.get("rule_desc") or rb.get("description") or "").lower()

    # rule.groups is a list in real Wazuh; may be a comma-str in flat dicts
    raw_groups = raw.get("rule_groups") or rb.get("groups") or []
    if isinstance(raw_groups, str):
        raw_groups = [g.strip() for g in raw_groups.split(",") if g.strip()]
    rule_groups: list[str] = [g.lower() for g in raw_groups]

    # MITRE ATT&CK
    mitre_block   = rb.get("mitre", {}) or {}
    mitre_tactics = [t.lower() for t in (mitre_block.get("tactic") or [])]
    mitre_ids     = list(mitre_block.get("id") or [])

    # Decoder name — identifies the log source pipeline
    decoder_name = (raw.get("decoder_name") or raw.get("decoder", {}).get("name") or "").lower()

    # Agent / host info
    agent_name = (raw.get("agent_name") or ag.get("name") or "").lower()
    agent_id   = str(raw.get("agent_id") or ag.get("id") or "0")

    # Network
    src_ip  = (raw.get("src_ip") or db.get("srcip") or db.get("src_ip")
               or raw.get("data", {}).get("srcip") or "")
    dst_ip  = (raw.get("dst_ip") or db.get("dstip") or db.get("dst_ip") or "")
    src_port = int(raw.get("src_port") or db.get("srcport") or 0)
    dst_port = int(raw.get("dst_port") or db.get("dstport") or 0)

    # Geo
    src_country = (raw.get("src_country") or
                   db.get("srcgeoip", {}).get("country_name") or "").lower()
    is_external_ip = bool(src_country) and src_country not in ("", "private", "reserved")

    # Process / execution
    process_name = (raw.get("process_name") or db.get("process") or
                    db.get("win", {}).get("eventdata", {}).get("image") or "").lower()
    command_line = (raw.get("command_line") or
                    db.get("win", {}).get("eventdata", {}).get("commandLine") or "").lower()

    # File / path
    file_path = (
        raw.get("file_path")
        or db.get("syscheck", {}).get("path")
        or db.get("path") or ""
    ).lower()
    file_count = int(raw.get("file_count") or 0)

    # Username
    username = (raw.get("username") or db.get("srcuser") or db.get("dstuser") or "").lower()

    # Login outcome
    is_login_success = bool(raw.get("is_login_success", False))

    timestamp = parse_ts(raw.get("timestamp", datetime.now()))

    # ── Derived behavioral tags ─────────────────────────────────────────
    is_auth_fail    = _any_in(rule_desc + " " + " ".join(rule_groups),
                              ["authentication failure", "failed password", "invalid user",
                               "login failed", "auth failure", "bad password", "wrong password"])
    is_auth_success = is_login_success or _any_in(rule_desc,
                              ["accepted password", "accepted publickey", "session opened",
                               "logged in", "new session", "successful login"])
    is_priv_cmd     = _any_in(rule_desc + " " + command_line + " " + file_path,
                              ["sudo", "su ", "visudo", "chown root", "passwd", "id_rsa",
                               "privilege escalation", "privilege_escalation"])
    is_sensitive    = _any_in(file_path, SENSITIVE_PATH_KW)
    is_suspicious   = _any_in(command_line + " " + process_name, SUSPICIOUS_CMDS)

    # ── Layer 1: derive attack class from groups + MITRE ────────────────
    attack_class = _derive_attack_class(rule_groups, mitre_tactics, rule_desc, is_auth_fail,
                                        is_auth_success, is_priv_cmd, is_suspicious)

    return {
        # Identity
        "timestamp":       timestamp,
        "rule_id":         rule_id,
        "rule_level":      rule_level,
        "rule_desc":       rule_desc,
        "rule_groups":     rule_groups,
        "mitre_tactics":   mitre_tactics,
        "mitre_ids":       mitre_ids,
        "decoder_name":    decoder_name,
        # Agent / host
        "agent_name":      agent_name,
        "agent_id":        agent_id,
        # Network
        "src_ip":          src_ip,
        "dst_ip":          dst_ip,
        "src_port":        src_port,
        "dst_port":        dst_port,
        "src_country":     src_country,
        "is_external_ip":  is_external_ip,
        # Process / execution
        "process_name":    process_name,
        "command_line":    command_line,
        # File
        "file_path":       file_path,
        "file_count":      file_count,
        # User
        "username":        username,
        # Behavioral tags
        "is_auth_fail":    is_auth_fail,
        "is_auth_success": is_auth_success,
        "is_priv_cmd":     is_priv_cmd,
        "is_sensitive":    is_sensitive,
        "is_suspicious":   is_suspicious,
        "is_login_success": is_login_success,
        # Derived attack class
        "attack_class":    attack_class,
    }


def _any_in(text: str, keywords) -> bool:
    return any(k in text for k in keywords)


def _derive_attack_class(
    groups: list[str],
    mitre_tactics: list[str],
    rule_desc: str,
    is_auth_fail: bool,
    is_auth_success: bool,
    is_priv_cmd: bool,
    is_suspicious: bool,
) -> str:
    """
    Determine the best-fit attack class for a single event.
    Priority: explicit Wazuh groups > MITRE tactics > keyword fallback.
    Returns 'unknown' if nothing matches — that is a valid and important class.
    """
    for group in groups:
        if group in ATTACK_CLASS_GROUPS:
            return ATTACK_CLASS_GROUPS[group]

    for tactic in mitre_tactics:
        if tactic in MITRE_TACTIC_MAP:
            return MITRE_TACTIC_MAP[tactic]

    # Keyword fallback — only fires when Wazuh metadata is absent
    if is_auth_fail:
        return "brute_force"
    if is_priv_cmd:
        return "priv_escalation"
    if is_suspicious:
        return "suspicious_exec"
    if is_auth_success:
        return "audit"

    return "unknown"


# ---------------------------------------------------------------------------
# Feature dataclass
# ---------------------------------------------------------------------------

@dataclass
class WindowFeatures:
    """
    Two-layer feature vector extracted from the PAST observation window.

    Layer 1 (L1_*) — Known attack intelligence
      Each L1 feature represents a count or rate of a specific known attack
      class. These let the model distinguish attack types without heuristics.

    Layer 2 (L2_*) — Behavioral / unknown attack intelligence
      These features capture anomalous patterns that could signal unknown
      threats even when no Wazuh rule fires.

    Ctx_* — Temporal and contextual features
      Time-of-day, weekday, and window metadata.

    All values are floats.
    """

    # ── Layer 1: Known attack class event counts ─────────────────────────
    L1_brute_force_count:      float = 0.0
    L1_credential_count:       float = 0.0
    L1_sqli_count:             float = 0.0
    L1_priv_escalation_count:  float = 0.0
    L1_lateral_movement_count: float = 0.0
    L1_web_attack_count:       float = 0.0
    L1_web_shell_count:        float = 0.0
    L1_malware_count:          float = 0.0
    L1_ransomware_count:       float = 0.0
    L1_exfiltration_count:     float = 0.0
    L1_recon_count:            float = 0.0
    L1_persistence_count:      float = 0.0
    L1_suspicious_exec_count:  float = 0.0
    L1_evasion_count:          float = 0.0
    L1_c2_count:               float = 0.0
    L1_unknown_count:          float = 0.0

    # L1 rates (per minute)
    L1_brute_force_rate:       float = 0.0
    L1_attack_event_rate:      float = 0.0   # all classified events / min

    # L1 MITRE tactic diversity (unique tactics seen)
    L1_mitre_tactic_diversity: float = 0.0
    L1_unique_rule_ids:        float = 0.0   # distinct Wazuh rule IDs

    # L1 max severity and its trend
    L1_max_rule_level:         float = 0.0
    L1_mean_rule_level:        float = 0.0
    L1_level_trend:            float = 0.0   # last_third_mean - first_third_mean

    # L1 auth signals
    L1_auth_fail_count:        float = 0.0
    L1_auth_success_count:     float = 0.0
    L1_auth_fail_rate:         float = 0.0
    L1_auth_fail_trend:        float = 0.0   # escalating failures?
    L1_fail_to_success_ratio:  float = 0.0   # high → brute force + success

    # ── Layer 2: Behavioral / anomaly intelligence ────────────────────────

    # IP diversity and spread
    L2_unique_src_ips:         float = 0.0
    L2_unique_dst_ips:         float = 0.0
    L2_ip_entropy:             float = 0.0   # Shannon entropy of src_ip distribution
    L2_external_ip_count:      float = 0.0   # events from non-RFC-1918 IPs
    L2_unique_countries:       float = 0.0

    # Temporal anomalies
    L2_is_night:               float = 0.0   # 1 if hour < 7 or >= 22
    L2_is_off_hours:           float = 0.0   # 1 if hour < 8 or >= 19
    L2_is_weekend:             float = 0.0
    L2_burst_score:            float = 0.0   # events in last 5 min / events per min avg

    # Activity volume and trends
    L2_total_events:           float = 0.0
    L2_event_rate:             float = 0.0
    L2_event_trend:            float = 0.0   # last_third - first_third count
    L2_total_volume_mb:        float = 0.0
    L2_volume_trend:           float = 0.0   # MB trend across window

    # File / path behavior
    L2_unique_paths:           float = 0.0
    L2_unique_dirs:            float = 0.0
    L2_dir_spread:             float = 0.0   # distinct top-level dirs
    L2_sensitive_path_count:   float = 0.0
    L2_path_entropy:           float = 0.0   # entropy of paths accessed

    # Process / execution anomalies
    L2_unique_processes:       float = 0.0
    L2_suspicious_cmd_count:   float = 0.0
    L2_priv_cmd_count:         float = 0.0

    # User behavior
    L2_unique_users:           float = 0.0
    L2_user_entropy:           float = 0.0   # entropy of user distribution

    # Host / agent diversity
    L2_unique_agents:          float = 0.0

    # Decoder diversity (log source diversity — evasion indicator)
    L2_unique_decoders:        float = 0.0

    # Port behavior
    L2_unique_dst_ports:       float = 0.0
    L2_high_risk_port_count:   float = 0.0   # 22, 3389, 445, 1433, 3306

    # ── Contextual ────────────────────────────────────────────────────────
    Ctx_hour:                  float = 0.0
    Ctx_day_of_week:           float = 0.0
    Ctx_window_span_min:       float = 0.0

    def to_list(self) -> list[float]:
        return list(asdict(self).values())

    @staticmethod
    def feature_names() -> list[str]:
        return [f.name for f in fields(WindowFeatures)]


# High-risk ports: SSH, RDP, SMB, MSSQL, MySQL, WinRM, VNC, Telnet
_HIGH_RISK_PORTS = frozenset([22, 23, 135, 139, 445, 1433, 3306, 3389, 5985, 5986, 5900])


def _shannon_entropy(counter: Counter) -> float:
    """Shannon entropy of a Counter distribution. Returns 0 for empty."""
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0)


class WindowFeatureExtractor:
    """
    Builds a WindowFeatures vector from a list of normalised log dicts.

    Usage
    -----
    >>> extractor = WindowFeatureExtractor()
    >>> features  = extractor.extract(window_logs)

    The logs must all fall within the PAST observation window,
    sorted oldest → newest.
    """

    def extract(self, logs: list[dict]) -> WindowFeatures:
        if not logs:
            return WindowFeatures()

        f = WindowFeatures()
        n = len(logs)

        # ── Time span ────────────────────────────────────────────────────
        t_start  = logs[0]["timestamp"]
        t_end    = logs[-1]["timestamp"]
        span_min = max((t_end - t_start).total_seconds() / 60.0, 1.0)

        # Trend slices
        third       = max(n // 3, 1)
        first_slice = logs[:third]
        last_slice  = logs[-third:]
        last_5min   = [l for l in logs
                       if (t_end - l["timestamp"]).total_seconds() <= 300]

        # ── Layer 1: Known attack intelligence ───────────────────────────
        class_counts: Counter = Counter(l["attack_class"] for l in logs)

        f.L1_brute_force_count      = float(class_counts.get("brute_force", 0))
        f.L1_credential_count       = float(class_counts.get("credential_stuffing", 0))
        f.L1_sqli_count             = float(class_counts.get("sqli", 0))
        f.L1_priv_escalation_count  = float(class_counts.get("priv_escalation", 0))
        f.L1_lateral_movement_count = float(class_counts.get("lateral_movement", 0))
        f.L1_web_attack_count       = float(class_counts.get("web_attack", 0))
        f.L1_web_shell_count        = float(class_counts.get("web_shell", 0))
        f.L1_malware_count          = float(class_counts.get("malware", 0))
        f.L1_ransomware_count       = float(class_counts.get("ransomware", 0))
        f.L1_exfiltration_count     = float(class_counts.get("exfiltration", 0))
        f.L1_recon_count            = float(class_counts.get("recon", 0))
        f.L1_persistence_count      = float(class_counts.get("persistence", 0))
        f.L1_suspicious_exec_count  = float(class_counts.get("suspicious_exec", 0))
        f.L1_evasion_count          = float(class_counts.get("evasion", 0))
        f.L1_c2_count               = float(class_counts.get("c2", 0))
        f.L1_unknown_count          = float(class_counts.get("unknown", 0))

        known_event_count = n - class_counts.get("unknown", 0) - class_counts.get("audit", 0)
        f.L1_attack_event_rate      = known_event_count / span_min
        f.L1_brute_force_rate       = f.L1_brute_force_count / span_min
        f.L1_unique_rule_ids        = float(len({l["rule_id"] for l in logs}))
        f.L1_mitre_tactic_diversity = float(len({t for l in logs for t in l["mitre_tactics"]}))

        levels = [l["rule_level"] for l in logs]
        f.L1_max_rule_level  = float(max(levels))
        f.L1_mean_rule_level = float(sum(levels) / n)
        f.L1_level_trend     = (
            sum(l["rule_level"] for l in last_slice) / len(last_slice)
            - sum(l["rule_level"] for l in first_slice) / len(first_slice)
        )

        f.L1_auth_fail_count    = float(sum(1 for l in logs if l["is_auth_fail"]))
        f.L1_auth_success_count = float(sum(1 for l in logs if l["is_auth_success"]))
        f.L1_auth_fail_rate     = f.L1_auth_fail_count / span_min
        f.L1_auth_fail_trend    = (
            sum(1 for l in last_slice if l["is_auth_fail"])
            - sum(1 for l in first_slice if l["is_auth_fail"])
        )
        f.L1_fail_to_success_ratio = (
            f.L1_auth_fail_count / max(f.L1_auth_success_count, 1)
        )

        # ── Layer 2: Behavioral intelligence ─────────────────────────────
        src_ip_counter = Counter(l["src_ip"] for l in logs if l["src_ip"])
        user_counter   = Counter(l["username"] for l in logs if l["username"])
        path_counter   = Counter(l["file_path"] for l in logs if l["file_path"])

        f.L2_unique_src_ips     = float(len(src_ip_counter))
        f.L2_unique_dst_ips     = float(len({l["dst_ip"] for l in logs if l["dst_ip"]}))
        f.L2_ip_entropy         = _shannon_entropy(src_ip_counter)
        f.L2_external_ip_count  = float(sum(1 for l in logs if l["is_external_ip"]))
        f.L2_unique_countries   = float(len({l["src_country"] for l in logs if l["src_country"]}))

        last = logs[-1]["timestamp"]
        h = last.hour
        f.L2_is_night     = float(h < 7 or h >= 22)
        f.L2_is_off_hours = float(h < 8 or h >= 19)
        f.L2_is_weekend   = float(last.weekday() >= 5)

        avg_rate = n / span_min
        last5_rate = len(last_5min) / 5.0 if last_5min else 0.0
        f.L2_burst_score = last5_rate / max(avg_rate, 0.01)

        f.L2_total_events  = float(n)
        f.L2_event_rate    = n / span_min
        f.L2_event_trend   = float(len(last_slice) - len(first_slice))
        f.L2_total_volume_mb = float(sum(l["file_count"] * 0.5 for l in logs))
        first_vol = sum(l["file_count"] * 0.5 for l in first_slice)
        last_vol  = sum(l["file_count"] * 0.5 for l in last_slice)
        f.L2_volume_trend  = last_vol - first_vol

        f.L2_unique_paths      = float(len(path_counter))
        top_dirs = {
            p.split("/")[1] for l in logs
            for p in [l["file_path"]] if p.count("/") >= 1
        }
        f.L2_unique_dirs       = float(len(top_dirs))
        f.L2_dir_spread        = float(len(top_dirs))
        f.L2_sensitive_path_count = float(sum(1 for l in logs if l["is_sensitive"]))
        f.L2_path_entropy      = _shannon_entropy(path_counter)

        f.L2_unique_processes  = float(len({l["process_name"] for l in logs if l["process_name"]}))
        f.L2_suspicious_cmd_count = float(sum(1 for l in logs if l["is_suspicious"]))
        f.L2_priv_cmd_count    = float(sum(1 for l in logs if l["is_priv_cmd"]))

        f.L2_unique_users      = float(len(user_counter))
        f.L2_user_entropy      = _shannon_entropy(user_counter)

        f.L2_unique_agents     = float(len({l["agent_id"] for l in logs if l["agent_id"] != "0"}))
        f.L2_unique_decoders   = float(len({l["decoder_name"] for l in logs if l["decoder_name"]}))

        dst_ports = [l["dst_port"] for l in logs if l["dst_port"]]
        f.L2_unique_dst_ports    = float(len(set(dst_ports)))
        f.L2_high_risk_port_count = float(sum(1 for p in dst_ports if p in _HIGH_RISK_PORTS))

        # ── Contextual ───────────────────────────────────────────────────
        f.Ctx_hour          = float(h)
        f.Ctx_day_of_week   = float(last.weekday())
        f.Ctx_window_span_min = span_min

        return f
