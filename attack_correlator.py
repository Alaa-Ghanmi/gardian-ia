"""
attack_correlator.py
====================
Links individual events into multi-stage attack sequences (kill chains)
and provides historical pattern matching for the predictor.

The correlation problem
-----------------------
A random forest trained on individual observation windows can detect that
"many auth failures → brute force coming". But it cannot reason about:

  "The current activity looks like Stage 2 of a 4-stage attack pattern
   we saw three times before. Stage 4 (exfiltration) typically starts
   92 minutes after Stage 2."

The AttackCorrelator bridges this gap by:
  1. Maintaining a timeline of seen events per source entity (IP, host, user).
  2. Matching the current window's feature signature against learned
     historical attack patterns (attack signatures).
  3. Estimating expected time to next stage if a known pattern is matched.

Design decisions
----------------
- Correlation is keyed on (src_ip, agent_id) pairs — the most reliable
  identifiers across log sources. User-based correlation is secondary.
- Pattern matching uses a simple cosine similarity over the attack-class
  count vector (L1 features). This is intentionally NOT a deep neural net
  because it needs to be inspectable and explainable to SOC analysts.
- The "historical patterns" store is populated from training sessions.
  In production this would be backed by a time-series DB (e.g. OpenSearch).

Limitations
-----------
- Source IP pivoting can be fooled by attackers using many IPs (botnets).
  In that case host-based correlation (agent_id) is more reliable.
- Cosine similarity over L1 feature counts does not capture timing.
  A production system should use DTW or LSTM-based sequence matching
  for proper temporal pattern comparison.
- Pattern matching is O(n × m) where n = current observations,
  m = stored patterns. Fine for hundreds of patterns, needs indexing beyond.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger("guardian.correlator")

# ---------------------------------------------------------------------------
# Kill-chain stage taxonomy
# ---------------------------------------------------------------------------

KILL_CHAIN_ORDER = [
    "recon",
    "initial_access",
    "exploitation",
    "post_compromise",
    "impact",
]

# Which attack classes map to which kill-chain stage
CLASS_TO_STAGE = {
    "recon":               "recon",
    "brute_force":         "initial_access",
    "credential_stuffing": "initial_access",
    "sqli":                "exploitation",
    "web_attack":          "exploitation",
    "web_shell":           "exploitation",
    "priv_escalation":     "exploitation",
    "lateral_movement":    "post_compromise",
    "persistence":         "post_compromise",
    "suspicious_exec":     "post_compromise",
    "evasion":             "post_compromise",
    "c2":                  "post_compromise",
    "exfiltration":        "impact",
    "malware":             "impact",
    "ransomware":          "impact",
}

NEXT_STAGE = {
    "recon":            "initial_access",
    "initial_access":   "exploitation",
    "exploitation":     "post_compromise",
    "post_compromise":  "impact",
    "impact":           None,
}

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class EntityTimeline:
    """Tracks recent attack-class events for one (src_ip, agent_id) pair."""
    entity_key:   tuple
    events:       deque = field(default_factory=lambda: deque(maxlen=500))
    last_updated: datetime = field(default_factory=datetime.now)

    def add(self, ts: datetime, attack_class: str, rule_level: int) -> None:
        self.events.append((ts, attack_class, rule_level))
        self.last_updated = ts

    def stages_seen(self) -> list[str]:
        seen = {CLASS_TO_STAGE.get(cls, "unknown")
                for _, cls, _ in self.events
                if cls not in ("unknown", "audit", "no_attack")}
        return [s for s in KILL_CHAIN_ORDER if s in seen]

    def current_stage(self) -> str:
        stages = self.stages_seen()
        return stages[-1] if stages else "none"

    def next_expected_stage(self) -> Optional[str]:
        return NEXT_STAGE.get(self.current_stage())

    def dominant_class(self) -> str:
        from collections import Counter
        counts = Counter(cls for _, cls, _ in self.events
                        if cls not in ("unknown", "audit", "no_attack"))
        return counts.most_common(1)[0][0] if counts else "unknown"


@dataclass
class AttackPattern:
    """A learned historical attack pattern extracted from a session."""
    pattern_id:       int
    stage_sequence:   list[str]          # e.g. ["recon", "initial_access", "exploitation"]
    dominant_class:   str
    l1_vector:        list[float]        # L1 feature counts (normalised)
    avg_lead_minutes: float              # avg time from this pattern to attack
    std_lead_minutes: float              # uncertainty
    sample_count:     int                # how many sessions matched this pattern
    description:      str


@dataclass
class CorrelationResult:
    """Output of correlate_window()."""
    entity_key:            tuple
    current_stage:         str
    next_expected_stage:   Optional[str]
    stages_seen:           list[str]
    matched_pattern:       Optional[AttackPattern]
    pattern_similarity:    float          # cosine similarity [0, 1]
    estimated_minutes:     Optional[float]  # None if no pattern matched
    estimated_minutes_std: Optional[float]
    campaign_risk:         float          # 0–1 composite risk from timeline
    evidence:              list[str]


# ---------------------------------------------------------------------------
# Correlator
# ---------------------------------------------------------------------------

class AttackCorrelator:
    """
    Maintains entity timelines and matches current windows against
    historical attack patterns.

    Usage
    -----
    >>> correlator = AttackCorrelator()
    >>> correlator.load_patterns(patterns_from_training)
    >>> result = correlator.correlate_window(logs, features)
    """

    def __init__(self, timeline_ttl_hours: float = 12.0) -> None:
        self._timelines:    dict[tuple, EntityTimeline] = {}
        self._patterns:     list[AttackPattern]         = []
        self._ttl_hours     = timeline_ttl_hours
        self._pattern_counter = 0

    # ── Pattern management ────────────────────────────────────────────────

    def load_patterns(self, patterns: list[AttackPattern]) -> None:
        self._patterns = patterns
        logger.info("Loaded %d historical attack patterns.", len(patterns))

    def add_pattern(self, pattern: AttackPattern) -> None:
        self._patterns.append(pattern)

    def patterns_from_sessions(self, sessions, features_by_session: dict) -> list[AttackPattern]:
        """
        Build patterns from training sessions and their feature vectors.

        Parameters
        ----------
        sessions           : list[AttackSession]
        features_by_session: dict[session_id → list[WindowFeatures]]
                             pre-attack features grouped by session

        Returns
        -------
        list[AttackPattern] ready for load_patterns()
        """
        patterns = []
        for sid, session in enumerate(sessions):
            flist = features_by_session.get(sid, [])
            if not flist:
                continue

            # Average the L1 counts across observation windows for this session
            avg_l1 = _mean_l1_vector([f.to_list() for f in flist], len(flist[0].to_list()))
            # Normalise to unit vector for cosine comparison
            norm = _l2_norm(avg_l1) or 1.0
            avg_l1_norm = [v / norm for v in avg_l1]

            self._pattern_counter += 1
            p = AttackPattern(
                pattern_id       = self._pattern_counter,
                stage_sequence   = session.stage_sequence,
                dominant_class   = session.dominant_class,
                l1_vector        = avg_l1_norm,
                avg_lead_minutes = 60.0,   # placeholder — computed from lead times in training
                std_lead_minutes = 20.0,   # placeholder
                sample_count     = len(flist),
                description      = (
                    f"{session.dominant_class} campaign: "
                    f"{' → '.join(session.stage_sequence)} "
                    f"({session.event_count} events, peak level {session.peak_level})"
                ),
            )
            patterns.append(p)

        self._patterns = patterns
        return patterns

    # ── Runtime correlation ───────────────────────────────────────────────

    def ingest_window(self, logs: list[dict]) -> None:
        """
        Update entity timelines with the events in the current window.
        Call this before correlate_window() to keep timelines current.
        """
        now = datetime.now()
        self._expire_timelines(now)

        for log in logs:
            if log["attack_class"] in ("unknown", "audit"):
                continue
            ip  = log["src_ip"]  or "unknown_ip"
            aid = log["agent_id"] or "unknown_host"
            key = (ip, aid)
            if key not in self._timelines:
                self._timelines[key] = EntityTimeline(entity_key=key)
            self._timelines[key].add(
                log["timestamp"], log["attack_class"], log["rule_level"]
            )

    def correlate_window(
        self,
        logs:     list[dict],
        features, # WindowFeatures
    ) -> CorrelationResult:
        """
        Match the current observation window against historical patterns
        and return a correlation result.

        Parameters
        ----------
        logs     : normalised logs from the current window
        features : WindowFeatures extracted from the same logs

        Returns
        -------
        CorrelationResult with stage estimation and matched pattern.
        """
        # Find dominant entity in current window
        from collections import Counter
        ip_counter = Counter(l["src_ip"] for l in logs if l["src_ip"])
        aid_counter = Counter(l["agent_id"] for l in logs if l["agent_id"] != "0")
        top_ip  = ip_counter.most_common(1)[0][0]  if ip_counter  else "unknown"
        top_aid = aid_counter.most_common(1)[0][0] if aid_counter else "unknown"
        key = (top_ip, top_aid)

        timeline = self._timelines.get(key)

        current_stage      = timeline.current_stage()      if timeline else "none"
        next_stage         = timeline.next_expected_stage() if timeline else None
        stages_seen        = timeline.stages_seen()         if timeline else []
        campaign_risk      = self._campaign_risk(timeline)  if timeline else 0.0

        # Pattern matching via cosine similarity on L1 feature vector
        current_l1 = features.to_list()[:16]  # first 16 = L1 class counts
        norm = _l2_norm(current_l1) or 1.0
        current_l1_norm = [v / norm for v in current_l1]

        best_pattern: Optional[AttackPattern] = None
        best_sim     = 0.0
        SIM_THRESHOLD = 0.45  # minimum cosine similarity to accept a match

        for pattern in self._patterns:
            sim = _cosine_similarity(current_l1_norm, pattern.l1_vector)
            if sim > best_sim:
                best_sim     = sim
                best_pattern = pattern

        matched = best_pattern if best_sim >= SIM_THRESHOLD else None
        est_min = matched.avg_lead_minutes if matched else None
        est_std = matched.std_lead_minutes if matched else None

        evidence = _build_evidence(logs, features, stages_seen, current_stage)

        return CorrelationResult(
            entity_key            = key,
            current_stage         = current_stage,
            next_expected_stage   = next_stage,
            stages_seen           = stages_seen,
            matched_pattern       = matched,
            pattern_similarity    = round(best_sim, 3),
            estimated_minutes     = est_min,
            estimated_minutes_std = est_std,
            campaign_risk         = round(campaign_risk, 3),
            evidence              = evidence,
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _expire_timelines(self, now: datetime) -> None:
        expired = [
            k for k, tl in self._timelines.items()
            if (now - tl.last_updated).total_seconds() > self._ttl_hours * 3600
        ]
        for k in expired:
            del self._timelines[k]

    @staticmethod
    def _campaign_risk(timeline: EntityTimeline) -> float:
        """
        Composite risk score [0, 1] for an entity based on its timeline.
        Considers: number of stages seen, how recent the activity is,
        and whether post-compromise stages are present.
        """
        stages = timeline.stages_seen()
        if not stages:
            return 0.0

        # More stages = higher risk (max 1.0 at 5 stages)
        stage_score = len(stages) / 5.0

        # Post-compromise or impact stages double the urgency
        high_stages = {"post_compromise", "impact"}
        severity = 2.0 if any(s in high_stages for s in stages) else 1.0

        # Recency — decay risk if last event was > 2h ago
        age_hours = (datetime.now() - timeline.last_updated).total_seconds() / 3600.0
        recency = max(0.0, 1.0 - age_hours / 6.0)

        return min(1.0, stage_score * severity * recency)


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = _l2_norm(a)
    norm_b = _l2_norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _mean_l1_vector(vecs: list[list[float]], dim: int) -> list[float]:
    if not vecs:
        return [0.0] * dim
    result = [0.0] * dim
    for vec in vecs:
        for i, v in enumerate(vec[:dim]):
            result[i] += v
    n = len(vecs)
    return [v / n for v in result]


def _build_evidence(
    logs: list[dict],
    features,
    stages_seen: list[str],
    current_stage: str,
) -> list[str]:
    """Build a list of human-readable evidence strings for the analyst."""
    ev = []
    fv = dict(zip(features.feature_names(), features.to_list()))

    if fv.get("L1_brute_force_count", 0) >= 3:
        ev.append(f"Brute-force signals: {int(fv['L1_brute_force_count'])} auth-failure events")
    if fv.get("L1_priv_escalation_count", 0) >= 1:
        ev.append(f"Privilege escalation signals: {int(fv['L1_priv_escalation_count'])} events")
    if fv.get("L1_recon_count", 0) >= 2:
        ev.append(f"Reconnaissance signals: {int(fv['L1_recon_count'])} events")
    if fv.get("L1_lateral_movement_count", 0) >= 1:
        ev.append(f"Lateral movement signals: {int(fv['L1_lateral_movement_count'])} events")
    if fv.get("L1_exfiltration_count", 0) >= 1:
        ev.append(f"Exfiltration signals: {int(fv['L1_exfiltration_count'])} events")
    if fv.get("L2_suspicious_cmd_count", 0) >= 1:
        ev.append(f"Suspicious commands detected: {int(fv['L2_suspicious_cmd_count'])}")
    if fv.get("L2_external_ip_count", 0) >= 1:
        ev.append(f"External (non-RFC-1918) IP involvement: {int(fv['L2_external_ip_count'])} events")
    if stages_seen:
        ev.append(f"Kill-chain stages observed: {' → '.join(stages_seen)} (currently at: {current_stage})")
    if fv.get("L2_burst_score", 0) > 3.0:
        ev.append(f"Activity burst detected: {fv['L2_burst_score']:.1f}× above baseline rate")

    return ev or ["No strong correlating evidence in current window."]
