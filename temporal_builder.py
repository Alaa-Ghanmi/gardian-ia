"""
temporal_builder.py
===================
Builds training samples using ATTACK SESSION RECONSTRUCTION.

The core problem with naive temporal labeling
---------------------------------------------
Naive approach: take logs from [T-60min, T], label them "attack" if
logs from [T, T+3h] look bad.

This fails because:
1. "Attack in next 3 hours" is an arbitrary fixed horizon that doesn't
   reflect how real attacks progress.
2. Logs BEFORE an attack may be from unrelated benign activity on the
   same host, creating false correlations. The model learns noise.
3. It cannot distinguish attack stages: recon vs. exploitation vs. impact.

What we do instead
------------------
STEP 1 — Reconstruct attack sessions from labeled attack logs.
  An attack "session" is a cluster of temporally adjacent logs that
  share at least one attack class indicator and come from overlapping
  IP/host contexts. Sessions have a START time and a END time.

STEP 2 — For each session, generate labeled training windows.
  For each attack session S:
    For each observation point T in [S.start - 4h, S.start - 10min]:
      past_logs  = logs in [T - obs_window, T]
      label      = AttackLabel(
                     will_attack=True,
                     attack_class=S.dominant_class,
                     minutes_to_attack=(S.start - T).minutes,
                     stage=estimate_stage(T, S),
                   )

  This creates samples where:
  - Features come from activity BEFORE the attack starts (no leakage)
  - Time-to-attack is the actual measured gap
  - The label knows the attack type and the attack stage

STEP 3 — Generate negative (no-attack) samples from quiet periods.
  Quiet periods = windows that are ≥ 2h from any known attack session.

STEP 4 — Encode time-to-attack as a regression target AND a class.
  Regression: predict exact minutes until attack (or 0 if no attack).
  Classification: predict the attack class (multi-class).

This design means the model learns:
  "When I see THIS behavior, an attack of type X starts in Y minutes."

Limitations
-----------
- Session reconstruction works best with labeled (known-attack) data.
  For unknown attacks the system falls back to behavioral anomaly scores.
- The heuristic session-gap threshold (SESSION_GAP_MINUTES) is tunable
  but not automatically learned. Ground truth incident labels from
  a SIEM or IR system would be more reliable.
- Time-to-attack estimates have uncertainty that grows as the horizon
  lengthens. The model's confidence intervals are wide beyond 2 hours.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Generator

from feature_extractor import WindowFeatureExtractor, normalise_log

logger = logging.getLogger("guardian.temporal_builder")

# ---------------------------------------------------------------------------
# Constants — all tunable
# ---------------------------------------------------------------------------

SESSION_GAP_MINUTES  = 30    # gap between events that splits two sessions
OBS_WINDOW_MINUTES   = 60    # past observation window length
MIN_LEAD_MINUTES     = 10    # shortest meaningful prediction lead time
MAX_LEAD_MINUTES     = 240   # beyond this, prediction is too uncertain
QUIET_BUFFER_HOURS   = 2.0   # min distance from attack to call a window "normal"
SAMPLES_PER_SESSION  = 6     # how many observation points per attack session


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AttackSession:
    """A reconstructed attack campaign from a cluster of attack logs."""
    start:          datetime
    end:            datetime
    dominant_class: str            # most frequent attack class in session
    classes_seen:   list[str]      # all attack classes in this session
    src_ips:        set            # IPs involved
    agent_ids:      set            # hosts involved
    peak_level:     int            # max Wazuh rule_level
    event_count:    int

    @property
    def duration_min(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0

    @property
    def stage_sequence(self) -> list[str]:
        """
        Attempt to infer kill-chain stage sequence from the classes seen.
        Order matters — recon before exploitation before impact.
        """
        _stage_order = {
            "recon":            1,
            "brute_force":      2,
            "credential_stuffing": 2,
            "sqli":             3,
            "web_attack":       3,
            "web_shell":        4,
            "priv_escalation":  4,
            "lateral_movement": 5,
            "persistence":      5,
            "suspicious_exec":  5,
            "evasion":          5,
            "c2":               6,
            "exfiltration":     7,
            "malware":          6,
            "ransomware":       8,
        }
        return sorted(set(self.classes_seen),
                      key=lambda c: _stage_order.get(c, 99))


@dataclass
class AttackLabel:
    """Ground-truth label for one training sample."""
    will_attack:        bool
    attack_class:       str     # "no_attack" | any attack class
    minutes_to_attack:  float   # 0 if no_attack; actual gap otherwise
    attack_stage:       str     # "recon" | "exploitation" | "impact" | "none"
    session_id:         int     # which session this sample belongs to (-1 = none)

    @property
    def binary_label(self) -> int:
        return int(self.will_attack)


@dataclass
class TrainingSample:
    features:  object        # WindowFeatures
    label:     AttackLabel


# ---------------------------------------------------------------------------
# Session reconstructor
# ---------------------------------------------------------------------------

class AttackSessionReconstructor:
    """
    Groups a list of normalised logs into attack sessions.

    Session definition:
      A contiguous cluster of logs where each consecutive pair is at most
      SESSION_GAP_MINUTES apart AND at least one of the logs carries a
      non-unknown, non-audit attack class.

    Why this matters:
      Without session reconstruction, a naive model might correlate
      totally unrelated benign traffic with an attack that happens to
      occur shortly after it. Sessions anchor each training sample to
      logs that are causally near the attack.
    """

    def __init__(self, gap_minutes: int = SESSION_GAP_MINUTES) -> None:
        self.gap_minutes = gap_minutes

    def reconstruct(self, all_logs: list[dict]) -> list[AttackSession]:
        """
        Parameters
        ----------
        all_logs : ALL normalised logs (sorted by timestamp),
                   including both attack and benign events.

        Returns
        -------
        List of AttackSession, sorted by start time.
        """
        if not all_logs:
            return []

        # Keep only logs that belong to a known attack class
        attack_logs = [l for l in all_logs
                       if l["attack_class"] not in ("unknown", "audit")]
        if not attack_logs:
            logger.info("No attack-class events found; no sessions reconstructed.")
            return []

        sessions: list[AttackSession] = []
        cluster: list[dict] = [attack_logs[0]]

        for log in attack_logs[1:]:
            gap = (log["timestamp"] - cluster[-1]["timestamp"]).total_seconds() / 60.0
            if gap <= self.gap_minutes:
                cluster.append(log)
            else:
                sessions.append(self._build_session(cluster))
                cluster = [log]

        sessions.append(self._build_session(cluster))
        logger.info("Reconstructed %d attack sessions from %d attack events.",
                    len(sessions), len(attack_logs))
        return sessions

    @staticmethod
    def _build_session(cluster: list[dict]) -> AttackSession:
        class_counter: dict[str, int] = defaultdict(int)
        for l in cluster:
            class_counter[l["attack_class"]] += 1
        dominant = max(class_counter, key=class_counter.__getitem__)

        return AttackSession(
            start          = cluster[0]["timestamp"],
            end            = cluster[-1]["timestamp"],
            dominant_class = dominant,
            classes_seen   = list(class_counter.keys()),
            src_ips        = {l["src_ip"] for l in cluster if l["src_ip"]},
            agent_ids      = {l["agent_id"] for l in cluster if l["agent_id"] != "0"},
            peak_level     = max(l["rule_level"] for l in cluster),
            event_count    = len(cluster),
        )


# ---------------------------------------------------------------------------
# Training sample builder
# ---------------------------------------------------------------------------

def _attack_stage(minutes_to_attack: float, dominant_class: str) -> str:
    """
    Estimate which kill-chain stage the observation window is at,
    given the dominant class and time until attack materialises.

    This is a heuristic — it improves interpretability but is not a
    ground-truth label. A production system would use analyst-confirmed
    incident timelines.
    """
    if dominant_class in ("recon",):
        return "recon"
    if dominant_class in ("brute_force", "credential_stuffing", "sqli", "web_attack"):
        return "initial_access"
    if dominant_class in ("priv_escalation", "web_shell", "suspicious_exec"):
        return "exploitation"
    if dominant_class in ("lateral_movement", "persistence", "evasion", "c2"):
        return "post_compromise"
    if dominant_class in ("exfiltration", "ransomware", "malware"):
        return "impact"
    return "unknown"


class TemporalSampleBuilder:
    """
    Generates (WindowFeatures, AttackLabel) training pairs using
    attack session reconstruction.

    Parameters
    ----------
    obs_window_min   : how far back from the prediction point to collect logs
    min_lead_min     : shortest lead time (predictions too close to attack start
                       are not useful — the attack is essentially in progress)
    max_lead_min     : longest lead time (beyond this, signal is too weak)
    samples_per_session : observation points to generate per session
    quiet_buffer_hrs : how far a "normal" window must be from any attack session
    """

    def __init__(
        self,
        obs_window_min:      int   = OBS_WINDOW_MINUTES,
        min_lead_min:        int   = MIN_LEAD_MINUTES,
        max_lead_min:        int   = MAX_LEAD_MINUTES,
        samples_per_session: int   = SAMPLES_PER_SESSION,
        quiet_buffer_hrs:    float = QUIET_BUFFER_HOURS,
    ) -> None:
        self.obs_window_min      = obs_window_min
        self.min_lead_min        = min_lead_min
        self.max_lead_min        = max_lead_min
        self.samples_per_session = samples_per_session
        self.quiet_buffer_hrs    = quiet_buffer_hrs
        self._extractor          = WindowFeatureExtractor()

    def build(
        self,
        all_logs:  list[dict],
        sessions:  list[AttackSession],
    ) -> list[TrainingSample]:
        """
        Build training samples from all_logs and reconstructed sessions.

        Attack samples: observation windows that precede a session start
          by [min_lead_min, max_lead_min] minutes.

        Negative samples: observation windows that are at least
          quiet_buffer_hrs from any session.
        """
        samples: list[TrainingSample] = []
        all_logs_sorted = sorted(all_logs, key=lambda l: l["timestamp"])

        # ── Attack samples ────────────────────────────────────────────────
        for sid, session in enumerate(sessions):
            attack_start = session.start

            # Generate multiple observation points in the lead-up to the attack
            lead_times = _linspace(
                self.min_lead_min,
                min(self.max_lead_min, (attack_start - all_logs_sorted[0]["timestamp"]).total_seconds() / 60.0),
                self.samples_per_session,
            )

            for lead_min in lead_times:
                pred_point = attack_start - timedelta(minutes=lead_min)
                win_start  = pred_point   - timedelta(minutes=self.obs_window_min)

                past_logs = [
                    l for l in all_logs_sorted
                    if win_start <= l["timestamp"] <= pred_point
                ]
                if len(past_logs) < 3:
                    continue

                features = self._extractor.extract(past_logs)
                label = AttackLabel(
                    will_attack       = True,
                    attack_class      = session.dominant_class,
                    minutes_to_attack = lead_min,
                    attack_stage      = _attack_stage(lead_min, session.dominant_class),
                    session_id        = sid,
                )
                samples.append(TrainingSample(features=features, label=label))

        # ── Negative (normal) samples ─────────────────────────────────────
        attack_times = {s.start for s in sessions} | {s.end for s in sessions}

        def _near_attack(t: datetime) -> bool:
            for s in sessions:
                if abs((t - s.start).total_seconds()) < self.quiet_buffer_hrs * 3600:
                    return True
                if abs((t - s.end).total_seconds()) < self.quiet_buffer_hrs * 3600:
                    return True
            return False

        # Sample observation points from the beginning of the log timeline
        if all_logs_sorted:
            t_min = all_logs_sorted[0]["timestamp"]
            t_max = all_logs_sorted[-1]["timestamp"]
            total_hours = (t_max - t_min).total_seconds() / 3600.0
            n_negatives = max(len(sessions) * self.samples_per_session, 20)
            step_hours  = total_hours / max(n_negatives, 1)

            for i in range(n_negatives):
                pred_point = t_min + timedelta(hours=i * step_hours + self.obs_window_min / 60.0)
                if _near_attack(pred_point):
                    continue
                win_start = pred_point - timedelta(minutes=self.obs_window_min)
                past_logs = [
                    l for l in all_logs_sorted
                    if win_start <= l["timestamp"] <= pred_point
                ]
                if len(past_logs) < 3:
                    continue
                features = self._extractor.extract(past_logs)
                label = AttackLabel(
                    will_attack       = False,
                    attack_class      = "no_attack",
                    minutes_to_attack = 0.0,
                    attack_stage      = "none",
                    session_id        = -1,
                )
                samples.append(TrainingSample(features=features, label=label))

        pos = sum(1 for s in samples if s.label.will_attack)
        logger.info(
            "Built %d training samples (%d positive, %d negative).",
            len(samples), pos, len(samples) - pos,
        )
        return samples


def _linspace(start: float, stop: float, n: int) -> list[float]:
    """Evenly spaced floats from start to stop (inclusive), n points."""
    if n <= 1 or stop <= start:
        return [start]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]
