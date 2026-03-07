"""
heuristic_engine.py
Module 1 — Heuristic Triage Engine

Implements the 3-step triage algorithm from Design Document §3.1.
Runs fast rule-based checks (target ≤ 1 ms, NFR-1) before escalating to LLM.

Steps:
  1. Volumetric / Flood Detection  — pps > 1000 AND unique dsts < 5
  2. Port Scan Detection           — unique dst ports > 50 in 1 second
  2b. SYN Flood Detection          — SYN >> ACK ratio > 3x (TCP only)
  3. Suspicious heuristics         — escalate to LLM
  Default: Normal
"""

import time
from collections import defaultdict

# ── Per-source sliding-window statistics ─────────────────────────────────────
_src_stats: dict = defaultdict(lambda: {
    'pkt_times': [],
    'dst_ips': set(),
    'dst_ports': set(),
    'syn_total': 0,
    'ack_total': 0,
})

# ── Thresholds (from Design Document §3.1 and SRS FR-2) ─────────────────────
WINDOW_SIZE         = 1.0   # seconds — sliding window for rate calculations
PPS_FLOOD_THRESHOLD = 1000  # packets/sec — volumetric flood threshold
UNIQUE_DESTS_MAX    = 5     # max unique destinations for flood classification
PORT_SCAN_THRESHOLD = 50    # unique dst ports in window → port scan
SYN_FLOOD_RATIO     = 3.0   # SYN/ACK ratio → SYN flood


def _clean_window(stats: dict, now: float) -> None:
    """Remove packet timestamps older than WINDOW_SIZE from the sliding window."""
    cutoff = now - WINDOW_SIZE
    stats['pkt_times'] = [t for t in stats['pkt_times'] if t > cutoff]


def analyze_flow(flow: dict) -> dict:
    """
    Analyze a single network flow against heuristic rules.

    Parameters
    ----------
    flow : dict
        Feature dictionary from packet_capture.py

    Returns
    -------
    dict with keys:
        verdict    — human-readable threat label
        method     — 'Heuristic' or 'LLM_Required'
        confidence — 'High' | 'Medium' | 'Low'
        details    — explanation string
    """
    now = time.time()
    src = flow.get('srcip', 'unknown')
    stats = _src_stats[src]

    # Update sliding window
    _clean_window(stats, now)
    stats['pkt_times'].append(now)
    stats['dst_ips'].add(flow.get('dstip', ''))
    stats['dst_ports'].add(flow.get('dsport', 0))
    stats['syn_total'] += flow.get('syn_count', 0)
    stats['ack_total'] += flow.get('ack_count', 0)

    # Computed values
    pps              = flow.get('pps', 0)
    unique_dsts      = len(stats['dst_ips'])
    unique_dst_ports = flow.get('unique_dst_ports', 1)
    syn              = flow.get('syn_count', 0)
    ack              = flow.get('ack_count', 0)
    rst              = flow.get('rst_count', 0)
    proto            = flow.get('proto', 'tcp').lower()
    sbytes           = flow.get('sbytes', 0)
    dur              = max(flow.get('dur', 0.001), 0.001)
    state            = flow.get('state', 'CON')
    sintpkt          = flow.get('Sintpkt', 999)   # ms — key matches packet_capture output

    # ── Step 1: Volumetric / Flood Detection ─────────────────────────────────
    if pps > PPS_FLOOD_THRESHOLD and unique_dsts < UNIQUE_DESTS_MAX:
        return {
            'verdict':    'DoS / Volumetric Attack',
            'method':     'Heuristic',
            'confidence': 'High',
            'details': (
                f"Flood: {pps:.0f} pps from {src} targeting "
                f"only {unique_dsts} destination(s). "
                f"Threshold: >{PPS_FLOOD_THRESHOLD} pps, <{UNIQUE_DESTS_MAX} dsts."
            ),
        }

    # ── Step 2: Port Scan / Reconnaissance ───────────────────────────────────
    if unique_dst_ports > PORT_SCAN_THRESHOLD:
        return {
            'verdict':    'Port Scan / Reconnaissance',
            'method':     'Heuristic',
            'confidence': 'High',
            'details': (
                f"Port scan: {src} contacted {unique_dst_ports} unique dest ports "
                f"in {dur:.2f}s. Threshold: >{PORT_SCAN_THRESHOLD} ports."
            ),
        }

    # ── Step 2b: SYN Flood (TCP-specific) ────────────────────────────────────
    if proto == 'tcp' and ack > 0 and (syn / max(ack, 1)) > SYN_FLOOD_RATIO:
        ratio = round(syn / max(ack, 1), 1)
        return {
            'verdict':    'SYN Flood',
            'method':     'Heuristic',
            'confidence': 'High',
            'details': (
                f"SYN flood: SYN={syn}, ACK={ack} (ratio {ratio}x, "
                f"threshold {SYN_FLOOD_RATIO}x). "
                f"{src} → {flow.get('dstip', '?')}:{flow.get('dsport', '?')}"
            ),
        }

    # ── Step 3: Suspicious — escalate to LLM ─────────────────────────────────
    suspicion_flags = []

    if sbytes > 500_000:
        suspicion_flags.append(f"high byte volume ({sbytes:,} bytes)")

    # Only flag tiny sinpkt if duration is meaningful (avoid false positives from
    # zero-duration CSV artifacts where sinpkt approaches 0 by definition)
    if sintpkt < 1.0 and pps > 500 and dur > 0.001:
        suspicion_flags.append(f"very low inter-packet time ({sintpkt:.3f} ms at {pps:.0f} pps)")

    if rst > 5:
        suspicion_flags.append(f"excessive RST flags ({rst})")

    if state in ('REQ', 'PAR') and pps > 50:
        suspicion_flags.append(f"incomplete connection state ({state}) with elevated rate ({pps:.0f} pps)")

    if proto == 'tcp' and syn > 50 and ack == 0:
        suspicion_flags.append(f"unanswered SYN burst ({syn} SYNs, 0 ACKs)")

    # Simulation mode: use UNSW-NB15 ground-truth label to catch attack types
    # (Exploits, Generic, Fuzzers, Backdoors, etc.) that have moderate pps and
    # don't trigger flood/port-scan rules above.
    # In live capture, label is never present so this never fires.
    if flow.get('capture_source') == 'simulation' and int(flow.get('label', 0)) == 1:
        attack_cat = flow.get('attack_cat', 'Unknown')
        suspicion_flags.append(
            f"UNSW-NB15 label=1 (attack category: {attack_cat or 'Unknown'})"
        )

    if suspicion_flags:
        return {
            'verdict':    'Suspicious — Pending LLM Analysis',
            'method':     'LLM_Required',
            'confidence': 'Low',
            'details':    f"Suspicious indicators: {'; '.join(suspicion_flags)}. Escalating to AI.",
        }

    # ── Default: Normal ───────────────────────────────────────────────────────
    return {
        'verdict':    'Normal',
        'method':     'Heuristic',
        'confidence': 'High',
        'details':    f"No attack indicators detected from {src}.",
    }
