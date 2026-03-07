"""
packet_capture.py
Module 1 — Real-Time Packet Capture and Flow Feature Extraction

Uses Scapy for live capture on Windows (requires Npcap + admin rights).
Falls back to CSV-based simulation if live capture is unavailable.

Architecture:
  - Background thread captures packets / replays CSV rows
  - Flows are assembled from raw packets using a 5-tuple key
  - Timed-out flows are flushed to a shared queue
  - Main app polls get_flows() to drain the queue
"""

import threading
import time
import queue
import pandas as pd
from collections import defaultdict
from datetime import datetime

# ── Try importing Scapy ──────────────────────────────────────────────────────
try:
    from scapy.all import sniff, IP, TCP, UDP, ICMP
    SCAPY_AVAILABLE = True
except Exception:
    SCAPY_AVAILABLE = False

# ── Shared state ─────────────────────────────────────────────────────────────
_flow_buffer = defaultdict(lambda: {
    'packets': [],
    'src_bytes': 0,
    'dst_bytes': 0,
    'start_time': None,
    'last_time': None,
    'dst_ports_seen': set(),
    'src_ports_seen': set(),
})

_flow_queue: queue.Queue = queue.Queue()
_capture_active = threading.Event()
_lock = threading.Lock()

FLOW_TIMEOUT = 5.0   # seconds of inactivity before a flow is flushed


# ── Packet handler (live capture path) ──────────────────────────────────────
def _packet_handler(pkt):
    """Process one captured packet and update the bidirectional flow buffer."""
    if not pkt.haslayer(IP):
        return

    ip = pkt[IP]
    now = time.time()
    proto = 'other'
    sport, dport = 0, 0
    flags = ''
    payload_size = len(pkt)

    if pkt.haslayer(TCP):
        proto = 'tcp'
        sport = pkt[TCP].sport
        dport = pkt[TCP].dport
        flags = str(pkt[TCP].flags)
    elif pkt.haslayer(UDP):
        proto = 'udp'
        sport = pkt[UDP].sport
        dport = pkt[UDP].dport
    elif pkt.haslayer(ICMP):
        proto = 'icmp'

    flow_key = (ip.src, ip.dst, sport, dport, proto)

    with _lock:
        flow = _flow_buffer[flow_key]
        if flow['start_time'] is None:
            flow['start_time'] = now
        flow['last_time'] = now
        flow['packets'].append({'size': payload_size, 'time': now, 'flags': flags})
        flow['src_bytes'] += payload_size
        flow['dst_ports_seen'].add(dport)
        flow['src_ports_seen'].add(sport)


# ── Flow flusher (background thread) ────────────────────────────────────────
def _flush_complete_flows():
    """Periodically move timed-out flows from the buffer to the queue."""
    while _capture_active.is_set():
        now = time.time()
        to_flush = []
        with _lock:
            for key, flow in list(_flow_buffer.items()):
                if flow['last_time'] and (now - flow['last_time']) > FLOW_TIMEOUT:
                    to_flush.append((key, dict(flow)))
                    del _flow_buffer[key]

        for key, flow in to_flush:
            feat = _extract_features(key, flow)
            _flow_queue.put(feat)

        time.sleep(1.0)


# ── Feature extraction ────────────────────────────────────────────────────────
def _extract_features(flow_key: tuple, flow: dict) -> dict:
    """Convert raw flow data into an UNSW-NB15-aligned feature dictionary."""
    src_ip, dst_ip, sport, dport, proto = flow_key
    pkts = flow['packets']
    n = len(pkts)
    duration = (flow['last_time'] - flow['start_time']) if n > 1 else 0.001
    src_bytes = flow['src_bytes']

    inter_arrivals = [
        pkts[i]['time'] - pkts[i - 1]['time'] for i in range(1, n)
    ] if n > 1 else [0]
    avg_iat = sum(inter_arrivals) / len(inter_arrivals) if inter_arrivals else 0

    syn_count = sum(1 for p in pkts if 'S' in p.get('flags', ''))
    ack_count = sum(1 for p in pkts if 'A' in p.get('flags', ''))
    fin_count = sum(1 for p in pkts if 'F' in p.get('flags', ''))
    rst_count = sum(1 for p in pkts if 'R' in p.get('flags', ''))
    psh_count = sum(1 for p in pkts if 'P' in p.get('flags', ''))

    pps = n / duration if duration > 0 else n
    bps = src_bytes / duration if duration > 0 else src_bytes

    return {
        'timestamp': datetime.fromtimestamp(flow['start_time']).strftime('%Y-%m-%d %H:%M:%S'),
        'srcip': src_ip,
        'dstip': dst_ip,
        'sport': sport,
        'dsport': dport,
        'proto': proto,
        'dur': round(duration, 4),
        'Spkts': n,
        'Dpkts': 0,
        'sbytes': src_bytes,
        'dbytes': 0,
        'smeansz': round(src_bytes / n, 2) if n > 0 else 0,
        'Sintpkt': round(avg_iat * 1000, 3),
        'Sload': round(bps, 2),
        'pps': round(pps, 2),
        'syn_count': syn_count,
        'ack_count': ack_count,
        'fin_count': fin_count,
        'rst_count': rst_count,
        'psh_count': psh_count,
        'unique_dst_ports': len(flow['dst_ports_seen']),
        'unique_src_ports': len(flow['src_ports_seen']),
        'state': _infer_state(proto, syn_count, ack_count, fin_count, rst_count),
        'capture_source': 'live',
    }


def _infer_state(proto: str, syn: int, ack: int, fin: int, rst: int) -> str:
    if proto != 'tcp':
        return 'CON'
    if rst > 0:
        return 'RST'
    if fin > 0:
        return 'FIN'
    if syn > 0 and ack > 0:
        return 'ACC'
    if syn > 0:
        return 'REQ'
    return 'CON'


# ── Public API — Live Capture ─────────────────────────────────────────────────
def start_live_capture(interface=None):
    """
    Start live packet capture in two background threads:
      Thread 1 — Scapy sniffer
      Thread 2 — Flow flusher
    Requires Npcap installed on Windows and admin privileges.
    """
    if not SCAPY_AVAILABLE:
        return False, "Scapy not installed. Run: pip install scapy"

    _capture_active.set()

    def _sniff():
        try:
            sniff(
                iface=interface,
                prn=_packet_handler,
                store=False,
                stop_filter=lambda _: not _capture_active.is_set(),
            )
        except Exception as e:
            _capture_active.clear()

    t1 = threading.Thread(target=_sniff, daemon=True)
    t2 = threading.Thread(target=_flush_complete_flows, daemon=True)
    t1.start()
    t2.start()
    return True, "Live capture started (Scapy)."


def stop_capture():
    """Signal all capture threads to stop."""
    _capture_active.clear()


def is_capture_active() -> bool:
    return _capture_active.is_set()


def get_flows(max_items: int = 50) -> list:
    """Drain up to max_items completed flows from the queue."""
    flows = []
    while not _flow_queue.empty() and len(flows) < max_items:
        try:
            flows.append(_flow_queue.get_nowait())
        except queue.Empty:
            break
    return flows


# ── Public API — CSV Simulation (Windows fallback) ───────────────────────────
_sim_thread = None


def start_csv_simulation(csv_path: str, delay: float = 0.4) -> tuple:
    """
    Simulate live traffic by replaying rows from the UNSW-NB15 CSV.
    Each row is fed into the flow queue with a small delay to mimic
    real-time arrival. No Npcap or admin rights required.

    UNSW-NB15 column names used (lowercase, no srcip/dstip in this version):
      spkts, dpkts, sbytes, dbytes, rate, sinpkt, sload, smean,
      dur, proto, state, attack_cat, label, etc.
    """
    global _sim_thread

    try:
        df = pd.read_csv(csv_path).sample(frac=1, random_state=99).reset_index(drop=True)
    except FileNotFoundError:
        return False, f"Dataset not found: {csv_path}"
    except Exception as e:
        return False, str(e)

    _capture_active.set()

    # ── IP pool for synthetic display (dataset has no srcip/dstip) ──────────
    _src_pool = [
        "10.0.0.1",  "10.0.0.2",  "10.0.0.5",  "10.0.0.10",
        "192.168.1.5","192.168.1.20","172.16.0.3","172.16.0.7",
    ]
    _dst_pool = [
        "192.168.1.1","192.168.1.254","10.10.0.1","10.10.0.2",
        "8.8.8.8",    "1.1.1.1",     "172.16.0.1","203.0.113.1",
    ]
    import random as _rng
    _rng.seed(42)

    def _simulate():
        for idx, row in df.iterrows():
            if not _capture_active.is_set():
                break

            # ── Read actual UNSW-NB15 column values (lowercase names) ──────
            dur_val    = float(row.get('dur',    0) or 0)
            spkts_val  = int(row.get('spkts',   1) or 1)
            dpkts_val  = int(row.get('dpkts',   0) or 0)
            sbytes_val = int(row.get('sbytes',  0) or 0)
            dbytes_val = int(row.get('dbytes',  0) or 0)
            rate_val   = float(row.get('rate',   0) or 0)   # actual pps from dataset
            sinpkt_val = float(row.get('sinpkt', 999) or 999)  # inter-pkt ms
            sload_val  = float(row.get('sload',  0) or 0)
            smean_val  = float(row.get('smean',  0) or 0)
            state_val  = str(row.get('state',   'CON'))
            proto_val  = str(row.get('proto',   'tcp')).lower()

            # Synthetic IPs / ports (for display only — not in this CSV)
            src_ip  = _src_pool[int(idx) % len(_src_pool)]
            dst_ip  = _dst_pool[int(idx) % len(_dst_pool)]
            s_port  = _rng.randint(1024, 65535)
            d_port  = _rng.choice([80, 443, 22, 53, 21, 8080, 3389, 445])

            # Infer TCP flag counts from state string
            syn_c = 1 if 'S' in state_val or state_val in ('REQ','INT') else 0
            ack_c = 1 if 'A' in state_val or state_val in ('ACC','FIN') else 0
            fin_c = 1 if 'F' in state_val or state_val == 'FIN' else 0
            rst_c = 1 if 'R' in state_val or state_val == 'RST' else 0

            feat = {
                'timestamp':        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'srcip':            src_ip,
                'dstip':            dst_ip,
                'sport':            s_port,
                'dsport':           d_port,
                'proto':            proto_val,
                'dur':              dur_val,
                'Spkts':            spkts_val,
                'Dpkts':            dpkts_val,
                'sbytes':           sbytes_val,
                'dbytes':           dbytes_val,
                'smeansz':          smean_val,
                'Sintpkt':          sinpkt_val,
                'Sload':            sload_val,
                'pps':              rate_val,        # use dataset's own rate column
                'syn_count':        syn_c,
                'ack_count':        ack_c,
                'fin_count':        fin_c,
                'rst_count':        rst_c,
                'psh_count':        0,
                'unique_dst_ports': 1,
                'unique_src_ports': 1,
                'state':            state_val,
                'attack_cat':       str(row.get('attack_cat', '')),
                'label':            int(row.get('label', 0) or 0),
                'capture_source':   'simulation',
            }
            _flow_queue.put(feat)
            time.sleep(delay)

    _sim_thread = threading.Thread(target=_simulate, daemon=True)
    _sim_thread.start()
    return True, "CSV simulation started (replaying UNSW-NB15 dataset)."
