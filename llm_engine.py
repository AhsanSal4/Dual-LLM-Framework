"""
llm_engine.py
Module 1 — LLM Cognitive Layer

Handles ambiguous / suspicious flows escalated from the heuristic engine.
Uses LangChain + Google Gemini 1.5 Flash with few-shot prompting.
Target response time: ≤ 5 seconds (NFR-2).
LLM output is always grounded — no hallucination (NFR-4).
"""

import os
import json
import re
import time
import threading

# ── Rate-limit state ──────────────────────────────────────────────────────────
# Free-tier limits: 15 RPM / 1500 RPD for gemini-2.0-flash
# We enforce 1 call per 8 s (~7.5 RPM) with a thread lock so concurrent
# Streamlit threads can't both slip through at the same instant.
_rate_lock:      threading.Lock = threading.Lock()
_last_call_time: float = 0.0
_backoff_until:  float = 0.0
_MIN_INTERVAL:   float = 8.0   # seconds between LLM calls (~7.5 RPM max)

# ── Few-shot examples (Design Document §1.3 — few-shot prompting) ────────────
FEW_SHOT_EXAMPLES = """\
Example 1:
Flow: proto=TCP, src=10.0.0.5:4321 → dst=192.168.1.1:22, dur=0.05s,
      sbytes=1,200, Spkts=12, syn_count=10, ack_count=1, state=REQ
Analysis:
  classification: "Exploits"
  confidence: "High"
  attack_indicators: ["10 SYNs / 1 ACK in 0.05s targeting port 22",
                      "Incomplete connection (REQ state)", "Possible SSH brute-force"]
  explanation: "High SYN-to-ACK ratio targeting SSH port 22 in very short duration
               strongly indicates a brute-force or credential-stuffing attempt."
  mitigation: ["Block source IP at perimeter firewall",
               "Rate-limit SSH (max 5 connections/min/IP)",
               "Enable fail2ban / intrusion prevention on SSH service"]

Example 2:
Flow: proto=UDP, src=172.16.0.3:53421 → dst=8.8.8.8:53, dur=2.1s,
      sbytes=320, Spkts=4, state=CON
Analysis:
  classification: "Normal"
  confidence: "High"
  attack_indicators: []
  explanation: "Small UDP packets to port 53 (DNS) with normal byte volume and
               timing. No anomalies detected."
  mitigation: ["No action required"]

Example 3:
Flow: proto=TCP, src=10.1.2.3:50234 → dst=192.168.5.10:445, dur=0.1s,
      sbytes=870,000, rst_count=8, state=RST
Analysis:
  classification: "Exploits"
  confidence: "Medium"
  attack_indicators: ["870 KB to SMB port 445 in 0.1s", "8 RST flags",
                      "Abnormally large payload for service negotiation"]
  explanation: "Very high byte volume to port 445 (SMB) with multiple RST flags
               may indicate an exploit attempt such as EternalBlue / MS17-010."
  mitigation: ["Apply MS17-010 patch immediately", "Block port 445 at network edge",
               "Capture full pcap for payload inspection"]
"""

# ── Prompt template ───────────────────────────────────────────────────────────
_PROMPT_TEMPLATE = """\
You are a senior cybersecurity analyst reviewing a suspicious network flow \
that was escalated by an automated heuristic IDS engine.

Reference examples of known attack patterns:
{few_shot_examples}

Analyze the following new flow:
{flow_description}

Respond ONLY with a single valid JSON object — no prose, no markdown fences — \
using this exact structure:
{{
  "classification": "<one of: Normal | DoS | Reconnaissance | Exploits | Fuzzers | Generic | Backdoors | Shellcode | Worms | Analysis | SYN Flood | Port Scan>",
  "confidence": "<High | Medium | Low>",
  "attack_indicators": ["<indicator 1>", "<indicator 2>"],
  "explanation": "<2-3 sentence plain-English explanation citing only the provided flow data>",
  "mitigation": ["<action 1>", "<action 2>", "<action 3>"]
}}
"""

# ── Singleton LCEL chain ──────────────────────────────────────────────────────
_lcel_chain = None


def _get_chain():
    """Lazily initialise and cache the LCEL chain."""
    global _lcel_chain
    if _lcel_chain is not None:
        return _lcel_chain

    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Add it to .streamlit/secrets.toml or set it as an environment variable."
        )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0.1,
        google_api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_template(_PROMPT_TEMPLATE)
    _lcel_chain = prompt | llm | StrOutputParser()
    return _lcel_chain


# ── Flow → readable text ──────────────────────────────────────────────────────
def flow_to_text(flow: dict) -> str:
    """
    Convert a flow feature dictionary to a human-readable description
    suitable for inclusion in the LLM prompt.
    """
    lines = [
        f"Protocol      : {flow.get('proto', 'unknown').upper()}",
        f"Source        : {flow.get('srcip', '?')}:{flow.get('sport', '?')}",
        f"Destination   : {flow.get('dstip', '?')}:{flow.get('dsport', '?')}",
        f"Duration      : {flow.get('dur', 0):.4f} s",
        f"Source bytes  : {flow.get('sbytes', 0):,}",
        f"Source packets: {flow.get('Spkts', 0)}",
        f"Packets/sec   : {flow.get('pps', 0):.1f}",
        f"Mean pkt size : {flow.get('smeansz', 0):.1f} bytes",
        f"Avg inter-pkt : {flow.get('Sintpkt', 0):.3f} ms",
        (
            f"TCP Flags     : SYN={flow.get('syn_count', 0)}, "
            f"ACK={flow.get('ack_count', 0)}, "
            f"FIN={flow.get('fin_count', 0)}, "
            f"RST={flow.get('rst_count', 0)}, "
            f"PSH={flow.get('psh_count', 0)}"
        ),
        f"Conn state    : {flow.get('state', '?')}",
        f"Unique dst ports contacted: {flow.get('unique_dst_ports', 1)}",
        f"Heuristic note: {flow.get('heuristic_details', 'Escalated by heuristic engine')}",
    ]
    return "\n".join(lines)


# ── Public API ────────────────────────────────────────────────────────────────
def analyze_flow_with_llm(flow: dict) -> dict:
    """
    Send a suspicious flow to Gemini 2.0 Flash for deep analysis.

    Returns a dict with keys:
        classification, confidence, attack_indicators,
        explanation, mitigation, flow_text, raw_response
    """
    flow_text = flow_to_text(flow)

    # ── Acquire shared rate-limit slot (used by Forensic Console too) ─────────
    remaining_backoff = acquire_call_slot()
    if remaining_backoff is not None:
        return {
            'classification':    'Rate Limited',
            'confidence':        'Low',
            'attack_indicators': [],
            'explanation':       (
                f"API rate limit active — retry in {remaining_backoff:.0f}s. "
                "15 requests/minute limit shared across Live Dashboard and Forensic Console."
            ),
            'mitigation':        ['Wait for backoff to expire, then retry.'],
            'flow_text':         flow_text,
            'raw_response':      'skipped (rate limited)',
        }

    chain = _get_chain()
    raw = ""

    try:
        raw = chain.invoke({
            "few_shot_examples": FEW_SHOT_EXAMPLES,
            "flow_description":  flow_text,
        })

        # Extract the JSON block from the response
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        if start == -1 or end <= start:
            raise ValueError("LLM did not return a JSON object.")

        result = json.loads(raw[start:end])
        result['flow_text']    = flow_text
        result['raw_response'] = raw
        return result

    except (json.JSONDecodeError, ValueError) as e:
        classification = _regex_field(raw, 'classification') or 'Unknown'
        confidence     = _regex_field(raw, 'confidence') or 'Low'
        return {
            'classification':    classification,
            'confidence':        confidence,
            'attack_indicators': [],
            'explanation':       f"LLM returned non-JSON response. Parse error: {e}",
            'mitigation':        ['Manual analyst review required.'],
            'flow_text':         flow_text,
            'raw_response':      raw,
        }
    except Exception as e:
        err = str(e)
        # Detect 429 — set backoff window so subsequent calls skip the API
        if '429' in err or 'RESOURCE_EXHAUSTED' in err:
            retry_delay = record_429(err)
            explanation = (
                f"API rate limit hit (429 — 15 requests/minute exceeded). "
                f"Backoff for {retry_delay:.0f}s. "
                "Wait for the backoff to expire, then retry."
            )
        else:
            explanation = f"LLM call failed: {err}"
        return {
            'classification':    'Analysis Failed',
            'confidence':        'Low',
            'attack_indicators': [],
            'explanation':       explanation,
            'mitigation':        ['Check API key and quota. Manual review required.'],
            'flow_text':         flow_text,
            'raw_response':      err,
        }


def _regex_field(text: str, field: str) -> str | None:
    """Best-effort regex extraction of a JSON string field value."""
    pattern = rf'"{field}"\s*:\s*"([^"]+)"'
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(1) if match else None


# ── Shared rate-limit helpers (used by both Live Dashboard and Forensic Console) ──
def acquire_call_slot() -> float | None:
    """
    Atomically acquire a rate-limited API call slot.
    Returns remaining backoff seconds if still in a 429 window (caller must abort).
    Returns None if slot was successfully acquired (after any needed sleep).
    """
    global _last_call_time, _backoff_until
    with _rate_lock:
        now = time.time()
        remaining = _backoff_until - now
        if remaining > 0:
            return remaining
        sleep_needed = max(0.0, _MIN_INTERVAL - (now - _last_call_time))
        _last_call_time = now + sleep_needed
    if sleep_needed > 0:
        time.sleep(sleep_needed)
    return None


def record_429(err_str: str) -> float:
    """
    Parse a 429 error string, set the shared backoff window, return retry_delay.
    """
    global _backoff_until
    retry_delay = 90.0
    m = re.search(r'retry[^0-9]+([0-9]+)', err_str, re.IGNORECASE)
    if m:
        retry_delay = max(90.0, float(m.group(1)) + 10)
    with _rate_lock:
        _backoff_until = time.time() + retry_delay
    return retry_delay
