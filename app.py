"""
app.py
Dual LLM Framework — Network Intrusion Detection & Digital Forensics

Streamlit application with three tabs:
  🖥️  Live Dashboard   — real-time packet capture → heuristic triage → LLM analysis
  🔍  Forensic Console — RAG-based natural-language queries over UNSW-NB15 (Module 2)
  📋  System Logs      — full detection history with filters and CSV export

Design Document §4 (Interface Design) — all three interface sections implemented here.
SRS FR-1 through FR-6, NFR-1 through NFR-5.
"""

import streamlit as st
import pandas as pd
import os
import torch
import gc
from datetime import datetime

# ── LangChain / Gemini ────────────────────────────────────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import google.generativeai as genai

# ── Local IDS modules ─────────────────────────────────────────────────────────
try:
    import packet_capture as pc
    import heuristic_engine as he
    import llm_engine as le
    MODULES_LOADED = True
    MODULE_ERROR = ""
except ImportError as e:
    MODULES_LOADED = False
    MODULE_ERROR = str(e)

# ── Optional auto-refresh (for live tab) ─────────────────────────────────────
try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# ═════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Network IDS — Dual LLM Framework",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ═════════════════════════════════════════════════════════════════════════════
# API KEY SETUP  (NFR-5: keys via env / secrets, never hardcoded)
# ═════════════════════════════════════════════════════════════════════════════
google_api_key = None
if "GOOGLE_API_KEY" in st.secrets:
    google_api_key = st.secrets["GOOGLE_API_KEY"]
else:
    google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key or google_api_key == "your-gemini-api-key-here":
    st.warning(
        "⚠️ **Google API Key not configured.** "
        "Add it to `.streamlit/secrets.toml` or enter it below.",
        icon="🔑",
    )
    user_key = st.text_input("Enter your Gemini API Key:", type="password")
    st.caption("Get a free key at [aistudio.google.com](https://aistudio.google.com) → Get API Key → Create API key")
    if user_key:
        os.environ["GOOGLE_API_KEY"] = user_key
        genai.configure(api_key=user_key)
        st.rerun()
    else:
        st.stop()
else:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    genai.configure(api_key=google_api_key)

# ═════════════════════════════════════════════════════════════════════════════
# SESSION STATE DEFAULTS
# ═════════════════════════════════════════════════════════════════════════════
_defaults = {
    "chat_history":   [],
    "system_logs":    [],
    "capture_active": False,
    "use_simulation": True,
    "llm_enabled":    True,
    "total_flows":    0,
    "attack_count":   0,
    "normal_count":   0,
    "llm_count":      0,
    "proto_counts":   {},
    "verdict_counts": {},
    "llm_session_cap": 20,   # max LLM calls per session to protect free-tier daily quota
}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ═════════════════════════════════════════════════════════════════════════════
# CACHED RAG CHAIN  (Module 2 — Forensic Engine)
# Design Doc §1.2 / SRS FR-4, FR-5, FR-6
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading forensic engine…")
def load_forensic_chain():
    """
    Load (or auto-build) ChromaDB and initialise the Gemini 2.0 Flash RAG chain.
    Retrieval k=15 per Design Document §2.2.
    On Streamlit Cloud the DB won't exist on first deploy — it is built
    automatically from the committed CSV (takes ~2 min on first run only).
    Returns (retriever, llm, prompt, error_message).
    """
    persist_dir = "./chroma_db"
    if not os.path.exists(persist_dir):
        # Auto-build — happens on first deploy or fresh clone (Design Doc §1.4)
        try:
            from build_chroma_db import build_and_persist_chroma_db
            build_and_persist_chroma_db()
        except Exception as build_err:
            return None, None, None, (
                f"Failed to auto-build ChromaDB: {build_err}. "
                "Run `python build_chroma_db.py` manually in the project folder."
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device},
        encode_kwargs={"batch_size": 64},
    )
    torch.cuda.empty_cache()
    gc.collect()

    docsearch = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    retriever = docsearch.as_retriever(search_kwargs={"k": 15})

    # Gemini 2.0 Flash — Design Document §1.3
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

    # SRS FR-6: RAG grounded prompt — LLM must cite only retrieved context (NFR-4)
    forensic_prompt = ChatPromptTemplate.from_template("""\
You are a highly skilled cybersecurity analyst. \
Analyze the following network traffic records retrieved from the UNSW-NB15 dataset:

{context}

Question: {question}

Provide a structured markdown response covering:

### 1. Threat Classification & Confidence
- **Classification:** (Normal / Exploits / DoS / Reconnaissance / Fuzzers / \
Generic / Shellcode / Worms / Backdoors / Analysis / Undetermined)
- **Confidence:** High / Medium / Low
- **Reasoning:** Cite specific feature values from the retrieved records only. \
Do not speculate beyond the provided data.

### 2. Key Indicators & Pattern Analysis
- List quantifiable anomalies, flag patterns, timing anomalies, and behavioural \
relationships observed across the retrieved records.

### 3. Actionable Mitigation Steps
- Concrete, prioritised incident-response actions.
- Further investigations or data sources needed for a complete picture.
""")

    return retriever, llm, forensic_prompt, None


def _run_forensic_query(retriever, llm, prompt, question: str) -> dict:
    """
    Execute a forensic RAG query using LCEL.
    Returns {"result": str, "source_documents": list, "error": str|None}
    """
    source_docs = retriever.invoke(question)
    context = "\n\n---\n\n".join([d.page_content for d in source_docs])
    chain = prompt | llm | StrOutputParser()
    try:
        result = chain.invoke({"context": context, "question": question})
        return {"result": result, "source_documents": source_docs, "error": None}
    except Exception as exc:
        err = str(exc)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            import re as _re
            m = _re.search(r'retry[^0-9]+([0-9]+)', err, _re.IGNORECASE)
            wait = int(m.group(1)) + 5 if m else 65
            friendly = (
                f"\u26a0\ufe0f **API quota exhausted (429).**\n\n"
                f"The Gemini free tier has a limit of **1,500 requests/day** shared across "
                f"the Live Dashboard and Forensic Console. Today\u2019s quota is used up.\n\n"
                f"**Retry in {wait}s**, or wait until midnight (Pacific time) for the daily "
                f"reset. Alternatively, enable billing on your Google AI project for higher limits."
            )
        else:
            friendly = f"LLM error: {err}"
        return {"result": friendly, "source_documents": source_docs, "error": err}


# ═════════════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════════════
st.title("🛡️ Dual LLM Framework — Network Intrusion Detection & Digital Forensics")
st.caption(
    "Real-time hybrid IDS (Scapy + heuristics + Gemini 2.0 Flash) "
    "· RAG forensic analysis over UNSW-NB15 "
    "· LangChain · ChromaDB · Honors Project 2026"
)

tab1, tab2, tab3 = st.tabs(["🖥️ Live Dashboard", "🔍 Forensic Console", "📋 System Logs"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE DASHBOARD  (Design Doc §4.1)
# SRS FR-1 (ingestion), FR-2 (triage), FR-3 (LLM reasoning)
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    # Auto-refresh every 3 s while capture is active
    if st.session_state.capture_active and AUTOREFRESH_AVAILABLE:
        st_autorefresh(interval=3000, key="live_refresh")

    st.subheader("Real-Time Network Traffic Monitoring")

    if not MODULES_LOADED:
        st.error(f"⚠️ IDS modules failed to load: {MODULE_ERROR}")
        st.stop()

    # ── Controls ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns([1, 1, 2, 2])

    with c1:
        if not st.session_state.capture_active:
            if st.button("▶ Start Capture", type="primary", use_container_width=True):
                csv_path = "./UNSW_NB15_attack_binary_bits.csv"
                if st.session_state.use_simulation:
                    ok, msg = pc.start_csv_simulation(csv_path, delay=0.3)
                else:
                    ok, msg = pc.start_live_capture()
                if ok:
                    st.session_state.capture_active = True
                    st.rerun()
                else:
                    st.error(msg)
        else:
            if st.button("⏹ Stop Capture", type="secondary", use_container_width=True):
                pc.stop_capture()
                st.session_state.capture_active = False
                st.rerun()

    with c2:
        if st.button("🗑 Clear All", use_container_width=True):
            st.session_state.system_logs    = []
            st.session_state.proto_counts   = {}
            st.session_state.verdict_counts = {}
            st.session_state.total_flows    = 0
            st.session_state.attack_count   = 0
            st.session_state.normal_count   = 0
            st.session_state.llm_count      = 0
            st.rerun()

    with c3:
        sim = st.toggle(
            "📁 Use CSV Simulation (no Npcap/admin required)",
            value=st.session_state.use_simulation,
            disabled=st.session_state.capture_active,
        )
        st.session_state.use_simulation = sim

    with c4:
        llm_on = st.toggle(
            "🤖 Enable LLM analysis for suspicious flows",
            value=st.session_state.llm_enabled,
        )
        st.session_state.llm_enabled = llm_on
        cap = st.slider(
            "Max LLM calls this session",
            min_value=5, max_value=100,
            value=st.session_state.llm_session_cap,
            step=5,
            help="Free tier: ~1500 calls/day shared. Lower = more quota preserved.",
            disabled=not llm_on,
        )
        st.session_state.llm_session_cap = cap

    # Status badges
    if st.session_state.capture_active:
        mode = "CSV Simulation" if st.session_state.use_simulation else "Live Scapy Capture"
        st.success(f"🟢 **ACTIVE** — {mode}", icon="📡")
        if not AUTOREFRESH_AVAILABLE:
            st.info(
                "Tip: Install `streamlit-autorefresh` for automatic page refresh.",
                icon="ℹ️",
            )
    else:
        st.info("🔴 Capture stopped — press ▶ Start Capture to begin monitoring.", icon="ℹ️")

    # LLM quota / backoff status
    if MODULES_LOADED:
        import time as _time
        remaining = le._backoff_until - _time.time()
        if remaining > 0:
            st.warning(
                f"⏳ **API rate-limit backoff** — LLM calls paused for **{remaining:.0f}s** more. "
                "Free-tier quota exhausted. Heuristic detections continue unaffected.",
                icon="🔴",
            )

    # ── Drain flow queue and run triage pipeline ──────────────────────────────
    if st.session_state.capture_active:
        new_flows = pc.get_flows(max_items=25)
        for flow in new_flows:
            st.session_state.total_flows += 1

            proto = flow.get("proto", "other").lower()
            st.session_state.proto_counts[proto] = (
                st.session_state.proto_counts.get(proto, 0) + 1
            )

            # FR-2: Heuristic triage (NFR-1: ≤ 1 ms)
            triage = he.analyze_flow(flow)

            log_entry = {
                "timestamp":       flow.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                "srcip":           flow.get("srcip", "?"),
                "dstip":           flow.get("dstip", "?"),
                "sport":           flow.get("sport", "?"),
                "dsport":          flow.get("dsport", "?"),
                "proto":           proto,
                "dur":             round(float(flow.get("dur", 0)), 4),
                "sbytes":          int(flow.get("sbytes", 0)),
                "pps":             round(float(flow.get("pps", 0)), 1),
                "state":           flow.get("state", "?"),
                "verdict":         triage["verdict"],
                "method":          triage["method"],
                "confidence":      triage["confidence"],
                "details":         triage["details"],
                "llm_class":       "",
                "llm_explanation": "",
                "attack_cat_gt":   flow.get("attack_cat", ""),
                "label_gt":        flow.get("label", ""),
            }

            # FR-3: LLM reasoning for ambiguous flows (NFR-2: ≤ 5 s)
            cap = st.session_state.llm_session_cap
            if triage["method"] == "LLM_Required" and st.session_state.llm_enabled:
                if st.session_state.llm_count >= cap:
                    # Cap reached — keep heuristic verdict, note reason
                    log_entry["method"]  = "Heuristic"
                    log_entry["details"] += f" | LLM skipped (session cap {cap} reached)"
                else:
                    flow["heuristic_details"] = triage["details"]
                    try:
                        llm_result = le.analyze_flow_with_llm(flow)
                        classification = llm_result.get("classification", "")
                        explanation    = llm_result.get("explanation", "")
                        # Detect internal LLM failure (returned as explanation text)
                        if "LLM call failed" in explanation or not classification or classification == "Unknown":
                            log_entry["method"]          = "LLM_Error"
                            log_entry["llm_explanation"] = explanation or llm_result.get("raw_response", "")
                            log_entry["details"]        += " | LLM failed — see llm_explanation"
                        else:
                            log_entry["verdict"]         = classification
                            log_entry["method"]          = "LLM"
                            log_entry["confidence"]      = llm_result.get("confidence", "Low")
                            log_entry["llm_class"]       = classification
                            log_entry["llm_explanation"] = explanation
                            indicators = llm_result.get("attack_indicators", [])
                            log_entry["details"] = "; ".join(indicators) if indicators else triage["details"]
                            st.session_state.llm_count += 1
                    except Exception as exc:
                        log_entry["method"]          = "LLM_Error"
                        log_entry["llm_explanation"] = str(exc)
                        log_entry["details"]        += f" | LLM exception: {type(exc).__name__}"

            # Update counters
            if "normal" in log_entry["verdict"].lower():
                st.session_state.normal_count += 1
            else:
                st.session_state.attack_count += 1

            v = log_entry["verdict"]
            st.session_state.verdict_counts[v] = (
                st.session_state.verdict_counts.get(v, 0) + 1
            )
            st.session_state.system_logs.append(log_entry)

    # ── Metrics ───────────────────────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("📦 Total Flows",      st.session_state.total_flows)
    m2.metric("🚨 Attacks Detected", st.session_state.attack_count)
    m3.metric("✅ Normal Flows",      st.session_state.normal_count)
    m4.metric("🤖 LLM Analyses",     st.session_state.llm_count)
    pct = (
        round(st.session_state.attack_count / st.session_state.total_flows * 100, 1)
        if st.session_state.total_flows > 0 else 0.0
    )
    m5.metric("⚠️ Attack Rate", f"{pct}%")

    # ── Charts ────────────────────────────────────────────────────────────────
    st.markdown("---")
    ch1, ch2 = st.columns(2)

    with ch1:
        st.markdown("**Protocol Distribution**")
        if st.session_state.proto_counts:
            st.bar_chart(
                pd.DataFrame(
                    list(st.session_state.proto_counts.items()),
                    columns=["Protocol", "Count"],
                ).set_index("Protocol")
            )
        else:
            st.caption("No traffic captured yet.")

    with ch2:
        st.markdown("**Verdict Distribution**")
        if st.session_state.verdict_counts:
            st.bar_chart(
                pd.DataFrame(
                    list(st.session_state.verdict_counts.items()),
                    columns=["Verdict", "Count"],
                ).set_index("Verdict")
            )
        else:
            st.caption("No detections yet.")

    # ── Recent Alerts feed ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("**🚨 Recent Alerts (latest 15 non-normal detections)**")
    alerts = [
        l for l in reversed(st.session_state.system_logs)
        if "normal" not in l["verdict"].lower()
    ][:15]

    if alerts:
        st.dataframe(
            pd.DataFrame(alerts)[
                ["timestamp", "srcip", "dstip", "proto", "verdict", "method", "confidence", "details"]
            ],
            use_container_width=True,
            hide_index=True,
            column_config={
                "verdict":    st.column_config.TextColumn("Verdict",  width="medium"),
                "details":    st.column_config.TextColumn("Details",  width="large"),
                "confidence": st.column_config.TextColumn("Conf.",    width="small"),
            },
        )
    else:
        st.caption("No alerts yet. Start capture to begin monitoring.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — FORENSIC CONSOLE  (Design Doc §4.2)
# SRS FR-4 (indexing), FR-5 (semantic search), FR-6 (report generation)
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Forensic Analysis Console")
    st.caption(
        "Semantic search over the UNSW-NB15 historical dataset. "
        "Ask questions in plain English — the RAG engine retrieves the 15 most "
        "similar flows and Gemini 1.5 Flash generates a grounded forensic report."
    )

    # Reload button — busts the @st.cache_resource cache so that a newly-built
    # ChromaDB (or updated API key) is picked up without restarting the server.
    _rld_col, _ = st.columns([1, 6])
    with _rld_col:
        if st.button("🔄 Reload Engine", help="Clear cached forensic engine and reconnect to ChromaDB"):
            load_forensic_chain.clear()
            st.rerun()

    retriever, llm, forensic_prompt, chain_err = load_forensic_chain()

    if chain_err:
        st.error(chain_err)
        st.info("Run `python build_chroma_db.py` in the project folder, then click **🔄 Reload Engine** above.")
        st.code("python build_chroma_db.py", language="bash")
    else:
        st.success("✅ Forensic engine ready — ChromaDB loaded, Gemini 2.0 Flash connected.", icon="🔍")

        with st.expander("💡 Example queries", expanded=False):
            st.markdown(
                "- *Show me Reconnaissance attempts targeting SSH ports*\n"
                "- *Describe DoS attack patterns — what features distinguish them from normal traffic?*\n"
                "- *Are there any Worm flows with lateral movement indicators?*\n"
                "- *What does normal TCP traffic look like in this dataset?*\n"
                "- *Explain suspicious FTP login activity in the logs*"
            )

        user_query = st.text_area(
            "Enter your forensic query:",
            placeholder='e.g. "Describe Backdoor traffic — what source IPs and ports are involved?"',
            height=90,
            key="forensic_query_input",
        )

        qc1, qc2 = st.columns([4, 1])
        with qc1:
            analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
        with qc2:
            if st.button("🗑 Clear", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        if analyze_btn:
            if not user_query.strip():
                st.warning("Please enter a query.")
            else:
                with st.spinner("Retrieving flows and generating forensic report…"):
                    response = _run_forensic_query(retriever, llm, forensic_prompt, user_query)
                if response.get("error"):
                    # Show friendly error — don't append broken result to history
                    st.error(response["result"])
                else:
                    st.session_state.chat_history.append({
                        "query":   user_query,
                        "result":  response["result"],
                        "sources": [
                            {"content": d.page_content, "metadata": d.metadata}
                            for d in response["source_documents"]
                        ],
                        "ts": datetime.now().strftime("%H:%M:%S"),
                    })
                    st.rerun()

        # ── Latest result ─────────────────────────────────────────────────────
        if st.session_state.chat_history:
            latest = st.session_state.chat_history[-1]
            st.markdown("---")
            st.markdown(f"**Query ({latest['ts']}):** {latest['query']}")
            st.markdown(latest["result"])

            with st.expander(f"📄 Source Documents ({len(latest['sources'])} retrieved)"):
                for i, src in enumerate(latest["sources"]):
                    meta = src.get("metadata", {})
                    st.markdown(
                        f"**Doc {i+1}** — attack_cat: `{meta.get('attack_cat','N/A')}` "
                        f"| label: `{meta.get('label','N/A')}` "
                        f"| proto: `{meta.get('proto','N/A')}`"
                    )
                    st.text(src["content"][:400] + ("…" if len(src["content"]) > 400 else ""))
                    st.divider()

        # ── Query history ─────────────────────────────────────────────────────
        if len(st.session_state.chat_history) > 1:
            st.markdown("---")
            st.markdown("**📚 Previous Queries**")
            for entry in reversed(st.session_state.chat_history[:-1]):
                preview = entry["query"][:65] + ("…" if len(entry["query"]) > 65 else "")
                with st.expander(f"[{entry.get('ts','')}] {preview}"):
                    st.markdown(f"**Question:** {entry['query']}")
                    st.markdown(entry["result"])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — SYSTEM LOGS  (Design Doc §4.3)
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("System Detection Logs")
    st.caption("Attack classification, detection method, feature values, and LLM reasoning — Design Document §4.3")

    if not st.session_state.system_logs:
        st.info("No logs yet. Go to the Live Dashboard tab and start capture.")
    else:
        logs_df = pd.DataFrame(st.session_state.system_logs)

        # Summary
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total Entries",        len(logs_df))
        s2.metric("Heuristic Detections", int((logs_df["method"] == "Heuristic").sum()))
        s3.metric("LLM Detections",       int((logs_df["method"] == "LLM").sum()))
        s4.metric("Unique Verdicts",      logs_df["verdict"].nunique())

        # Filters
        st.markdown("**Filters**")
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            vf = st.multiselect("Verdict",           options=sorted(logs_df["verdict"].unique()), default=[])
        with fc2:
            pf = st.multiselect("Protocol",          options=sorted(logs_df["proto"].unique()),   default=[])
        with fc3:
            mf = st.multiselect("Detection Method",  options=sorted(logs_df["method"].unique()),  default=[])

        filtered = logs_df.copy()
        if vf:
            filtered = filtered[filtered["verdict"].isin(vf)]
        if pf:
            filtered = filtered[filtered["proto"].isin(pf)]
        if mf:
            filtered = filtered[filtered["method"].isin(mf)]

        st.markdown(f"Showing **{len(filtered):,}** of **{len(logs_df):,}** entries")

        display_cols = [
            "timestamp", "srcip", "sport", "dstip", "dsport", "proto",
            "dur", "sbytes", "pps", "state",
            "verdict", "method", "confidence", "details",
            "llm_class", "llm_explanation",
            "attack_cat_gt", "label_gt",
        ]
        available = [c for c in display_cols if c in filtered.columns]
        st.dataframe(
            filtered[available],
            use_container_width=True,
            hide_index=True,
            height=520,
            column_config={
                "verdict":         st.column_config.TextColumn("Verdict",         width="medium"),
                "details":         st.column_config.TextColumn("Details",         width="large"),
                "llm_explanation": st.column_config.TextColumn("LLM Explanation", width="large"),
                "sbytes":          st.column_config.NumberColumn("Bytes",         format="%d"),
                "pps":             st.column_config.NumberColumn("PPS",           format="%.1f"),
                "label_gt":        st.column_config.TextColumn("GT Label",        width="small"),
                "attack_cat_gt":   st.column_config.TextColumn("GT Category",     width="medium"),
            },
        )

        csv_bytes = filtered[available].to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Logs as CSV",
            data=csv_bytes,
            file_name=f"ids_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Dual LLM Framework for Network Intrusion Detection & Digital Forensics | "
    "UNSW-NB15 Dataset | Google Gemini 1.5 Flash | LangChain | ChromaDB | "
    "Honors Project 2026"
)
