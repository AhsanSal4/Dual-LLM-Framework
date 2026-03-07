# build_chroma_db.py
"""
Offline script — Module 2: Forensic Engine Indexer

Reads the UNSW-NB15 dataset, converts each flow row into a natural-language
text document, embeds it with all-MiniLM-L6-v2, and stores the vectors plus
metadata (attack_cat, proto, label) in a persisted ChromaDB collection.

Run once before launching the Streamlit app:
    python build_chroma_db.py

Design Document §1.4: ChromaDB, all-MiniLM-L6-v2 (384-dim), sub-second retrieval
SRS FR-4: UNSW-NB15 → text template → HuggingFace embeddings → ChromaDB
"""

import pandas as pd
import os
import torch
import gc
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- Configuration ---
DATA_FILE = 'UNSW_NB15_attack_binary_bits.csv'
FEATURES_FILE = 'NUSW-NB15_features1.csv'
PERSIST_DIRECTORY = "./chroma_db"
SAMPLE_SIZE = 10000
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- Feature Descriptions (copied from your app.py) ---
feature_descriptions_map = {
    'srcip': 'Source IP address', 'sport': 'Source port number', 'dstip': 'Destination IP address',
    'dsport': 'Destination port number', 'proto': 'Transaction protocol',
    'state': 'Indicates to the state and its dependent protocol, e.g. ACC, CLO, CON, ECO, ECR, FIN, INT, MAS, PAR, REQ, RST, TST, TXD, URH, URN, and (-) (if not used state)',
    'dur': 'Record total duration', 'sbytes': 'Source to destination transaction bytes',
    'dbytes': 'Destination to source transaction bytes', 'sttl': 'Source to destination time to live value',
    'dttl': 'Destination to source time to live value', 'sloss': 'Source packets retransmitted or dropped',
    'dloss': 'Destination packets retransmitted or dropped', 'service': 'http, ftp, smtp, ssh, dns, ftp-data ,irc and (-) if not much used service',
    'Sload': 'Source bits per second', 'Dload': 'Destination bits per second',
    'Spkts': 'Source to destination packet count', 'Dpkts': 'Destination to source packet count',
    'swin': 'Source TCP window advertisement value', 'dwin': 'Destination TCP window advertisement value',
    'stcpb': 'Source TCP base sequence number', 'dtcpb': 'Destination TCP base sequence number',
    'smeansz': 'Mean of the flow packet size transmitted by the src',
    'dmeansz': 'Mean of the flow packet size transmitted by the dst', # Added missing description
    'trans_depth': 'Represents the pipelined depth into the connection of http request/response transaction',
    'res_bdy_len': 'Actual uncompressed content size of the data transferred from the server’s http service.',
    'Sjit': 'Source jitter (mSec)', 'Djit': 'Destination jitter (mSec)', 'Stime': 'record start time',
    'Ltime': 'record last time', 'Sintpkt': 'Source interpacket arrival time (mSec)',
    'Dintpkt': 'Destination interpacket arrival time (mSec)',
    'tcprtt': 'TCP connection setup round-trip time, the sum of ’synack’ and ’ackdat’.',
    'synack': 'TCP connection setup time, the time between the SYN and the SYN_ACK packets.',
    'ackdat': 'TCP connection setup time, the time between the SYN_ACK and the ACK packets.',
    'is_sm_ips_ports': 'If source (1) and destination (3)IP addresses equal and port numbers (2)(4) equal then, this variable takes value 1 else 0',
    'ct_state_ttl': 'No. for each state (6) according to specific range of values for source/destination time to live (10) (11).',
    'ct_flw_http_mthd': 'No. of flows that has methods such as Get and Post in http service.',
    'is_ftp_login': 'If the ftp session is accessed by user and password then 1 else 0.',
    'ct_ftp_cmd': 'No of flows that has a command in ftp session.',
    'ct_srv_src': 'No. of connections that contain the same service (14) and source address (1) in 100 connections according to the last time (26).',
    'ct_srv_dst': 'No. of connections that contain the same service (14) and destination address (3) in 100 connections according to the last time (26).',
    'ct_dst_ltm': 'No. of connections of the same destination address (3) in 100 connections according to the last time (26).',
    'ct_src_ltm': 'No. of connections of the same source address (1) in 100 connections according to the last time (26).',
    'ct_src_dport_ltm': 'No of connections of the same source address (1) and the destination port (4) in 100 connections according to the last time (26).',
    'ct_dst_sport_ltm': 'No of connections of the same destination address (3) and the source port (2) in 100 connections according to the last time (26).',
    'ct_dst_src_ltm': 'No of connections of the same source (1) and the destination (3) address in in 100 connections according to the last time (26).',
    'attack_cat': 'The name of each attack category. In this data set , nine categories e.g. Fuzzers, Analysis, Backdoors, DoS Exploits, Generic, Reconnaissance, Shellcode and Worms',
    'label': '0 for normal and 1 for attack records',
    'b3': 'Binary representation bit 3 of attack category (for Fuzzers, Analysis, Backdoors, DoS Exploits, Generic, Reconnaissance, Shellcode and Worms)',
    'b2': 'Binary representation bit 2 of attack category',
    'b1': 'Binary representation bit 1 of attack category',
    'b0': 'Binary representation bit 0 of attack category'
}

# Columns stored as ChromaDB metadata (not embedded into text body)
_METADATA_COLS = {'attack_cat', 'label', 'b3', 'b2', 'b1', 'b0'}


def row_to_text(row, feature_descriptions: dict) -> str:
    """Convert one dataset row to a human-readable flow description.
    Metadata columns (attack_cat, label, binary bits) are excluded from
    the text body so the embedding reflects traffic features only.
    """
    lines = []
    for col in row.index:
        if col in _METADATA_COLS:
            continue
        desc = feature_descriptions.get(col, col)
        lines.append(f"{desc}: {row[col]}")
    return "\n".join(lines)


def row_to_metadata(row) -> dict:
    """Extract label fields as ChromaDB document metadata for filtered retrieval."""
    return {
        'attack_cat': str(row.get('attack_cat', 'Normal')).strip() or 'Normal',
        'label':      int(row.get('label', 0) or 0),
        'proto':      str(row.get('proto', 'unknown')).strip(),
        'b3':         int(row.get('b3', 0) or 0),
        'b2':         int(row.get('b2', 0) or 0),
        'b1':         int(row.get('b1', 0) or 0),
        'b0':         int(row.get('b0', 0) or 0),
    }

def build_and_persist_chroma_db():
    print(f"--- Starting Chroma DB build process ({SAMPLE_SIZE} records) ---")
    print("SRS FR-4: UNSW-NB15 → text template → HuggingFace embeddings → ChromaDB")

    # Load data
    try:
        df_full = pd.read_csv(DATA_FILE)
    except FileNotFoundError:
        print(f"Error: {DATA_FILE} not found. Ensure it is in the project directory.")
        return

    df_sample = df_full.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    print(f"Loaded and sampled {len(df_sample)} records from {len(df_full):,} total.")

    # Convert rows to LangChain Document objects with metadata
    print("Converting rows to Document objects with metadata...")
    documents = [
        Document(
            page_content=row_to_text(row, feature_descriptions_map),
            metadata=row_to_metadata(row),
        )
        for _, row in df_sample.iterrows()
    ]
    print(f"Generated {len(documents)} documents.")

    # Initialise embedding model (GPU if available, else CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Embedding device: {device}")
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device},
        encode_kwargs={'batch_size': 64},
    )

    torch.cuda.empty_cache()
    gc.collect()

    # Ensure persist directory exists
    os.makedirs(PERSIST_DIRECTORY, exist_ok=True)

    # Build and persist ChromaDB
    print(f"Building ChromaDB at '{PERSIST_DIRECTORY}' ...")
    docsearch = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY,
    )
    docsearch.persist()
    print(f"ChromaDB built and persisted — {len(documents)} vectors stored.")
    print("Attack categories indexed:",
          df_sample['attack_cat'].value_counts().to_dict() if 'attack_cat' in df_sample.columns else 'N/A')

if __name__ == "__main__":
    build_and_persist_chroma_db()