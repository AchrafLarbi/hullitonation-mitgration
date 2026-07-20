#!/usr/bin/env python3
"""
Phase 3: Reproducible Evaluation + Response Diff Analysis
=========================================================
Converted from phase3-rag-bio-v3.ipynb for SSH / GPU background execution.

New models vs notebook:
  - OpenBioLLM    (aaditya/Llama3-OpenBioLLM-8B)    — medical fine-tuned (Llama-3 base)
  - BioMistral    (BioMistral/BioMistral-7B)         — medical fine-tuned (Mistral base)
  - Qwen2-General (Qwen/Qwen2-7B-Instruct)           — NO medical fine-tuning (baseline)
                  Qwen2 architecture (Alibaba) — entirely different family from Mistral/Llama,
                  fully public, no license gate, ~7B params

Usage (SSH / GPU):
  # foreground
  python phase3_rag_bio_v3.py

  # background (nohup)
  nohup python phase3_rag_bio_v3.py --eval_mode full_eval > run.log 2>&1 &

  # screen session
  screen -S pfe_eval
  python phase3_rag_bio_v3.py --models BioMistral OpenBioLLM Llama3-General

CLI options:
  --data_dir          Path to dataset directory
  --output_dir        Path to output directory
  --eval_mode         quick_debug_sample | full_eval
  --quick_n           Sample size for quick mode  (default 120)
  --seed              Random seed                 (default 42)
  --models            Space-separated model keys to test
  --hf_token          HuggingFace token (or set HF_TOKEN env var)
  --no_embedding_cmp  Disable fast/biomed embedding comparison
"""

# ── Standard library ──────────────────────────────────────────────────────────
import argparse
import gc
import hashlib
import json
import logging
import os
import re
import shutil
import string
import sys
import time
import unicodedata
import difflib
from collections import Counter
from pathlib import Path

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import faiss
import torch
import matplotlib
matplotlib.use('Agg')                     # no display required on remote GPU
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import textwrap

from huggingface_hub import snapshot_download, login as hf_login
from sentence_transformers import SentenceTransformer
try:
    from tqdm.auto import tqdm as _tqdm_auto
except ImportError:
    _tqdm_auto = None

# ── Fast download: hf_transfer (Rust backend, 3-5x faster than urllib) ────────
# Install with: pip install hf-transfer
# If not installed the code falls back gracefully to the default downloader.
try:
    import hf_transfer as _hft          # noqa: F401 — just check availability
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '1')
except ImportError:
    pass  # hf-transfer not installed; standard download will be used
from rouge_score import rouge_scorer
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    pipeline,
)

# ─────────────────────────────────────────────────────────────────────────────
# 0)  CLI / Logging
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Phase-3 RAG evaluation (GPU/SSH)")
    p.add_argument('--data_dir',     default=None,
                   help='Dataset directory (default: auto-detect Kaggle or ./data/)')
    p.add_argument('--output_dir',   default=None,
                   help='Output directory   (default: auto-detect Kaggle or ./)')
    p.add_argument('--eval_mode',    default='full_eval',
                   choices=['quick_debug_sample', 'full_eval'])
    p.add_argument('--quick_n',      type=int,  default=120)
    p.add_argument('--seed',         type=int,  default=42)
    p.add_argument('--models',       nargs='+',
                   default=['BioMistral', 'OpenBioLLM', 'Qwen2-General'],
                   help='Model keys to test (space-separated)')
    p.add_argument('--hf_token',     default=None,
                   help='HuggingFace token (overrides HF_TOKEN env var)')
    p.add_argument('--no_embedding_cmp', action='store_true',
                   help='Run only the default (fast) embedding profile')
    p.add_argument('--download_only', action='store_true',
                   help='Pre-download all models and exit (run this before nohup)')
    p.add_argument('--download_workers', type=int, default=8,
                   help='Parallel shard download workers (default 8, max ~16)')
    return p.parse_args()


def _setup_logging(output_dir: str):
    log_path = os.path.join(output_dir, 'phase3_run.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(levelname)-8s  %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1)  Global Configuration
# ─────────────────────────────────────────────────────────────────────────────

def build_config(args):
    """Return a plain dict holding all runtime configuration."""
    IS_KAGGLE = os.path.exists('/kaggle')

    data_dir = args.data_dir or 'data/'
    output_dir = args.output_dir or 'output/'

    hf_token = args.hf_token or os.getenv('HF_TOKEN', None)

    # ── Embedding models ──────────────────────────────────────────────────────
    EMBEDDING_MODEL_REGISTRY = {
        'fast':   'sentence-transformers/all-MiniLM-L6-v2',
        'biomed': 'NeuML/pubmedbert-base-embeddings',
    }
    EMBEDDING_BATCH_SIZES = {'fast': 128, 'biomed': 64}
    DEFAULT_EMBEDDING_PROFILE = 'fast'
    EMBEDDING_PROFILES_TO_COMPARE = ['fast', 'biomed']
    ENABLE_EMBEDDING_COMPARISON = not args.no_embedding_cmp

    # ── LLM models ────────────────────────────────────────────────────────────
    # OpenBioLLM-8B  : aaditya/Llama3-OpenBioLLM-8B
    #   Pal (2024) — fine-tuned on curated biomedical corpora from Meta Llama-3-8B-Instruct
    #   (Llama-3 architecture, Meta)
    #
    # BioMistral-7B  : BioMistral/BioMistral-7B
    #   Labrak et al. (2024) ACL Findings — fine-tuned on medical data from Mistral-7B-v0.1
    #   (Mistral architecture, Mistral AI)
    #
    # Qwen2-General  : Qwen/Qwen2-7B-Instruct
    #   Qwen Team, Alibaba Group (2024) "Qwen2 Technical Report". arXiv:2407.10671
    #   Qwen2 architecture — entirely different family from Mistral and Llama:
    #     • Grouped Query Attention (GQA) with 64 heads / 8 KV heads
    #     • Dual chunk attention + YARN for long-context
    #     • Trained on 7T tokens (multilingual, general-domain)
    #   NO medical fine-tuning. Used as non-medical BASELINE to measure how much
    #   domain specialisation contributes across all prompt techniques.
    #   Fully public — no HuggingFace license gate.
    MODEL_REGISTRY = {
        'OpenBioLLM':   'aaditya/Llama3-OpenBioLLM-8B',
        'BioMistral':   'BioMistral/BioMistral-7B',
        'Qwen2-General': 'Qwen/Qwen2-7B-Instruct',
    }
    MODELS_TO_TEST = [m for m in args.models if m in MODEL_REGISTRY]
    if not MODELS_TO_TEST:
        raise ValueError(f"No valid model key found in {args.models}. "
                         f"Valid keys: {list(MODEL_REGISTRY.keys())}")

    # ── Evaluation thresholds ─────────────────────────────────────────────────
    THRESHOLDS = {
        'cosine_correct': 0.65,
        'cosine_partial': 0.25,
        'f1_partial':     0.10,
    }

    # ── Retrieval settings ────────────────────────────────────────────────────
    RETRIEVAL_MIN_SCORE = 0.25
    RETRIEVAL_TOP_K     = 10
    CRAG_RETRIEVE_SCORE = 0.20
    CRAG_CORRECT_SCORE  = 0.60
    CRAG_AMBIGUOUS_MIN  = 0.35
    RERANKER_ENABLED    = True
    RERANKER_TOP_N      = 18
    RERANKER_MODEL_CANDIDATES = [
        'ncbi/MedCPT-Cross-Encoder',
        'cross-encoder/ms-marco-MiniLM-L-6-v2',
    ]

    # ── Prompt techniques ─────────────────────────────────────────────────────
    RAG_RETRIEVAL_TYPES = ['rag', 'self-rag', 'corrective-rag', 'speculative-rag', 'react']
    PROMPTS_TO_TEST = [
        'zero-shot', 'few-shot', 'cot',
        'rag', 'self-rag', 'corrective-rag', 'speculative-rag', 'react',
    ]
    TECHNIQUE_NAMES = {
        'zero-shot':       'Zero-Shot',
        'few-shot':        'Few-Shot (ICL)',
        'cot':             'CoT',
        'rag':             'RAG',
        'self-rag':        'Self-RAG',
        'corrective-rag':  'Corrective-RAG',
        'speculative-rag': 'Speculative-RAG',
        'react':           'ReAct',
    }

    # ── Sampling ──────────────────────────────────────────────────────────────
    EVAL_MODE         = args.eval_mode
    QUICK_SAMPLE_SIZE = args.quick_n
    QUICK_SAMPLE_SEED = args.seed
    QUAL_CASES        = 5
    QUAL_SEED         = args.seed

    # ── Generation ───────────────────────────────────────────────────────────
    GENERATION_TEMPERATURE = 0.4
    N_CONSISTENCY_VOTES    = 3
    MAX_GEN_TOKENS         = 1024
    _MIN_FREE_BYTES        = 16 * 1024 ** 3    # 16 GB minimum free disk space
    DOWNLOAD_WORKERS       = args.download_workers
    DOWNLOAD_ONLY          = args.download_only

    cfg = dict(locals())
    cfg.pop('args')
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 2)  Load Datasets
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets(cfg, log):
    data_dir = cfg['data_dir']
    rag_file  = os.path.join(data_dir, 'rag_corpus_final.jsonl')
    pe_file   = os.path.join(data_dir, 'pe_sft_final.jsonl')
    eval_file = os.path.join(data_dir, 'eval_rare_cases.jsonl')

    log.info('Loading datasets...')
    try:
        df_rag  = pd.read_json(rag_file,  lines=True)
        df_pe   = pd.read_json(pe_file,   lines=True)
        df_eval = pd.read_json(eval_file, lines=True)
    except Exception as e:
        log.warning(f'Fallback to synthetic data: {e}')
        df_rag  = pd.DataFrame([{'id': 1, 'text': 'Rare disease symptoms and diagnosis text',
                                  'source_id': 'PMID:1'}])
        df_pe   = pd.DataFrame([{'instruction': 'Example differential diagnosis format'}])
        df_eval = pd.DataFrame([
            {'id': 'sample_1',
             'input': 'Patient with progressive ataxia and neuropathy',
             'target': 'friedreich ataxia'},
            {'id': 'sample_2',
             'input': 'Splenomegaly, anemia, bone pain',
             'target': 'gaucher disease'},
        ])

    df_clean = df_rag.dropna(subset=['text']).copy() if 'text' in df_rag.columns else df_rag.copy()
    if 'text' in df_clean.columns:
        df_clean['text'] = df_clean['text'].astype(str).str.strip()
    documents = df_clean.to_dict(orient='records')

    log.info(f'RAG corpus : {df_rag.shape}')
    log.info(f'PE SFT     : {df_pe.shape}')
    log.info(f'Eval set   : {df_eval.shape}')
    log.info(f'Documents ready for embedding: {len(documents)}')
    return df_rag, df_pe, df_eval, documents


# ─────────────────────────────────────────────────────────────────────────────
# 3)  FAISS Index
# ─────────────────────────────────────────────────────────────────────────────

def _index_path_for_profile(profile, faiss_index_dir):
    tag_map = {'fast': 'all_minilm_l6_v2', 'biomed': 'pubmedbert_base'}
    tag = tag_map.get(profile, profile.replace('/', '_').replace('-', '_'))
    return os.path.join(faiss_index_dir, f'rag_corpus_{tag}.index')


def _build_and_save_index(backend, documents, log):
    log.info(f"Building FAISS index for profile '{backend['profile']}'...")
    texts = [d.get('text', '') for d in documents]
    idx = faiss.IndexFlatIP(backend['dim'])
    if texts:
        emb = backend['embedding_model'].encode(
            texts, batch_size=backend['batch_size'],
            show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True,
        )
        emb = np.ascontiguousarray(emb, dtype=np.float32)
        idx.add(emb)
        faiss.write_index(idx, backend['index_path'])
        log.info(f"FAISS index saved: {backend['index_path']}")
    return idx


def _load_or_create_backend(profile, cfg, documents, device, log):
    model_name = cfg['EMBEDDING_MODEL_REGISTRY'][profile]
    batch_size = cfg['EMBEDDING_BATCH_SIZES'].get(profile, 64)
    index_path = _index_path_for_profile(profile, cfg['faiss_index_dir'])

    log.info(f"Downloading/Loading embedding model '{profile}': {model_name}")
    t0 = time.time()
    local_model = SentenceTransformer(
        model_name, cache_folder=cfg['hf_cache_dir'],
        device=device, token=cfg['hf_token'],
    )
    # get_sentence_embedding_dimension() is deprecated; use get_embedding_dimension() when available
    if hasattr(local_model, 'get_embedding_dimension'):
        dim = local_model.get_embedding_dimension()
    else:
        dim = local_model.get_sentence_embedding_dimension()
    log.info(f"Embedding model '{profile}' ready in {time.time()-t0:.1f}s | dim={dim}")

    backend = {
        'profile': profile, 'model_name': model_name, 'batch_size': batch_size,
        'index_path': index_path, 'dim': dim, 'embedding_model': local_model,
    }

    if os.path.exists(index_path):
        idx = faiss.read_index(index_path)
        if getattr(idx, 'd', None) != dim:
            log.warning(f"Dimension mismatch for '{profile}'; rebuilding index...")
            idx = _build_and_save_index(backend, documents, log)
        else:
            log.info(f"Loaded FAISS index for '{profile}': {index_path}")
    else:
        idx = _build_and_save_index(backend, documents, log)

    backend['index'] = idx
    log.info(f"Index size for '{profile}': {idx.ntotal}")
    return backend


# ─────────────────────────────────────────────────────────────────────────────
# 4)  Retrieval + Prompt Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _lexical_overlap_score(query: str, text: str, normalize_fn) -> float:
    q_toks = {t for t in normalize_fn(query).split() if len(t) >= 3}
    d_toks = {t for t in normalize_fn(text).split() if len(t) >= 3}
    if not q_toks or not d_toks:
        return 0.0
    return len(q_toks & d_toks) / max(len(q_toks), 1)


def _hybrid_rerank(query, rows, cfg, log, normalize_fn):
    if not rows:
        return rows
    _RERANKER_MODEL = None
    if cfg['RERANKER_ENABLED']:
        from sentence_transformers import CrossEncoder
        for m_name in cfg['RERANKER_MODEL_CANDIDATES']:
            try:
                _dev = 'cuda' if torch.cuda.is_available() else 'cpu'
                _RERANKER_MODEL = CrossEncoder(m_name, device=_dev, max_length=384)
                log.info(f"Cross-encoder reranker loaded: {m_name}")
                break
            except Exception as e:
                log.warning(f"Reranker unavailable [{m_name}]: {e}")

    n = min(cfg['RERANKER_TOP_N'], len(rows))
    if _RERANKER_MODEL is not None:
        pairs = [(str(query), f"{r.get('title','')} {r.get('text','')}"[:1400]) for r in rows[:n]]
        try:
            ce_raw = _RERANKER_MODEL.predict(pairs, batch_size=16, show_progress_bar=False)
        except TypeError:
            ce_raw = _RERANKER_MODEL.predict(pairs, batch_size=16)
        except Exception:
            ce_raw = None
        if ce_raw is not None:
            ce_raw = np.asarray(ce_raw, dtype=np.float32)
            cmin, cmax = float(np.min(ce_raw)), float(np.max(ce_raw))
            cden = max(cmax - cmin, 1e-8)
            for i in range(n):
                rows[i]['cross_score'] = float((ce_raw[i] - cmin) / cden)

    dense_scores = [float(r.get('score', 0.0)) for r in rows]
    d_min, d_max = min(dense_scores), max(dense_scores)
    reranked = []
    for r in rows:
        dense  = float(r.get('score', 0.0))
        cross  = float(r.get('cross_score', 0.0))
        d_norm = (dense - d_min) / (d_max - d_min + 1e-8)
        lex    = _lexical_overlap_score(query, r.get('text', ''), normalize_fn)
        hybrid = (0.52 * d_norm + 0.20 * lex + 0.28 * cross
                  if cross > 0.0 else 0.72 * d_norm + 0.28 * lex)
        out = dict(r)
        out['hybrid_score'] = float(hybrid)
        reranked.append(out)
    reranked.sort(key=lambda x: x['hybrid_score'], reverse=True)
    return reranked


def retrieve_documents(query, retrieval_backends, documents, cfg, log,
                       normalize_fn, k=None, embedding_profile=None, min_score=None):
    k_use   = int(k) if k is not None else cfg['RETRIEVAL_TOP_K']
    min_s   = float(min_score) if min_score is not None else cfg['RETRIEVAL_MIN_SCORE']
    k_fetch = min(max(k_use * 6, k_use), 64)
    profile = str(embedding_profile or cfg['DEFAULT_EMBEDDING_PROFILE']).strip().lower()
    backend = retrieval_backends[profile]
    local_model, local_index = backend['embedding_model'], backend['index']
    if local_index.ntotal == 0:
        return []
    q = local_model.encode([str(query)], convert_to_numpy=True, normalize_embeddings=True)
    q = np.ascontiguousarray(q, dtype=np.float32)
    scores, indices = local_index.search(q, min(k_fetch, local_index.ntotal))
    candidates = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if not (0 <= idx < len(documents)):
            continue
        if score < min_s and rank > max(10, k_use * 2):
            continue
        doc = documents[idx]
        candidates.append({
            'score': float(score),
            'text': str(doc.get('text', ''))[:1200],
            'title': str(doc.get('title', ''))[:220],
            'source_id': doc.get('source_id', doc.get('id', '')),
            'cross_score': 0.0,
        })
    return _hybrid_rerank(query, candidates, cfg, log, normalize_fn)[:k_use]


def expand_query_with_hpo(query, pipe, log):
    expansion_prompt = (
        "You are a medical NLP expert. Extract the key clinical phenotype terms from "
        "the following patient description. "
        "Use standard terminology (e.g. HPO terms). One term per line, no explanations:\n\n"
        f"{str(query)[:600]}\n\nTerms:"
    )
    try:
        gen_kwargs = {'return_full_text': False, 'max_new_tokens': 80, 'truncation': True}
        pad_id = getattr(pipe.tokenizer, 'pad_token_id',
                         getattr(pipe.tokenizer, 'eos_token_id', None))
        if pad_id is not None:
            gen_kwargs['pad_token_id'] = pad_id
        out = pipe(expansion_prompt, **gen_kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        terms = out[0]['generated_text'].strip()
        return f"{query}\n\nKey phenotype terms: {terms}"
    except Exception as e:
        log.debug(f'HPO expansion failed: {e}')
        return query


def _extract_diagnosis_from_pe_output(output_text):
    text = str(output_text).strip()
    m = re.search(
        r'(?:most likely diagnosis|diagnosis|likely)\s*(?:is)?\s*[:\-]?\s*\*{0,2}'
        r'([A-Z][^*\n\.]{3,80}?)\*{0,2}(?:\.|,|\n|$)',
        text, flags=re.IGNORECASE,
    )
    if m:
        name = m.group(1).strip().rstrip('.,;').strip('*').strip()
        if len(name) < 80 and not re.search(r'\b(is|are|the|and|with)\b', name[:30], re.IGNORECASE):
            return name
    m = re.search(r'\*\*([A-Z][^*]{3,70}?)\*\*', text)
    if m:
        name = m.group(1).strip()
        if len(name) < 80:
            return name
    return None


def get_few_shot_examples(df_pe, cfg, num_shots=3):
    if len(df_pe) == 0:
        return ''
    has_output = 'output' in df_pe.columns
    candidates = df_pe.sample(frac=1, random_state=cfg['QUICK_SAMPLE_SEED']).reset_index(drop=True)
    examples = []
    for _, row in candidates.iterrows():
        if len(examples) >= num_shots:
            break
        question = str(row.get('instruction', row.get('input', ''))).strip()[:500]
        if not question:
            continue
        diag = _extract_diagnosis_from_pe_output(row.get('output', '')) if has_output else None
        if diag:
            examples.append(f'Patient: {question}\nFinal diagnosis: {diag}')
    if not examples and 'instruction' in df_pe.columns:
        shots = df_pe.sample(n=min(num_shots, len(df_pe)),
                              random_state=cfg['QUICK_SAMPLE_SEED'])['instruction'].astype(str).tolist()
        examples = [f'Patient: {s[:500]}' for s in shots]
    return '\n---\n'.join(examples)


def make_cache_key(patient_case, prompt_type, llm_model, embedding_profile='na'):
    raw = f'{llm_model}||{prompt_type}||{embedding_profile}||{patient_case}'
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def build_prompt(patient_case, prompt_type, df_pe, cfg, retrieved_docs_text=''):
    format_constraint = (
        "IMPORTANT: You MUST commit to exactly one rare disease diagnosis. "
        "Even if uncertain, name the single most likely rare disease based on all available evidence. "
        "End your response exactly with 'Final diagnosis: [Exact Disease Name]'. "
        "Only output 'Final diagnosis: Insufficient information' if the clinical description "
        "contains NO phenotypic features whatsoever."
    )
    evidence_block = str(retrieved_docs_text).strip() or '[No documents retrieved above threshold]'
    prompt = 'You are an expert medical diagnostician specializing in rare diseases.\n\n'

    if prompt_type == 'zero-shot':
        prompt += (f'Patient: {patient_case}\n\n'
                   f'{format_constraint}\n\nFinal diagnosis:')

    elif prompt_type == 'few-shot':
        examples = get_few_shot_examples(df_pe, cfg)
        prompt += (f'Examples (follow the same format for your answer):\n{examples}\n\n'
                   f'Patient: {patient_case}\n\n'
                   f'{format_constraint}\n\nFinal diagnosis:')

    elif prompt_type == 'cot':
        prompt += (
            f'Patient: {patient_case}\n\n'
            'Follow this five-question Chain-of-Thought protocol '
            '(Wang et al., arXiv:2503.12286):\n\n'
            'Q1. What are the key phenotypic features and clinical findings '
            'described in this note?\n'
            'Q2. Which rare diseases or genetic disorders are associated with '
            'these phenotypic features?\n'
            'Q3. What genetic or molecular mechanisms or causative genes are '
            'linked to the candidate diseases?\n'
            'Q4. What clinical evidence differentiates the most likely diagnosis '
            'from the other differentials?\n'
            'Q5. What is the final diagnosis or most likely rare disease?\n\n'
            f'{format_constraint}\n\nFinal diagnosis:'
        )

    elif prompt_type == 'rag':
        prompt += (
            'Retrieved evidence (passages from medical literature; may contain noise):\n'
            f'{evidence_block}\n\n'
            f'Patient: {patient_case}\n\n'
            'Use only evidence that is supported by the patient description. '
            'If retrieved evidence is weak, conflicting, or irrelevant, abstain.\n\n'
            f'{format_constraint}\n\nFinal diagnosis:'
        )

    elif prompt_type == 'self-rag':
        prompt += (
            'Use the Self-RAG reflection approach (Asai et al., ICLR 2024).\n\n'
            f'Retrieved passages (numbered, with cosine relevance scores):\n{evidence_block}\n\n'
            f'Patient: {patient_case}\n\n'
            'Reflection steps:\n'
            'Step 1 [IsREL]  — For each numbered passage, label [Relevant] or [Not Relevant]\n'
            'Step 2          — Using only [Relevant] passages, state your candidate diagnosis\n'
            'Step 3 [IsSUP]  — Is your candidate [Supported] / [Partially Supported] / [Not Supported]?\n'
            'Step 4 [IsUSE]  — If [Not Supported], revise using your biomedical knowledge\n\n'
            f'{format_constraint}\n\nFinal diagnosis:'
        )

    elif prompt_type == 'corrective-rag':
        prompt += (
            'Use the Corrective RAG approach (Yan et al., ICLR 2024).\n\n'
            f'{evidence_block}\n\n'
            f'Patient: {patient_case}\n\n'
            'Instructions based on retrieval quality label above:\n'
            '• CORRECT   → rely primarily on the filtered evidence\n'
            '• AMBIGUOUS → combine the evidence with your biomedical knowledge\n'
            '• INCORRECT → ignore retrieved documents; reason from medical knowledge\n\n'
            f'{format_constraint}\n\nFinal diagnosis:'
        )

    elif prompt_type == 'speculative-rag-draft':
        prompt += (
            '[Speculative RAG — DRAFT PHASE | Shi et al. 2024]\n'
            'You are the DRAFTER. Generate a concise candidate diagnosis from the '
            'high-relevance evidence subset below.\n\n'
            f'Evidence subset (highest-scored documents):\n{evidence_block}\n\n'
            f'Patient: {patient_case}\n\n'
            f'{format_constraint}\n\nFinal diagnosis:'
        )

    elif prompt_type == 'speculative-rag-verify':
        prompt += (
            '[Speculative RAG — VERIFY PHASE | Shi et al. 2024]\n'
            'You are the VERIFIER. Review the draft diagnosis and all retrieved evidence '
            'below; confirm the draft if well-supported, or provide the corrected diagnosis.\n\n'
            f'{evidence_block}\n\n'
            f'Patient: {patient_case}\n\n'
            'Verification:\n'
            '• Draft is supported by evidence → confirm it\n'
            '• Evidence contradicts the draft → provide the correct diagnosis\n'
            '• Evidence is insufficient → reason from your biomedical knowledge\n\n'
            f'{format_constraint}\n\nFinal diagnosis:'
        )

    elif prompt_type == 'react':
        prompt += (
            'Use the ReAct framework (Yao et al., ICLR 2023).\n\n'
            f'Retrieved evidence:\n{evidence_block}\n\n'
            f'Patient: {patient_case}\n\n'
            'Complete exactly 3 reasoning cycles, then commit to a diagnosis:\n'
            'Thought 1: Identify the key phenotypic features from the patient description.\n'
            'Action 1: Check which retrieved passages match these features.\n'
            'Observation 1: [Your finding]\n\n'
            'Thought 2: Build a differential diagnosis using the evidence.\n'
            'Action 2: Compare candidate diseases against the clinical presentation.\n'
            'Observation 2: [Your finding]\n\n'
            'Thought 3: Select the most likely rare disease.\n'
            'Action 3: Confirm the diagnosis is consistent with all evidence.\n'
            'Observation 3: [Your finding]\n\n'
            f'{format_constraint}\n\nFinal diagnosis:'
        )

    else:
        prompt += f'Patient: {patient_case}\n\n{format_constraint}\n\nFinal diagnosis:'

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# 5)  Model Loading + Inference
# ─────────────────────────────────────────────────────────────────────────────

def _check_disk_space(hf_cache_dir, min_bytes, model_id, log):
    try:
        free = shutil.disk_usage(hf_cache_dir).free
    except Exception:
        return
    required_gb = min_bytes / (1024 ** 3)
    free_gb = free / (1024 ** 3)
    if free < min_bytes:
        raise RuntimeError(
            f'Insufficient disk space for {model_id}. '
            f'Free: {free_gb:.1f} GB  Required: >={required_gb:.0f} GB'
        )
    log.info(f'Disk space OK: {free_gb:.1f} GB free')


def _cleanup_partial_download(model_id, hf_cache_dir, log):
    safe_id = model_id.replace('/', '--')
    repo_cache = os.path.join(hf_cache_dir, f'models--{safe_id}')
    if os.path.exists(repo_cache):
        try:
            shutil.rmtree(repo_cache)
            log.info(f'Cleaned up partial download: {repo_cache}')
        except Exception as e:
            log.warning(f'Could not remove partial cache {repo_cache}: {e}')


MODEL_DOWNLOAD_ALLOW_PATTERNS = [
    'config.json', 'generation_config.json',
    'tokenizer.json', 'tokenizer.model', 'tokenizer_config.json',
    'special_tokens_map.json', 'vocab.json', 'vocab.txt', 'merges.txt',
    '*.safetensors', '*.safetensors.index.json', '*.bin', '*.bin.index.json',
]
MODEL_DOWNLOAD_IGNORE_PATTERNS = ['*.gguf', '*.onnx', '*.ot', '*.h5', '*.msgpack', '*.tflite']


def _download_snapshot(model_id, cfg, log, model_snapshots):
    if model_id in model_snapshots:
        return model_snapshots[model_id]
    log.info(f'Downloading model snapshot: {model_id}')
    log.info(f'  hf_transfer fast-download: {os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "0")}')
    log.info(f'  parallel workers         : {cfg["DOWNLOAD_WORKERS"]}')
    _check_disk_space(cfg['hf_cache_dir'], cfg['_MIN_FREE_BYTES'], model_id, log)
    t0_dl = time.time()

    dl_kwargs = dict(
        repo_id=model_id,
        cache_dir=cfg['hf_cache_dir'],
        token=cfg['hf_token'],
        allow_patterns=MODEL_DOWNLOAD_ALLOW_PATTERNS,
        ignore_patterns=MODEL_DOWNLOAD_IGNORE_PATTERNS,
    )
    # max_workers speeds up multi-shard downloads (safetensors shards in parallel)
    try:
        snap = snapshot_download(**dl_kwargs, max_workers=cfg['DOWNLOAD_WORKERS'])
    except TypeError:
        # older huggingface_hub version without max_workers
        try:
            snap = snapshot_download(**dl_kwargs)
        except Exception as exc:
            _cleanup_partial_download(model_id, cfg['hf_cache_dir'], log)
            raise RuntimeError(
                f'Failed to download {model_id}. '
                'Check HF_TOKEN and license acceptance.'
            ) from exc
    except Exception as exc:
        _cleanup_partial_download(model_id, cfg['hf_cache_dir'], log)
        raise RuntimeError(
            f'Failed to download {model_id}. '
            'Check HF_TOKEN and license acceptance.'
        ) from exc

    elapsed = time.time() - t0_dl
    log.info(f'Download finished in {elapsed:.1f}s  ({elapsed/60:.1f} min)')
    model_snapshots[model_id] = snap
    log.info(f'Snapshot ready: {snap}')
    return snap


def pre_download_all_models(cfg, log, model_snapshots):
    """Pre-download every model in MODEL_REGISTRY before running evaluation.
    Run with --download_only to fetch everything while you still have a terminal,
    then launch the full evaluation with nohup."""
    log.info('=' * 70)
    log.info('PRE-DOWNLOAD MODE  (--download_only)')
    log.info('Tip: pip install hf-transfer  for 3-5x faster downloads')
    log.info('=' * 70)
    for key, model_id in cfg['MODEL_REGISTRY'].items():
        log.info(f'\n>>> [{key}]  {model_id}')
        try:
            _download_snapshot(model_id, cfg, log, model_snapshots)
            log.info(f'[{key}] download OK')
        except Exception as e:
            log.error(f'[{key}] download FAILED: {e}')
    log.info('\nAll downloads attempted. Check errors above if any model failed.')
    log.info('Now run without --download_only to start evaluation.')


def _load_model_safe(model_source, dtype, trust_remote_code, cfg=None):
    base_kw = {
        'device_map': 'auto', 'low_cpu_mem_usage': True,
        'trust_remote_code': trust_remote_code, 'torch_dtype': dtype,
        'local_files_only': True,
    }
    if cfg is not None:
        base_kw['config'] = cfg
    try:
        return AutoModelForCausalLM.from_pretrained(model_source, use_safetensors=True, **base_kw)
    except Exception as e:
        if any(k in str(e).lower() for k in ('safetensor', 'no file named', 'not found')):
            return AutoModelForCausalLM.from_pretrained(model_source, use_safetensors=False, **base_kw)
        raise


def _prepare_tokenizer_and_gen_config(model, tokenizer):
    if getattr(tokenizer, 'pad_token_id', None) is None:
        if getattr(tokenizer, 'eos_token_id', None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.pad_token   = tokenizer.eos_token
    pad_id = getattr(tokenizer, 'pad_token_id', None)
    eos_id = getattr(tokenizer, 'eos_token_id', None)
    kw = {'max_new_tokens': 1024, 'do_sample': True, 'temperature': 0.4}
    if pad_id is not None:
        kw['pad_token_id'] = pad_id
    if eos_id is not None:
        kw['eos_token_id'] = eos_id
    model.generation_config = GenerationConfig(**kw)


def load_generation_pipeline(model_key, cfg, log, model_pipes, model_snapshots):
    if model_key in model_pipes:
        return model_pipes[model_key]

    model_id = cfg['MODEL_REGISTRY'][model_key]
    log.info(f'Loading model: {model_key} -> {model_id}')
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    snap = _download_snapshot(model_id, cfg, log, model_snapshots)

    tokenizer = AutoTokenizer.from_pretrained(snap, trust_remote_code=False, local_files_only=True)
    try:
        model = _load_model_safe(snap, dtype, trust_remote_code=False)
    except Exception:
        mcfg = AutoConfig.from_pretrained(snap, trust_remote_code=True, local_files_only=True)
        if isinstance(getattr(mcfg, 'rope_scaling', None), dict):
            rs = dict(mcfg.rope_scaling)
            if 'type' not in rs and 'rope_type' in rs:
                rs['type'] = rs['rope_type']
            mcfg.rope_scaling = rs
        model = _load_model_safe(snap, dtype, trust_remote_code=True, cfg=mcfg)

    _prepare_tokenizer_and_gen_config(model, tokenizer)
    gen_pipe = pipeline('text-generation', model=model, tokenizer=tokenizer,
                        return_full_text=False)
    model_pipes[model_key] = gen_pipe
    return gen_pipe


# ── Self-Consistency Voting ────────────────────────────────────────────────────
_VOTE_ABSTENTION_KEYS = [
    "insufficient information", "cannot determine", "unable to determine",
    "i am unable", "i cannot", "not enough information", "there is not enough",
    "no diagnosis", "unable to diagnose", "cannot be determined",
]


def _extract_vote_candidate(text: str) -> str:
    raw = str(text).strip()
    if not raw:
        return ""
    m = re.search(r"final\s+diagnosis\s*[:\-]\s*(.+)", raw, flags=re.IGNORECASE)
    if m:
        return m.group(1).split("\n")[0].strip()
    return raw.split("\n")[-1].strip() or raw[:120].strip()


def _looks_like_abstention(text: str) -> bool:
    low = str(text).lower()
    return any(k in low for k in _VOTE_ABSTENTION_KEYS)


def _score_vote_output(text: str) -> float:
    cand = _extract_vote_candidate(text)
    cand_n = cand.lower().strip() if cand else ""
    score = 0.0
    if not cand_n:
        return -5.0
    if _looks_like_abstention(cand):
        score -= 4.0
    tok_len = len(cand_n.split())
    if tok_len <= 8:
        score += 1.2
    elif tok_len <= 14:
        score += 0.6
    else:
        score -= 0.8
    if re.search(r"\b(step|because|therefore|evidence|reasoning)\b", cand_n):
        score -= 0.8
    return float(score)


def _build_gen_config(pipe, *, max_new_tokens, do_sample, temperature=1.0):
    base = getattr(getattr(pipe, 'model', None), 'generation_config', None)
    cfg_g = GenerationConfig(**base.to_dict()) if base is not None else GenerationConfig()
    cfg_g.max_new_tokens = int(max_new_tokens)
    cfg_g.max_length     = None
    cfg_g.do_sample      = bool(do_sample)
    cfg_g.temperature    = float(temperature) if cfg_g.do_sample else 1.0
    tok = getattr(pipe, 'tokenizer', None)
    if tok is not None:
        if getattr(tok, 'pad_token_id', None) is not None:
            cfg_g.pad_token_id = int(tok.pad_token_id)
        if getattr(tok, 'eos_token_id', None) is not None:
            cfg_g.eos_token_id = int(tok.eos_token_id)
    return cfg_g


def call_llm(prompt_text, model_key, cfg, log, model_pipes, model_snapshots):
    pipe = load_generation_pipeline(model_key, cfg, log, model_pipes, model_snapshots)
    outputs = []
    for _ in range(cfg['N_CONSISTENCY_VOTES']):
        try:
            gen_cfg = _build_gen_config(pipe, max_new_tokens=cfg['MAX_GEN_TOKENS'],
                                         do_sample=True, temperature=cfg['GENERATION_TEMPERATURE'])
            out = pipe(prompt_text, generation_config=gen_cfg,
                       return_full_text=False, truncation=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            outputs.append(out[0]['generated_text'].strip())
        except Exception as e:
            log.debug(f'Generation error: {e}')
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    if not outputs:
        return ""
    scored = sorted(((_score_vote_output(o), o) for o in outputs),
                    key=lambda x: x[0], reverse=True)
    return scored[0][1]


def _save_cache(cache_file, llm_memory):
    with open(cache_file, 'w', encoding='utf-8') as f:
        json.dump(llm_memory, f, ensure_ascii=False)


def process_patient_case(patient_case, prompt_type, llm_model,
                         cfg, log, retrieval_backends, documents,
                         df_pe, llm_memory, cache_file,
                         model_pipes, model_snapshots,
                         normalize_fn, extract_final_diagnosis_fn,
                         target_val='', embedding_profile=None):
    retrieval_profile = str(embedding_profile or cfg['DEFAULT_EMBEDDING_PROFILE']).strip().lower()
    cache_profile = retrieval_profile if prompt_type in cfg['RAG_RETRIEVAL_TYPES'] else 'na'
    key = make_cache_key(patient_case, prompt_type, llm_model, embedding_profile=cache_profile)
    if key in llm_memory:
        return llm_memory[key], 0.0

    t0 = time.time()

    _retrieval_query = patient_case
    if prompt_type in cfg['RAG_RETRIEVAL_TYPES']:
        try:
            _pipe = load_generation_pipeline(llm_model, cfg, log, model_pipes, model_snapshots)
            _retrieval_query = expand_query_with_hpo(patient_case, _pipe, log)
        except Exception:
            pass

    def _ret(min_s=None):
        return retrieve_documents(
            _retrieval_query, retrieval_backends, documents, cfg, log,
            normalize_fn, k=cfg['RETRIEVAL_TOP_K'],
            embedding_profile=retrieval_profile,
            min_score=min_s or cfg['RETRIEVAL_MIN_SCORE'],
        )

    def _docs_text(docs):
        return '\n'.join([f"[score={d['score']:.3f}] [Source: {d['source_id']}] {d['text']}"
                          for d in docs])

    def _call(pt, evidence=''):
        return call_llm(build_prompt(patient_case, pt, df_pe, cfg, evidence),
                        llm_model, cfg, log, model_pipes, model_snapshots)

    if prompt_type == 'rag':
        pred = _call('rag', _docs_text(_ret()))

    elif prompt_type == 'self-rag':
        docs = _ret()
        numbered = ('\n\n'.join([
            f"[{i+1}] (cosine_score={d['score']:.3f}) [Source: {d['source_id']}]\n{d['text']}"
            for i, d in enumerate(docs)
        ]) if docs else '[No documents retrieved above threshold]')
        pred = _call('self-rag', numbered)

    elif prompt_type == 'corrective-rag':
        docs = _ret(min_s=cfg['CRAG_RETRIEVE_SCORE'])
        correct_docs  = [d for d in docs if d['score'] >= cfg['CRAG_CORRECT_SCORE']]
        ambiguous_docs = [d for d in docs if
                          cfg['CRAG_AMBIGUOUS_MIN'] <= d['score'] < cfg['CRAG_CORRECT_SCORE']]
        if correct_docs:
            quality_label = (f"CORRECT - {len(correct_docs)} high-confidence document(s) "
                             f"(cosine >= {cfg['CRAG_CORRECT_SCORE']})")
            use_docs = correct_docs
        elif ambiguous_docs:
            quality_label = (f"AMBIGUOUS - {len(ambiguous_docs)} document(s) with uncertain "
                             f"relevance; combine with biomedical knowledge")
            use_docs = ambiguous_docs
        else:
            quality_label = ("INCORRECT - no documents passed the relevance threshold; "
                             "reason from parametric medical knowledge only")
            use_docs = []
        filtered = _docs_text(use_docs) if use_docs else '[No documents passed the confidence filter]'
        crag_block = f"Retrieval quality: {quality_label}\n\nFiltered evidence:\n{filtered}"
        pred = _call('corrective-rag', crag_block)

    elif prompt_type == 'speculative-rag':
        docs = _ret()
        n_draft = max(1, len(docs) // 2) if len(docs) > 1 else len(docs)
        draft_docs = docs[:n_draft]
        draft_key  = make_cache_key(patient_case, 'speculative-rag-draft',
                                    llm_model, embedding_profile=cache_profile)
        if draft_key in llm_memory:
            draft_answer = llm_memory[draft_key]
        else:
            draft_answer = _call('speculative-rag-draft', _docs_text(draft_docs))
            llm_memory[draft_key] = draft_answer
            _save_cache(cache_file, llm_memory)
        draft_diag  = extract_final_diagnosis_fn(draft_answer)
        verify_block = (f"Draft diagnosis (to verify): {draft_diag}\n\n"
                        f"All retrieved evidence:\n{_docs_text(docs)}")
        pred = _call('speculative-rag-verify', verify_block)

    elif prompt_type == 'react':
        docs = _ret()
        pred = _call('react', '\n'.join([f"[score={d['score']:.3f}] {d['text']}" for d in docs]))

    else:
        pred = _call(prompt_type)

    inf_t = time.time() - t0
    llm_memory[key] = pred
    _save_cache(cache_file, llm_memory)
    return pred, inf_t


# ─────────────────────────────────────────────────────────────────────────────
# 6)  Metrics and Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def normalize_text(text):
    t = str(text).lower().strip()
    t = t.replace('-', ' ').replace('–', ' ').replace('—', ' ')
    t = t.translate(str.maketrans('', '', string.punctuation))
    t = re.sub(r'\s+', ' ', t).strip()
    return t


DISEASE_ALIASES = {
    "dlbcl": "diffuse large b cell lymphoma",
    "sle": "systemic lupus erythematosus",
    "all": "acute lymphoblastic leukemia",
    "aml": "acute myeloid leukemia",
    "cml": "chronic myeloid leukemia",
    "cll": "chronic lymphocytic leukemia",
    "hus": "hemolytic uremic syndrome",
    "tma": "thrombotic microangiopathy",
    "gbs": "guillain barre syndrome",
    "mfs": "miller fisher syndrome",
    "sma": "spinal muscular atrophy",
    "als": "amyotrophic lateral sclerosis",
    "pnh": "paroxysmal nocturnal hemoglobinuria",
    "ttp": "thrombotic thrombocytopenic purpura",
    "hsp": "henoch schonlein purpura",
    "iga vasculitis": "iga vasculitis henoch schonlein purpura",
    "nmo": "neuromyelitis optica",
    "nmosd": "neuromyelitis optica spectrum disorder",
    "pbc": "primary biliary cholangitis",
    "psc": "primary sclerosing cholangitis",
    "anca vasculitis": "anca associated vasculitis",
    "gpa": "granulomatosis with polyangiitis",
    "mpa": "microscopic polyangiitis",
    "egpa": "eosinophilic granulomatosis with polyangiitis",
    "ecd": "erdheim chester disease",
    "lch": "langerhans cell histiocytosis",
    "hlh": "hemophagocytic lymphohistiocytosis",
    "mas": "macrophage activation syndrome",
    "igg4 rd": "igg4 related disease",
    "igg4 related disease": "igg4 related disease",
    "behcets": "behcet disease",
    "behcets disease": "behcet disease",
    "behcet syndrome": "behcet disease",
    "schnitzlers syndrome": "schnitzler syndrome",
    "stills disease": "adult onset still disease",
    "adult onset stills disease": "adult onset still disease",
    "castlemans disease": "castleman disease",
    "wilsons disease": "wilson disease",
    "addisons disease": "addison disease",
    "cushings syndrome": "cushing syndrome",
    "pagets disease": "paget disease",
    "fabrys disease": "fabry disease",
    "gauchers disease": "gaucher disease",
    "gaucher disease": "gaucher disease",
    "hodgkins disease": "hodgkin lymphoma",
    "hodgkins lymphoma": "hodgkin lymphoma",
    "non hodgkins lymphoma": "non hodgkin lymphoma",
    "burgers disease": "buerger disease",
    "buergers disease": "buerger disease",
    "kikuchi fujimoto": "kikuchi fujimoto disease",
    "kikuchi disease": "kikuchi fujimoto disease",
    "takayasus arteritis": "takayasu arteritis",
    "wegeners granulomatosis": "granulomatosis with polyangiitis",
    "churg strauss": "eosinophilic granulomatosis with polyangiitis",
    "von hippel lindau": "von hippel lindau disease",
    "vhl": "von hippel lindau disease",
    "tuberous sclerosis": "tuberous sclerosis complex",
    "tsc": "tuberous sclerosis complex",
    "marfans syndrome": "marfan syndrome",
    "marfan": "marfan syndrome",
    "ehlers danlos": "ehlers danlos syndrome",
    "eds": "ehlers danlos syndrome",
    "pheochromocytoma": "pheochromocytoma",
    "paraganglioma": "paraganglioma",
    "myasthenia gravis": "myasthenia gravis",
    "mg": "myasthenia gravis",
    "lupus": "systemic lupus erythematosus",
    "fabry disease": "fabry disease",
}

DISEASE_ALIAS_MAP = {normalize_text(k): normalize_text(v) for k, v in DISEASE_ALIASES.items()}
DISEASE_CANONICAL_NAMES = sorted(set(DISEASE_ALIAS_MAP.values()))
DISEASE_CANONICAL_LOOKUP = {n: n for n in DISEASE_CANONICAL_NAMES}


def _preclean_disease_phrase(text: str) -> str:
    t = normalize_text(text)
    for prefix in ['the diagnosis is', 'the most likely diagnosis is',
                   'final diagnosis', 'diagnosis', 'the patient has',
                   'the patient likely has', 'most likely',
                   'based on the clinical presentation']:
        if t.startswith(prefix):
            t = t[len(prefix):].strip().lstrip(':').strip()
    return t


def _best_dictionary_match(candidate: str):
    candidate_norm = normalize_text(candidate)
    if not candidate_norm:
        return None
    toks = set(candidate_norm.split())
    best_name, best_score = None, -1.0
    for n in DISEASE_CANONICAL_NAMES:
        ntoks = set(n.split())
        inter = len(toks & ntoks)
        union = len(toks | ntoks)
        if union == 0:
            continue
        j = inter / union
        contain = 0.25 if (n in candidate_norm or candidate_norm in n) else 0.0
        score = j + contain
        if score > best_score:
            best_name, best_score = n, score
    if best_name is None:
        return None
    if best_score >= 0.58:
        return best_name
    if best_score >= 0.48 and len(set(best_name.split()) & toks) >= 2:
        return best_name
    return None


def canonicalize_disease_name(text: str) -> str:
    raw = str(text).strip()
    if not raw:
        return ""
    base = _preclean_disease_phrase(raw)
    if not base:
        return ""
    if base in DISEASE_ALIAS_MAP:
        alias_norm = normalize_text(DISEASE_ALIAS_MAP[base])
        return DISEASE_CANONICAL_LOOKUP.get(alias_norm, DISEASE_ALIAS_MAP[base])
    best = _best_dictionary_match(base)
    if best:
        return DISEASE_CANONICAL_LOOKUP.get(best, best)
    return base


def extract_final_diagnosis(text):
    text = str(text).strip()
    if not text:
        return ''
    m = re.search(r'(?i)final\s+diagnosis\s*[:\-–—]\s*(.+)', text)
    if m:
        diag = m.group(1).strip()
        diag = re.split(r'[\n\r]', diag)[0].strip()
        diag = re.sub(r'\s*\(?\s*(Explanation|Reasoning|Note|Because|Step|Confidence|References?)\b.*',
                      '', diag, flags=re.IGNORECASE)
        diag = diag.strip(' .')
        if len(diag) >= 2:
            return diag
    m = re.search(r'(?i)(?:the\s+)?most\s+likely\s+diagnosis\s+is\s+(.+?)[\.\n]', text)
    if m:
        return m.group(1).strip(' .')
    m = re.search(r'(?i)(?:the\s+)?diagnosis\s+is\s+(?:consistent\s+with\s+)?(.+?)[\.\n]', text)
    if m:
        return m.group(1).strip(' .')
    if len(text) < 100:
        cleaned = text.strip()
        cleaned = re.sub(r'^(?:the\s+)?(?:most\s+likely\s+)?(?:diagnosis\s+is\s+)?',
                         '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip(' .')
    sentences = re.split(r'[.\n]', text)
    for s in reversed(sentences):
        s = s.strip()
        if len(s) >= 3:
            return s
    return text[:100]


def get_canonical_prediction(prediction: str) -> str:
    return canonicalize_disease_name(extract_final_diagnosis(prediction))


def get_canonical_target(target: str) -> str:
    return canonicalize_disease_name(target)


def compute_exact_match(prediction, target):
    p = normalize_text(get_canonical_prediction(prediction))
    t = normalize_text(get_canonical_target(target))
    if not p or not t:
        return 0.0
    return 1.0 if (t in p or p in t) else 0.0


def compute_token_f1(prediction, target):
    p_toks = normalize_text(get_canonical_prediction(prediction)).split()
    t_toks = normalize_text(get_canonical_target(target)).split()
    if not p_toks or not t_toks:
        return 0.0
    pc = Counter(p_toks)
    tc = Counter(t_toks)
    common = sum((pc & tc).values())
    if common == 0:
        return 0.0
    precision = common / len(p_toks)
    recall    = common / len(t_toks)
    return 2 * precision * recall / (precision + recall)


_rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


def compute_rouge_l(prediction, target):
    p = normalize_text(get_canonical_prediction(prediction))
    t = normalize_text(get_canonical_target(target))
    return _rouge_scorer_obj.score(t, p)['rougeL'].fmeasure


def compute_cosine_similarity(prediction, target, retrieval_backends, cfg, embedding_profile=None):
    backend = retrieval_backends[str(embedding_profile or cfg['DEFAULT_EMBEDDING_PROFILE'])]
    em = backend['embedding_model']
    p = str(get_canonical_prediction(prediction))
    t = str(get_canonical_target(target))
    emb = em.encode([p, t], normalize_embeddings=True)
    return float(np.dot(emb[0], emb[1]))


_ABSTENTION_PHRASES = [
    'information insuffisante', 'informations insuffisantes',
    'je ne sais pas', 'impossible de determiner',
    'cannot determine', 'insufficient information',
    'not enough information', 'unable to determine',
    'i dont know', 'i do not know', 'no diagnosis', 'unable to diagnose',
    'i am unable', 'i cannot', 'cannot be determined', 'cannot make a diagnosis',
    'there is not enough', 'there is insufficient', 'based on the limited',
    'based on limited information', 'more information', 'additional information',
    'the provided information', 'with the information provided',
    'without more', 'without additional', 'need more', 'require more',
    'impossible to determine', 'not possible to determine',
    'no sufficient', 'insufficient data', 'the diagnosis cannot',
    'diagnosis cannot be made',
]


def _target_mentioned_in_full_response(prediction, target):
    pred_n = normalize_text(prediction)
    targ_n = normalize_text(get_canonical_target(target))
    if not pred_n or not targ_n:
        return False
    if targ_n in pred_n:
        return True
    toks = targ_n.split()
    if len(toks) >= 2 and sum(1 for tok in toks if tok in pred_n) / len(toks) >= 0.70:
        return True
    return False


def detect_hallucination(prediction, target, cosine_sim, cfg, em_score=None, f1_score=None):
    pred_l = normalize_text(extract_final_diagnosis(prediction))
    is_abstention = any(k in pred_l for k in _ABSTENTION_PHRASES)
    em_v = 0.0 if em_score is None else float(em_score)
    f1_v = 0.0 if f1_score is None else float(f1_score)

    if em_v >= 1.0 or cosine_sim >= cfg['THRESHOLDS']['cosine_correct']:
        return {'is_hallucination': False, 'is_abstention': False,
                'is_correct': True, 'confidence_category': 'correct'}
    if cosine_sim >= cfg['THRESHOLDS']['cosine_partial'] or f1_v >= cfg['THRESHOLDS']['f1_partial']:
        return {'is_hallucination': False, 'is_abstention': False,
                'is_correct': False, 'confidence_category': 'partially_correct'}
    if _target_mentioned_in_full_response(prediction, target):
        return {'is_hallucination': False, 'is_abstention': False,
                'is_correct': False, 'confidence_category': 'partially_correct'}
    if is_abstention:
        return {'is_hallucination': False, 'is_abstention': True,
                'is_correct': False, 'confidence_category': 'abstention'}
    return {'is_hallucination': True, 'is_abstention': False,
            'is_correct': False, 'confidence_category': 'hallucination'}


def compute_hallucination_gravity(em_score, f1_score, cosine_sim, category):
    em  = float(np.clip(float(em_score),  0.0, 1.0))
    f1  = float(np.clip(float(f1_score),  0.0, 1.0))
    cos = float(np.clip(float(cosine_sim), 0.0, 1.0))
    cat = str(category).strip().lower()
    if cat == 'correct':
        return 0.00
    if cat == 'abstention':
        return 0.10
    if cat == 'partially_correct':
        return float(np.clip(0.35 + 0.35 * (1.0 - cos) + 0.30 * (1.0 - f1), 0.35, 0.70))
    return float(np.clip(0.70 + 0.20 * (1.0 - cos) + 0.10 * (1.0 - em), 0.70, 1.00))


def format_eval_input(inp):
    if isinstance(inp, dict):
        q = str(inp.get('question', '')).strip()
        case_text = str(inp.get('case_report', '')).strip()
        if case_text:
            return f'{q}\n\nCase Report: {case_text}'.strip()
        options = inp.get('options', {})
        if isinstance(options, dict) and options:
            opts = ' '.join([f'{k}:{v}' for k, v in options.items()])
            return f'{q} {opts}'.strip()
        return q
    return str(inp)


def get_eval_subset(df, cfg):
    if cfg['EVAL_MODE'] == 'full_eval':
        return df.copy().reset_index(drop=True)
    n = min(cfg['QUICK_SAMPLE_SIZE'], len(df))
    subset = df.sample(n=n, random_state=cfg['QUICK_SAMPLE_SEED']).copy()
    subset['sample_seed'] = cfg['QUICK_SAMPLE_SEED']
    return subset.reset_index(drop=True)


def evaluate_pipeline(df_eval_subset, prompt_type, llm_model, cfg, log,
                      retrieval_backends, documents, df_pe, llm_memory,
                      cache_file, model_pipes, model_snapshots,
                      embedding_profile=None, technique_label=None):
    details = []
    active_profile = str(embedding_profile or cfg['DEFAULT_EMBEDDING_PROFILE']).strip().lower()
    total_time = total_em = total_f1 = total_rouge = total_cos = total_hgi = 0.0
    total_correct = total_partial = total_hallu = total_abst = 0
    display_technique = technique_label or cfg['TECHNIQUE_NAMES'].get(prompt_type, prompt_type)

    for i, row in df_eval_subset.iterrows():
        raw_input = row.get('input', None)
        if isinstance(raw_input, dict):
            inp_obj = raw_input
        elif raw_input is not None and not (isinstance(raw_input, float) and np.isnan(raw_input)):
            inp_obj = row.to_dict()
        else:
            inp_obj = row.to_dict()

        inp  = format_eval_input(inp_obj)
        targ = row.get('target', row.get('answer', ''))
        targ = (str(targ[0]).lower() if isinstance(targ, list) and len(targ) > 0
                else str(targ).lower())

        pred_raw, inf_t = process_patient_case(
            inp, prompt_type, llm_model, cfg, log,
            retrieval_backends, documents, df_pe, llm_memory, cache_file,
            model_pipes, model_snapshots,
            normalize_fn=normalize_text,
            extract_final_diagnosis_fn=extract_final_diagnosis,
            target_val=targ, embedding_profile=active_profile,
        )

        em  = compute_exact_match(pred_raw, targ)
        f1  = compute_token_f1(pred_raw, targ)
        rg  = compute_rouge_l(pred_raw, targ)
        cs  = compute_cosine_similarity(pred_raw, targ, retrieval_backends, cfg,
                                         embedding_profile=active_profile)
        hal = detect_hallucination(pred_raw, targ, cs, cfg, em_score=em, f1_score=f1)
        hgi = compute_hallucination_gravity(em, f1, cs, hal['confidence_category'])

        total_time += inf_t; total_em += em; total_f1 += f1
        total_rouge += rg; total_cos += cs; total_hgi += hgi

        cat = hal['confidence_category']
        total_correct += int(cat == 'correct')
        total_partial += int(cat == 'partially_correct')
        total_abst    += int(cat == 'abstention')
        total_hallu   += int(cat == 'hallucination')

        details.append({
            'idx': i, 'case_id': row.get('id', f'case_{i}'),
            'input': inp, 'target': targ, 'prediction': str(pred_raw),
            'prediction_clean': extract_final_diagnosis(str(pred_raw)),
            'prediction_canonical': get_canonical_prediction(str(pred_raw)),
            'target_canonical': get_canonical_target(targ),
            'technique': display_technique,
            'technique_base': cfg['TECHNIQUE_NAMES'].get(prompt_type, prompt_type),
            'prompt_type': prompt_type, 'model': llm_model,
            'embedding_profile': active_profile,
            'embedding_model': cfg['EMBEDDING_MODEL_REGISTRY'].get(active_profile, ''),
            'exact_match': em, 'token_f1': f1, 'rouge_l': rg,
            'cosine_sim': cs, 'hgi': hgi, 'category': cat,
            'is_hallucination': hal['is_hallucination'],
            'is_abstention': hal['is_abstention'],
            'inference_time': inf_t,
        })

        n_done = len(details)
        if n_done % 10 == 0:
            log.info(f'[{display_technique}] {n_done}/{len(df_eval_subset)}'
                     f'  EM={total_em/n_done:.3f}  Cos={total_cos/n_done:.3f}')

    n = max(len(details), 1)
    metrics = {
        'exact_match_rate':      total_em / n,
        'avg_token_f1':          total_f1 / n,
        'avg_rouge_l':           total_rouge / n,
        'avg_cosine_similarity': total_cos / n,
        'accuracy_rate':         total_correct / n,
        'partial_accuracy_rate': total_partial / n,
        'abstention_rate':       total_abst / n,
        'hallucination_rate':    total_hallu / n,
        'avg_hgi':               total_hgi / n,
        'avg_inference_time':    total_time / n,
    }
    return metrics, details


# ─────────────────────────────────────────────────────────────────────────────
# 7)  Run Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def profiles_for_prompt(prompt_type, cfg):
    if cfg['ENABLE_EMBEDDING_COMPARISON'] and prompt_type in cfg['RAG_RETRIEVAL_TYPES']:
        return cfg['EMBEDDING_PROFILES_TO_COMPARE']
    return [cfg['DEFAULT_EMBEDDING_PROFILE']]


def run_evaluation(df_eval, cfg, log, retrieval_backends, documents, df_pe,
                   llm_memory, cache_file, model_pipes, model_snapshots):
    eval_subset = get_eval_subset(df_eval, cfg)
    log.info(f"Evaluation mode: {cfg['EVAL_MODE']} | cases: {len(eval_subset)}")
    log.info(f"Models: {cfg['MODELS_TO_TEST']}")

    all_agg, all_details = [], []

    for model_key in cfg['MODELS_TO_TEST']:
        log.info('=' * 80)
        log.info(f'Model: {model_key}')
        log.info('=' * 80)

        for p in cfg['PROMPTS_TO_TEST']:
            for emb_profile in profiles_for_prompt(p, cfg):
                technique_label = cfg['TECHNIQUE_NAMES'][p]
                if cfg['ENABLE_EMBEDDING_COMPARISON'] and p in ['rag', 'react']:
                    technique_label = f"{technique_label} [{emb_profile}]"

                log.info(f'Running {technique_label}...')
                try:
                    metrics, run_details = evaluate_pipeline(
                        eval_subset, p, model_key, cfg, log,
                        retrieval_backends, documents, df_pe,
                        llm_memory, cache_file, model_pipes, model_snapshots,
                        embedding_profile=emb_profile,
                        technique_label=technique_label,
                    )
                except Exception as e:
                    log.error(f'Skipping {model_key} | {technique_label}: {e}')
                    continue

                acc_pct   = metrics['accuracy_rate']          * 100
                part_pct  = metrics['partial_accuracy_rate']  * 100
                abst_pct  = metrics['abstention_rate']        * 100
                hallu_pct = metrics['hallucination_rate']     * 100

                all_agg.append({
                    'Model': model_key, 'Technique': technique_label,
                    'Prompt_Type': p, 'Embedding_Profile': emb_profile,
                    'Embedding_Model': cfg['EMBEDDING_MODEL_REGISTRY'].get(emb_profile, ''),
                    'Exact_Match (%)':      metrics['exact_match_rate'] * 100,
                    'Token_F1 (%)':         metrics['avg_token_f1'] * 100,
                    'ROUGE_L (%)':          metrics['avg_rouge_l'] * 100,
                    'Cosine_Sim (%)':       metrics['avg_cosine_similarity'] * 100,
                    'Accuracy_Rate (%)':    acc_pct,
                    'Partial_Acc_Rate (%)': part_pct,
                    'Abstention_Rate (%)':  abst_pct,
                    'Hallucination_Rate (%)': hallu_pct,
                    'HGI (%)':              metrics['avg_hgi'] * 100,
                    'Avg_Inference_Time(s)': metrics['avg_inference_time'],
                })
                all_details.extend(run_details)

        # free GPU memory before next model
        if model_key in model_pipes:
            del model_pipes[model_key]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df_results = pd.DataFrame(all_agg)
    df_details = pd.DataFrame(all_details)

    if not df_results.empty:
        cat_sum = (df_results['Accuracy_Rate (%)'] + df_results['Partial_Acc_Rate (%)'] +
                   df_results['Abstention_Rate (%)'] + df_results['Hallucination_Rate (%)'])
        if not (cat_sum.round(1) == 100.0).all():
            log.warning('Category rates do NOT sum to 100% — check evaluate_pipeline logic')
        else:
            log.info('Category-rate integrity check passed (Acc + Partial + Abs + Hallu = 100%)')

    log.info('\n' + df_results.sort_values(['Model', 'Technique']).reset_index(drop=True).to_string())
    return df_results, df_details, eval_subset


# ─────────────────────────────────────────────────────────────────────────────
# 8)  Qualitative Samples
# ─────────────────────────────────────────────────────────────────────────────

def run_qualitative(eval_subset, cfg, log, retrieval_backends, documents, df_pe,
                    llm_memory, cache_file, model_pipes, model_snapshots):
    sample_qual = eval_subset.sample(
        n=min(cfg['QUAL_CASES'], len(eval_subset)),
        random_state=cfg['QUAL_SEED'],
    ).reset_index(drop=True)

    qual_rows, pretty_blocks = [], []

    for i, row in sample_qual.iterrows():
        case_id = row.get('id', f'qual_{i}')
        inp = format_eval_input(row.get('input', row.get('question', '')))
        target = row.get('target', row.get('answer', ''))
        target = (str(target[0]).lower() if isinstance(target, list) and len(target) > 0
                  else str(target).lower())

        lines = ['=' * 100, f'Case {i+1}: {case_id}', f'Target: {target}',
                 f'Input: {inp[:1200]}', '-' * 100]

        for model_key in cfg['MODELS_TO_TEST']:
            lines.append(f'## Model: {model_key}')
            for p in cfg['PROMPTS_TO_TEST']:
                for emb_profile in profiles_for_prompt(p, cfg):
                    technique_label = cfg['TECHNIQUE_NAMES'][p]
                    if cfg['ENABLE_EMBEDDING_COMPARISON'] and p in ['rag', 'react']:
                        technique_label = f"{technique_label} [{emb_profile}]"
                    try:
                        pred_raw, inf_t = process_patient_case(
                            inp, p, model_key, cfg, log, retrieval_backends, documents,
                            df_pe, llm_memory, cache_file, model_pipes, model_snapshots,
                            normalize_fn=normalize_text,
                            extract_final_diagnosis_fn=extract_final_diagnosis,
                            target_val=target, embedding_profile=emb_profile,
                        )
                    except Exception as e:
                        lines += [f'[{technique_label}]', f'Skipped: {e}', '-' * 80]
                        continue

                    pred_clean = extract_final_diagnosis(pred_raw)
                    cs = compute_cosine_similarity(pred_raw, target, retrieval_backends, cfg,
                                                   embedding_profile=emb_profile)
                    hal = detect_hallucination(pred_raw, target, cs, cfg)
                    qual_rows.append({
                        'case_id': case_id, 'target': target, 'input': inp,
                        'model': model_key, 'technique': technique_label,
                        'technique_base': cfg['TECHNIQUE_NAMES'][p],
                        'prompt_type': p, 'embedding_profile': emb_profile,
                        'embedding_model': cfg['EMBEDDING_MODEL_REGISTRY'].get(emb_profile, ''),
                        'response_raw': str(pred_raw), 'prediction_clean': str(pred_clean),
                        'category': hal['confidence_category'],
                        'cosine_similarity': float(cs), 'inference_time': float(inf_t),
                    })
                    lines += [f'[{technique_label}]', f'Response: {str(pred_raw)[:1500]}',
                              f'Extracted: {pred_clean}',
                              f'Category: {hal["confidence_category"]} | Cosine: {cs:.3f}',
                              '-' * 80]

        block = '\n'.join(lines)
        pretty_blocks.append(block)
        log.info('\n' + block)

    return pd.DataFrame(qual_rows), pretty_blocks


# ─────────────────────────────────────────────────────────────────────────────
# 9)  HGI Analysis + Plots
# ─────────────────────────────────────────────────────────────────────────────

def _hgi_row(row):
    cat = str(row.get('category', '')).strip().lower()
    em  = float(np.clip(float(row.get('exact_match', 0.0)), 0.0, 1.0))
    f1  = float(np.clip(float(row.get('token_f1', 0.0)), 0.0, 1.0))
    cos = float(np.clip(float(row.get('cosine_sim', row.get('cosine_similarity', 0.0))), 0.0, 1.0))
    if cat == 'correct':      return 0.00
    if cat == 'abstention':   return 0.10
    if cat == 'partially_correct':
        return float(np.clip(0.35 + 0.35 * (1.0 - cos) + 0.30 * (1.0 - f1), 0.35, 0.70))
    return float(np.clip(0.70 + 0.20 * (1.0 - cos) + 0.10 * (1.0 - em), 0.70, 1.00))


def run_hgi_analysis(df_results, df_details, cfg, log):
    out_dir = cfg['output_dir']
    df_dh = df_details.copy()
    df_dh['hallucination_gravity'] = df_dh.apply(_hgi_row, axis=1) * 100

    summary = (
        df_dh.groupby(['model', 'technique'], as_index=False)
        .agg(HGI_mean=('hallucination_gravity', 'mean'),
             HGI_median=('hallucination_gravity', 'median'),
             HGI_std=('hallucination_gravity', 'std'),
             Cases=('hallucination_gravity', 'count'))
    )
    summary['HGI_std'] = summary['HGI_std'].fillna(0.0)

    log.info('\n' + '=' * 80 + '\nHGI SUMMARY\n' + '=' * 80)
    log.info('\n' + summary.sort_values(['model', 'HGI_mean', 'technique']).to_string())

    # save CSVs
    df_dh.to_csv(os.path.join(out_dir, 'phase3_details_with_gravity.csv'), index=False)
    summary.to_csv(os.path.join(out_dir, 'phase3_hgi_summary.csv'), index=False)

    # merge HGI into results
    if 'Model' in df_results.columns and 'Technique' in df_results.columns:
        df_rh = df_results.merge(summary, left_on=['Model', 'Technique'],
                                 right_on=['model', 'technique'], how='left'
                                 ).drop(columns=['model', 'technique'])
    else:
        df_rh = df_results.copy()
    df_rh.to_csv(os.path.join(out_dir, 'phase3_results_with_hgi.csv'), index=False)

    # Plots
    sns.set_theme(style='whitegrid')

    # Plot 1 — HGI bar chart
    plt.figure(figsize=(14, 6))
    sns.barplot(data=summary, x='technique', y='HGI_mean', hue='model')
    plt.title('Hallucination Gravity Index (HGI) — Mean by Technique and Model')
    plt.xlabel('Technique'); plt.ylabel('HGI Mean (%)'); plt.xticks(rotation=20, ha='right')
    plt.ylim(0, 100); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot_hgi_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 2 — HGI heatmap
    try:
        pivot = summary.pivot(index='model', columns='technique', values='HGI_mean')
        plt.figure(figsize=(14, 4))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Reds', vmin=0, vmax=100)
        plt.title('HGI Heatmap (%)'); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'plot_hgi_heatmap.png'), dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        log.warning(f'HGI heatmap failed: {e}')

    # Plot 3 — HGI distribution
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df_dh, x='technique', y='hallucination_gravity', hue='model')
    plt.title('HGI Severity Distribution by Technique')
    plt.xlabel('Technique'); plt.ylabel('HGI (%)'); plt.xticks(rotation=20, ha='right')
    plt.ylim(0, 100); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot_hgi_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Plot 4 — Combined metrics
    candidate_metrics = [
        'Exact_Match (%)', 'Token_F1 (%)', 'ROUGE_L (%)', 'Cosine_Sim (%)',
        'Accuracy_Rate (%)', 'Partial_Acc_Rate (%)', 'Abstention_Rate (%)',
        'Hallucination_Rate (%)', 'HGI_mean',
    ]
    plot_metrics = [m for m in candidate_metrics if m in df_rh.columns]
    melted = df_rh.melt(id_vars=['Model', 'Technique'], value_vars=plot_metrics,
                         var_name='Metric', value_name='Value')
    plt.figure(figsize=(18, 7))
    sns.barplot(data=melted, x='Metric', y='Value', hue='Model', errorbar=None)
    plt.title('Final Combined Metrics (%)'); plt.xlabel('Metric'); plt.ylabel('Value (%)')
    plt.ylim(0, 100); plt.xticks(rotation=25, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'plot_final_combined_metrics_with_hgi.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    log.info(f'Plots saved to {out_dir}')
    return df_rh


# ─────────────────────────────────────────────────────────────────────────────
# 10) Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = _parse_args()

    # Build config early so we can set up directories and logging
    IS_KAGGLE = os.path.exists('/kaggle')
    output_dir = args.output_dir or ('/kaggle/working/' if os.path.exists('/kaggle/working/') else '.')
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    log = _setup_logging(output_dir)
    log.info('Phase-3 RAG evaluation script started')

    cfg = build_config(args)

    # Materialise derived paths into cfg
    cfg['output_dir']       = output_dir
    cfg['hf_cache_dir']     = os.path.join(output_dir, 'hf_cache')
    cfg['faiss_index_dir']  = os.path.join(output_dir, 'faiss_index')
    cfg['cache_dir']        = os.path.join(output_dir, 'cache_llm')
    cfg['cache_file']       = os.path.join(cfg['cache_dir'],
                                            'predictions_comparative_phase3.json')
    for d in (cfg['hf_cache_dir'], cfg['faiss_index_dir'], cfg['cache_dir']):
        Path(d).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('HF_HOME', cfg['hf_cache_dir'])
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    # ── Authenticate with HuggingFace Hub ─────────────────────────────────────
    if cfg.get('hf_token') and not os.environ.get('HF_TOKEN'):
        hf_login(token=cfg['hf_token'], add_to_git_credential=False)
        log.info('Authenticated with HuggingFace Hub via --hf_token')
    elif os.environ.get('HF_TOKEN'):
        log.info('Authenticated with HuggingFace Hub via HF_TOKEN env var')
    else:
        log.warning('No HF_TOKEN provided — unauthenticated requests may be rate-limited or blocked')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg['device'] = device
    log.info(f'Running on device: {device}')
    log.info(f"Models to test  : {cfg['MODELS_TO_TEST']}")
    log.info(f"Eval mode       : {cfg['EVAL_MODE']} | n={cfg['QUICK_SAMPLE_SIZE']} "
             f"| seed={cfg['QUICK_SAMPLE_SEED']}")

    # ── Pre-download mode: fetch all models then exit ─────────────────────────
    # Usage:  python phase1_rag-pf.py --download_only
    # After this finishes, run the full evaluation with nohup (no download needed).
    model_snapshots: dict = {}
    if cfg.get('DOWNLOAD_ONLY'):
        pre_download_all_models(cfg, log, model_snapshots)
        return

    # ── Load datasets ─────────────────────────────────────────────────────────
    df_rag, df_pe, df_eval, documents = load_datasets(cfg, log)

    # ── Load embedding backends ───────────────────────────────────────────────
    retrieval_backends = {}
    profiles_needed = set([cfg['DEFAULT_EMBEDDING_PROFILE']])
    if cfg['ENABLE_EMBEDDING_COMPARISON']:
        profiles_needed.update(cfg['EMBEDDING_PROFILES_TO_COMPARE'])

    for profile in profiles_needed:
        retrieval_backends[profile] = _load_or_create_backend(
            profile, cfg, documents, device, log
        )

    # ── Load LLM prediction cache ─────────────────────────────────────────────
    try:
        llm_memory = json.load(open(cfg['cache_file'], 'r', encoding='utf-8')) \
            if os.path.exists(cfg['cache_file']) else {}
    except Exception:
        llm_memory = {}

    model_pipes = {}   # key -> pipeline (freed between models)
    # model_snapshots already initialised above (reused if --download_only was skipped)

    # ── Run evaluation ────────────────────────────────────────────────────────
    df_results, df_details, eval_subset = run_evaluation(
        df_eval, cfg, log, retrieval_backends, documents, df_pe,
        llm_memory, cfg['cache_file'], model_pipes, model_snapshots,
    )

    # ── Save results ──────────────────────────────────────────────────────────
    df_results.to_csv(os.path.join(output_dir, 'phase3_results_aggregated.csv'), index=False)
    df_details.to_csv(os.path.join(output_dir, 'phase3_results_details.csv'), index=False)
    log.info(f"Saved: {os.path.join(output_dir, 'phase3_results_aggregated.csv')}")
    log.info(f"Saved: {os.path.join(output_dir, 'phase3_results_details.csv')}")

    # ── Qualitative samples ───────────────────────────────────────────────────
    df_qual, pretty_blocks = run_qualitative(
        eval_subset, cfg, log, retrieval_backends, documents, df_pe,
        llm_memory, cfg['cache_file'], model_pipes, model_snapshots,
    )
    if not df_qual.empty:
        df_qual.to_csv(os.path.join(output_dir, 'sample_case_responses.csv'), index=False)
        df_qual.to_json(os.path.join(output_dir, 'sample_case_responses.json'),
                        orient='records', force_ascii=False, indent=2)
    with open(os.path.join(output_dir, 'sample_case_responses_pretty.txt'), 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(pretty_blocks))

    # ── HGI analysis ──────────────────────────────────────────────────────────
    if not df_results.empty and not df_details.empty:
        run_hgi_analysis(df_results, df_details, cfg, log)

    log.info('Phase-3 evaluation complete. All outputs saved to: ' + output_dir)


if __name__ == '__main__':
    main()
