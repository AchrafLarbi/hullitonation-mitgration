# ===========================================================================
# 🧬 Rare Disease Diagnostic Assistant — FULL Kaggle Notebook
# ===========================================================================
# Run this in a Kaggle notebook with GPU T4 enabled.
# It loads BioMistral-7B on GPU + the full KG-RAG pipeline.
# Gradio share=True gives a PUBLIC URL — no ngrok needed.
#
# SETUP:
#   1. New Kaggle Notebook → Settings → GPU T4 x2 → Internet ON
#   2. Cell 1: !pip install -q gradio transformers accelerate bitsandbytes \
#              sentence-transformers faiss-cpu networkx pandas python-docx
#   3. Cell 2: Paste this entire script and run
#   4. You'll get a public URL like: https://xxxxx.gradio.live
# ===========================================================================

import os, json, re, unicodedata, time, gc
import numpy as np
import networkx as nx
import pandas as pd
import torch
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, GenerationConfig
from sentence_transformers import SentenceTransformer
import faiss
import gradio as gr

# ── Config ───────────────────────────────────────────────────────────────────
# Points to your existing Kaggle dataset "Pfe-rarediease"
# Make sure to add it: Notebook → Add data → Your datasets → Pfe-rarediease
KAGGLE_DS    = "/kaggle/input/datasets/larbimohammedachraf/pfe-rarediease"
FAISS_DIR    = "/kaggle/input/datasets/larbimohammedachraf/open-eval/faiss_index"
EMBEDDING_ID = "NeuML/pubmedbert-base-embeddings"
LLM_MODEL_ID = "BioMistral/BioMistral-7B"

CORPUS_PATH = os.path.join(KAGGLE_DS, "rag_corpus_final.jsonl")
KG_PATH     = os.path.join(KAGGLE_DS, "knowledge_graph.json")
IDX_PATH    = os.path.join(FAISS_DIR, "rag_corpus_pubmedbert_base.index")
# FAISS index might be in pfe-rarediease or open-eval dataset
DATASET_IDX_CANDIDATES = [
    os.path.join(KAGGLE_DS, "faiss_index", "rag_pubmedbert.index"),
    "/kaggle/input/open-eval/faiss_index/rag_pubmedbert.index",
    "/kaggle/input/open-eval/faiss_index/rag_corpus_pubmedbert_base.index",
]

TOP_K      = 5
MIN_SCORE  = 0.20
MAX_NEW_TOKENS = 1024
GENERATION_TEMPERATURE = 0.4
N_VOTES    = 5

_FMT = (
    "IMPORTANT: You MUST commit to exactly one rare disease diagnosis. "
    "Even if uncertain, name the single most likely rare disease based on all available evidence. "
    "End your response exactly with 'Final diagnosis: [Exact Disease Name]'. "
    "Only output 'Final diagnosis: Insufficient information' if the clinical description "
    "contains NO phenotypic features whatsoever."
)

# ═══════════════════════════  TEXT UTILS  ═══════════════════════════════════
def normalize_text(t):
    t = str(t).lower()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"'", "", t)
    t = re.sub(r"[-\u2013\u2014]", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def _is_question_line(line):
    """Return True if the line looks like a CoT question, not a diagnosis."""
    s = line.strip()
    if s.endswith("?"):
        return True
    if re.match(r"^Q\d", s, re.I):
        return True
    if re.match(r"^\d+[\).:]\s*(What|Which|How|Why|Where|When|Are|Is|Do|Does)\b", s, re.I):
        return True
    return False

def extract_diagnosis(text):
    raw = str(text).strip()
    if not raw: return ""
    def _c(s):
        s = str(s).strip().split("\n")[0].strip()
        s = re.sub(r'^[\[\("\']+|[\]\)"\'\.]+$', "", s).strip()
        s = re.split(r"\b(?:because|based on|given|considering)\b", s, 1, re.I)[0]
        # Split on semicolons, periods, AND commas — model often lists multiple diseases
        s = re.split(r"[;.,]", s, 1)[0]
        # Remove numbered prefixes like "1." "2."
        s = re.sub(r'^\s*\d+[\.)\-]\s*', '', s)
        s = re.sub(r"\s+", " ", s).strip(" :-")
        if _is_question_line(s):
            return None
        # Hard cap: a disease name should never be > 80 chars
        if len(s) > 80:
            s = s[:80].rsplit(' ', 1)[0].strip()
        return s if 2 < len(s) <= 80 else None
    # Try explicit "Final diagnosis:" patterns first
    for p in [r"final\s+diagnosis\s*[:\-]\s*(.+)", r"diagnosis\s+is\s*[:\-]?\s*(.+)",
              r"diagnosis\s*[:\-]\s*(.+)", r"most\s+likely\s+(?:diagnosis|disease)\s+is\s*[:\-]?\s*(.+)"]:
        ms = list(re.finditer(p, raw, re.I))
        if ms:
            c = _c(ms[-1].group(1))
            if c: return c
    # Fallback: use LAST non-question line (matches notebook's _extract_vote_candidate)
    non_q_lines = [l.strip() for l in raw.split("\n") if l.strip() and not _is_question_line(l)]
    for ln in reversed(non_q_lines):
        c = _c(ln)
        if c: return c
    # Ultimate fallback: last line of raw output (truncated)
    last_line = raw.split("\n")[-1].strip()
    if last_line and len(last_line) > 80:
        last_line = last_line[:80].rsplit(' ', 1)[0].strip()
    return last_line if last_line else raw[:80].strip()

# Template echo blocklist — model sometimes echoes prompt placeholders
_TEMPLATE_ECHOES = {
    "most likely disease", "second most likely disease", "third most likely disease",
    "exact disease name", "disease name", "differential diagnoses",
}

# ═══════════════════════════  LOAD EVERYTHING  ═════════════════════════════
print("=" * 60)
print("🧬 Rare Disease Diagnostic Assistant — Loading")
print("=" * 60)

# 1. Verify dataset
print("\n⏳ [1/5] Checking Kaggle dataset...")
assert os.path.exists(CORPUS_PATH), f"Missing: {CORPUS_PATH}\nAdd dataset 'Pfe-rarediease' to your notebook!"
assert os.path.exists(KG_PATH), f"Missing: {KG_PATH}"
print(f"✅ Dataset found at {KAGGLE_DS}")

# 2. Corpus
print("⏳ [2/5] Loading RAG corpus...")
documents = []
with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            documents.append(json.loads(line))
        except json.JSONDecodeError:
            continue
print(f"✅ Corpus: {len(documents)} documents")

# 3. Embeddings + FAISS
print("⏳ [3/5] Loading PubMedBERT + FAISS index...")
emb_model = SentenceTransformer(EMBEDDING_ID, device="cpu")

if os.path.exists(IDX_PATH):
    faiss_idx = faiss.read_index(IDX_PATH)
    print(f"✅ FAISS loaded from cache: {faiss_idx.ntotal} vectors")
else:
    # Search candidate paths from Kaggle datasets
    found_idx = None
    for candidate in DATASET_IDX_CANDIDATES:
        if os.path.exists(candidate):
            found_idx = candidate
            break
    if found_idx:
        faiss_idx = faiss.read_index(found_idx)
        faiss.write_index(faiss_idx, IDX_PATH)
        print(f"✅ FAISS loaded from {found_idx}: {faiss_idx.ntotal} vectors")
    else:
        print(f"⏳ Building FAISS index ({len(documents)} docs)...")
        texts = [str(d.get("text", "")) for d in documents]
        embs = emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                 batch_size=64, show_progress_bar=True)
        faiss_idx = faiss.IndexFlatIP(embs.shape[1])
        faiss_idx.add(embs)
        faiss.write_index(faiss_idx, IDX_PATH)
        print(f"✅ FAISS built: {faiss_idx.ntotal} vectors")

# 4. Knowledge Graph
print("⏳ [4/5] Loading Knowledge Graph...")
G = nx.Graph()
kg_names = []
kg_idx = None

if os.path.exists(KG_PATH):
    with open(KG_PATH, "r", encoding="utf-8") as f:
        kg = json.load(f)
    for node in kg.get("nodes", []):
        G.add_node(node["id"], **node)
    for edge in kg.get("edges", []):
        G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", "related_to"))
    for nid, attr in G.nodes(data=True):
        name = attr.get("name") or attr.get("label") or ""
        if isinstance(name, str) and name.strip():
            kg_names.append((nid, name.strip()))
    if kg_names:
        kg_texts = [n for _, n in kg_names]
        kg_embs = emb_model.encode(kg_texts, convert_to_numpy=True, normalize_embeddings=True, batch_size=128)
        kg_idx = faiss.IndexFlatIP(kg_embs.shape[1])
        kg_idx.add(kg_embs)
        if G.number_of_edges() == 0:
            sims, inds = kg_idx.search(np.ascontiguousarray(kg_embs, dtype=np.float32),
                                        min(5, len(kg_names)))
            for i in range(len(kg_names)):
                for sim, j in zip(sims[i], inds[i]):
                    if j != i and sim >= 0.60:
                        s, d = kg_names[i][0], kg_names[j][0]
                        if not G.has_edge(s, d):
                            G.add_edge(s, d, relation="semantic_related", weight=float(sim))
print(f"✅ KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# 5. BioMistral-7B on GPU (float16 — full precision for better quality)
print("⏳ [5/5] Loading BioMistral-7B on GPU (float16)...")

tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()
device = next(model.parameters()).device
print(f"✅ BioMistral-7B loaded on {device}!")

# Set up GenerationConfig
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token

_gen_cfg_kw = {"max_new_tokens": 1024, "max_length": None, "do_sample": True, "temperature": 0.1}
if tokenizer.pad_token_id is not None:
    _gen_cfg_kw["pad_token_id"] = tokenizer.pad_token_id
if tokenizer.eos_token_id is not None:
    _gen_cfg_kw["eos_token_id"] = tokenizer.eos_token_id
model.generation_config = GenerationConfig(**_gen_cfg_kw)

# Use HuggingFace pipeline (matches notebook exactly — no apply_chat_template)
gen_pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)
print("✅ Generation pipeline ready!")

print("\n" + "=" * 60)
print("✅ ALL READY — Starting Gradio UI...")
print("=" * 60)

# ═══════════════════════════  RETRIEVAL & KG  ══════════════════════════════
def retrieve(query, k=TOP_K):
    if faiss_idx is None or faiss_idx.ntotal == 0: return []
    q = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = faiss_idx.search(np.ascontiguousarray(q, dtype=np.float32), min(k*3, faiss_idx.ntotal))
    res = []
    for s, i in zip(scores[0], indices[0]):
        if 0 <= i < len(documents) and s >= MIN_SCORE:
            d = documents[i]
            res.append({"score": float(s), "text": str(d.get("text",""))[:800],
                        "title": str(d.get("title",""))[:200]})
    return res[:k]

def link_kg(query, top_k=6):
    if kg_idx is None or kg_idx.ntotal == 0: return []
    q = emb_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, indices = kg_idx.search(np.ascontiguousarray(q, dtype=np.float32), min(top_k, kg_idx.ntotal))
    return [(kg_names[i][0], kg_names[i][1], float(s))
            for s, i in zip(scores[0], indices[0]) if s >= 0.30 and 0 <= i < len(kg_names)]

def get_triples(node_ids, mx=20):
    if G.number_of_edges() == 0:
        return [(G.nodes[n].get("name",str(n)), "type_of", G.nodes[n].get("type","entity"))
                for n in node_ids if G.has_node(n)][:mx]
    triples, vis, front = [], set(), list(dict.fromkeys(node_ids))
    for _ in range(2):
        nf = []
        for nd in front:
            if nd in vis or not G.has_node(nd): continue
            vis.add(nd)
            for nb in G.neighbors(nd):
                if len(triples) >= mx: break
                triples.append((G.nodes[nd].get("name",str(nd)),
                                G.edges[nd,nb].get("relation","related_to"),
                                G.nodes[nb].get("name",str(nb))))
                if nb not in vis: nf.append(nb)
        front = nf
        if not front or len(triples) >= mx: break
    return triples

def community_summary(seeds):
    lines, seen = [], set()
    for s in seeds:
        if not G.has_node(s): continue
        for nd, _ in sorted(nx.single_source_shortest_path_length(G, s, cutoff=2).items(), key=lambda x:x[1]):
            if nd in seen: continue
            seen.add(nd)
            a = G.nodes[nd]
            nm, tp = a.get("name",str(nd)), a.get("type","entity")
            nbs = [G.nodes[n].get("name",str(n)) for n in list(G.neighbors(nd))[:5]]
            lines.append(f"[{tp}] {nm} — {', '.join(nbs)}" if nbs else f"[{tp}] {nm}")
    return "\n".join(lines) if lines else "[No community context]"

def fmt_passages(docs, n=5):
    if not docs: return "[No passages retrieved]"
    return "\n".join(f"[{d['score']:.3f}] {d.get('title','')}: {d.get('text','')[:400]}" for d in docs[:n])

# ═══════════════════════════  LLM (GPU)  ═══════════════════════════════════
def _build_generation_config(max_new_tokens, do_sample, temperature=1.0):
    """Build GenerationConfig matching the notebook's approach."""
    base_cfg = getattr(model, "generation_config", None)
    cfg = GenerationConfig(**base_cfg.to_dict()) if base_cfg is not None else GenerationConfig()
    cfg.max_new_tokens = int(max_new_tokens)
    cfg.max_length = None
    cfg.do_sample = bool(do_sample)
    cfg.temperature = float(temperature) if cfg.do_sample else 1.0
    if tokenizer.pad_token_id is not None:
        cfg.pad_token_id = int(tokenizer.pad_token_id)
    if tokenizer.eos_token_id is not None:
        cfg.eos_token_id = int(tokenizer.eos_token_id)
    return cfg

def call_llm(prompt, max_tok=MAX_NEW_TOKENS, temp=GENERATION_TEMPERATURE, do_sample=True):
    """Generate text using BioMistral-7B via HF pipeline (matches notebook)."""
    try:
        gen_cfg = _build_generation_config(max_tok, do_sample, temp)
        out = gen_pipe(
            prompt,
            generation_config=gen_cfg,
            return_full_text=False,
            truncation=True,
        )
        torch.cuda.empty_cache()
        return out[0]["generated_text"].strip()
    except Exception as e:
        torch.cuda.empty_cache()
        return f"[LLM Error: {e}]"

# ── Self-consistency voting (Mod 3, matches notebook) ────────────────────────
_VOTE_ABSTENTION_KEYS = [
    "insufficient information", "cannot determine", "unable to determine",
    "i am unable", "i cannot", "not enough information", "there is not enough",
    "no diagnosis", "unable to diagnose", "cannot be determined",
]

def _score_vote_output(text):
    cand = extract_diagnosis(text)
    cand_n = normalize_text(cand)
    score = 0.0
    if not cand_n:
        return -5.0
    if any(k in cand_n for k in _VOTE_ABSTENTION_KEYS):
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
    # Penalize outputs that look like questions rather than diagnoses
    if "?" in cand_n or re.match(r"^q\d", cand_n):
        score -= 5.0
    return float(score)

def _get_differential_with_rag(patient_query, exclude_diseases):
    """Find ONE differential diagnosis using the full RAG + KG pipeline.
    The LLM gets the same evidence as the 4 architectures: retrieved passages
    from the 130K doc corpus + KG triples from the knowledge graph.
    Falls back to KG/corpus semantic search if LLM fails."""
    if not patient_query:
        return ""
    # Step 1: Retrieve evidence (same pipeline as the 4 architectures)
    expanded = expand_query_with_hpo(patient_query)
    docs = retrieve(expanded, 10)
    evidence = fmt_passages(docs, 6)

    # Step 2: Get KG context (triples)
    linked = link_kg(patient_query, 8)
    seed_ids = [nid for nid, _, _ in linked]
    triples = get_triples(seed_ids, 15)
    kg_block = "\n".join(f"  ({s}, {r}, {o})" for s, r, o in triples) if triples else ""

    # Step 3: Build context-rich prompt
    exclude_str = ", ".join(exclude_diseases)
    prompt = "You are an expert medical diagnostician specializing in rare diseases.\n\n"
    if kg_block:
        prompt += f"=== Knowledge Graph Evidence ===\n{kg_block}\n\n"
    prompt += (
        f"=== Retrieved Medical Literature ===\n{evidence}\n\n"
        f"Patient: {str(patient_query)[:500]}\n\n"
        f"The following diagnosis has already been considered: {exclude_str}.\n"
        "Based on the medical evidence above, what is the NEXT most likely "
        "rare disease diagnosis for this patient?\n\n"
        f"{_FMT}\n\nFinal diagnosis:"
    )
    out = call_llm(prompt, max_tok=200, temp=0.4, do_sample=True)
    if not out.startswith("[LLM Error"):
        diag = extract_diagnosis(out)
        diag_n = normalize_text(diag)
        is_abstention = any(k in diag_n for k in _VOTE_ABSTENTION_KEYS)
        is_excluded = any(normalize_text(e) == diag_n for e in exclude_diseases)
        is_echo = diag_n in _TEMPLATE_ECHOES
        if diag and not is_abstention and not is_excluded and not is_echo and len(diag_n) > 3:
            return diag

    # Fallback: KG + RAG corpus semantic search
    exclude_n = {normalize_text(e) for e in exclude_diseases}
    seen = set(exclude_n)
    # KG search
    if kg_idx is not None and kg_idx.ntotal > 0:
        q_emb = emb_model.encode([patient_query], convert_to_numpy=True, normalize_embeddings=True)
        k = min(20, kg_idx.ntotal)
        scores, indices = kg_idx.search(np.ascontiguousarray(q_emb, dtype=np.float32), k)
        for sim, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(kg_names) or float(sim) < 0.25:
                continue
            name = kg_names[idx][1]
            name_n = normalize_text(name)
            if name_n in seen or len(name_n) < 4:
                continue
            if any(en in name_n or name_n in en for en in exclude_n):
                continue
            return name
    # RAG corpus title search
    for d in docs:
        title = str(d.get("title", "")).strip()
        title_n = normalize_text(title)
        if title_n and title_n not in seen and len(title_n) > 3:
            if not any(en in title_n or title_n in en for en in exclude_n):
                return title
    return ""

def _get_confidence_ratings(patient_query, diagnoses):
    """Ask LLM to rate confidence (0-100%) for each diagnosis."""
    if not diagnoses:
        return {}
    diag_list = "\n".join(f"{i+1}. {d}" for i, d in enumerate(diagnoses))
    prompt = (
        f"Patient: {str(patient_query)[:400]}\n\n"
        "Rate the likelihood of each diagnosis as a percentage (0-100%):\n"
        f"{diag_list}\n\n"
        f"1. {diagnoses[0]}:"
    )
    out = call_llm(prompt, max_tok=80, temp=0.2, do_sample=False)
    if out.startswith("[LLM Error"):
        return {d: max(85 - i*25, 10) for i, d in enumerate(diagnoses)}
    full = f"1. {diagnoses[0]}:" + out
    numbers = [int(m.group(1)) for m in re.finditer(r'(\d{1,3})\s*%', full)
               if 1 <= int(m.group(1)) <= 100]
    ratings = {}
    for i, d in enumerate(diagnoses):
        ratings[d] = numbers[i] if i < len(numbers) else max(85 - i*25, 10)
    return ratings

def generate_response(prompt, n_votes=N_VOTES):
    """Self-consistency voting. Returns (best_raw, top3_with_scores).
    top3_with_scores = [(diagnosis, confidence%), ...] from vote counting.
    Each vote produces a diagnosis; we count how many times each appears."""
    outputs = []
    for _ in range(n_votes):
        out = call_llm(prompt, MAX_NEW_TOKENS, GENERATION_TEMPERATURE, do_sample=True)
        if not out.startswith("[LLM Error"):
            outputs.append(out)
    if not outputs:
        return "", []

    # Extract diagnosis from each output
    all_diags = []
    best_raw = outputs[0]
    best_score = -999
    for o in outputs:
        s = _score_vote_output(o)
        if s > best_score:
            best_score = s
            best_raw = o
        d = extract_diagnosis(o)
        if d:
            d_n = normalize_text(d)
            if d_n not in _TEMPLATE_ECHOES and not any(k in d_n for k in _VOTE_ABSTENTION_KEYS) and len(d_n) > 3:
                all_diags.append(d)

    if not all_diags:
        return best_raw, []

    # Count votes for each unique diagnosis (normalized)
    norm_to_original = {}
    vote_counts = Counter()
    for d in all_diags:
        d_n = normalize_text(d)
        vote_counts[d_n] += 1
        if d_n not in norm_to_original or len(d) < len(norm_to_original[d_n]):
            norm_to_original[d_n] = d  # keep shortest (cleanest) form

    # Rank by vote count, compute confidence as percentage
    total = len(all_diags)
    top3 = []
    for d_n, count in vote_counts.most_common(3):
        confidence = round((count / total) * 100)
        top3.append((norm_to_original[d_n], confidence))

    return best_raw, top3

# ── HPO query expansion (Mod 2, matches notebook) ────────────────────────────
def expand_query_with_hpo(query):
    """Ask the LLM to extract 5 HPO-style phenotype terms from the note."""
    expansion_prompt = (
        "List the 5 most specific medical or phenotypic terms from this clinical description. "
        "Use standard terminology (e.g. HPO terms). One term per line, no explanations:\n\n"
        f"{str(query)[:600]}\n\nTerms:"
    )
    terms = call_llm(expansion_prompt, max_tok=80, temp=1.0, do_sample=False)
    if terms.startswith("[LLM Error"):
        return query
    return f"{query}\n\nKey phenotypes: {terms}"

# ═══════════════════════════  4 ARCHITECTURES  ═════════════════════════════
# Aligned with phenopacket-bio-eval.ipynb

# Architecture 1: GraphRAG
def run_graphrag(q):
    # 1. Find related diseases in KG
    linked = link_kg(q, 8)
    seed_ids = [nid for nid,_,_ in linked]
    # 2. Build community narrative
    comm = community_summary(seed_ids)
    # 3. Retrieve documents from 130K corpus
    expanded = expand_query_with_hpo(q)
    docs = retrieve(expanded, 10)
    p = fmt_passages(docs, 6)
    # 4. Combine KG community + documents in ONE prompt
    prompt = (
        "You are an expert medical diagnostician specializing in rare diseases.\n\n"
        "Community context (knowledge graph):\n"
        f"{comm}\n\n"
        "Dense Retrieval Evidence:\n"
        f"{p}\n\n"
        f"Patient: {q}\n\n"
        f"{_FMT}\n\nFinal diagnosis:"
    )
    raw, top3 = generate_response(prompt)
    return raw, top3, comm, p

# Architecture 2: KAPING
def run_kaping(q):
    # 1. Entity linking
    linked = link_kg(q, 8)
    seed_ids = [nid for nid,_,_ in linked]
    # 2. Extract triples instead of community
    triples = get_triples(seed_ids, 25)
    if triples:
        kg_block = "\n".join(f"  ({s}, {r}, {o})" for s,r,o in triples)
    else:
        kg_block = "  (No KG triples found for these symptoms)"
    # 3. Retrieve documents
    expanded = expand_query_with_hpo(q)
    docs = retrieve(expanded, 10)
    p = fmt_passages(docs, 6)
    # 4. Structured triples + documents prompt
    prompt = (
        "You are an expert medical diagnostician specializing in rare diseases.\n\n"
        "Structured knowledge graph triples:\n"
        f"{kg_block}\n\n"
        "Supporting Literature Passages:\n"
        f"{p}\n\n"
        f"Patient: {q}\n\n"
        "Instructions:\n"
        "1. Identify KG triples most relevant to the patient\n"
        "2. Cross-reference triples with literature\n"
        "3. Name the single most probable rare disease\n\n"
        f"{_FMT}\n\nFinal diagnosis:"
    )
    raw, top3 = generate_response(prompt)
    return raw, top3, kg_block, p

# Architecture 3: RAG-driven CoT (THE WINNER)
def run_rag_cot(q):
    # 1. Retrieve first
    expanded = expand_query_with_hpo(q)
    docs = retrieve(expanded, 10)
    evidence = fmt_passages(docs, 6)
    # 2. Then reason with 5-question CoT
    prompt = (
        "You are an expert medical diagnostician specializing in rare diseases.\n\n"
        "Retrieved domain-specific evidence:\n"
        f"{evidence}\n\n"
        f"Patient: {q}\n\n"
        "Q1. What are the key phenotypic features and clinical findings?\n"
        "Q2. Which body systems are involved?\n"
        "Q3. What rare diseases in the EVIDENCE match these findings?\n"
        "Q4. Which disease BEST fits all the clinical evidence?\n"
        "Q5. What is the final diagnosis?\n\n"
        f"{_FMT}\n\nFinal diagnosis:"
    )
    raw, top3 = generate_response(prompt)
    return raw, top3, "", evidence

# Architecture 4: CoT-driven RAG
def run_cot_rag(q):
    # 1. Reason first: extract HPO terms (deterministic, no retrieval)
    extraction_prompt = (
        "You are a genetic counselor specializing in rare diseases.\n\n"
        "Extract key phenotypic features (HPO terms) from the following "
        "clinical note. List only the phenotypic features, one per line. "
        "No explanations.\n\n"
        f"{q}\n\nPhenotypic features:"
    )
    phenotype_output = call_llm(extraction_prompt, max_tok=512, temp=1.0, do_sample=False)
    if phenotype_output.startswith("[LLM Error"):
        phenotype_output = q
    # 2. Retrieve using the phenotypes as query
    docs = retrieve(phenotype_output, 10)
    passage_ctx = fmt_passages(docs, 6)
    # 3. Synthesize phenotypes + evidence → diagnosis
    synthesis_prompt = (
        "You are an expert medical diagnostician specializing in rare diseases.\n\n"
        "Extracted phenotypic features:\n"
        f"{phenotype_output}\n\n"
        "Retrieved biomedical evidence:\n"
        f"{passage_ctx}\n\n"
        f"Patient: {q}\n\n"
        "Based on the extracted phenotypes and retrieved evidence, "
        "identify the single most probable rare disease diagnosis.\n\n"
        f"{_FMT}\n\nFinal diagnosis:"
    )
    raw, top3 = generate_response(synthesis_prompt)
    return raw, top3, phenotype_output, passage_ctx

ARCHS = {
    "GraphRAG": run_graphrag,
    "KG-Augmented RAG / KAPING": run_kaping,
    "RAG-driven CoT": run_rag_cot,
    "CoT-driven RAG": run_cot_rag,
}

# ═══════════════════════════  MAIN  ════════════════════════════════════════
def diagnose_all(note):
    if not note.strip():
        return "⚠️ Enter a clinical description.", "", "", "", "", ""
    t0 = time.time()

    results = {}
    for arch_name, arch_fn in ARCHS.items():
        at = time.time()
        raw, top3, ctx, ps = arch_fn(note)
        results[arch_name] = {
            "top3": top3,  # [(diag, confidence%), ...]
            "raw": raw,
            "context": ctx,
            "passages": ps,
            "time": time.time() - at,
        }

    total_time = time.time() - t0
    medals = ["🥇", "🥈", "🥉"]

    # Build summary with per-architecture top 3
    summary = f"## 🏥 Comparative Diagnosis — All 4 Architectures\n\n"
    summary += f"**Model:** BioMistral-7B (Kaggle T4 GPU) | **Total Time:** {total_time:.1f}s\n\n"

    for i, (name, r) in enumerate(results.items(), 1):
        summary += f"### {i}. {name} — ⏱️ {r['time']:.1f}s\n\n"
        if r["top3"]:
            summary += "| Rank | Diagnosis | Confidence |\n"
            summary += "|------|-----------|------------|\n"
            for rank, (d, pct) in enumerate(r["top3"]):
                medal = medals[rank] if rank < 3 else ""
                summary += f"| {medal} {rank+1} | `{d}` | {pct}% |\n"
        else:
            summary += "> ⚠️ No diagnosis extracted\n"
        summary += "\n"

    # Cross-architecture consensus on Top-1 diagnoses
    all_top1 = [r["top3"][0][0] for r in results.values() if r["top3"]]
    if all_top1:
        diag_counts = Counter(normalize_text(d) for d in all_top1)
        consensus_n, consensus_count = diag_counts.most_common(1)[0]
        consensus_label = next(d for d in all_top1 if normalize_text(d) == consensus_n)
        summary += f"---\n### 🎯 Cross-Architecture Consensus ({consensus_count}/4 agree): `{consensus_label}`\n"

    details = ""
    for name, r in results.items():
        top1_diag = r["top3"][0][0] if r["top3"] else "(none)"
        details += f"### {name}\n**Top Diagnosis:** {top1_diag}\n"
        details += f"```\n{r['raw']}\n```\n---\n"

    # KG context
    graphrag_ctx = results.get("GraphRAG", {}).get("context", "")
    kaping_ctx = results.get("KG-Augmented RAG / KAPING", {}).get("context", "")
    cot_pheno = results.get("CoT-driven RAG", {}).get("context", "")
    kg_combined = ""
    if graphrag_ctx:
        kg_combined += f"=== GraphRAG Community ===\n{graphrag_ctx}\n\n"
    if kaping_ctx:
        kg_combined += f"=== KAPING Triples ===\n{kaping_ctx}\n\n"
    if cot_pheno:
        kg_combined += f"=== CoT Phenotypes ===\n{cot_pheno}\n\n"

    passages = ""
    for r in results.values():
        if r["passages"]:
            passages = r["passages"]
            break

    # All top-1 diagnoses with confidence (compact)
    diag_list_str = ""
    for name, r in results.items():
        if r["top3"]:
            entries = ", ".join(f"{d} ({pct}%)" for d, pct in r["top3"])
            diag_list_str += f"{name}: {entries}\n"

    return summary, diag_list_str.strip(), details, kg_combined, passages, f"Total: {total_time:.1f}s"

# ── DOCX import helper ────────────────────────────────────────────────────────
def import_docx(file):
    """Extract all text from a .docx file and return it as a string."""
    if file is None:
        return ""
    try:
        from docx import Document
        doc = Document(file.name)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return text.strip() if text.strip() else "⚠️ The uploaded file contains no text."
    except ImportError:
        return "⚠️ python-docx not installed. Run: pip install python-docx"
    except Exception as e:
        return f"⚠️ Error reading file: {e}"

# ═══════════════════════════  UI  ══════════════════════════════════════════

CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; }
#header { 
    background: linear-gradient(135deg, #0d4e6f 0%, #1a7a5a 50%, #0d6e4e 100%);
    padding: 24px 32px; border-radius: 12px; margin-bottom: 16px;
    color: white; text-align: center;
}
#header h1 { color: white !important; font-size: 2em; margin: 0 0 4px 0; }
#header h3 { color: #b8e6d0 !important; font-weight: 400; margin: 0; font-size: 1em; }
#header p { color: #d0ece0 !important; margin: 8px 0 0 0; font-size: 0.85em; }
.arch-table th { background: #f0faf5 !important; font-weight: 600; }
.arch-table td { font-size: 0.9em; }
#run-btn { 
    background: linear-gradient(135deg, #0d6e4e, #1a9a6a) !important;
    border: none !important; font-size: 1.1em !important;
    padding: 12px 0 !important; border-radius: 8px !important;
    transition: transform 0.1s ease !important;
}
#run-btn:hover { transform: translateY(-1px) !important; }
#results-panel { min-height: 400px; }
.diagnosis-box textarea { 
    font-family: 'Segoe UI', system-ui, sans-serif !important;
    font-size: 0.95em !important; line-height: 1.6 !important;
}
.evidence-box textarea {
    font-family: 'Consolas', 'Monaco', monospace !important;
    font-size: 0.82em !important; line-height: 1.5 !important;
    background: #f8fffe !important;
}
.timer-box input {
    text-align: center !important; font-weight: 600 !important;
    font-size: 1.1em !important; color: #0d6e4e !important;
}
footer { display: none !important; }
"""

EXAMPLES = [
    "A 3-year-old boy with progressive motor weakness, calf pseudohypertrophy, CK 15000 U/L, Gowers' sign. Affected maternal uncle.",
    "Progressive ataxia, neuropathy, cardiomyopathy, diabetes. Adolescent onset. Frataxin mutation.",
    "45F with recurrent abdominal pain, peripheral neuropathy, dark urine. Elevated porphobilinogen.",
    "Infant with hepatosplenomegaly, Gaucher cells in marrow, glucocerebrosidase deficiency.",
]

with gr.Blocks(
    title="Rare Disease Diagnostic Assistant",
    theme=gr.themes.Soft(primary_hue="teal", secondary_hue="cyan"),
    css=CUSTOM_CSS,
) as demo:

    # ── Header ──
    gr.HTML("""
    <div id="header">
        <h1>🧬 Rare Disease Diagnostic Assistant</h1>
        <h3>KG-RAG Pipeline — Mitigating Hallucinations in Medical LLMs</h3>
        <p>BioMistral-7B · PubMedBERT Embeddings · 130K Document Corpus · 335-Node Knowledge Graph</p>
    </div>
    """)

    # ── Architecture overview (collapsed) ──
    with gr.Accordion("ℹ️ Architecture Overview", open=False):
        gr.Markdown("""
| # | Architecture | Strategy |
|:-:|:------------|:---------|
| 1 | **GraphRAG** | KG community summaries → dense retrieval → diagnosis |
| 2 | **KAPING** | Structured KG triples → literature cross-reference → diagnosis |
| 3 | **RAG-driven CoT** | Retrieve evidence → 5-question Chain-of-Thought → diagnosis |
| 4 | **CoT-driven RAG** | Extract phenotypes → targeted retrieval → synthesis → diagnosis |

Each architecture runs **5 self-consistency votes** and returns a **Top 3 differential** ranked by vote confidence.
        """)

    # ── Main layout ──
    with gr.Row(equal_height=False):
        # LEFT: Input panel
        with gr.Column(scale=1, min_width=350):
            gr.Markdown("#### 📋 Patient Input")
            inp = gr.Textbox(
                label="Clinical Description",
                lines=12,
                placeholder="Paste or type the clinical case report here...\n\nInclude: symptoms, lab results, imaging findings, patient demographics, medical history...",
                show_label=False,
            )
            with gr.Row():
                docx_file = gr.File(
                    label="📄 Import .docx",
                    file_types=[".docx"],
                    file_count="single",
                    scale=1,
                )
            run_btn = gr.Button(
                "🔍  Run All 4 Architectures",
                variant="primary",
                size="lg",
                elem_id="run-btn",
            )
            timer = gr.Textbox(
                label="⏱️ Total Time",
                interactive=False,
                elem_classes=["timer-box"],
            )
            gr.Examples(EXAMPLES, [inp], label="📝 Quick Examples")

        # RIGHT: Results panel
        with gr.Column(scale=2, min_width=500):
            gr.Markdown("#### 📊 Diagnostic Results")
            summary = gr.Markdown(
                value="*Run a diagnosis to see comparative results across all 4 architectures...*",
                elem_id="results-panel",
            )
            with gr.Accordion("📌 Summary — All Diagnoses with Confidence", open=False):
                diag_out = gr.Textbox(
                    label="Per-Architecture Top 3",
                    lines=8,
                    max_lines=16,
                    interactive=False,
                    elem_classes=["diagnosis-box"],
                )

    # ── Detail panels ──
    with gr.Row():
        with gr.Column():
            with gr.Accordion("📄 Raw LLM Outputs", open=False):
                details_out = gr.Markdown()

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("🕸️ Knowledge Graph Context", open=False):
                ctx_out = gr.Textbox(
                    label="KG Triples & Community Summaries",
                    lines=12,
                    max_lines=30,
                    interactive=False,
                    elem_classes=["evidence-box"],
                )
        with gr.Column(scale=1):
            with gr.Accordion("📚 Retrieved Evidence Passages", open=False):
                pass_out = gr.Textbox(
                    label="PubMed/OMIM Passages",
                    lines=12,
                    max_lines=30,
                    interactive=False,
                    elem_classes=["evidence-box"],
                )

    # ── Footer ──
    gr.Markdown(
        "<center style='color:#999; font-size:0.8em; margin-top:16px;'>"
        "⚠️ Research demo — NOT for clinical use. "
        "Built with BioMistral-7B + PubMedBERT on Kaggle T4 GPU."
        "</center>"
    )

    # ── Events ──
    docx_file.change(fn=import_docx, inputs=[docx_file], outputs=[inp])
    run_btn.click(
        fn=diagnose_all,
        inputs=[inp],
        outputs=[summary, diag_out, details_out, ctx_out, pass_out, timer],
    )

# Launch with share=True — creates public URL, no ngrok needed!
demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
