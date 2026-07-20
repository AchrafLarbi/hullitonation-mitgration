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
#              sentence-transformers faiss-cpu networkx pandas
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
    """Self-consistency voting (notebook-style). Returns (best_raw, diagnosis)."""
    outputs = []
    for _ in range(n_votes):
        out = call_llm(prompt, MAX_NEW_TOKENS, GENERATION_TEMPERATURE, do_sample=True)
        if not out.startswith("[LLM Error"):
            outputs.append(out)
    if not outputs:
        return "", ""
    scored = sorted(((_score_vote_output(o), o) for o in outputs),
                    key=lambda x: x[0], reverse=True)
    best_raw = scored[0][1]
    diag = extract_diagnosis(best_raw)
    # Fallback: if extract_diagnosis failed, try all outputs
    if not diag:
        for _, o in scored:
            diag = extract_diagnosis(o)
            if diag:
                break
    return best_raw, diag if diag else ""

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

# Architecture 1: GraphRAG (Edge et al., 2024)
def run_graphrag(q):
    # Step 1 — Entity linking
    linked = link_kg(q, 8)
    seed_ids = [nid for nid,_,_ in linked]
    # Step 2 — Community summary (GraphRAG local search)
    comm = community_summary(seed_ids)
    # Step 3 — Dense retrieval using HPO-expanded query
    expanded = expand_query_with_hpo(q)
    docs = retrieve(expanded, 10)
    p = fmt_passages(docs, 6)
    # Step 4 — Prompt (Edge et al. 2024, §4.1)
    prompt = (
        "You are an expert medical diagnostician specializing in rare diseases.\n\n"
        "=== GraphRAG: Community Knowledge (Edge et al., 2024) ===\n"
        "Disease-phenotype community context derived from the medical knowledge graph:\n"
        f"{comm}\n\n"
        "=== Dense Retrieval Evidence ===\n"
        f"{p}\n\n"
        f"Patient: {q}\n\n"
        "Instructions: Use the knowledge graph context to understand disease associations "
        "and the retrieved passages to ground your reasoning. Identify the most specific "
        "matching rare disease and commit to one diagnosis.\n\n"
        f"{_FMT}\n\nFinal diagnosis:"
    )
    raw, diag = generate_response(prompt)
    return raw, diag, comm, p

# Architecture 2: KG-Augmented RAG / KAPING (Baek et al., 2023)
def run_kaping(q):
    # Step 1 — Entity linking
    linked = link_kg(q, 8)
    seed_ids = [nid for nid,_,_ in linked]
    # Step 2 & 3 — BFS triple extraction + verbalization
    triples = get_triples(seed_ids, 25)
    if triples:
        kg_block = "\n".join(f"  ({s}, {r}, {o})" for s,r,o in triples)
    else:
        kg_block = "  (No KG triples found for these symptoms)"
    # Step 4 extension — dense passage retrieval with HPO-expanded query
    expanded = expand_query_with_hpo(q)
    docs = retrieve(expanded, 10)
    p = fmt_passages(docs, 6)
    # Prompt aligned with KAPING §4 zero-shot QA format
    prompt = (
        "You are an expert medical diagnostician specializing in rare diseases.\n\n"
        "=== KG-Augmented RAG — KAPING (Baek et al., 2023) ===\n"
        "Structured knowledge graph triples (subject, relation, object):\n"
        f"{kg_block}\n\n"
        "=== Supporting Literature Passages ===\n"
        f"{p}\n\n"
        f"Patient: {q}\n\n"
        "Instructions (KAPING §3.2):\n"
        "1. Identify the KG triples most relevant to the patient's symptoms.\n"
        "2. Cross-reference relevant triples with the literature passages.\n"
        "3. Name the single most probable rare disease grounded in both sources.\n\n"
        f"{_FMT}\n\nFinal diagnosis:"
    )
    raw, diag = generate_response(prompt)
    return raw, diag, kg_block, p

# Architecture 3: RAG-driven CoT (Wang et al., 2025)
# Exact prompt from the working notebook (advanced-technique-bio-final-med.ipynb)
def run_rag_cot(q):
    # Step 1: HPO query expansion + retrieval
    expanded = expand_query_with_hpo(q)
    docs = retrieve(expanded, 10)
    evidence = fmt_passages(docs, 6)
    # Step 2 + 3: evidence-first prompt with the paper's five-question CoT
    prompt = (
        "You are an expert medical diagnostician specializing in rare diseases.\n\n"
        "=== RAG-driven CoT (Wang et al., arXiv:2503.12286) ===\n"
        "Retrieved domain-specific evidence (HPO/OMIM-enriched passages):\n"
        f"{evidence}\n\n"
        f"Patient description:\n{q}\n\n"
        "Follow this five-question Chain-of-Thought protocol:\n\n"
        "Q1. What are the key phenotypic features and clinical findings described in this note?\n"
        "Q2. Which rare diseases or genetic disorders are associated with these phenotypic features?\n"
        "Q3. What genetic or molecular mechanisms or causative genes are linked to the candidate diseases?\n"
        "Q4. What clinical evidence differentiates the most likely diagnosis from the other differentials?\n"
        "Q5. What is the final diagnosis or most likely rare disease?\n\n"
        f"{_FMT}\n\nFinal diagnosis:"
    )
    raw, diag = generate_response(prompt)
    return raw, diag, "", evidence

# Architecture 4: CoT-driven RAG (Wang et al., 2025)
def run_cot_rag(q):
    # Step 1: Extract key phenotypic features (HPO terms) — deterministic
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
    # Step 2: Retrieve evidence using phenotype output as query
    docs = retrieve(phenotype_output, 10)
    passage_ctx = fmt_passages(docs, 6)
    # Step 3: Synthesis — phenotypes + evidence → single diagnosis
    synthesis_prompt = (
        "You are an expert medical diagnostician specializing in rare diseases.\n\n"
        "=== CoT-driven RAG (Wang et al., arXiv:2503.12286) ===\n\n"
        "Extracted phenotypic features (HPO terms):\n"
        f"{phenotype_output}\n\n"
        "Retrieved biomedical evidence:\n"
        f"{passage_ctx}\n\n"
        f"Patient description:\n{q}\n\n"
        "Based on the extracted phenotypes and retrieved evidence, identify "
        "the single most probable rare disease diagnosis.\n\n"
        f"{_FMT}\n\nFinal diagnosis:"
    )
    raw, diag = generate_response(synthesis_prompt)
    return raw, diag, phenotype_output, passage_ctx

ARCHS = {
    "GraphRAG (Edge et al., 2024)": run_graphrag,
    "KG-Augmented RAG / KAPING (Baek et al., 2023)": run_kaping,
    "RAG-driven CoT (Wang et al., 2025)": run_rag_cot,
    "CoT-driven RAG (Wang et al., 2025)": run_cot_rag,
}

# ═══════════════════════════  MAIN  ════════════════════════════════════════
def diagnose_all(note):
    if not note.strip():
        return "⚠️ Enter a clinical description.", "", "", "", "", ""
    t0 = time.time()

    results = {}
    for arch_name, arch_fn in ARCHS.items():
        at = time.time()
        raw, diag, ctx, ps = arch_fn(note)
        results[arch_name] = {
            "diagnosis": diag if diag else extract_diagnosis(raw),
            "raw": raw,
            "context": ctx,
            "passages": ps,
            "time": time.time() - at,
        }

    # Consensus Top 1 from all architectures
    all_diags = [r["diagnosis"] for r in results.values() if r["diagnosis"]]
    top1 = ""
    if all_diags:
        diag_counts = Counter(normalize_text(d) for d in all_diags)
        consensus_n = diag_counts.most_common(1)[0][0]
        top1 = next(d for d in all_diags if normalize_text(d) == consensus_n)

    # Build shared top-3: Top 1 from voting, Top 2/3 via RAG+KG-grounded LLM
    shared_top3 = []
    if top1:
        # Top 2: LLM with full RAG + KG context, excluding Top 1
        top2 = _get_differential_with_rag(note, [top1])
        # Top 3: LLM with full RAG + KG context, excluding Top 1 + Top 2
        exclude = [top1]
        if top2:
            exclude.append(top2)
        top3_d = _get_differential_with_rag(note, exclude) if top2 else ""

        diag_list = [d for d in [top1, top2, top3_d] if d]

        # One LLM call to rate confidence of all 3 diagnoses
        ratings = _get_confidence_ratings(note, diag_list)
        for d in diag_list:
            shared_top3.append((d, ratings.get(d, 50)))

    total_time = time.time() - t0

    # Build summary
    summary = f"## 🏥 Comparative Diagnosis — All 4 Architectures\n\n"
    summary += f"**Model:** BioMistral-7B (Kaggle T4 GPU) | **Total Time:** {total_time:.1f}s\n\n"
    summary += "| # | Architecture | Diagnosis | Time |\n"
    summary += "|---|---|---|---|\n"
    for i, (name, r) in enumerate(results.items(), 1):
        short_name = name.split(" (")[0]
        summary += f"| {i} | **{short_name}** | `{r['diagnosis']}` | {r['time']:.1f}s |\n"

    if all_diags:
        consensus = Counter(normalize_text(d) for d in all_diags).most_common(1)[0]
        summary += f"\n### 🎯 Consensus ({consensus[1]}/4 agree): `{consensus[0]}`\n"

    if shared_top3:
        summary += "\n### 📊 Top 3 Differential Diagnoses (LLM Confidence)\n\n"
        summary += "| Rank | Diagnosis | Confidence |\n"
        summary += "|------|-----------|------------|\n"
        medals = ["🥇", "🥈", "🥉"]
        for rank, (d, pct) in enumerate(shared_top3):
            medal = medals[rank] if rank < 3 else ""
            summary += f"| {medal} {rank+1} | `{d}` | {pct}% |\n"

    # Detailed raw outputs
    details = ""
    for name, r in results.items():
        short = name.split(" (")[0]
        details += f"### {short}\n**Diagnosis:** {r['diagnosis']}\n"
        details += f"```\n{r['raw']}\n```\n---\n"

    # KG context
    graphrag_ctx = results.get("GraphRAG (Edge et al., 2024)", {}).get("context", "")
    kaping_ctx = results.get("KG-Augmented RAG / KAPING (Baek et al., 2023)", {}).get("context", "")
    cot_pheno = results.get("CoT-driven RAG (Wang et al., 2025)", {}).get("context", "")
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

    # Diagnoses with confidence
    diag_list_str = ""
    if shared_top3:
        for rank, (d, pct) in enumerate(shared_top3, 1):
            diag_list_str += f"{rank}. {d} ({pct}%)\n"

    return summary, diag_list_str.strip(), details, kg_combined, passages, f"Total: {total_time:.1f}s"

# ═══════════════════════════  UI  ══════════════════════════════════════════
EXAMPLES = [
    "A 3-year-old boy with progressive motor weakness, calf pseudohypertrophy, CK 15000 U/L, Gowers' sign. Affected maternal uncle.",
    "Progressive ataxia, neuropathy, cardiomyopathy, diabetes. Adolescent onset. Frataxin mutation.",
    "45F with recurrent abdominal pain, peripheral neuropathy, dark urine. Elevated porphobilinogen.",
    "Infant with hepatosplenomegaly, Gaucher cells in marrow, glucocerebrosidase deficiency.",
]

with gr.Blocks(title="Rare Disease Diagnostic Assistant",
               theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue")) as demo:
    gr.Markdown("""# 🧬 Rare Disease Diagnostic Assistant
### KG-RAG Pipeline — Mitigating Hallucinations in Medical LLMs

**BioMistral-7B** (Kaggle T4 GPU) + **PubMedBERT** + **130K docs** + **335-node KG**

Runs **all 4 architectures** simultaneously and shows comparative results.

| # | Architecture | Paper | Strategy |
|---|---|---|---|
| 1 | **GraphRAG** | Edge et al., 2024 | KG community summaries + retrieval |
| 2 | **KAPING** | Baek et al., 2023 | Structured KG triples + retrieval |
| 3 | **RAG-driven CoT** | Wang et al., 2025 | Retrieve → 5-question CoT |
| 4 | **CoT-driven RAG** | Wang et al., 2025 | Decompose → retrieve → synthesize |

⚠️ **Research demo — NOT for clinical use.**
""")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Textbox(label="📋 Clinical Description", lines=10,
                             placeholder="Enter symptoms, labs, imaging findings...")
            run_btn = gr.Button("🔍 Run All 4 Architectures", variant="primary", size="lg")
            timer = gr.Textbox(label="⏱️ Timing", interactive=False)
        with gr.Column(scale=2):
            summary = gr.Markdown(label="Comparative Results")
            diag_out = gr.Textbox(label="📌 All Diagnoses (Top 3 with Confidence)", lines=6, interactive=False)
    with gr.Accordion("📄 Detailed Raw Outputs", open=False):
        details_out = gr.Markdown()
    with gr.Accordion("📚 KG Context & Evidence", open=False):
        ctx_out = gr.Textbox(label="Knowledge Graph Context", lines=8, interactive=False)
        pass_out = gr.Textbox(label="Retrieved Passages", lines=8, interactive=False)
    gr.Examples(EXAMPLES, [inp], label="📝 Example Cases")
    run_btn.click(fn=diagnose_all, inputs=[inp],
                  outputs=[summary, diag_out, details_out, ctx_out, pass_out, timer])

# Launch with share=True — creates public URL, no ngrok needed!
demo.launch(share=True, server_name="0.0.0.0", server_port=7860)
