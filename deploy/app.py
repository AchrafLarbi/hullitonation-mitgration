"""
Rare Disease Diagnostic Assistant — Gradio Web App
BioMistral-7B on ZeroGPU (free A10G on demand).
Embeddings + FAISS + KG on CPU. LLM inference on GPU via @spaces.GPU.
"""

import os, json, re, unicodedata, time, gc
import numpy as np
import networkx as nx
import pandas as pd
import gradio as gr
import torch
import spaces
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import faiss

# ── Config ───────────────────────────────────────────────────────────────────
DATASET_REPO    = "achraf2203/pfe-raredisease"
DATA_DIR        = os.environ.get("DATA_DIR", "data")
FAISS_DIR       = os.path.join("workspace", "faiss_index")
DATASET_FAISS   = os.path.join(DATA_DIR, "faiss_index", "rag_pubmedbert.index")
EMBEDDING_ID    = "NeuML/pubmedbert-base-embeddings"
LLM_MODEL_ID    = "BioMistral/BioMistral-7B"

TOP_K           = 5
MIN_SCORE       = 0.25
MAX_TOKENS      = 512
N_VOTES         = 3

for d in [DATA_DIR, FAISS_DIR]:
    os.makedirs(d, exist_ok=True)

CORPUS_PATH = os.path.join(DATA_DIR, "rag_corpus_final.jsonl")
KG_PATH     = os.path.join(DATA_DIR, "knowledge_graph.json")

_FMT = (
    "IMPORTANT: You MUST commit to exactly one rare disease diagnosis. "
    "Even if uncertain, name the single most likely rare disease. "
    "End with 'Final diagnosis: [Disease Name]'."
)

# ═══════════════════════════  TEXT UTILS  ═══════════════════════════════════
def normalize_text(t):
    t = str(t).lower()
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"'", "", t)
    t = re.sub(r"[-\u2013\u2014]", " ", t)
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()

def extract_diagnosis(text):
    raw = str(text).strip()
    if not raw: return ""
    def _c(s):
        s = str(s).strip().split("\n")[0].strip()
        s = re.sub(r'^[\[\("\']+|[\]\)"\'\.]+$', "", s).strip()
        s = re.split(r"\b(?:because|based on|given|considering)\b", s, 1, re.I)[0]
        s = re.split(r"[;.]", s, 1)[0]
        s = re.sub(r"\s+", " ", s).strip(" :-")
        return s if len(s) > 2 else None
    for p in [r"final\s+diagnosis\s*[:\-]\s*(.+)", r"diagnosis\s+is\s*[:\-]?\s*(.+)",
              r"diagnosis\s*[:\-]\s*(.+)", r"most\s+likely\s+(?:diagnosis|disease)\s+is\s*[:\-]?\s*(.+)"]:
        ms = list(re.finditer(p, raw, re.I))
        if ms:
            c = _c(ms[-1].group(1))
            if c: return c
    if len(raw) < 120:
        c = _c(raw.split("\n")[0].strip())
        if c: return c
    for ln in reversed([l.strip() for l in raw.split("\n") if l.strip()]):
        c = _c(ln)
        if c and 2 < len(c) < 220: return c
    return raw.strip()

# ═══════════════════════════  LOAD LLM (CPU, moved to GPU on call)  ════════
print("Loading BioMistral-7B tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading BioMistral-7B model (float16 → moved to GPU on demand)...")
llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_ID,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
)
llm_model.eval()
print("✅ BioMistral-7B loaded (CPU). Will use GPU via @spaces.GPU on inference.")

# ═══════════════════════════  GLOBAL STATE  ════════════════════════════════
emb_model = None
faiss_idx = None
documents = []
G = nx.Graph()
kg_names = []
kg_idx = None
loaded = False

def load_all():
    global emb_model, faiss_idx, documents, G, kg_names, kg_idx, loaded
    if loaded:
        yield "✅ Already loaded!"
        return

    # Dataset
    yield "⏳ Downloading dataset..."
    if not (os.path.exists(CORPUS_PATH) and os.path.exists(KG_PATH)):
        snapshot_download(repo_id=DATASET_REPO, repo_type="dataset", local_dir=DATA_DIR)

    # Corpus
    yield "⏳ Loading RAG corpus..."
    try:
        documents = pd.read_json(CORPUS_PATH, lines=True).to_dict("records")
        yield f"✅ Corpus: {len(documents)} docs"
    except Exception as e:
        yield f"⚠️ Corpus error: {e}"

    # Embeddings
    yield "⏳ Loading PubMedBERT embeddings (CPU)..."
    emb_model = SentenceTransformer(EMBEDDING_ID, device="cpu")

    idx_path = os.path.join(FAISS_DIR, "rag_pubmedbert.index")
    if os.path.exists(idx_path):
        yield "⏳ Loading FAISS index from workspace cache..."
        faiss_idx = faiss.read_index(idx_path)
    elif os.path.exists(DATASET_FAISS):
        yield "⏳ Loading pre-built FAISS index from dataset..."
        faiss_idx = faiss.read_index(DATASET_FAISS)
        faiss.write_index(faiss_idx, idx_path)
    elif documents:
        n = len(documents)
        yield f"⏳ Building FAISS index ({n} docs, ~15-20 min on CPU)..."
        texts = [str(d.get("text", "")) for d in documents]
        embs = emb_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                                 batch_size=64, show_progress_bar=True)
        faiss_idx = faiss.IndexFlatIP(embs.shape[1])
        faiss_idx.add(embs)
        faiss.write_index(faiss_idx, idx_path)
        yield f"✅ FAISS index built: {faiss_idx.ntotal} vectors"

    # KG
    yield "⏳ Loading Knowledge Graph..."
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
        yield f"✅ KG: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges"

    loaded = True
    yield "✅ All ready! BioMistral-7B on ZeroGPU. Enter a clinical case and click 'Run Diagnosis'."

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

# ═══════════════════════════  LLM (ZeroGPU)  ═══════════════════════════════
@spaces.GPU(duration=120)
def call_llm_gpu(prompt, max_tok=MAX_TOKENS, temp=0.7):
    """Runs BioMistral-7B on ZeroGPU A10G. GPU allocated only during this call."""
    llm_model.to("cuda")
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True, max_length=2048,
    ).to("cuda")

    with torch.no_grad():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=max_tok,
            temperature=temp if temp > 0 else 1.0,
            do_sample=temp > 0,
            top_p=0.9 if temp > 0 else 1.0,
            repetition_penalty=1.1,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    llm_model.to("cpu")
    torch.cuda.empty_cache()
    return text

def call_llm(prompt, max_tok=MAX_TOKENS, temp=0.7):
    try:
        return call_llm_gpu(prompt, max_tok, temp)
    except Exception as e:
        return f"[LLM Error: {e}]"

def vote(prompt, n=N_VOTES):
    abst = ["insufficient information", "cannot determine", "unable to determine", "i cannot", "no diagnosis"]
    cands, raws = [], []
    for _ in range(n):
        r = call_llm(prompt, MAX_TOKENS, 0.7)
        raws.append(r)
        cands.append(normalize_text(extract_diagnosis(r)))
    if not cands: return ""
    non_a = [c for c in cands if c and not any(k in c for k in abst)]
    winner = Counter(non_a).most_common(1)[0][0] if non_a else cands[0]
    for r, c in zip(raws, cands):
        if c == winner: return r
    return raws[0]

def expand_hpo(query):
    r = call_llm(f"List 5 specific medical/phenotypic terms from this description. One per line:\n\n{query[:600]}\n\nTerms:", 80, 0.1)
    return f"{query}\n\nKey phenotypes: {r}" if not r.startswith("[") else query

# ═══════════════════════════  4 ARCHITECTURES  ═════════════════════════════
def run_graphrag(q):
    seeds = [nid for nid,_,_ in link_kg(q, 8)]
    comm = community_summary(seeds)
    docs = retrieve(expand_hpo(q), 8)
    p = fmt_passages(docs)
    prompt = f"You are an expert rare disease diagnostician.\n\n=== GraphRAG Community ===\n{comm}\n\n=== Evidence ===\n{p}\n\nPatient: {q}\n\n{_FMT}\n\nFinal diagnosis:"
    return vote(prompt), comm, p

def run_kaping(q):
    seeds = [nid for nid,_,_ in link_kg(q, 8)]
    tr = get_triples(seeds)
    ts = "\n".join(f"  ({s},{r},{o})" for s,r,o in tr) if tr else "  (No triples)"
    docs = retrieve(expand_hpo(q), 8)
    p = fmt_passages(docs)
    prompt = f"You are an expert rare disease diagnostician.\n\n=== KG Triples (KAPING) ===\n{ts}\n\n=== Evidence ===\n{p}\n\nPatient: {q}\n\n{_FMT}\n\nFinal diagnosis:"
    return vote(prompt), ts, p

def run_rag_cot(q):
    docs = retrieve(expand_hpo(q), 8)
    p = fmt_passages(docs)
    prompt = (f"You are an expert rare disease diagnostician.\n\n=== RAG-driven CoT ===\nEvidence:\n{p}\n\n"
              f"Patient: {q}\n\n"
              f"Using the evidence above, reason through these steps internally:\n"
              f"1) Identify the key phenotypic features and clinical findings.\n"
              f"2) Determine which rare diseases are associated with these features.\n"
              f"3) Consider genetic/molecular mechanisms linked to candidate diseases.\n"
              f"4) Evaluate differentiating evidence between candidates.\n"
              f"5) Select the single most likely rare disease.\n\n"
              f"Do NOT output the reasoning steps or questions. Only output your final answer.\n\n"
              f"{_FMT}\n\nFinal diagnosis:")
    return vote(prompt), "", p

def run_cot_rag(q):
    pheno = call_llm(f"Extract key phenotypic features (HPO terms) from this clinical note. One per line:\n\n{q}\n\nPhenotypic features:", 200, 0.1)
    if pheno.startswith("["): pheno = q
    docs = retrieve(pheno, 8)
    p = fmt_passages(docs)
    prompt = f"You are an expert rare disease diagnostician.\n\n=== CoT-driven RAG ===\nPhenotypes:\n{pheno}\n\nEvidence:\n{p}\n\nPatient: {q}\n\n{_FMT}\n\nFinal diagnosis:"
    return vote(prompt), pheno, p

ARCHS = {
    "GraphRAG (Edge et al., 2024)": run_graphrag,
    "KG-Augmented RAG / KAPING (Baek et al., 2023)": run_kaping,
    "RAG-driven CoT (Wang et al., 2025)": run_rag_cot,
    "CoT-driven RAG (Wang et al., 2025)": run_cot_rag,
}

# ═══════════════════════════  MAIN  ════════════════════════════════════════
def diagnose(note, arch):
    if not loaded: return "❌ Click **Load Models** first.", "", "", ""
    if not note.strip(): return "⚠️ Enter a clinical description.", "", "", ""
    t0 = time.time()
    raw, ctx, ps = ARCHS[arch](note)
    diag = extract_diagnosis(raw)
    return (f"## 🏥 Diagnosis Result\n\n**Architecture:** {arch}\n\n"
            f"**Model:** BioMistral-7B (ZeroGPU A10G)\n\n"
            f"**Final Diagnosis:** `{diag}`\n\n**Time:** {time.time()-t0:.1f}s\n\n"
            f"---\n### Raw Output\n```\n{raw}\n```"), diag, ctx, ps

# ═══════════════════════════  UI  ══════════════════════════════════════════
EXAMPLES = [
    ["A 3-year-old boy with progressive motor weakness, calf pseudohypertrophy, CK 15000 U/L, Gowers' sign. Affected maternal uncle.", "GraphRAG (Edge et al., 2024)"],
    ["Progressive ataxia, neuropathy, cardiomyopathy, diabetes. Adolescent onset. Frataxin mutation.", "KG-Augmented RAG / KAPING (Baek et al., 2023)"],
    ["45F with recurrent abdominal pain, peripheral neuropathy, dark urine. Elevated porphobilinogen.", "RAG-driven CoT (Wang et al., 2025)"],
    ["Infant with hepatosplenomegaly, Gaucher cells in marrow, glucocerebrosidase deficiency.", "CoT-driven RAG (Wang et al., 2025)"],
]

with gr.Blocks(title="Rare Disease Diagnostic Assistant",
               theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue")) as demo:
    gr.Markdown("""# 🧬 Rare Disease Diagnostic Assistant
### KG-RAG Pipeline — Mitigating Hallucinations in Medical LLMs

**BioMistral-7B** (ZeroGPU A10G) + **PubMedBERT** + **130K docs** + **335-node KG**

| Architecture | Paper | Strategy |
|---|---|---|
| **GraphRAG** | Edge et al., 2024 | KG community summaries + retrieval |
| **KAPING** | Baek et al., 2023 | Structured KG triples + retrieval |
| **RAG-driven CoT** | Wang et al., 2025 | Retrieve → 5-question CoT |
| **CoT-driven RAG** | Wang et al., 2025 | Decompose → retrieve → synthesize |

⚠️ **Research demo — NOT for clinical use.**
""")
    with gr.Row():
        load_btn = gr.Button("🚀 Load Models (click first)", variant="primary", scale=1)
        status = gr.Textbox(label="Status", interactive=False, scale=3)
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="📋 Clinical Description", lines=8, placeholder="Enter symptoms, labs, imaging...")
            arch = gr.Dropdown(list(ARCHS.keys()), value=list(ARCHS.keys())[0], label="🔬 Architecture")
            run_btn = gr.Button("🔍 Run Diagnosis", variant="primary", size="lg")
        with gr.Column():
            result = gr.Markdown(label="Result")
            diag_out = gr.Textbox(label="📌 Extracted Diagnosis", interactive=False)
    with gr.Accordion("📚 Context & Evidence", open=False):
        ctx_out = gr.Textbox(label="KG Context / Phenotypes", lines=6, interactive=False)
        pass_out = gr.Textbox(label="Retrieved Passages", lines=8, interactive=False)
    gr.Examples(EXAMPLES, [inp, arch], label="📝 Example Cases")
    load_btn.click(fn=load_all, outputs=status)
    run_btn.click(fn=diagnose, inputs=[inp, arch], outputs=[result, diag_out, ctx_out, pass_out])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
