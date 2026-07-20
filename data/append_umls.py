"""
Append UMLS rare disease concepts to the final_ready datasets.
- Adds extracted UMLS concepts to rag_corpus_final.jsonl
- Updates generation_report.json with new counts
- Updates rag_data_manifest.json to include UMLS as a source
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "resources" / "final_ready"
UMLS_JSON = ROOT / "resources" / "ontologies" / "umls" / "extracted" / "umls_rare_disease_concepts.json"


def build_umls_rag_rows():
    """Convert each UMLS concept into a RAG document."""
    if not UMLS_JSON.exists():
        print(f"Error: {UMLS_JSON} not found!")
        return
        
    data = json.loads(UMLS_JSON.read_text(encoding="utf-8"))
    concepts = data.get("concepts", [])
    
    for concept in concepts:
        cui = concept.get("cui", "")
        name = concept.get("name", "")
        if not cui or not name:
            continue
            
        semantic_types = concept.get("semantic_types", [])
        sources = concept.get("sources", [])
        definitions = concept.get("definitions", [])
        
        # Build text description
        text_parts = [f"UMLS Medical Concept: {name}"]
        text_parts.append(f"Concept Unique Identifier (CUI): {cui}")
        
        if semantic_types:
            text_parts.append(f"Semantic Types: {', '.join(semantic_types)}")
        
        if definitions:
            text_parts.append(f"Definition: {' '.join(definitions)}")
            
        text = ". ".join(text_parts) + "."

        yield {
            "id": f"umls_{cui.lower()}",
            "source_dataset": "umls_rare_disease_concepts",
            "source_id": cui,
            "title": name,
            "text": text,
            "metadata": {
                "cui": cui,
                "semantic_types": semantic_types,
                "sources": sources,
                "atom_count": concept.get("atom_count", 0)
            },
        }


def main():
    # 1) Read existing RAG corpus line count
    rag_path = OUT_DIR / "rag_corpus_final.jsonl"
    existing_count = 0
    with rag_path.open("r", encoding="utf-8") as f:
        for _ in f:
            existing_count += 1
    print(f"Existing RAG corpus: {existing_count} records")

    # 2) Append UMLS rows
    umls_count = 0
    with rag_path.open("a", encoding="utf-8") as f:
        for row in build_umls_rag_rows():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            umls_count += 1
    print(f"Appended {umls_count} UMLS concepts")
    
    if umls_count == 0:
        print("No UMLS concepts appended. Make sure the extraction was successful.")
        return
        
    print(f"New RAG corpus total: {existing_count + umls_count} records")

    # 3) Update generation_report.json
    report_path = OUT_DIR / "generation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rag_corpus_final"] = existing_count + umls_count
    report["umls_appended"] = umls_count
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Updated generation_report.json")

    # 4) Update rag_data_manifest.json
    manifest_path = OUT_DIR / "rag_data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Add UMLS to rag_train_corpus if not already present
    umls_entry = {
        "name": "UMLS Rare Disease Concepts",
        "path": "resources/ontologies/umls/extracted/umls_rare_disease_concepts.json",
        "format": "json",
        "estimated_records": umls_count,
        "field_mapping": {
            "doc_id": "cui",
            "title": "name",
            "text": "definitions",
            "semantic_types": "semantic_types"
        },
        "use": "unified_medical_concept_knowledge"
    }

    # Check if UMLS already in corpus list
    existing_names = [e.get("name", "") for e in manifest.get("rag_train_corpus", [])]
    if "UMLS Rare Disease Concepts" not in existing_names:
        manifest["rag_train_corpus"].append(umls_entry)

    # Add UMLS to grounding_knowledge
    umls_grounding = {
        "name": "UMLS Metathesaurus Concepts",
        "path": "resources/ontologies/umls/extracted/umls_rare_disease_concepts.json",
        "format": "json",
        "use": "CUI_normalization_and_semantic_typing"
    }
    existing_grounding_names = [e.get("name", "") for e in manifest.get("grounding_knowledge", [])]
    if "UMLS Metathesaurus Concepts" not in existing_grounding_names:
        manifest["grounding_knowledge"].append(umls_grounding)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Updated rag_data_manifest.json")

    print("\n✅ UMLS integration complete!")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
