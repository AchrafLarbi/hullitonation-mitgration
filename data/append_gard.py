"""
Append GARD rare disease entries to the final_ready datasets.
- Adds 6,137 disease reference documents to rag_corpus_final.jsonl
- Updates generation_report.json with new counts
- Updates rag_data_manifest.json to include GARD as a source
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "resources" / "final_ready"
GARD_CSV = ROOT / "resources" / "corpora" / "gard" / "GARD_Rare_Disease_List_Feb2026_v3.csv"


def build_gard_rag_rows():
    """Convert each GARD disease into a RAG document."""
    with GARD_CSV.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gard_id = row.get("ID", "").strip()
            name = row.get("DisplayName", "").strip()
            synonyms_raw = row.get("Synonyms", "").strip()
            url = row.get("URL", "").strip()

            if not gard_id or not name:
                continue

            # Parse synonyms (pipe-separated)
            synonyms = [s.strip() for s in synonyms_raw.split("|") if s.strip()] if synonyms_raw else []

            # Build a rich text description for RAG retrieval
            text_parts = [f"Rare Disease: {name}"]
            if synonyms:
                text_parts.append(f"Also known as: {', '.join(synonyms)}")
            text_parts.append(f"GARD ID: {gard_id}")
            text_parts.append(f"Source: NIH Genetic and Rare Diseases Information Center (GARD)")
            text_parts.append(f"Classification: Rare disease (affecting fewer than 200,000 individuals in the US)")
            if url:
                text_parts.append(f"Reference: {url}")

            text = ". ".join(text_parts) + "."

            yield {
                "id": f"gard_{gard_id.replace(':', '_').lower()}",
                "source_dataset": "gard_rare_disease_list",
                "source_id": gard_id,
                "title": name,
                "text": text,
                "metadata": {
                    "gard_id": gard_id,
                    "disease_name": name,
                    "synonyms": synonyms,
                    "url": url,
                    "source": "NIH/NCATS GARD",
                    "version": "Feb2026_v3",
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

    # 2) Append GARD rows
    gard_count = 0
    with rag_path.open("a", encoding="utf-8") as f:
        for row in build_gard_rag_rows():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            gard_count += 1
    print(f"Appended {gard_count} GARD disease entries")
    print(f"New RAG corpus total: {existing_count + gard_count} records")

    # 3) Update generation_report.json
    report_path = OUT_DIR / "generation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["rag_corpus_final"] = existing_count + gard_count
    report["gard_appended"] = gard_count
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Updated generation_report.json")

    # 4) Update rag_data_manifest.json
    manifest_path = OUT_DIR / "rag_data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Add GARD to rag_train_corpus if not already present
    gard_entry = {
        "name": "GARD Rare Disease Reference List",
        "path": "resources/corpora/gard/GARD_Rare_Disease_List_Feb2026_v3.csv",
        "format": "csv",
        "estimated_records": gard_count,
        "field_mapping": {
            "doc_id": "ID",
            "title": "DisplayName",
            "synonyms": "Synonyms (pipe-separated)",
            "url": "URL"
        },
        "use": "disease_dictionary_and_synonym_expansion"
    }

    # Check if GARD already in corpus list
    existing_names = [e.get("name", "") for e in manifest.get("rag_train_corpus", [])]
    if "GARD Rare Disease Reference List" not in existing_names:
        manifest["rag_train_corpus"].append(gard_entry)

    # Add GARD to grounding_knowledge
    gard_grounding = {
        "name": "GARD disease dictionary",
        "path": "resources/corpora/gard/GARD_Rare_Disease_List_Feb2026_v3.csv",
        "format": "csv",
        "use": "rare_disease_name_normalization_and_synonym_expansion"
    }
    existing_grounding_names = [e.get("name", "") for e in manifest.get("grounding_knowledge", [])]
    if "GARD disease dictionary" not in existing_grounding_names:
        manifest["grounding_knowledge"].append(gard_grounding)

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Updated rag_data_manifest.json")

    print("\n✅ GARD integration complete!")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
