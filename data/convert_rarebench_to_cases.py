"""
Convert eval_rare_disease.jsonl (phenotype IDs → disease codes)
into eval_rare_cases.jsonl format (textual clinical description → disease name).

Steps:
  1. Parse hp.obo to build HPO ID → term name mapping
  2. Parse phenotype.hpoa to build OMIM/ORPHA → disease name mapping
  3. For each RareBench entry, resolve phenotypes to readable names,
     build a clinical-style description, and resolve the target to a disease name.
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
ONTOLOGY_DIR = DATA_DIR / "ontologies" / "hpo"


def load_hpo_names(obo_path: Path) -> dict[str, str]:
    """Parse hp.obo and return {HP:XXXXXXX: 'Term Name'}."""
    hpo_map = {}
    current_id = None
    with obo_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "[Term]":
                current_id = None
            elif line.startswith("id: HP:"):
                current_id = line[4:]  # e.g. "HP:0001522"
            elif line.startswith("name: ") and current_id:
                hpo_map[current_id] = line[6:]
    print(f"  Loaded {len(hpo_map)} HPO terms from {obo_path.name}")
    return hpo_map


def load_disease_names(hpoa_path: Path) -> dict[str, str]:
    """Parse phenotype.hpoa and return {OMIM:XXXXXX: 'Disease Name', ORPHA:XXX: ...}."""
    disease_map = {}
    with hpoa_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            db_id = parts[0]       # e.g. "OMIM:619340"
            name = parts[1]        # e.g. "Developmental and epileptic encephalopathy 96"
            if db_id not in disease_map:
                disease_map[db_id] = name
    print(f"  Loaded {len(disease_map)} disease names from {hpoa_path.name}")
    return disease_map


def resolve_target(target_codes: list, disease_map: dict) -> str:
    """Resolve OMIM/ORPHA/CCRD codes to a disease name. Returns the first match."""
    for code in target_codes:
        # Try direct lookup
        if code in disease_map:
            return disease_map[code]
        # ORPHA:27 → try ORPHA:27
        if code.startswith("ORPHA:"):
            orpha_num = code.split(":")[1]
            for key, name in disease_map.items():
                if key.startswith("ORPHA:") and key.split(":")[1] == orpha_num:
                    return name
    # Fallback: return the first code as-is
    return target_codes[0] if target_codes else "Unknown"


def build_clinical_description(phenotype_ids: list, hpo_map: dict, department: str | None) -> str:
    """Convert a list of HPO IDs into a clinical-style textual description."""
    phenotype_names = []
    unresolved = []
    for hpo_id in phenotype_ids:
        name = hpo_map.get(hpo_id)
        if name:
            phenotype_names.append(name)
        else:
            unresolved.append(hpo_id)

    # Build the description
    parts = []
    parts.append("A patient presents with the following clinical phenotypes:")
    
    if phenotype_names:
        for i, name in enumerate(phenotype_names, 1):
            parts.append(f"  {i}. {name}")
    
    if unresolved:
        parts.append(f"\nAdditional phenotype codes: {', '.join(unresolved)}")

    if department:
        parts.append(f"\nDepartment: {department}")

    return "\n".join(parts)


def main():
    print("Loading ontology data...")
    hpo_map = load_hpo_names(ONTOLOGY_DIR / "hp.obo")
    disease_map = load_disease_names(ONTOLOGY_DIR / "phenotype.hpoa")

    input_file = DATA_DIR / "eval_rare_disease.jsonl"
    output_file = DATA_DIR / "eval_rare_cases_from_rarebench.jsonl"

    print(f"\nConverting {input_file.name} -> {output_file.name}...")

    converted = 0
    unresolved_targets = 0

    with input_file.open("r", encoding="utf-8") as fin, \
         output_file.open("w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            phenotype_ids = entry["input"].get("phenotype_ids", [])
            department = entry["input"].get("department")
            target_codes = entry.get("target", [])

            # Build clinical description from phenotype IDs
            case_description = build_clinical_description(phenotype_ids, hpo_map, department)

            # Resolve target disease code to a name
            disease_name = resolve_target(target_codes, disease_map)
            if disease_name == target_codes[0] if target_codes else True:
                unresolved_targets += 1

            # Build output in eval_rare_cases format
            new_entry = {
                "id": entry["id"],
                "source_dataset": entry["source_dataset"],
                "split": "test",
                "task": "rare_disease_diagnosis",
                "input": {
                    "question": "Analyze this clinical case report and diagnose the rare disease.",
                    "case_report": case_description
                },
                "target": disease_name
            }

            fout.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
            converted += 1

    print(f"\nDone! Converted {converted} entries.")
    print(f"  Unresolved target names: {unresolved_targets}")

    # Show a sample
    with output_file.open("r", encoding="utf-8") as f:
        sample = json.loads(f.readline())
    print("\nSample output:")
    print(json.dumps(sample, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
