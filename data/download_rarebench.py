import os
import json
import urllib.request
import zipfile

def main():
    url = "https://huggingface.co/datasets/chenxz/RareBench/resolve/main/data.zip"
    zip_path = "data/data.zip"
    extract_dir = "data/rarebench"
    
    print(f"Downloading RareBench data from {url}...")
    if not os.path.exists("data"):
        os.makedirs("data")
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
    
    print(f"Extracting to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    datasets_to_process = {
        "rarebench_ramedis": os.path.join(extract_dir, "data", "RAMEDIS.jsonl"),
        "rarebench_mme": os.path.join(extract_dir, "data", "MME.jsonl")
    }
    
    output_file = "data/eval_rare_disease.jsonl"
    print(f"Generating evaluation cases at {output_file}...")
    
    cases_generated = 0
    sample_case = None
    
    with open(output_file, 'w', encoding='utf-8') as out_f:
        for ds_name, file_path in datasets_to_process.items():
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found.")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as in_f:
                for idx, line in enumerate(in_f):
                    data = json.loads(line)
                    ds_str = ds_name.replace('rarebench_', '').upper()
                    case_id = f"rarebench_{ds_str}_{idx+1}"
                    
                    eval_case = {
                        "id": case_id,
                        "source_dataset": ds_name,
                        "split": "test",
                        "task": "rare_disease_phenotype_to_diagnosis",
                        "input": {
                            "phenotype_ids": data.get("Phenotype", []),
                            "department": data.get("Department", None)
                        },
                        "target": data.get("RareDisease", [])
                    }
                    
                    out_f.write(json.dumps(eval_case) + '\n')
                    cases_generated += 1
                    
                    if sample_case is None:
                        sample_case = eval_case

    print(f"\nDone! Generated {cases_generated} evaluation cases.")
    if sample_case:
        print("\nSample Case:")
        print(json.dumps(sample_case, indent=2))

if __name__ == '__main__':
    main()
