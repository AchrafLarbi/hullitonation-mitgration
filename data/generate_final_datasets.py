import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "resources" / "final_ready"


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_json_array(path: Path):
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buf = ""
        eof = False

        while True:
            if not buf and eof:
                return
            if not buf:
                chunk = f.read(1 << 16)
                if chunk == "":
                    return
                buf += chunk

            i = 0
            n = len(buf)
            while i < n and buf[i].isspace():
                i += 1
            if i >= n:
                buf = ""
                continue

            if buf[i] == "[":
                i += 1
            buf = buf[i:]

            while True:
                j = 0
                n = len(buf)
                while j < n and buf[j].isspace():
                    j += 1
                if j >= n:
                    if eof:
                        return
                    break

                if buf[j] == ',':
                    buf = buf[j + 1 :]
                    continue
                if buf[j] == ']':
                    return

                try:
                    obj, end = decoder.raw_decode(buf[j:])
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break

                yield obj
                buf = buf[j + end :]

            chunk = f.read(1 << 16)
            if chunk == "":
                eof = True
            else:
                buf += chunk


def safe_text(v):
    return "" if v is None else str(v)


def write_jsonl(path: Path, rows):
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_rag_rows():
    # 1) rag_chunks
    rag_chunks = ROOT / "pubmed" / "pubmed_rag_data" / "rag_chunks.json"
    for x in iter_json_array(rag_chunks):
        yield {
            "id": safe_text(x.get("id")),
            "source_dataset": "pubmed_rag_chunks",
            "source_id": safe_text(x.get("pmid")),
            "title": safe_text(x.get("title")),
            "text": safe_text(x.get("text")),
            "metadata": x.get("metadata", {}),
        }

    # 2) RDS case reports (jsonl)
    rds = ROOT / "RDS.json"
    for x in iter_jsonl(rds):
        yield {
            "id": safe_text(x.get("_id")),
            "source_dataset": "rds_case_reports",
            "source_id": safe_text(x.get("Orpha_id")),
            "title": safe_text(x.get("diagnosis")),
            "text": safe_text(x.get("case_report")),
            "metadata": {
                "diagnosis": x.get("diagnosis"),
                "orpha_name": x.get("Orpha_name"),
                "orpha_id": x.get("Orpha_id"),
                "age": x.get("age"),
                "gender": x.get("gender"),
                "pub_date": x.get("pub_date"),
            },
        }

    # 3) PubMed abstracts
    pubmed_articles = ROOT / "pubmed" / "pubmed_rag_data" / "pubmed_articles.json"
    for x in iter_json_array(pubmed_articles):
        yield {
            "id": f"pmid:{safe_text(x.get('pmid'))}",
            "source_dataset": "pubmed_articles",
            "source_id": safe_text(x.get("pmid")),
            "title": safe_text(x.get("title")),
            "text": safe_text(x.get("abstract")),
            "metadata": {
                "authors": x.get("authors"),
                "journal": x.get("journal"),
                "date": x.get("date"),
                "mesh_terms": x.get("mesh_terms"),
                "hpo_terms": x.get("hpo_terms"),
                "omim_ids": x.get("omim_ids"),
            },
        }


def format_mcq_prompt(question, options_dict):
    parts = [f"Question: {safe_text(question)}", "Options:"]
    for k, v in options_dict.items():
        parts.append(f"{k}. {safe_text(v)}")
    return "\n".join(parts)


def build_pe_rows():
    # 1) CoT data
    cot = ROOT / "CoT_Rare20Conditions_9.8k.json"
    for x in iter_json_array(cot):
        yield {
            "id": safe_text(x.get("id")),
            "source_dataset": "cot_rare20conditions",
            "task": "cot_generation",
            "instruction": safe_text(x.get("question")),
            "input": "",
            "output": safe_text(x.get("answer")),
            "metadata": x.get("metadata", {}),
        }

    # 2) MedQA train
    medqa_train = ROOT / "resources" / "benchmarks" / "medqa" / "medqa_train.json"
    for i, x in enumerate(iter_json_array(medqa_train), start=1):
        options = x.get("options", {}) or {}
        instruction = format_mcq_prompt(x.get("question"), options)
        answer = x.get("answer") or x.get("answer_idx")
        yield {
            "id": f"medqa_train_{i}",
            "source_dataset": "medqa_train",
            "task": "medical_mcq",
            "instruction": instruction,
            "input": "",
            "output": safe_text(answer),
            "metadata": {
                "answer_idx": x.get("answer_idx"),
                "meta_info": x.get("meta_info"),
            },
        }

    # 3) MedMCQA train
    medmcqa_train = ROOT / "resources" / "benchmarks" / "medmcqa" / "medmcqa_train.json"
    key_by_index = ["opa", "opb", "opc", "opd"]
    for i, x in enumerate(iter_json_array(medmcqa_train), start=1):
        opts = {
            "A": x.get("opa"),
            "B": x.get("opb"),
            "C": x.get("opc"),
            "D": x.get("opd"),
        }
        cop = x.get("cop")
        answer = ""
        if isinstance(cop, int) and 0 <= cop <= 3:
            answer = safe_text(x.get(key_by_index[cop]))

        output = answer
        if x.get("exp"):
            output = f"Answer: {answer}\nExplanation: {safe_text(x.get('exp'))}"

        yield {
            "id": safe_text(x.get("id") or f"medmcqa_train_{i}"),
            "source_dataset": "medmcqa_train",
            "task": "medical_mcq_with_rationale",
            "instruction": format_mcq_prompt(x.get("question"), opts),
            "input": "",
            "output": output,
            "metadata": {
                "correct_option_index": cop,
                "subject_name": x.get("subject_name"),
                "topic_name": x.get("topic_name"),
                "choice_type": x.get("choice_type"),
            },
        }

    # 4) PubMedQA train
    pubmedqa_train = ROOT / "resources" / "benchmarks" / "pubmedqa" / "pubmedqa_train.json"
    for i, x in enumerate(iter_json_array(pubmedqa_train), start=1):
        ctx = x.get("context", {}) or {}
        contexts = ctx.get("contexts", []) or []
        joined_context = "\n".join([safe_text(c) for c in contexts])
        answer = x.get("final_decision") or x.get("answer") or ""

        yield {
            "id": f"pubmedqa_train_{i}",
            "source_dataset": "pubmedqa_train",
            "task": "biomedical_yes_no_maybe",
            "instruction": f"Question: {safe_text(x.get('question'))}",
            "input": f"Context:\n{joined_context}",
            "output": safe_text(answer),
            "metadata": {
                "pubid": x.get("pubid"),
                "labels": ctx.get("labels"),
                "meshes": ctx.get("meshes"),
            },
        }


def build_eval_rows():
    # MedQA test
    medqa_test = ROOT / "resources" / "benchmarks" / "medqa" / "medqa_test.json"
    for i, x in enumerate(iter_json_array(medqa_test), start=1):
        yield {
            "id": f"medqa_test_{i}",
            "source_dataset": "medqa_test",
            "split": "test",
            "task": "medical_mcq",
            "input": {
                "question": x.get("question"),
                "options": x.get("options"),
            },
            "target": x.get("answer") or x.get("answer_idx"),
        }

    # MedMCQA validation + test
    for split_name in ["medmcqa_validation", "medmcqa_test"]:
        path = ROOT / "resources" / "benchmarks" / "medmcqa" / f"{split_name}.json"
        for i, x in enumerate(iter_json_array(path), start=1):
            yield {
                "id": f"{split_name}_{i}",
                "source_dataset": split_name,
                "split": "validation" if "validation" in split_name else "test",
                "task": "medical_mcq",
                "input": {
                    "question": x.get("question"),
                    "options": {
                        "A": x.get("opa"),
                        "B": x.get("opb"),
                        "C": x.get("opc"),
                        "D": x.get("opd"),
                    },
                },
                "target": x.get("cop"),
            }

    # RareBench public splits
    rb_dir = ROOT / "resources" / "benchmarks" / "rarebench" / "RareBench_HF" / "data" / "data"
    for split in ["RAMEDIS", "MME", "HMS", "LIRICAL"]:
        path = rb_dir / f"{split}.jsonl"
        for i, x in enumerate(iter_jsonl(path), start=1):
            yield {
                "id": f"rarebench_{split}_{i}",
                "source_dataset": f"rarebench_{split.lower()}",
                "split": "test",
                "task": "rare_disease_phenotype_to_diagnosis",
                "input": {
                    "phenotype_ids": x.get("Phenotype", []),
                    "department": x.get("Department"),
                },
                "target": x.get("RareDisease", []),
            }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rag_out = OUT_DIR / "rag_corpus_final.jsonl"
    pe_out = OUT_DIR / "pe_sft_final.jsonl"
    eval_out = OUT_DIR / "eval_final.jsonl"

    rag_count = write_jsonl(rag_out, build_rag_rows())
    pe_count = write_jsonl(pe_out, build_pe_rows())
    eval_count = write_jsonl(eval_out, build_eval_rows())

    report = {
        "rag_corpus_final": rag_count,
        "pe_sft_final": pe_count,
        "eval_final": eval_count,
        "outputs": {
            "rag": str(rag_out.relative_to(ROOT)).replace("\\", "/"),
            "pe": str(pe_out.relative_to(ROOT)).replace("\\", "/"),
            "eval": str(eval_out.relative_to(ROOT)).replace("\\", "/"),
        },
    }

    report_path = OUT_DIR / "generation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
