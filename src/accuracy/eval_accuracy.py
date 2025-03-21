import json
from rouge_score import rouge_scorer
from bert_score import score

# 1) Load JSONL files
references = []
with open('answer.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        references.append(json.loads(line.strip()))

ref_dict = {item["id"]: item["answer"] for item in references}

# List of generated JSONL filenames
generated_files = [
    "vanilla.jsonl",
    "matkv.jsonl",   # add more files as needed
    "matkv-reverse.jsonl"
]

# Iterate over each generated file and compute metrics
for gen_file in generated_files:
    generated = []
    with open(gen_file, 'r', encoding='utf-8') as f:
        for line in f:
            generated.append(json.loads(line.strip()))

    # Create a mapping for generated responses: id -> answer
    gen_dict = {item["id"]: item["answer"] for item in generated}

    # Get the list of common IDs
    common_ids = sorted(list(set(ref_dict.keys()) & set(gen_dict.keys())))
    ref_texts = [ref_dict[i] for i in common_ids]
    gen_texts = [gen_dict[i] for i in common_ids]

    # 2) ROUGE Evaluation
    rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = [rouge.score(ref, gen) for ref, gen in zip(ref_texts, gen_texts)]
    
    avg_rouge1_f = sum(s['rouge1'].fmeasure for s in rouge_scores) / len(rouge_scores)
    avg_rouge2_f = sum(s['rouge2'].fmeasure for s in rouge_scores) / len(rouge_scores)
    avg_rougeL_f = sum(s['rougeL'].fmeasure for s in rouge_scores) / len(rouge_scores)

    # 3) BERTScore Evaluation
    # You can change model_type if needed (e.g., "roberta-base")
    P, R, F1 = score(gen_texts, ref_texts, model_type="bert-base-uncased")
    avg_precision = P.mean().item()
    avg_recall = R.mean().item()
    avg_f1 = F1.mean().item()

    # Output the evaluation for the current file
    print(f"=== Evaluation for {gen_file} ===")
    print("ROUGE Scores (Average F-measure):")
    print(f"ROUGE-1: {avg_rouge1_f:.4f}")
    print(f"ROUGE-2: {avg_rouge2_f:.4f}")
    print(f"ROUGE-L: {avg_rougeL_f:.4f}\n")

    print("BERTScore (Average):")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall   : {avg_recall:.4f}")
    print(f"F1       : {avg_f1:.4f}")
    print("\n" + "-"*40 + "\n")
