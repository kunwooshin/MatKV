import json
from rouge_score import rouge_scorer
from bert_score import score

import re
import string
from collections import Counter

# List of generated JSONL filenames
generated_files = [
    "vanilla-top5-instruct.jsonl",
    "matkv-top5-instruct.jsonl"
]

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth, **kwargs):
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def qa_f1_score(prediction, ground_truth, **kwargs):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    return f1_score(prediction_tokens, ground_truth_tokens)

# 1) Load JSONL files
references = []
cnt = 0
with open('./results/answer_2wiki.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        # if cnt == 10:
        #     break
        references.append(json.loads(line.strip()))
        # cnt += 1

ref_dict = {item["id"]: item["answer"] for item in references}

# Iterate over each generated file and compute metrics
for gen_file in generated_files:
    generated = []
    with open(f"./results/{gen_file}", 'r', encoding='utf-8') as f:
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

    # 4) QA-style F1 Evaluation (token overlap)
    qa_f1s = [qa_f1_score(gen, ref) for gen, ref in zip(gen_texts, ref_texts)]
    avg_qa_f1 = sum(qa_f1s) / len(qa_f1s)

    # Output the evaluation for the current file
    print(f"=== Evaluation for {gen_file} ===")
    print("ROUGE Scores (Average F-measure):")
    print(f"ROUGE-1: {avg_rouge1_f:.4f}")
    print(f"ROUGE-2: {avg_rouge2_f:.4f}")
    print(f"ROUGE-L: {avg_rougeL_f:.4f}\n")

    print("BERTScore (Average):")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall   : {avg_recall:.4f}")
    print(f"F1       : {avg_f1:.4f}\n")
    
    print("QA-style F1:")
    print(f"F1       : {avg_qa_f1:.4f}")
    print("\n" + "-"*40 + "\n")
