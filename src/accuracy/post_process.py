import json
from collections import OrderedDict
import os

input_files = [
    "answer-vanilla.jsonl",
    "answer-matkv.jsonl",
    "answer-matkv-reverse.jsonl",
]

for input_file in input_files:
    base_name = input_file.replace("answer-", "")
    output_file = os.path.splitext(base_name)[0] + ".jsonl"

    with open('./raw/'+input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for i, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue

            data = json.loads(line)

            answer = data.get("answer", "")
            idx = answer.find("\n\n")
            if idx != -1:
                answer = answer[:idx]
            data["answer"] = answer.strip()
          
            new_data = OrderedDict()
            new_data["id"] = i
            for k, v in data.items():
                if k != "id":
                    new_data[k] = v

            fout.write(json.dumps(new_data, ensure_ascii=False) + "\n")
