#!/bin/python

from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch

# top_0 = 0.9
# top_k = 4
model_name = "meta-llama/Llama-3.1-8B"
model_name = "Qwen/Qwen2.5-7B-Instruct"
max_new_tokens = 1024


def load_json(file_path):
    import json
    with open(file_path, 'r') as f:
        return json.load(f)
    
def load_jsonl(file_path):
    import json
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

def hf_load_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto", device_map="auto")
    return model

def hf_single_generate_test(model, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    messages = [ { "role": "user", "content": prompt } ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    outputs = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    print(response)
    return

def hf_batch_generate(model, prompts, batch_size=1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    for i in range(0, len(prompts), batch_size):
        formatted_prompts = []
        for prompt in prompts[i:i+batch_size]:
            formatted_prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        inputs = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            padding_side="left"
            )
        inputs = inputs.to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=True,
                )
            
        sequences = generated_ids.sequences
        scores = generated_ids.scores
        outputs = [
            seq[len(inputs.input_ids):] for j, seq in enumerate(sequences)
        ]
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        print(f"{i}-{i+batch_size-1}/{len(prompts)}: {len(response)}")

def vllm_single_generate_test():
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=max_new_tokens, repetition_penalty=1.05)
    llm = LLM(model=model_name)

    prompt = "what is LLM?"
    messages = [
{"role": "system", "content": "You are friendly neighborhood LLM assistant."},
{"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
        )
    outputs = llm.generate([text], sampling_params)
    generated_texts = []
    for output in outputs:
        generated_texts.append(output.outputs[0].text)
    print(generated_texts)

def vllm_batch_generate(llm, prompts, batch_size=1, output_size=500, print_header=False):
    from vllm import LLM, SamplingParams
    import time

    total_prefill_time = 0
    total_decode_time = 0
    total_time = 0
    total_prompt_tokens = 0
    total_generated_tokens = 0
    total_tokens = 0

    sampling_params = SamplingParams(temperature=0.1, top_p=0.9, max_tokens=output_size, repetition_penalty=1.05)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    st = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        formatted_prompts = []
        for prompt in prompts[i:i+batch_size]:
            formatted_prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)

        with torch.no_grad():
            outputs = llm.generate(formatted_prompts, sampling_params)

        generated_texts = []
        prefill_time = 0
        decode_time = 0
        generated_token_count = 0
        for output in outputs:
            generated_texts.append(output.outputs[0].text)
            first_scheduled_time = output.metrics.first_scheduled_time
            first_token_time = output.metrics.first_token_time
            last_token_time = output.metrics.last_token_time

# 'arrival_time', 'finished_time', 'first_scheduled_time', 'first_token_time',
# 'last_token_time', 'model_execute_time', 'model_forward_time', 'scheduler_time',
# 'time_in_queue']

            prefill_time += first_token_time - first_scheduled_time
            decode_time += output.metrics.finished_time - first_token_time
            generated_token_count += len(output.outputs[0].token_ids)

        prompt_token_counts = sum([len(tokenizer.encode(prompt)) for prompt in formatted_prompts])

        total_prefill_time += prefill_time
        total_decode_time += decode_time
        total_time += prefill_time + decode_time
        total_prompt_tokens += prompt_token_counts
        total_generated_tokens += generated_token_count
        total_tokens += prompt_token_counts + generated_token_count

        prompt_tokens_per_sec = prompt_token_counts / prefill_time
        generated_tokens_per_sec = generated_token_count / decode_time
        # print(generated_texts)
        print(f"@@@ {i}-{i+batch_size-1}/{len(prompts)}: {len(generated_texts)} prefill={prefill_time:.2f}/{prompt_tokens_per_sec:.1f}t decode={decode_time:.2f}/{generated_tokens_per_sec:.1f}t tokens={prompt_token_counts}/{generated_token_count}")

    et = time.perf_counter()
    dt = et - st
    prompt_tokens_per_sec = total_prompt_tokens / total_prefill_time
    generated_tokens_per_sec = total_generated_tokens / total_decode_time
    time_per = total_time / len(prompts)
    prefill_time_per = total_prefill_time / len(prompts)
    decode_time_per = total_decode_time / len(prompts)

    print(f"@@@ batch={batch_size} ellapsed={dt:.2f}sec prompts={len(prompts)} ellapsed_per_prompts={dt/len(prompts):.2f}sec")
    print(f"@@@ total:   time={total_time:.2f} /prompt={time_per:.2f} tokens={total_tokens} tokens/sec={prompt_tokens_per_sec+generated_tokens_per_sec:.2f}")
    print(f"@@@ prefill: time={total_prefill_time:.2f} /prompt={prefill_time_per:.2f} tokens={total_prompt_tokens} tokens/sec={prompt_tokens_per_sec:.2f} combined_tokens/sec={prompt_tokens_per_sec*batch_size:.2f}")
    print(f"@@@ decode:  time={total_decode_time:.2f} /prompt={decode_time_per:.2f} tokens={total_generated_tokens} tokens/sec={generated_tokens_per_sec:.2f} combined_tokens/sec={generated_tokens_per_sec*batch_size:.2f}")
    # want to print out cvs
    # %%%, batch, prompts, ellapsed, ellapsed_per_prompts, total_time, time_per, tokens, tokens_per_sec, prefill_time, prefill_time_per, prompt_tokens, prompt_tokens_per_request, prompt_tokens_per_sec, combined_tokens_per_sec, decode_time, decode_time_per, generated_tokens, generated_tokens_per_request, generated_tokens_per_sec, combined_tokens_per_sec
    if print_header:
        print(f"%%%, batch, prompts, ellapsed, ellapsed_per_prompt, total_time, time_per, tokens, tokens_per_sec, prefill_time, prefill_time_per, prompt_tokens, prompt_tokens_per_request, prompt_tokens_per_sec, combined_prompt_tokens_per_sec, decode_time, decode_time_per, generated_tokens, generated_tokens_per_request, generated_tokens_per_sec, combined_generated_tokens_per_sec")
    print(f"%%%, {batch_size}, {len(prompts)}, {dt:.2f}, {dt/len(prompts):.2f}, {total_time:.2f}, {time_per:.2f}, {total_tokens}, {prompt_tokens_per_sec+generated_tokens_per_sec:.2f}, {total_prefill_time:.2f}, {prefill_time_per:.2f}, {total_prompt_tokens}, {int(total_prompt_tokens/len(prompts))}, {prompt_tokens_per_sec:.2f}, {prompt_tokens_per_sec*batch_size:.2f}, {total_decode_time:.2f}, {decode_time_per:.2f}, {total_generated_tokens}, {int(total_generated_tokens/len(prompts))}, {generated_tokens_per_sec:.2f}, {generated_tokens_per_sec*batch_size:.2f}")

def get_api_key(file_name, env_name):
    import os
    try:
        with open(file_name, 'r') as f:
            api_key = f.read().strip()
    except Exception:
        api_key = os.environ.get(env_name, "")
    if not api_key:
        raise Exception(f"{env_name} is missing")
    return api_key

def openai_generate(prompt, model="o3-mini"):
    import openai

    cli = openai.OpenAI(
        api_key=get_api_key(".openai-api-key", "OPENAI_API_KEY")
    )
    messages = [
        # {"role": "system", "content": "You're a smooth operator."},
        {"role": "user", "content": prompt},
    ]
    response = cli.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response.choices[0].message.content

def generate_prompts(count=200, model="o1-mini"):
    import json

    print("getting movie titles...")
    titles = openai_generate(f"Give me a list of {count} movies made after 2000 in the order of popularity. In the following parsable json format: [ {{ 'rank': 1, 'title': 'The Dark Knight', 'year': 2008 }}, ...].")
    print(titles)
    titles = json.loads(titles)
    print(f"got movie titles: {len(titles)}...")
    summaries = []
    for movie in titles:
        print(f"  getting summary of {movie['title']}...")
        while True:
            summary = openai_generate(f"Summarize the movie {movie['title']} made in {movie['year']}, in 1000 tokens, 2000 tokens, 4000 tokens and 8000 tokens. In the following parsable json format. {{'movie': 'The Cabin in the Woods', 'year': 2012, 'summary_1000': '...', 'summary_2000': '...'}}")
            print(summary)
            try: 
                summary = json.loads(summary)
                summaries.append(summary)
                break
            except:
                print("retrying...")
                continue
    out = {
        "titles": titles,
        "summaries": summaries
    }
    print("saving...")
    with open("movie-prompts.json", "w") as f:
        json.dump(out, f)
    return

def gen_prompts_from_files(dir_path, output_file="file-prompts.jsonl", count_per_item=10):
    import os
    import json
    import random
    import sys
    
    # Get all text files in the directory
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) 
             if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.txt')]
    
    # Shuffle files to get random combinations
    random.shuffle(files)
    
    prompts = []
    for i in range(0, len(files), count_per_item):
        batch_files = files[i:i+count_per_item]
        if len(batch_files) < count_per_item:  # Skip incomplete batches
            continue
            
        # Combine the content of count_per_item files
        combined_content = ""
        for file_path in batch_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_content = f.read()
                    file_name = os.path.basename(file_path)
                    combined_content += f"{file_content}\n\n"
            except Exception as e:
                print(f"Error reading {file_path}: {e}", file=sys.stderr)
                
        # Create a prompt item
        prompt = [{"role": "user", "content": combined_content}]
        prompts.append(prompt)
    
    # Save to JSONL
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt in prompts:
            f.write(json.dumps(prompt) + '\n')
    return prompts

def load_movie_prompts(file_name, size=1000):
    import json
    with open(file_name, 'r') as f:
        movies = json.load(f)
    prompts = []
    for movie in movies["summaries"]:
        prompt = [
            {"role": "user", "content": movie["summary_8000"][:size] + "\n\nSummarize the above."}
        ]
        prompts.append(prompt)
    return prompts

def load_jsonl_prompts(file_name, size=None):
    prompts = load_jsonl(file_name)
    if size is None:
        return prompts
    else:
        for prompt in prompts:
            for item in prompt:
                if item["role"] == "user":
                    item["content"] = item["content"][:size] + "\n\nSummarize the above."
        return prompts

def main():
    import sys

    def usage():
        print("Usage: python batch.py [hf-batch | vllm-batch batch-size prompt-size output-size | gen-prompts-from-files <jsonl-name> <dir> <count> | load-movies]")

    cmd = sys.argv[1] if len(sys.argv) > 1 else None
    if cmd == "hf-batch":
        batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 1
        prompts = load_jsonl("/home/sjung/src/sqlizer/sqlizer-qna.jsonl")
        model = hf_load_model(model_name)
        hf_batch_generate(model, prompts, batch_size=batch_size)
    elif cmd == "vllm-batch":
        if len(sys.argv) < 6:
            usage()
            sys.exit(1)
        prompts_file = sys.argv[2]        
        batch_size = int(sys.argv[3])
        prompt_size = int(sys.argv[4])
        output_size = int(sys.argv[5])
        print_header = True if len(sys.argv) > 6 and sys.argv[6] in [ '1', 't', 'true', 'T', 'True' ] else False
        prompts = load_jsonl_prompts(prompts_file, size=prompt_size)
        llm = LLM(model=model_name)
        vllm_batch_generate(llm, prompts, batch_size=batch_size, output_size=output_size, print_header=print_header)
    elif cmd == "gen-prompts":
        generate_prompts(200, "o1-mini")
        # print(openai_generate("Give me a list of 200 movies made after 2000 in the order of popularity, in json format."))
        # print(openai_generate("Summarize the movie The Cabin in the Woods made in 2012, in 1000 tokens, 2000 tokens, 4000 tokens and 8000 tokens. In the following json format. {'movie': 'The Cabin in the Woods', 'year': 2012, 'summary_1000': '...', 'summary_2000': '...'}"))
    elif cmd == "gen-prompts-from-files":
        if len(sys.argv) < 5:
            usage()
            sys.exit(1)
        jsonl_name = sys.argv[2]
        dir_name = sys.argv[3]
        file_count = int(sys.argv[4])
        gen_prompts_from_files(dir_name, jsonl_name, file_count)
    elif cmd == "load-movies":
        prompts = load_movie_prompts("movie-prompts.json", size=8000)
        print(prompts[0])
    else:
        usage()
    return

if __name__ == "__main__":
    main()

# EOF
