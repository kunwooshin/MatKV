# MatKV
## Installation
### Package Version Compatibility
   ```bash
    $ conda create -n matkv python=3.12.7
    $ conda activate matkv
    $ pip3 install -r requirements.txt
   ```

### HuggingFace cli for LLaMA 3 model
https://huggingface.co/meta-llama/Llama-3.1-70B
   ```bash
   $ huggingface-cli login # Insert your HuggingFace access token
   $ huggingface-cli download meta-llama/Llama-3.1-70B
   $ huggingface-cli download meta-llama/Llama-3.1-8B
   $ huggingface-cli download meta-llama/Llama-3.2-3B
   ```

### DeepNVMe for optimized tensor I/O
https://github.com/deepspeedai/DeepSpeedExamples/tree/master/deepnvme/file_access
NVMe Storage is required for optimized tensor I/O using DeepNVMe's `async_io` operator.
   ```base
   $ apt install libaio-dev
   ```
If NVMe Storage is not available:
(1) Use `save_kv_cache()` instead of `save_kv_cache_aio()` inside `preprocess_documents()` in `preprocessing.py`.

   ```python
   def process_documents(self):
       ...
         self.save_kv_cache(chunks)
       ...
   ```

(2) Use `load_kv_cache()` instead of `load_kv_cache_aio()` inside `load_all_caches()` in `eval_batch.py` (and also in `eval_pp.py`).

   ```python
   def load_all_caches(self, docs: List[Document]):
        return [self.load_kv_cache(doc.id) for doc in docs]
   ```

## Run experiment
### Preprocessing
Chunk documents and update vector DB. Then, store key-value tensors for each chunk on SSD.
   ```bash
   $ ./src/preprocessing.sh
   ```
Make sure to set a separate directory to store the vector DB and cache for each configuration (including different LLMs and datasets).

### Inference Latency Experiments
(1) Script file for batch processing. Set `use_past_cache = True` for MatKV, `False` for Vanilla.
   ```bash
   $ ./src/eval_batch.sh
   ```

(2) Overlapping using `multiprocessing` (only for MatKV):
   ```bash
   $ ./src/eval_pp.sh
   ```

### Power Consumption Experiments
Use `./power_monitor-smi.sh` instead of `./power_monitor.sh` in both `eval_batch_power_consumption.sh` and `./eval_pp_power_consumption.sh`.

   ```bash 
   $ ./src/eval_batch_power_consumption.sh
   $ ./src/eval_pp_power_consumption.sh
   ```
Ensure that the appropriate arguments are set within each script file.

### Generation Accuracy Experiments
Preprocess new documents from `./data/LongBench-HotpotQA/documents` and replace the path of the question source file in `eval_batch.sh`.

Answers generated from (1) Vanilla (2) MatKV (3) MatKV with reversed document orders are saved in the `./scr/accuracy` directory along with ground truth answers provided by the benchmark.

Note: Answers generated by LLM (`./src/accuracy/raw/`) are post-processed to extract the short answer without explanations for better comparison with the ground truth (which also contains short answers only.). See `./src/accuracy/post_process.py`.

Use `eval_accuracy.py` to evaluate the accuracy of generated answers using two metrics: ROUGE-L and BERTScore.
Note: The experiment in the paper was conducted using the `Llama-3.1-8B` model with top-2 retrieval and a chunk size of 1,024.
