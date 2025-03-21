# MatKV
## Installation
### package version compatibility
   ```bash
    $ conda create -n matkv python=3.12.7
    $ conda activate matkv
    $ pip3 install -r requirements.txt
   ```
### HuggingFace cli for LLaMA 3 model
https://huggingface.co/meta-llama/Llama-3.1-70B
   ```bash
   $ huggingface-cli login # insert your HuggingFace access token
   $ huggingface-cli download meta-llama/Llama-3.1-70B
   $ huggingface-cli download meta-llama/Llama-3.1-8B
   $ huggingface-cli download meta-llama/Llama-3.2-3B
   ```
### DeepNVMe for optimized tensor I/O
https://github.com/deepspeedai/DeepSpeedExamples/tree/master/deepnvme/file_access
NVMe Storage is required for optimized tensor I/O using DeepNVMe's async_io operator
   ```base
   $ apt install libaio-dev
   ```
If NVMe Storage is not available
(1) you should use save_kv_cache() instead of save_kv_cache_aio() inside preprocess_documents() in preprocessing.py

   ```python
   def process_documents(self):
       ...
         self.save_kv_cache(chunks)
   ```

(2) you should use load_kv_cache() instead of load_kv_cache_aio() inside load_all_caches() in eval_batch.py (also in eval_pp.py)

   ```python
   def load_all_caches(self, docs: List[Document]):
        return [self.load_kv_cache(doc.id) for doc in docs]
   ```
### Run experiment
   ```bash
   $ ./preprocessing.sh # (1) update vector DB (2) store key-value tensors for each document chunk in SSD
   ```
   ```bash
   $ ./eval_batch.sh
   ```
   ```bash
   $ ./eval_pp.sh # pipelined process (Overlapping)
   ```
