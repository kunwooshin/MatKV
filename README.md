# MatKV
## Installation
### package version compatibility
   ```bash
    $ conda create -n matkv python=3.12.7
    $ conda activate matkv
    $ pip3 install -r requirements.txt
   ```
### HuggingFace cli for LLaMA 3 model
   ```bash
   $ huggingface-cli login # insert your HuggingFace access token
   $ huggingface-cli download meta-llama/Llama-3.1-70B
   $ huggingface-cli download meta-llama/Llama-3.1-8B
   $ huggingface-cli download meta-llama/Llama-3.2-3B
   ```
### DeepNVMe for optimized tensor I/O
NVMe Storage required.
aio

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
