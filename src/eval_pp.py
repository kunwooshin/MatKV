import fire
import time
import os
import chromadb
import torch
import json
import multiprocessing
from tqdm import tqdm
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, logging, BitsAndBytesConfig
from deepspeed.ops.op_builder import AsyncIOBuilder
import pathlib

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit로 로드
    bnb_4bit_compute_dtype=torch.float16,  # 연산 시 float16 사용
    bnb_4bit_use_double_quant=True,  # 더블 양자화 사용 (메모리 최적화)
    bnb_4bit_quant_type="nf4"  # NormalFloat4 (nf4) 양자화 적용
)

def file_read(inp_f, handle, gpu_buffer):
    handle.sync_pread(gpu_buffer, inp_f)
    return gpu_buffer.cuda()

def get_seq_len(flattened_tensor, num_layers, num_kv_heads, dim):
    """ 저장된 Flatten된 텐서에서 Token 수(seq_len) 자동 추출 """
    total_params = flattened_tensor.shape[0]  # 1D 텐서 크기
    per_layer_size = num_kv_heads * dim * 2  # Key + Value 크기
    
    seq_len = total_params // (num_layers * per_layer_size)  # 토큰 수 계산
    return seq_len

def restore_tensor_shape(flattened_tensor, num_layers, num_kv_heads, dim):
    restored_cache = []
    offset = 0
    seq_len = get_seq_len(flattened_tensor, num_layers, num_kv_heads, dim)
    each_layer = num_kv_heads * seq_len * dim
    for _ in range(num_layers):
        key = flattened_tensor[offset : offset + each_layer].reshape(1, num_kv_heads, seq_len, dim)
        offset += each_layer
        value = flattened_tensor[offset : offset + each_layer].reshape(1, num_kv_heads, seq_len, dim)
        offset += each_layer
        
        restored_cache.append((key, value))

    return tuple(restored_cache)


def load_model(model_name):
    """ 메인 프로세스에서 모델을 미리 로드 """
    print(f"LOADING MODEL {model_name} ...", flush =True)
    init_time = time.perf_counter()
    logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})

    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        # quantization_config=bnb_config,
        device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    print(time.perf_counter() - init_time, flush =True)
    print(f"MODEL LOADED", flush =True)
    
    return tokenizer, model

###########################################
# Query Parsing, Document, Chroma client
###########################################
def parse_json_query(json_query: str):
    parsed = json.loads(json_query)
    return parsed['query']

class Document:
    def __init__(self, id: str, text: str):
        self.id = id
        self.text = text

def get_chroma_client(dir: str):
    chroma_client = chromadb.PersistentClient(path=dir)
    return chroma_client.get_or_create_collection(name="doc_collection")

########################################
# (1) 검색/캐시 전용 클래스 (모델 없음)
########################################
class QueryProcessorForSearch:
    def __init__(self, db_dir: str, cache_dir: str, top_k: int = 4):
        self.cache_dir = cache_dir
        self.top_k = top_k
        
        print(f"[{os.getpid()}] Initializing QueryProcessorForSearch...")
        # DB
        self.vectordb = get_chroma_client(db_dir)
        print(f"[{os.getpid()}] DB ready (Search mode).")

    def find_top_k_docs(self, queries: List[str]) -> List[List[Document]]:
        outputs = self.vectordb.query(query_texts=queries, n_results=self.top_k)
        batch_docs = []
        for i in range(len(queries)):
            ids = outputs["ids"][i]
            texts = outputs["documents"][i]
            docs = [Document(id, txt) for id, txt in zip(ids, texts)]
            batch_docs.append(docs)
        return batch_docs
    
    def load_all_caches(self, docs: List[Document], aio_handle, config):
        return [self.load_kv_cache_aio(doc.id, aio_handle, config) for doc in docs]
    
    def load_kv_cache_aio(self, doc_id: str, aio_handle, config):
        in_file = os.path.join(self.cache_dir, f"{doc_id}.pt")
        file_sz = os.path.getsize(in_file)

        num_elements = file_sz // 2
        bounce_buffer = torch.empty(num_elements, dtype=torch.float16).pin_memory()
        
        loaded_tensor = file_read(in_file, aio_handle, bounce_buffer)
        kv_cache = restore_tensor_shape(loaded_tensor, config["num_layers"], config["num_kv_heads"], config["dim"])
        return kv_cache
    
    def load_caches(self, docs: List[Document]):
        """
        Load .pt cache from disk (CPU) for each doc
        """
        all_caches = []
        for doc in docs:
            fname = os.path.join(self.cache_dir, f"{doc.id}.pt")
            if not os.path.exists(fname):
                all_caches.append(None)
            else:
                ckpt = torch.load(fname, weights_only=True, map_location="cuda")
                all_caches.append(ckpt)
        return all_caches

    ############################
    # 문서별 KV concat
    ############################
    def concat_caches_single(self, caches):
        num_layers = len(caches[0])
        concat_result = []
        for layer_idx in range(num_layers):
            keys = torch.cat([cache[layer_idx][0] for cache in caches], dim=2)
            values = torch.cat([cache[layer_idx][1] for cache in caches], dim=2)
            concat_result.append((keys, values))
        return concat_result

    ############################
    # 배치(쿼리들)에 대해 concat
    ############################
    def concat_caches(self, batch_caches):
        """
        batch_caches: shape [batch_size, top_k(=1..n)?]
        """
        batch_size = len(batch_caches)
        num_layers = len(batch_caches[0][0]) if batch_caches[0][0] else 0

        batch_keys_list = [[] for _ in range(num_layers)]
        batch_values_list = [[] for _ in range(num_layers)]

        for i in range(batch_size):
            request_caches = batch_caches[i]
            concatenated_request = self.concat_caches_single(request_caches)
            
            for layer in range(num_layers):
                batch_keys_list[layer].append(concatenated_request[layer][0]) 
                batch_values_list[layer].append(concatenated_request[layer][1])

        return (batch_keys_list, batch_values_list)  # 시연용
    
    def seperate_query_and_doc(self, docs: List[Document], query: str):
        """
        문서 + 쿼리를 (doc_text, query_text)로 분리
        """
        doc_text = "".join(d.text for d in docs)
        query_text = f"\n\nAnswer the following Question, given the relevant documents.\nQuestion: {query}\nAnswer:"
        return (doc_text, query_text)

###########################################
# QueryProcessor (기존 코드 기반)
###########################################
class QueryProcessorForDecode:
    def __init__(self, tokenizer, model, use_past_cache=True):
        self.use_past_cache = use_past_cache

        print(f"[{os.getpid()}] Initializing QueryProcessorForDecode (GPU model)...")
        # 토크나이저
        self.tokenizer = tokenizer
        self.model = model
        
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": self.tokenizer.eos_token})
            
        self.model.resize_token_embeddings(len(self.tokenizer))
        
    ############################
    # seq_len 패딩
    ############################
    def pad_past_key_values(self, batch_keys_list, batch_values_list):
        """
        배치 내 요청마다 seq_len이 다를 경우, 최대 길이에 맞춰 padding 후 `past_key_values` 형태로 변환
        """
        num_layers = len(batch_keys_list)
        batch_size = len(batch_keys_list[0])
        
        max_doc_length = max(k.shape[2] for k in batch_keys_list[0])
        past_key_values = []
        
        for layer_idx in range(num_layers):
            keys = []
            values = []
            padding_counts_per_request = []

            for i in range(batch_size):
                past_k = batch_keys_list[layer_idx][i] 
                past_v = batch_values_list[layer_idx][i] 
                doc_length = past_k.shape[2]
                pad_len = max_doc_length - doc_length

                if pad_len > 0:
                    pad_shape = (past_k.shape[0], past_k.shape[1], pad_len, past_k.shape[3])  
                    pad_tensor_k = torch.zeros(pad_shape, dtype=past_k.dtype, device="cuda")
                    pad_tensor_v = torch.zeros(pad_shape, dtype=past_v.dtype, device="cuda")
                    past_k = torch.cat([pad_tensor_k, past_k], dim=2)
                    past_v = torch.cat([pad_tensor_v, past_v], dim=2) 

                keys.append(past_k)
                values.append(past_v)
                padding_counts_per_request.append(pad_len)

            past_key_values.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
            
        return tuple(past_key_values)


    ############################
    # 디코딩
    ############################
    def generate_response(self, inputs, pad_token_id, past_kv_caches_no_pad=None, max_new_tokens: int = 100):

        doc_inputs = [d for (d, q) in inputs]
        query_inputs = [q for (d, q) in inputs]
        
        tokenized_docs = self.tokenizer(doc_inputs, return_tensors="pt", padding=True, truncation=True).to("cuda")
        tokenized_q = self.tokenizer(query_inputs, return_tensors="pt", padding=True, truncation=True).to("cuda")

        doc_input_ids = tokenized_docs["input_ids"]
        query_input_ids = tokenized_q["input_ids"]
        
        past_kv_caches = self.pad_past_key_values(past_kv_caches_no_pad[0], past_kv_caches_no_pad[1])
        # if past_kv_caches:
        #     past_kv_caches = tuple(
        #         (k.to("cuda"), v.to("cuda")) for k, v in past_kv_caches
        #     )
        final_input_ids = torch.cat([doc_input_ids, query_input_ids], dim=1)
        attention_mask = (final_input_ids != self.tokenizer.pad_token_id).long()

        # print(f"[{os.getpid()}] Generating response: final_input_ids shape = {final_input_ids.shape}", flush=True)

        with torch.no_grad():
            output_tokens = self.model.generate(
                input_ids=final_input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                past_key_values=past_kv_caches,
                pad_token_id=pad_token_id,
            )
        # txt = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        # print(f"[{os.getpid()}] OUTPUT: {txt[-30:]}")


###########################################
# (A) prepare_process
###########################################
def prepare_process(
    query_file: str,
    bsz: int,
    total_num: int,
    db_dir: str,
    cache_dir: str,
    top_k: int,
    config: dict,
    queue_out: multiprocessing.Queue,
):
    """
    1) 파일에서 쿼리 bsz개씩 읽기
    2) DB 검색 -> 문서 -> 캐시 로드 + concat
    3) (batch_inputs, past_kv) 형태로 queue_out에 put
    """
    pid = os.getpid()
    aio_handle = AsyncIOBuilder().load().aio_handle()
    print(f"[{pid}] prepare_process started.")
    print("PREPARE STARTS: ", time.perf_counter())
    search_proc = QueryProcessorForSearch(db_dir, cache_dir, top_k=top_k)
    
    count = 0
    batch_queries = []

    with open(query_file, 'r') as f:
        for line in tqdm(f, total=total_num, desc="[Reader]"):
            query_str = parse_json_query(line)
            batch_queries.append(query_str)
            count += 1

            if len(batch_queries) == bsz:
                # DB 검색
                batch_docs = search_proc.find_top_k_docs(batch_queries)
                # 캐시 로드
                caches = []
                for docs in batch_docs:
                    loaded = search_proc.load_all_caches(docs, aio_handle, config)
                    # loaded = search_proc.load_caches(docs)
                    caches.append(loaded)
                    
                concat_kv = search_proc.concat_caches(caches)

                # (문서, 쿼리) 형태로 만들기
                batch_inputs = [
                    search_proc.seperate_query_and_doc(docs, q)
                    for q, docs in zip(batch_queries, batch_docs)
                ]
                # while queue_out.qsize() > 0:
                #     time.sleep(0.5) 
                # queue_out에 전달
                queue_out.put((batch_inputs, concat_kv))

                batch_queries = []

            if count >= total_num:
                break

    # 남은 것 처리
    if batch_queries:
        batch_docs = search_proc.find_top_k_docs(batch_queries)
        caches = []
        for docs in batch_docs:
            loaded = search_proc.load_caches(docs)
            caches.append(loaded)
        concat_kv = search_proc.concat_caches(caches)

        batch_inputs = [
            search_proc.seperate_query_and_doc(docs, q)
            for q, docs in zip(batch_queries, batch_docs)
        ]
        queue_out.put((batch_inputs, concat_kv))

    # 종료
    queue_out.put(None)
    print(f"[{pid}] prepare_process done. total={count}")


########################################
# (B) decode_process
########################################
def decode_process(
    use_past_cache: bool,
    queue_out: multiprocessing.Queue,
    max_new_tokens: int,
    tokenizer, model):

    pid = os.getpid()
    print(f"[{pid}] decode_process started.")
    decode_proc = QueryProcessorForDecode(tokenizer, model, use_past_cache=use_past_cache)
    
    batch_count = 0
    # t0 = time.perf_counter()

    while True:
        item = queue_out.get()
        if item is None:
            print("LAST DECODE ENDS: ", time.perf_counter())
            print(f"[{pid}] decode_process got None. Exiting.")
            break

        batch_count += 1
        (batch_inputs, all_concat) = item

        decode_proc.generate_response(
            inputs=batch_inputs,       
            pad_token_id=tokenizer.eos_token_id,
            past_kv_caches_no_pad=all_concat,
            max_new_tokens=max_new_tokens,
        )
        
    # elapsed = time.perf_counter() - t0
    # print(f"[{pid}] decode_process done. batch_count={batch_count}, time={elapsed:.2f}s.")


###########################################
# 메인 함수
###########################################
def main_multiprocess(
    query_file: str,
    db_dir: str,
    cache_dir: str,
    top_k: int = 4,
    use_past_cache: bool = True,
    bsz: int = 1,
    max_new_tokens: int = 100,
    total_num: int = 256, #FIXME
    model_name: str = "meta-llama/Llama-3.1-8B"
    # model_name: str = "meta-llama/Llama-3.1-70B"
    # model_name: str = "meta-llama/Llama-3.2-3B" #FIXME
):
    tokenizer, model = load_model(model_name)
    # Queue
    model_config = {'num_layers': model.config.num_hidden_layers,
              'dim': model.config.hidden_size//model.config.num_attention_heads,
              'num_kv_heads': model.config.num_key_value_heads
    }
    
    queue_out = multiprocessing.Queue(maxsize=1) 

    p_prepare = multiprocessing.Process(
        target=prepare_process,
        args=(query_file, bsz, total_num, db_dir, cache_dir, top_k, model_config, queue_out),
        daemon=False
    )

    p_decode = multiprocessing.Process(
        target=decode_process,
        args=(use_past_cache, queue_out, max_new_tokens, tokenizer, model),
        daemon=False
    )

    t0 = time.perf_counter()
    p_prepare.start()
    p_decode.start()

    p_prepare.join()
    p_decode.join()
    print(f"TOTAL: {time.perf_counter()-t0} seconds")
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")  # Windows/Mac은 spawn 권장
    fire.Fire(main_multiprocess)
