import fire
import time
import os
import chromadb
import torch
import json
from tqdm import tqdm
from typing import List
from transformers import LlamaForCausalLM, AutoTokenizer, DynamicCache, AutoModelForCausalLM, logging, BitsAndBytesConfig
import concurrent.futures
import torch
import os
from deepspeed.ops.op_builder import GDSBuilder, AsyncIOBuilder
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
        # print(restored_cache[0][0].shape)
    return tuple(restored_cache)
    
def get_chroma_client(dir: str):
    chroma_client = chromadb.PersistentClient(path=dir)
    return chroma_client.get_or_create_collection(name="doc_collection")

class Document:
    def __init__(self, id: str, text: str):
        self.id = id
        self.text = text

def parse_json_query(json_query: str):
    parsed = json.loads(json_query)
    return parsed['query']

class QueryProcessor():
    def __init__(
        self,
        query_file: str,
        db_dir: str,
        cache_dir: str,
        top_k: int = 4,
        # model_name: str = "meta-llama/Llama-3.1-8B",
        model_name: str = "meta-llama/Llama-3.1-70B",
        # model_name: str = "meta-llama/Llama-3.2-3B",
        use_past_cache: bool = True,
        log_file: str = "log.txt"
    ):
        self.query_file = query_file
        self.cache_dir = cache_dir
        self.top_k = top_k
        self.use_past_cache = use_past_cache
        self.vectordb = get_chroma_client(db_dir)
        self.log_file = log_file
        
        print(f"LOADING MODEL {model_name} ...", flush =True)
        init_time = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        # BATCH
        self.tokenizer.padding_side = "left"
        
        self.model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
        )

        self.model = torch.compile(self.model)
        config = self.model.config
        self.num_layers = config.num_hidden_layers
        self.dim = config.hidden_size//config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        # print(self.num_layers, self.dim, self.num_kv_heads) # 32 128 8
        # self.gds_handle = GDSBuilder().load().gds_handle()
        self.aio_handle = AsyncIOBuilder().load().aio_handle()
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        
        self.model.resize_token_embeddings(len(self.tokenizer))
            
        print(time.perf_counter() - init_time, flush =True)
        print(f"MODEL LOADED", flush =True)

    def process_query(self, bsz: int = 1, max_new_tokens: int = 100, total_num: int = 100):
        elapsed = 0.0
        cache_elapsed = 0.0
        batch_count = 0
        # BATCH
        total_time = [0.0, 0.0]
        
        self.tokenizer.padding_side = "left"
        
        with open(self.query_file) as f:
            batch_queries = []

            for line in tqdm(f, total=total_num):
                parsed_query = parse_json_query(line)
                batch_queries.append(parsed_query)
                
                if len(batch_queries) == bsz:
                    batch_count += 1
                    batch_top_k_docs = self.find_top_k_docs(batch_queries)
                    start = time.perf_counter()
                    ##### KV-SSD #####
                    if self.use_past_cache:
                        ### CACHE LOAD ###
                        cache_load_start = time.perf_counter()
                        caches = [self.load_all_caches(batch_top_k_docs[idx]) for idx in range(len(batch_queries))]
                        if bsz == 1:
                            past_kv_caches = DynamicCache.from_legacy_cache(self.concat_caches_single(caches[0]))
                            # print(len(past_kv_caches))
                            # print(len(past_kv_caches[0]))
                            # print(past_kv_caches[0][0].shape)
                        else:
                            past_kv_caches_no_pad = self.concat_caches(caches)
                            past_kv_caches = self.pad_past_key_values(past_kv_caches_no_pad[0], past_kv_caches_no_pad[1]) 
            
                        batch_inputs = [self.seperate_query_and_doc(docs, query)
                                        for docs, query in zip(batch_top_k_docs, batch_queries)]
                        
                        cache_load_end = time.perf_counter()
                        cache_elapsed += (cache_load_end - cache_load_start)
                        
                        total_time = self.generate_response(batch_inputs, past_kv_caches=past_kv_caches, max_new_tokens=max_new_tokens, total_time=total_time)
                        
                    ##### Vanilla #####
                    else:
                        batch_inputs = [
                        self.concatenate_query_and_doc(batch_top_k_docs[idx], batch_queries[idx])
                        for idx in range(len(batch_queries))
                    ]
                        total_time = self.generate_response(batch_inputs, 
                        max_new_tokens=max_new_tokens, total_time=total_time)

                    end = time.perf_counter()
                    elapsed += (end - start)
                    # print(f"Batch {batch_count} processed in {end - start} seconds", flush=True)

                    batch_queries = []

                if batch_count * bsz >= total_num:
                    break
            
        avg_time_per_query = elapsed / batch_count
        avg_cache_time_per_query = cache_elapsed / batch_count if cache_elapsed > 0 else 0
        
        # print(f"Avg prefill-sub per query: {total_time[0]/total_num}", flush=True)
        # print(f"Avg prefill-after-tokenize per query: {total_time[1]/total_num}", flush=True)
        print(f"Avg cache load time per batch: {avg_cache_time_per_query:.4f}", flush=True)
        print(f"Avg prefill per batch: {total_time[0]/batch_count:.4f}", flush=True)
        print(f"Avg decode per batch: {total_time[1]/batch_count:.4f}", flush=True)
        print('-'*15)
        print(f"Total cache load time: {cache_elapsed:.4f}", flush=True)
        print(f"Total prefill time: {total_time[0]:.4f}", flush=True)
        print(f"Total decode time: {total_time[1]:.4f}", flush=True)


    def find_top_k_docs(self, queries: List[str]):
        '''
        Select top k documents for each query from the vector db (batch processing)
        '''
        outputs = self.vectordb.query(query_texts=queries, n_results=self.top_k)

        batch_docs = []
        for i in range(len(queries)):
            ids = outputs['ids'][i]
            documents = outputs["documents"][i]
            # ids = outputs['ids'][i][::-1] # use this for matkv-reverse
            # documents = outputs["documents"][i][::-1] # use this for matkv-reverse
            docs = [Document(id, text) for id, text in zip(ids, documents)]
            batch_docs.append(docs)
        return batch_docs

    def concatenate_query_and_doc(self, docs: List[Document], query: str):
        input_text = "".join([doc.text for doc in docs])
        input_text += f"\n\nPlease answer the user's question based on these documents above. Only generate the short answer without explanation. \n\nQuestion: {query}\n\nAnswer:"

        return input_text

    def seperate_query_and_doc(self, docs: List[Document], query: str):
        doc = "".join([doc.text for doc in docs])
        q = f"\n\nPlease answer the user's question based on these documents above. Only generate the short answer without explanation. \n\nQuestion: {query}\n\nAnswer:"
        return doc, q
    
    def load_all_caches(self, docs: List[Document]):
        return [self.load_kv_cache_aio(doc.id) for doc in docs] # load_kv_cache(doc.id) # if aio is unavailable

    def load_kv_cache(self, doc_id: str):
        cache_file = os.path.join(self.cache_dir, f"{doc_id}.pt")
        return torch.load(cache_file, weights_only=True, map_location="cuda")

    def load_kv_cache_gds(self, doc_id: str):
        in_file = os.path.join(self.cache_dir, f"{doc_id}.pt")
        file_sz = os.path.getsize(in_file)

        file_sz = file_sz//2
        
        gds_buffer = self.gds_handle.new_pinned_device_tensor(file_sz, torch.empty(0, dtype=torch.float16, device='cuda',requires_grad=False)) 
        # buffer register failed:device pointer already registered
        loaded_tensor = file_read(in_file, self.gds_handle, gds_buffer)
        
        kv_cache = restore_tensor_shape(loaded_tensor, self.num_layers, self.num_kv_heads, self.dim)
        return kv_cache
    
    def load_kv_cache_aio(self, doc_id: str):
        in_file = os.path.join(self.cache_dir, f"{doc_id}.pt")
        file_sz = os.path.getsize(in_file)

        num_elements = file_sz // 2
        bounce_buffer = torch.empty(num_elements, dtype=torch.float16).pin_memory()
        
        loaded_tensor = file_read(in_file, self.aio_handle, bounce_buffer)
        
        kv_cache = restore_tensor_shape(loaded_tensor, self.num_layers, self.num_kv_heads, self.dim)
        return kv_cache
    
    def concat_caches_single(self, caches):
        '''
        concatenate the cache 
        '''
        if len(caches) == 0:
            return None

        num_layers = len(caches[0])
        concatenated = []
        for layer in range(num_layers):
            keys = torch.cat([cache[layer][0] for cache in caches], dim=2)
            values = torch.cat([cache[layer][1] for cache in caches], dim=2)
            concatenated.append((keys, values))
        return concatenated

    def concat_caches(self, batch_caches):
        """
        batch_caches: List[List[List[Tuple[Tensor, Tensor]]]]
        - batch_size(4)개의 요청이 각각 top_k(3)개의 문서에 대해 가져온 KV 캐시 리스트
        - 즉, batch_caches[i]는 i번째 요청의 KV 캐시 리스트(top-3)
        
        반환값: Tuple[List[Tensor], List[Tensor]]
        - past_key_values에 올바르게 들어갈 수 있도록 변환
        """
        if len(batch_caches) == 0:
            print("concat_caches: No caches to concatenate")
            return None  

        batch_size = len(batch_caches)  # 4
        num_layers = len(batch_caches[0][0])  # 16
        
        # batch_size만큼 KV 캐시를 각각 concat해서 저장할 리스트
        batch_keys_list = [[] for _ in range(num_layers)] 
        batch_values_list = [[] for _ in range(num_layers)]

        for i in range(batch_size):  # 각 요청별 처리
            request_caches = batch_caches[i]  # i번째 요청의 top-3 문서 캐시 리스트
            concatenated_request = self.concat_caches_single(request_caches)
            
            for layer in range(num_layers):
                batch_keys_list[layer].append(concatenated_request[layer][0])  # Key 저장
                batch_values_list[layer].append(concatenated_request[layer][1])  # Value 저장

        return (batch_keys_list, batch_values_list)
    
    def pad_past_key_values(self, batch_keys_list, batch_values_list):
        """
        배치 내 요청마다 seq_len이 다를 경우, 최대 길이에 맞춰 padding 후 `past_key_values` 형태로 변환
        """
        num_layers = len(batch_keys_list)  # 총 레이어 개수 (16개)
        batch_size = len(batch_keys_list[0])  # 배치 크기 (4개 요청)
        
        # 각 레이어별로 최대 seq_len 찾기
        max_doc_length = max(k.shape[2] for k in batch_keys_list[0])
        left_padding_counts = []
        # Padding 적용
        past_key_values = []
        
        for layer_idx in range(num_layers):
            keys = []
            values = []
            # padding_counts_per_request = []

            for i in range(batch_size):
                past_k = batch_keys_list[layer_idx][i]
                past_v = batch_values_list[layer_idx][i]
                doc_length = past_k.shape[2]
                pad_len = max_doc_length - doc_length

                # Zero-padding 적용
                if pad_len > 0:
                    pad_shape = (past_k.shape[0], past_k.shape[1], pad_len, past_k.shape[3])  # (1, num_heads, pad_len, head_dim)
                    pad_tensor_k = torch.zeros(pad_shape, dtype=past_k.dtype, device=past_k.device)
                    pad_tensor_v = torch.zeros(pad_shape, dtype=past_v.dtype, device=past_v.device)
                    past_k = torch.cat([pad_tensor_k, past_k], dim=2) 
                    past_v = torch.cat([pad_tensor_v, past_v], dim=2) 

                keys.append(past_k)
                values.append(past_v)

            past_key_values.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
        
        return tuple(past_key_values)
        
    def generate_response(
        self, 
        inputs, 
        past_kv_caches=None,
        max_new_tokens: int = 100,
        total_time: list = [0.0, 0.0],
        ):

        start_prefill = time.perf_counter()
        self.tokenizer.padding_side = "left"
        
        if past_kv_caches is None:
            generated_tokens = []
            tokens = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, padding_side="left").to("cuda")
            
        else:       
            doc_inputs = [doc for doc, _ in inputs]
            query_inputs = [query for _, query in inputs]
            
            tokenized_docs = self.tokenizer(doc_inputs, return_tensors="pt", padding=True, truncation=True, padding_side="left").to("cuda")
            tokenized_queries = self.tokenizer(query_inputs, return_tensors="pt", padding=True, truncation=True, padding_side="left").to("cuda")
            
            doc_input_ids = tokenized_docs["input_ids"]
            query_input_ids = tokenized_queries["input_ids"]

            input_ids = torch.cat([doc_input_ids, query_input_ids], dim=1)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            # 최종 모델 입력 구성
            tokens = {
                "input_ids": input_ids, # torch.Size([1, 1054])
                "attention_mask": attention_mask
            }
            
        prompt_length = tokens['input_ids'].shape[1]
        
        with torch.no_grad():    
            output_tokens = self.model.generate(
                **tokens, 
                max_new_tokens=1,
                use_cache=True,
                past_key_values=past_kv_caches,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                return_legacy_cache=True,
                )

        next_token_id = output_tokens.sequences[:, -1].unsqueeze(-1)
        past_key_values = output_tokens.past_key_values
        attention_mask = (output_tokens.sequences != self.tokenizer.pad_token_id).long()
        # print(past_key_values[0][0].shape)
        end_prefill = time.perf_counter()    
        unit_prefill = end_prefill - start_prefill
        # print(f"prefill 1 request: {unit_prefill:6f} seconds")
        total_time[0] += unit_prefill
        
        start_decode = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids = output_tokens.sequences,
                attention_mask = attention_mask,
                max_new_tokens=max_new_tokens-1,
                use_cache=True,
                # use_cache=False,
                past_key_values=past_key_values,
                # pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                return_legacy_cache=True,
            )
        # print(outputs.past_key_values[0][0].shape)
        
        end_decode = time.perf_counter()
        unit_decode = end_decode - start_decode
            
        total_time[1] += unit_decode
        
        # ONLY FOR TESTING ACCURACY (HotpotQA)
        # generated_answers = outputs.sequences[:, prompt_length:]
        # generated_text = self.tokenizer.batch_decode(generated_answers, skip_special_tokens=True)
            
        # with open(self.log_file, "a", encoding="utf-8") as lf:
        #     for line in generated_text:
        #         json_line = {"answer": line}
        #         lf.write(json.dumps(json_line, ensure_ascii=False) + "\n")
        return total_time
                
def main(
    query_file: str,
    db_dir: str,
    cache_dir: str,
    top_k: int = 4,
    use_past_cache: bool = True,
    bsz: int = 1,
    max_new_tokens: int = 100,
    total_num: int = 100,
    log_file: str = "log.txt"
):
    logging.set_verbosity_error()
    processor = QueryProcessor(
        query_file=query_file,
        db_dir=db_dir,
        cache_dir=cache_dir,
        top_k=top_k,
        use_past_cache=use_past_cache,
        log_file=log_file
    )
    t0 = time.perf_counter()
    processor.process_query(bsz=bsz, max_new_tokens=max_new_tokens, total_num=total_num)
    print(f"TOTAL: {time.perf_counter()-t0:.4f} seconds")
    
if __name__ == "__main__":
    fire.Fire(main)
