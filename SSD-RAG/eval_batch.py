import fire
import time
import os
import chromadb
import torch
import json
from tqdm import tqdm
from typing import List
from transformers import LlamaForCausalLM, AutoTokenizer, DynamicCache, AutoModelForCausalLM

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
        model_name: str = "meta-llama/Llama-3.1-8B",
        # model_name: str = "meta-llama/Llama-3.2-1B",
        use_past_cache: bool = True,
    ):
        self.query_file = query_file
        self.cache_dir = cache_dir
        self.top_k = top_k
        self.use_past_cache = use_past_cache
        self.vectordb = get_chroma_client(db_dir)
        
        print("Loading model...", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        # BATCH
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.model = torch.compile(self.model)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        
        self.model.resize_token_embeddings(len(self.tokenizer))
            
        print("Model loaded", flush=True)

    def process_query(self, bsz: int = 1, max_new_tokens: int = 100, total_num: int = 100):
        elapsed = 0.0
        cache_elapsed = 0.0
        batch_count = 0
        # BATCH
        total_time = [0.0, 0.0]
        
        with torch.no_grad():
            _ = self.model(torch.tensor([[1]]).cuda())
        torch.cuda.synchronize()
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
                    ##################
                    ##### KV-SSD #####
                    ##################
                    if self.use_past_cache:
                        ### CACHE LOAD ###
                        cache_load_start = time.perf_counter()
                        caches = [self.load_all_caches(batch_top_k_docs[idx]) for idx in range(len(batch_queries))] 
                        cache_load_start_sub = time.perf_counter()
                        if bsz == 1:
                            past_kv_caches = DynamicCache.from_legacy_cache(self.concat_caches_single(caches[0]))
                        else:
                            past_kv_caches_no_pad = self.concat_caches(caches)
                            past_kv_caches = self.pad_past_key_values(past_kv_caches_no_pad[0], past_kv_caches_no_pad[1]) 
            
                        batch_inputs = [self.seperate_query_and_doc(docs, query)
                                        for docs, query in zip(batch_top_k_docs, batch_queries)]
                        
                        cache_load_end = time.perf_counter()
                        cache_elapsed += (cache_load_end - cache_load_start)
                        print(f"Cache load time: {cache_load_end - cache_load_start_sub} seconds", flush=True)
                        print(f"Cache load time: {cache_load_end - cache_load_start} seconds", flush=True)
                        ### GENERATE ###
                        total_time = self.generate_response(batch_inputs, past_kv_caches=past_kv_caches, max_new_tokens=max_new_tokens, total_time=total_time)
                        
                    ##################
                    ##### Vanilla #####
                    ##################
                    else:
                        batch_inputs = [
                        self.concatenate_query_and_doc(batch_top_k_docs[idx], batch_queries[idx])
                        for idx in range(len(batch_queries))
                    ]
                        total_time = self.generate_response(batch_inputs, 
                        max_new_tokens=max_new_tokens,total_time=total_time)

                    end = time.perf_counter()
                    elapsed += (end - start)
                    print(f"Batch {batch_count} processed in {end - start} seconds", flush=True)

                    batch_queries = []

                if batch_count * bsz >= total_num:
                    break
            
        avg_time_per_query = elapsed / batch_count
        avg_cache_time_per_query = cache_elapsed / batch_count if cache_elapsed > 0 else 0
        
        # print(f"Avg prefill-sub per query: {total_time[0]/total_num}", flush=True)
        # print(f"Avg prefill-after-tokenize per query: {total_time[1]/total_num}", flush=True)
        print(f"Avg prefill per query: {total_time[0]/batch_count}", flush=True)
        print(f"Avg decode per query: {total_time[1]/batch_count}", flush=True)
        
        # print(f"Avg elapsed time per batch: {elapsed / batch_count}", flush=True)
        # print(f"Avg elapsed time per query: {avg_time_per_query}", flush=True) 
        print(f"Avg cache load time per query: {avg_cache_time_per_query}", flush=True)
        

    def find_top_k_docs(self, queries: List[str]):
        '''
        Select top k documents for each query from the vector db (batch processing)
        '''
        outputs = self.vectordb.query(query_texts=queries, n_results=self.top_k)

        batch_docs = []
        for i in range(len(queries)):
            ids = outputs['ids'][i]
            documents = outputs["documents"][i]
            docs = [Document(id, text) for id, text in zip(ids, documents)]
            batch_docs.append(docs)
        return batch_docs

    def concatenate_query_and_doc(self, docs: List[Document], query: str):
        input_text = "".join([doc.text for doc in docs])
        input_text += f"\n\nAnswer the following Question, given the relevant documents above. Answer without explanation. \n\nQuestion: {query}\n\nAnswer:"
        return input_text

    def seperate_query_and_doc(self, docs: List[Document], query: str):
        doc = "".join([doc.text for doc in docs])
        q = f"\n\nAnswer the following Question, given the relevant documents above. Answer without explanation. \n\nQuestion: {query}\n\nAnswer:"
        return doc, q
    
    def load_all_caches(self, docs: List[Document]):
        '''
        Load past kv cache from disk for all documents
        '''
        # print("Loading all caches", flush=True)
        return [self.load_kv_cache(doc.id) for doc in docs]

    def load_kv_cache(self, doc_id: str):
        '''
        Load past kv cache from disk
        '''
        cache_file = os.path.join(self.cache_dir, f"{doc_id}.pt")
        return torch.load(cache_file, weights_only=True, map_location="cuda")

    def concat_caches_single(self, caches):
        '''
        concatenate the cache 
        '''
        if len(caches) == 0:
            return None
        print(f"Concat {len(caches)} caches", flush=True)
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
        # num_heads = batch_caches[0][0][0][0].shape[1]  # 8
        # head_dim = batch_caches[0][0][0][0].shape[-1]  # 64
        # device = batch_caches[0][0][0][0].device  # cuda:0
        
        # batch_size만큼 KV 캐시를 각각 concat해서 저장할 리스트
        batch_keys_list = [[] for _ in range(num_layers)] 
        batch_values_list = [[] for _ in range(num_layers)]

        for i in range(batch_size):  # 각 요청별 처리
            request_caches = batch_caches[i]  # i번째 요청의 top-3 문서 캐시 리스트
            concatenated_request = self.concat_caches_single(request_caches)
            
            for layer in range(num_layers):
                batch_keys_list[layer].append(concatenated_request[layer][0])  # Key 저장
                batch_values_list[layer].append(concatenated_request[layer][1])  # Value 저장
        
#         batch_keys_list = [
#     [k1, k2, k3, k4],  # Layer 1 # (1, num_heads, seq_len, head_dim)
#     [k1, k2, k3, k4],  # Layer 2
#     ...
#     [k1, k2, k3, k4],  # Layer 16
# ] - 이 때, k1, k2, k3, k4 길이가 모두 다름

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
                # padding_counts_per_request.append(pad_len)
            # 배치 차원으로 `stack`
            past_key_values.append((torch.cat(keys, dim=0), torch.cat(values, dim=0)))
            
            # if layer_idx == 0:  # 첫 번째 레이어의 패딩 개수를 저장 (모든 레이어가 동일해야 함)
            #     left_padding_counts = padding_counts_per_request
        # print(len(past_key_values)) # layer
        # print(len(past_key_values[0])) # key, value
        # print(past_key_values[0][0].shape) # torch.Size([1, 8, 1022, 128])
        
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
            
        with torch.no_grad():
            output_tokens = self.model.generate(
                **tokens, 
                max_new_tokens=1,
                use_cache=True,
                past_key_values=past_kv_caches,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                )

        next_token_id = output_tokens.sequences[:, -1].unsqueeze(-1)
        past_key_values = output_tokens.past_key_values
        attention_mask = (output_tokens.sequences != self.tokenizer.pad_token_id).long()
        # print(past_key_values[0][0].shape)
        end_prefill = time.perf_counter()    
        unit_prefill = end_prefill - start_prefill
        print(f"prefill 1 request: {unit_prefill:6f} seconds")
        total_time[0] += unit_prefill

        start_decode = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids = output_tokens.sequences,
                attention_mask = attention_mask,
                max_new_tokens=max_new_tokens-1,
                use_cache=True,
                past_key_values=past_key_values,
                pad_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
            )
        # print(outputs.past_key_values[0][0].shape)
        
        end_decode = time.perf_counter()
        unit_decode = end_decode - start_decode
        print(f"decode 1 request: {unit_decode:6f} seconds")
        total_time[1] += unit_decode
        # generated_text = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        # print("Generated Text:", generated_text[0][1000:])
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
):
    
    processor = QueryProcessor(
        query_file=query_file,
        db_dir=db_dir,
        cache_dir=cache_dir,
        top_k=top_k,
        use_past_cache=use_past_cache,
    )
    processor.process_query(bsz=bsz, max_new_tokens=max_new_tokens, total_num=total_num)

if __name__ == "__main__":
    fire.Fire(main)
