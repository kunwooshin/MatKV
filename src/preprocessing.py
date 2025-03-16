import fire
import os
import chromadb
import torch
from tqdm import tqdm
from typing import List
from transformers import LlamaForCausalLM, AutoTokenizer, DynamicCache, AutoModelForCausalLM, BitsAndBytesConfig
import time
from deepspeed.ops.op_builder import AsyncIOBuilder
from deepspeed.ops.op_builder import GDSBuilder
import pathlib

# 4-bit quantization 설정
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 4-bit로 로드
    bnb_4bit_compute_dtype=torch.float16,  # 연산 시 float16 사용
    bnb_4bit_use_double_quant=True,  # 더블 양자화 사용 (메모리 최적화)
    bnb_4bit_quant_type="nf4"  # NormalFloat4 (nf4) 양자화 적용
)

def file_write(out_f, tensor, handle, gpu_buffer):
    gpu_buffer.copy_(tensor)
    handle.sync_pwrite(gpu_buffer, out_f)
    
def get_chroma_client(dir: str):
  chroma_client = chromadb.PersistentClient(path = dir)
  return chroma_client.get_or_create_collection(name = "doc_collection")

class DocumentChunk():
  def __init__(
    self,
    id: str,
    text: str, 
  ):
    self.id = id
    self.text = text


class DocumentPreprocessor():
  def __init__(
      self, 
      docs_dir: str,
      db_dir: str, 
      cache_dir: str,
      model_name: str = "meta-llama/Llama-3.1-8B", 
      chunk_size: int = 1024,
  ):
    self.docs_dir = docs_dir
    self.cache_dir = cache_dir
    self.chunk_size = chunk_size
    self.vectordb = get_chroma_client(db_dir)
    #model_name = "meta-llama/Llama-2-7b-hf"
    print("Load model")
    self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    self.tokenizer.padding_side = "left"
    self.model = AutoModelForCausalLM.from_pretrained(
      model_name, 
      torch_dtype=torch.float16,
      # quantization_config=bnb_config,
      device_map="auto",
    )

    print("Model loaded")

  def process_documents(self):
    start_time = time.time()
    files = os.listdir(self.docs_dir)[:200] #FIXME
    print(f"Processing {len(files)} documents...") #FIXME

    for filename in tqdm(files):
    # for filename in tqdm(os.listdir(self.docs_dir)): #FIXME
      chunks = self.split_document(filename)
      if not chunks:
        continue
      self.save_to_vectordb(chunks)
      self.save_kv_cache_aio(chunks)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Processing completed in {elapsed_time:.2f} seconds.")

  def split_document(
      self,
      filename: str, 
  ):
    with open(os.path.join(self.docs_dir, filename)) as f:
      text = f.read()
      tokens = self.tokenizer.encode(text, add_special_tokens=False)
      chunks = []
      for i in range(0, len(tokens), self.chunk_size):
        chunk_tokens = tokens[i:i+self.chunk_size]
        # 청크가 900토큰 미만이면 건너뛰기
        if len(chunk_tokens) < 900: # 500
            continue
        chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(DocumentChunk(
            id=f"{filename}-{i}",
            text=chunk_text
        ))
      return chunks

  def save_to_vectordb(self, chunks: List[DocumentChunk]):
    self.vectordb.upsert(
      documents=[chunk.text for chunk in chunks],
      ids=[chunk.id for chunk in chunks]
    )

  def save_kv_cache(self, chunks: List[DocumentChunk]):
    for chunk in chunks:
      input = self.tokenizer(chunk.text, return_tensors="pt", padding_side="left").to("cuda")
      with torch.no_grad():
        output = self.model(**input, use_cache = True)

      cache = output.past_key_values.to_legacy_cache()
      torch.save(cache, os.path.join(self.cache_dir, f"{chunk.id}.pt"))
      '''
        cache = torch.load(os.path.join(self.cache_dir, f"{chunk.id}.pt"))
        past_kv_cache = DynamicCache.from_legacy_cache(loaded)
        self.model(**input, use_cache = True, past_kv_cache = past_kv_cache)
      '''
  def save_kv_cache_aio(self, chunks: List[DocumentChunk]):
  
    for chunk in chunks:
      output_file = os.path.join(self.cache_dir, f"{chunk.id}.pt")
      pathlib.Path(output_file).unlink(missing_ok=True)
      input = self.tokenizer(chunk.text, return_tensors="pt", padding_side="left").to("cuda")
      with torch.no_grad():
        output = self.model(**input, use_cache = True)
      
      cache = output.past_key_values.to_legacy_cache()
      # 바이너리 파일로 읽어오기 위해 바이너리 파일로 저장 (확장자명만 .pt)
      cache_tensors = [t.flatten() for layer in cache for t in layer]  # 하나의 1D 텐서로 변환; .pin_memory()하려면 
      cache_tensor = torch.cat(cache_tensors)
      
      aio_handle = AsyncIOBuilder().load().aio_handle()
      bounce_buffer = torch.empty(cache_tensor.shape[0], dtype=torch.float16).pin_memory()

      file_write(output_file, cache_tensor, aio_handle, bounce_buffer)
      
  def save_kv_cache_gds(self, chunks: List[DocumentChunk]):
  
    for chunk in chunks:
      output_file = os.path.join(self.cache_dir, f"{chunk.id}.pt")
      pathlib.Path(output_file).unlink(missing_ok=True)
      input = self.tokenizer(chunk.text, return_tensors="pt", padding_side="left").to("cuda")
      with torch.no_grad():
        output = self.model(**input, use_cache = True)
      
      cache = output.past_key_values.to_legacy_cache()
      # 바이너리 파일로 읽어오기 위해 바이너리 파일로 저장 (확장자명만 .pt)
      cache_tensors = [t.flatten() for layer in cache for t in layer]  # 하나의 1D 텐서로 변환; .pin_memory()하려면 
      cache_tensor = torch.cat(cache_tensors)
      file_sz = cache_tensor.numel()
      # print(file_sz)
      gds_handle = GDSBuilder().load().gds_handle()
      gds_buffer = gds_handle.new_pinned_device_tensor(file_sz, torch.empty(0, dtype=torch.float16, device='cuda', requires_grad=False))
      # print(gds_buffer.shape[0])
      file_write(output_file, cache_tensor, gds_handle, gds_buffer)
      

def main(
    docs_dir: str,
    db_dir: str, 
    cache_dir: str,
    model_name: str = "meta-llama/Llama-3.1-8B",
    chunk_size: int = 1024,
):
    preprocessor = DocumentPreprocessor(
      docs_dir=docs_dir,
      db_dir=db_dir,
      cache_dir=cache_dir,
      chunk_size=chunk_size,
      model_name=model_name
    )

    preprocessor.process_documents()

if __name__ == "__main__":
  fire.Fire(main)
