import fire
import os
import chromadb
import torch
from tqdm import tqdm
from typing import List
from transformers import LlamaForCausalLM, AutoTokenizer, DynamicCache, AutoModelForCausalLM
import time


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
      chunk_size: int = 512,
  ):
    self.docs_dir = docs_dir
    self.cache_dir = cache_dir
    self.chunk_size = chunk_size
    self.vectordb = get_chroma_client(db_dir)
    #model_name = "meta-llama/Llama-2-7b-hf"
    print("Load model")
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    self.model = AutoModelForCausalLM.from_pretrained(
      model_name, 
      torch_dtype=torch.float16,
      device_map="auto",
    )
    print("Model loaded")

  def process_documents(self):
    start_time = time.time()
    files = sorted(os.listdir(self.docs_dir))[:100] #FIXME
    print(f"Processing {len(files)} documents...") #FIXME

    for filename in tqdm(files):
    # for filename in tqdm(os.listdir(self.docs_dir)): #FIXME
      chunks = self.split_document(filename)
      self.save_to_vectordb(chunks)
      self.save_kv_cache(chunks)

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
      chunks = [
        DocumentChunk(
          id=f"{filename}-{i}",
          text=self.tokenizer.decode(
            tokens[i:i+self.chunk_size], skip_special_tokens=True
          )
        ) for i in range(0, len(tokens), self.chunk_size)
      ]
      return chunks

  def save_to_vectordb(self, chunks: List[DocumentChunk]):
    self.vectordb.upsert(
      documents=[chunk.text for chunk in chunks],
      ids=[chunk.id for chunk in chunks]
    )

  def save_kv_cache(self, chunks: List[DocumentChunk]):
    for chunk in chunks:
      input = self.tokenizer(chunk.text, return_tensors="pt").to("cuda")
      with torch.no_grad():
        output = self.model(**input, use_cache = True)

      cache = output.past_key_values.to_legacy_cache()
      torch.save(cache, os.path.join(self.cache_dir, f"{chunk.id}.pt"))
      '''
        cache = torch.load(os.path.join(self.cache_dir, f"{chunk.id}.pt"))
        past_kv_cache = DynamicCache.from_legacy_cache(loaded)
        self.model(**input, use_cache = True, past_kv_cache = past_kv_cache)
      '''

def main(
    docs_dir: str,
    db_dir: str, 
    cache_dir: str,
    model_name: str = "meta-llama/Llama-3.1-8B",
    chunk_size: int = 512,
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
