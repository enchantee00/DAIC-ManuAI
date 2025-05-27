import torch
import faiss
import re
import pickle

from vllm import LLM, SamplingParams
from sentence_transformers import SentenceTransformer


from config import UPSTAGE_API_KEY, UPSTAGE_API_URL, LLM_MODEL_PATH, EMBEDDING_MODEL_PATH, CHROMA_DB_PATH

class ModelManager:
    def __init__(self):
        self.llm = None
        self.sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
        self.sentence_model = None
        
        self._load()
        

    def _load(self):
        # vLLM 초기화
        self.llm = LLM(model=LLM_MODEL_PATH, tensor_parallel_size=2, gpu_memory_utilization=0.8, max_model_len=1024)

        device = torch.device("cuda:2")  # 원하는 GPU 번호로 설정
        self.sentence_model = SentenceTransformer(EMBEDDING_MODEL_PATH, device=device)


    def load_faiss_db(self, faiss_db_directory: str):
        with open(faiss_db_directory + "_index_to_docstore_id.pkl", "rb") as f:
            index_to_docstore_id = pickle.load(f)

        with open(faiss_db_directory + "_docstore.pkl", "rb") as f:
            docstore = pickle.load(f)

        index = faiss.read_index(faiss_db_directory + "_faiss_db.index")

        db = FAISS(
            embedding_function=self.embeddings,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

        retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 5})
        return db, retriever


    def generate_response(self, prompt, retrieved_text, question):
        prompt = prompt.format(
            context=retrieved_text,
            question=question
        )
        # outputs = await asyncio.to_thread(llm.generate, [prompt], sampling_params)
        outputs = self.llm.generate([prompt], self.sampling_params)
        response = outputs[0].outputs[0].text.strip()
        return response