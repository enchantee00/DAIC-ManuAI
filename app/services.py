import time

from fastapi import HTTPException
from langchain.schema import HumanMessage

from prompt import *
from schemas import QuestionRequest
from config import UPSTAGE_API_KEY, UPSTAGE_API_URL
import requests
import json


def process_pdf(file, filename: str):
    files = {
        "document": (filename, file.file, file.content_type)
    }
    data = {
        "ocr": "force",
        "base64_encoding": "['figure']",
        "model": "document-parse"
    }
    headers = {
        "Authorization": f"Bearer {UPSTAGE_API_KEY}"
    }

    response = requests.post(UPSTAGE_API_URL, headers=headers, files=files, data=data)

    with open("filename.json", "w", encoding="utf-8") as f:
        json.dump(response.json(), f, ensure_ascii=False, indent=4)


def ask_llm(request: QuestionRequest, model_manager):
    start_time = time.time()  # 시작 시간 기록

    question = request.question


    # 해당 카테고리의 FAISS DB 로드
    db_path = FAISS_DB_PATHS[category]
    db, retriever = model_manager.load_faiss_db(db_path)
    retrieved_docs = retriever.get_relevant_documents(question)
    retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)
    print(f"~faiss: {processing_time}")

    title_prompt = vllm_llama_title()
    # response_prompt = vllm_llama_response()
    response_prompt = vllm_llama_response_without_history()


    # title, answer = await asyncio.gather(
    #     generate_title(session_id, title_prompt, question, sampling_params),
    #     generate_response(response_prompt, history_text, retrieved_text, question, sampling_params)
    # )

    title = model_manager.generate_title(session_id, title_prompt, question), 
    # answer = generate_response(response_prompt, history_text, retrieved_text, question, sampling_params)
    answer = model_manager.generate_response_without_history(response_prompt, retrieved_text, question)


    if isinstance(title, tuple):
        title = title[0]

    memory.save_context({"input": question}, {"output": answer})

    # 처리 시간 계산
    end_time = time.time()
    processing_time = round(end_time - start_time, 2)  
    print(f"processing_time: {processing_time}")

    return {
        "answer": answer
    }