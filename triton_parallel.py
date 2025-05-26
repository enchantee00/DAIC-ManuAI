import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
import tritonclient.grpc.aio as grpcclient
import numpy as np
import json
import uvicorn
import uuid


app = FastAPI()

# Triton client
triton_client: grpcclient.InferenceServerClient = None

# 비동기 요청 큐
request_queue = asyncio.Queue()

# 요청 데이터 구조
class Request(BaseModel):
    prompt: str

class InferenceTask:
    def __init__(self, prompt: str):
        self.id = str(uuid.uuid4())
        self.prompt = prompt
        self.future = asyncio.get_event_loop().create_future()

# FastAPI 요청 처리
@app.post("/generate/")
async def generate(req: Request):
    task = InferenceTask(req.prompt)
    await request_queue.put(task)
    result = await task.future  # 응답 올 때까지 대기
    return {"response": result}

# Worker: 큐에서 여러 요청 모아서 Triton에 stream_infer()
async def stream_worker():


    while True:
        batch = []
        # 50ms 동안 모아서 배치 구성
        try:
            while len(batch) < 4:  # 최대 batch size
                task = await asyncio.wait_for(request_queue.get(), timeout=0.05)
                batch.append(task)
        except asyncio.TimeoutError:
            if not batch:
                continue  # 아무 것도 없음

        # Triton 요청 구성
        inputs_list = []
        for task in batch:
            inputs = []

            prompt_np = np.array([task.prompt.encode("utf-8")], dtype=np.object_)
            inputs.append(grpcclient.InferInput("text_input", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(prompt_np)

            inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
            inputs[-1].set_data_from_numpy(np.array([True], dtype=bool))

            sampling = {"temperature": "0.1", "top_p": "0.95", "max_tokens": "100"}
            sampling_bytes = np.array([json.dumps(sampling).encode("utf-8")], dtype=np.object_)
            inputs.append(grpcclient.InferInput("sampling_parameters", [1], "BYTES"))
            inputs[-1].set_data_from_numpy(sampling_bytes)

            inputs.append(grpcclient.InferInput("exclude_input_in_output", [1], "BOOL"))
            inputs[-1].set_data_from_numpy(np.array([True], dtype=bool))

            outputs = [grpcclient.InferRequestedOutput("text_output")]

            inputs_list.append({
                "model_name": "vllm_model",
                "inputs": inputs,
                "outputs": outputs,
                "request_id": task.id,  # 간단히 식별
                "parameters": sampling,
            })

        # Triton stream 요청
        response_iterator = triton_client.stream_infer(inputs_iterator=iter(inputs_list))
        task_map = {task.id: task for task in batch}
        # breakpoint()
        print("📡 Triton 응답 수신 대기 중...")
        async for response in response_iterator:
            print("🟢 응답 수신됨")
            result, error = response
            if error:
                print("Error:", error)
                continue
            text = result.as_numpy("text_output")[0].decode("utf-8")
            prompt_id = result.get_response().id
            # task 찾기
            # 안전한 매핑
            task = task_map.get(prompt_id)
            if task:
                task.future.set_result(text)
            else:
                print(f"⚠️ Unknown response ID: {prompt_id}")

# FastAPI 서버 시작 시 worker 실행
@app.on_event("startup")
async def startup_event():
    global triton_client
    # Triton client를 현재 루프에서 생성
    triton_client = grpcclient.InferenceServerClient("localhost:8001")
    asyncio.create_task(stream_worker())

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
