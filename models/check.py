from vllm import LLM, SamplingParams

# 1. 모델 불러오기 (vLLM은 모델을 자동으로 GPU에 분산 로드)
llm = LLM(
    model="MLP-KTLim/llama-3-Korean-Bllossom-8B",  # Hugging Face 모델 이름 (또는 local 경로)
    tensor_parallel_size=2,            # 사용할 GPU 수
)

# 2. 생성 파라미터 설정
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256,
)

# 3. 프롬프트 작성
prompt = "한국에서 외국인 근로자를 고용할 때 지켜야 할 법적 절차는 무엇인가요?"

# 4. 텍스트 생성
outputs = llm.generate([prompt], sampling_params)

# 5. 출력 결과 확인
for output in outputs:
    print("=== Generated Output ===")
    print(output.outputs[0].text)
