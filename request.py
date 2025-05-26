import asyncio
import httpx
import time
import traceback

URL = "http://0.0.0.0:8003/generate/"
PROMPTS = [
    "Explain quantum computing in simple terms.",
    "What are the benefits of eating vegetables?",
    "Who won the world cup in 2022?",
    "Write a haiku about winter.",
    "Summarize the plot of Inception.",
    "What's the capital of Australia?",
    "Tell me a joke about computers.",
    "How do neural networks learn?",
    "What is the Pythagorean theorem?",
    "Describe black holes to a 10-year-old.",
]

async def send_request(prompt: str, session: httpx.AsyncClient, index: int):
    try:
        start = time.time()
        response = await session.post(URL, json={"prompt": prompt}, timeout=60)
        duration = time.time() - start
        print(f"[{index}] Response ({duration:.2f}s): {response.json()['response'][:60]}...")
    except Exception as e:
        print(f"[{index}] Error: {e}")
        traceback.print_exc()

async def main():
    async with httpx.AsyncClient() as session:
        tasks = [send_request(p, session, i) for i, p in enumerate(PROMPTS)]
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    asyncio.run(main())
