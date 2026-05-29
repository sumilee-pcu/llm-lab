import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama


load_dotenv()

model = os.getenv("OLLAMA_MODEL", "gemma3n:e4b")
llm = ChatOllama(model=model, temperature=0)

response = llm.invoke(
    "한국어로 한 문장만 답하세요: 21번 교재의 로컬 Ollama 실습 환경이 준비되었나요?"
)

print(response.content)
