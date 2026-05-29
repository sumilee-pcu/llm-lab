import os
from provider_config import get_gemini_chat_model
# CLAUDE: Claude를 사용하려면 'langchain_anthropic'에서 관련 클래스를 임포트해야 합니다.
# CLAUDE: from langchain_anthropic import ChatAnthropic
# GEMINI: Gemini를 사용하려면 'langchain_google_genai'에서 관련 클래스를 임포트해야 합니다.
# GEMINI: from langchain_google_genai import ChatGoogleGenerativeAI

# [추가] .env 파일 로드를 위한 라이브러리 임포트 (Best Practice)
from dotenv import load_dotenv
load_dotenv()

# Gemini API 키 설정
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_API_KEY")
# CLAUDE: Claude API 키를 설정해야 합니다.
# CLAUDE: os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY", "YOUR_CLAUDE_KEY")
# GEMINI: Google API 키를 설정해야 합니다.
# GEMINI: os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY", "YOUR_GEMINI_KEY")


# 1. 답변 생성에 사용할 언어 모델(LLM)을 정의합니다.
llm = get_gemini_chat_model(temperature=0, max_output_tokens=1024)

# CLAUDE: LLM 모델을 ChatAnthropic으로 변경합니다. (모델명 지정 필수)
# CLAUDE: llm = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)

# GEMINI: LLM 모델을 ChatGoogleGenerativeAI로 변경합니다. (모델명 지정 필수)
# GEMINI: llm = ChatGoogleGenerativeAI(model="gemini-3.0", temperature=0)


def update_summary(summary: str, user_input: str, assistant_output: str) -> str:
    """2026-05 기준: deprecated ConversationSummaryMemory 대신 직접 요약 상태를 갱신합니다."""
    prompt = f"""아래 기존 대화 요약과 새 대화를 합쳐 3문장 이내의 한국어 요약 메모리로 갱신하세요.

[기존 요약]
{summary or "없음"}

[새 사용자 발화]
{user_input}

[새 AI 응답]
{assistant_output}
"""
    return llm.invoke(prompt).content.strip()


def conversation_turn(summary: str, user_input: str) -> tuple[str, str]:
    prompt = f"""당신은 친절한 AI 학습 코치입니다.
아래 대화 요약을 기억으로 사용해 사용자 질문에 답하세요.

[대화 요약 메모리]
{summary or "아직 요약된 대화가 없습니다."}

[사용자]
{user_input}
"""
    response = llm.invoke(prompt).content.strip()
    return response, update_summary(summary, user_input, response)

# --- 대화 시작 ---

# 첫 번째 대화
# 최신 LangChain은 .invoke()를 사용하며, ConversationChain은 딕셔너리 입력을 받습니다.
print("--- 1번째 대화 ---")
summary_memory = ""
response, summary_memory = conversation_turn(
    summary_memory,
    "안녕? 내 이름은 이서준이야. 나는 AI 기술을 활용한 챗봇 개발에 관심이 많아."
)
print(response)
print("\n[요약 메모리]\n", summary_memory)

# 두 번째 대화
print("\n--- 2번째 대화 ---")
response, summary_memory = conversation_turn(
    summary_memory,
    "특히 RAG 기술에 대해 더 깊이 알고 싶어."
)
print(response)
print("\n[요약 메모리]\n", summary_memory)

# 세 번째 대화
# AI는 이전 대화 내용을 요약해서 기억하고 있으므로, '그 기술'이 RAG를 의미하는지 압니다.
print("\n--- 3번째 대화 ---")
response, summary_memory = conversation_turn(
    summary_memory,
    "그 기술을 우리 회사 고객 지원팀에 어떻게 적용할 수 있을까?"
)

print("\n--- 최종 응답 ---")
print(response)
