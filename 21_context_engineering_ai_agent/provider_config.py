import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_GEMINI_EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL", "models/embedding-001"
)


def require_google_api_key() -> str:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY가 설정되어 있지 않습니다. .env 파일에 Google AI Studio API 키를 넣어주세요."
        )
    return api_key


def get_gemini_chat_model(
    *,
    temperature: float = 0,
    max_output_tokens: int | None = None,
) -> ChatGoogleGenerativeAI:
    require_google_api_key()
    kwargs = {
        "model": DEFAULT_GEMINI_MODEL,
        "temperature": temperature,
    }
    if max_output_tokens is not None:
        kwargs["max_output_tokens"] = max_output_tokens
    return ChatGoogleGenerativeAI(**kwargs)


def get_gemini_embeddings() -> GoogleGenerativeAIEmbeddings:
    require_google_api_key()
    return GoogleGenerativeAIEmbeddings(model=DEFAULT_GEMINI_EMBEDDING_MODEL)
