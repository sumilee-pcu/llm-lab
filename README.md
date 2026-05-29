# LLM Lab

LLM/RAG/AI agent textbook practice workspace.

## Structure

- `21_context_engineering_ai_agent/` - 21번 교재, `컨텍스트 엔지니어링으로 완성하는 AI 에이전트`

Future textbook practices should be added as numbered folders, for example:

- `10_rag_advanced/`
- `18_practical_rag/`

## Current Policy

- Default paid API path: Gemini API via `GOOGLE_API_KEY`
- Local no-cost smoke path: Ollama via `gemma3n:e4b`
- OpenAI API keys are not required for the updated 21번 교재 examples.

## 21번 교재 Status

- Folder: `21_context_engineering_ai_agent/`
- Verification date: 2026-05-29
- Full harness: `../contextengineeringbook/.venv/bin/python run_all_examples.py`
- Result: `ALL_EXAMPLES_PASSED`
- Real API path verified with Gemini chat and `models/gemini-embedding-2` embeddings.
