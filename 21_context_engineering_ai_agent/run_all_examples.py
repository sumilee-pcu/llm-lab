import subprocess
import sys
from pathlib import Path


PYTHON = Path("../contextengineeringbook/.venv/bin/python")

EXAMPLES = [
    ("checkVersion.py", None, 30),
    ("local_ollama_smoke.py", None, 30),
    ("embedding_diagnostic.py", None, 60),
    ("1.1.3.py", None, 90),
    ("1.2.1-case1.py", None, 90),
    ("1.2.1-case2.py", None, 90),
    ("1.2.2-case1.py", None, 90),
    ("1.2.2-case2.py", None, 90),
    ("1.3-case1.py", None, 90),
    ("1.3-case2.py", None, 90),
    ("2.3.2.py", None, 90),
    ("2.5.1.py", None, 90),
    ("4.2.2.py", None, 90),
    ("4.3.py", None, 120),
    ("5.2.2.py", None, 120),
    ("5.2.3.py", None, 120),
    ("6.2.1.py", None, 120),
    ("6.2.2.py", None, 120),
    ("7.4.2.py", "내일 오후 3시 회의 일정 추가\n종료\n", 30),
    ("agent.py", "y\n", 120),
    ("agent_full_code.py", "y\n", 180),
    ("refund_agent.py", None, 120),
    ("knowledge_curator_agent.py", None, 180),
    ("verify_db.py", None, 30),
]


def summarize_output(text: str, limit: int = 2000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated] ..."


def main() -> int:
    failures = []
    for script, stdin, timeout in EXAMPLES:
        print(f"\n===== RUN {script} =====", flush=True)
        try:
            result = subprocess.run(
                [str(PYTHON), script],
                input=stdin,
                text=True,
                capture_output=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            failures.append((script, "TIMEOUT", str(exc)))
            print(f"TIMEOUT after {timeout}s")
            continue

        output = "\n".join(part for part in [result.stdout, result.stderr] if part)
        print(summarize_output(output), flush=True)
        if result.returncode != 0:
            failures.append((script, f"exit {result.returncode}", output))

    print("\n===== SUMMARY =====")
    if not failures:
        print("ALL_EXAMPLES_PASSED")
        return 0

    for script, status, output in failures:
        print(f"{script}: {status}")
        print(summarize_output(output, 1200))
    return 1


if __name__ == "__main__":
    sys.exit(main())
