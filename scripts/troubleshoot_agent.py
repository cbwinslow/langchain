"""Simple AI agent for deployment troubleshooting."""

import os
import sys
import textwrap
import subprocess

PROVIDER = os.getenv("LLM_PROVIDER", "ollama")

try:
    from langchain_groq import ChatGroq
except Exception:  # pragma: no cover - optional
    ChatGroq = None

try:
    from langchain_ollama import ChatOllama
except Exception:  # pragma: no cover - optional
    ChatOllama = None

try:
    from langchain_localai import ChatLocalAI
except Exception:  # pragma: no cover - optional
    ChatLocalAI = None

try:
    from langchain_openrouter import ChatOpenRouter
except Exception:  # pragma: no cover - optional
    ChatOpenRouter = None

PROVIDERS = {
    "groq": ChatGroq,
    "ollama": ChatOllama,
    "localai": ChatLocalAI,
    "openrouter": ChatOpenRouter,
}


def gather_diagnostics() -> str:
    """Collect basic system diagnostics for troubleshooting."""
    info = []
    for cmd in ["uname -a", "docker --version", "docker-compose --version"]:
        try:
            out = subprocess.check_output(cmd, shell=True, text=True).strip()
            info.append(f"$ {cmd}\n{out}")
        except Exception as exc:  # pragma: no cover - best effort
            info.append(f"$ {cmd}\nERROR: {exc}")
    return "\n".join(info)

def get_llm(provider: str):
    """Return an instantiated chat model for the given provider."""
    cls = PROVIDERS.get(provider)
    if cls is None:
        raise ValueError(f"Unsupported provider: {provider}")
    return cls()


def main() -> None:
    error_log = sys.stdin.read()
    diagnostics = gather_diagnostics()
    prompt = textwrap.dedent(
        f"""Troubleshoot the following deployment issue.\n\nSYSTEM INFO:\n{diagnostics}\n\nERROR LOG:\n{error_log}\n"""
    )
    try:
        llm = get_llm(PROVIDER)
        response = llm.invoke(prompt)
        print(str(response))
    except Exception as e:
        print(f"[Agent Error] {e}")


if __name__ == "__main__":
    main()
