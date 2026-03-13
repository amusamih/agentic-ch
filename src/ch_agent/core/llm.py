from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv

# OpenAI SDK (installed in Step 6)
from openai import OpenAI


@dataclass
class LLMReply:
    text: str
    model: str
    usage: Optional[Dict[str, Any]] = None


def load_env(project_root: str) -> None:
    """
    Loads environment variables from <project_root>/.env if present.
    """
    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path, override=False)


def get_openai_model() -> str:
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def make_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env file.")
    return OpenAI(api_key=api_key)


def openai_chat(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 400,
) -> LLMReply:
    """
    Uses the Chat Completions API (simple + stable).
    """
    m = model or get_openai_model()
    resp = client.chat.completions.create(
        model=m,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = resp.choices[0].message.content or ""
    usage = None
    if getattr(resp, "usage", None) is not None:
        usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
    return LLMReply(text=text, model=m, usage=usage)
