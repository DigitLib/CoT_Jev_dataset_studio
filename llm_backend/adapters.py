"""
adapters.py - Standardized reasoning channel adapters for local and remote LLM backends.

Provides unified parsing and normalization for models featuring explicit thinking / reasoning
channels (DeepSeek-R1, Spark-X, Qwen-2.5-Coder / Thinking, OpenAI o-series, and Anthropic).
"""

from typing import Dict, Any, Tuple, Optional
import re


class ReasoningTraceAdapter:
    """
    Standardizes reasoning trace extraction across diverse provider formats:
      - OpenAI / vLLM 'reasoning_content' or 'reasoning' fields
      - Chat templates embedding <think>...</think> in response content
      - Chat templates pre-opening <think> where only </think> is returned
      - ClearCoT <reasoning>...</reasoning> blocks
    """

    @classmethod
    def parse_chat_response(cls, data: Dict[str, Any]) -> str:
        """
        Extracts content and reasoning from an OpenAI-compatible /chat/completions payload,
        formatting it into a canonical string containing <think>...</think> if reasoning was present.
        """
        choices = data.get("choices", [])
        if not choices:
            return ""
        msg = choices[0].get("message", {})
        content = (msg.get("content") or "").strip()
        reasoning = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()

        # Handle models whose chat template opens <think> in prompt without server-side tag synthesis
        if not reasoning and "</think>" in content and "<think>" not in content:
            content = "<think>\n" + content

        if reasoning and content:
            if "<think>" not in content and "<reasoning>" not in content:
                return f"<think>\n{reasoning}\n</think>\n{content}".strip()
            return content
        elif reasoning and not content:
            return f"<think>\n{reasoning}\n</think>".strip()
        return content

    @classmethod
    def extract_trace_and_answer(cls, raw: str) -> Tuple[str, str]:
        """
        Splits a raw response into (reasoning_trace, final_answer).
        Checks <think>, <reasoning>, and <thought> tags, falling back to empty trace.
        """
        if not raw:
            return "", ""

        text = raw.strip()

        # 1. Check for <think>...</think>
        m_think = re.search(r"<think>(.*?)</think>", text, re.DOTALL | re.IGNORECASE)
        if m_think:
            trace = m_think.group(1).strip()
            ans = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            ans = re.sub(r"</?(?:answer)>", "", ans, flags=re.IGNORECASE).strip()
            return trace, ans

        # 2. Check for <reasoning>...</reasoning>
        m_reas = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
        if m_reas:
            trace = m_reas.group(1).strip()
            ans = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            ans = re.sub(r"</?(?:answer)>", "", ans, flags=re.IGNORECASE).strip()
            return trace, ans

        # 3. Check for <thought>...</thought>
        m_th = re.search(r"<thought>(.*?)</thought>", text, re.DOTALL | re.IGNORECASE)
        if m_th:
            trace = m_th.group(1).strip()
            ans = re.sub(r"<thought>.*?</thought>", "", text, flags=re.DOTALL | re.IGNORECASE).strip()
            ans = re.sub(r"</?(?:answer)>", "", ans, flags=re.IGNORECASE).strip()
            return trace, ans

        # 4. Check for <answer>...</answer> without explicit thinking tags
        m_ans = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
        if m_ans:
            ans = m_ans.group(1).strip()
            trace = text[:m_ans.start()].strip()
            return trace, ans

        return "", text
