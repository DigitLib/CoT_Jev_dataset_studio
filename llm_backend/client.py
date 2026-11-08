"""
Universal OpenAI-compatible client for local backends (Ollama, LM Studio, vLLM).
Uses httpx for lightweight, dependency-free local communication.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import httpx
import json
import os
import logging
import re
import asyncio
import concurrent.futures
import time

logger = logging.getLogger(__name__)


def clean_model_display_name(model: Any) -> str:
    """Extract a clean, human-readable display name for a model string or file path."""
    if not model or not isinstance(model, str):
        return "Default Model"
    s = model.strip().rstrip("/\\")
    if "/" in s or "\\" in s:
        parts = [p for p in s.replace("\\", "/").split("/") if p]
        if parts:
            last = parts[-1]
            if last.lower() in ("v1", "models", "checkpoint") and len(parts) > 1:
                return parts[-2]
            return last
    return s


def provenance_model_name(model: Any) -> str:
    """
    Model identifier safe to store in dataset provenance: a local filesystem path (as vLLM reports when it was
    started with --model /abs/path and no --served-model-name) is reduced to its folder name, so no local
    paths or user names end up in published datasets. Hub ids such as "org/model" are kept unchanged.
    """
    if not model or not isinstance(model, str):
        return ""
    s = model.strip()
    looks_like_path = (s.startswith(("/", "~", "./", "../")) or "\\" in s or re.match(r"^[A-Za-z]:[\\/]", s)
                       or s.count("/") > 1)
    return clean_model_display_name(s) if looks_like_path else s


class LocalLLMClient:
    _HEALTH_CACHE: Dict[str, Tuple[float, bool]] = {}
    _MODELS_CACHE: Dict[str, Tuple[float, List[str]]] = {}
    CACHE_TTL: float = 15.0  # seconds

    @classmethod
    def invalidate_cache(cls, base_url: Optional[str] = None):
        """Invalidate in-memory health and model caches."""
        if base_url:
            cleaned = base_url.rstrip("/")
            cls._HEALTH_CACHE.pop(cleaned, None)
            cls._MODELS_CACHE.pop(cleaned, None)
        else:
            cls._HEALTH_CACHE.clear()
            cls._MODELS_CACHE.clear()

    # Chat-template switches per model family (vLLM / SGLang `chat_template_kwargs`). Spark-X thinks by default
    # ("high" effort); the studio asks for JSON and its own reasoning format, so thinking is turned off unless
    # STUDIO_ENABLE_THINKING=1 or the client is given chat_template_kwargs explicitly.
    MODEL_TEMPLATE_DEFAULTS: Dict[str, Dict[str, Any]] = {
        "spark-x": {"enable_thinking": False},
    }

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.chat_template_kwargs = chat_template_kwargs
        self.base_url = base_url.rstrip("/")
        # local servers ignore the key; hosted OpenAI-compatible APIs read it from DTSTUDIO_LLM_API_KEY
        self.api_key = api_key or os.environ.get("DTSTUDIO_LLM_API_KEY") or "not-needed"
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def health_check(self, force: bool = False, ttl: Optional[float] = None) -> bool:
        """Check if local endpoint is responsive, with in-memory TTL caching."""
        cache_ttl = ttl if ttl is not None else self.CACHE_TTL
        now = time.time()
        if not force and self.base_url in self._HEALTH_CACHE:
            ts, val = self._HEALTH_CACHE[self.base_url]
            if (now - ts) < cache_ttl:
                return val

        is_healthy = False
        try:
            with httpx.Client(timeout=2.0) as client:
                res = client.get(f"{self.base_url}/models", headers=self.headers)
                is_healthy = (res.status_code == 200)
        except Exception:
            is_healthy = False

        self._HEALTH_CACHE[self.base_url] = (now, is_healthy)
        return is_healthy

    def list_models(self, force: bool = False, ttl: Optional[float] = None) -> List[str]:
        """Fetch available models from the local server, with in-memory TTL caching."""
        cache_ttl = ttl if ttl is not None else self.CACHE_TTL
        now = time.time()
        if not force and self.base_url in self._MODELS_CACHE:
            ts, val = self._MODELS_CACHE[self.base_url]
            if (now - ts) < cache_ttl:
                return val

        models: List[str] = []
        try:
            with httpx.Client(timeout=3.0) as client:
                res = client.get(f"{self.base_url}/models", headers=self.headers)
                if res.status_code == 200:
                    data = res.json()
                    models = [item["id"] for item in data.get("data", []) if "id" in item]
        except Exception as e:
            logger.warning(f"Failed to fetch models from {self.base_url}: {e}")

        self._MODELS_CACHE[self.base_url] = (now, models)
        return models

    def _template_kwargs(self, model: str, per_call: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """chat_template_kwargs for this request: per-call value, client setting, else the model family default."""
        import os
        if per_call is not None:
            return dict(per_call) or None
        if self.chat_template_kwargs is not None:
            return self.chat_template_kwargs or None
        if os.environ.get("STUDIO_ENABLE_THINKING", "").strip() in ("1", "true", "yes"):
            return None
        name = clean_model_display_name(model).lower()
        for key, kw in self.MODEL_TEMPLATE_DEFAULTS.items():
            if key in name:
                return dict(kw)
        return None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "qwen2.5:7b",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Execute chat completion request against local engine.
        Handles engine-specific quirks (e.g. models in LM Studio like Gemma rejecting system prompts).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        tkw = self._template_kwargs(model, chat_template_kwargs)
        if tkw:
            payload["chat_template_kwargs"] = tkw

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                )

                # Retry strategy for backends/models that hit context limits or reject "system" role / "response_format"
                if res.status_code == 400:
                    err_lower = res.text.lower()
                    is_context_err = any(k in err_lower for k in ("context length", "output tokens", "reduce the length", "max_tokens", "max_model_len"))
                    if is_context_err:
                        reduced_tokens = 384 if max_tokens > 384 else max(128, max_tokens // 2)
                        logger.warning(
                            f"LLM request returned 400 context limit error. Retrying with reduced max_tokens={reduced_tokens}..."
                        )
                        retry_payload = dict(payload)
                        retry_payload["max_tokens"] = reduced_tokens
                        res = client.post(
                            f"{self.base_url}/chat/completions",
                            headers=self.headers,
                            json=retry_payload,
                        )
                    elif system_prompt or json_mode:
                        logger.warning(
                            f"Initial LLM request returned 400 ({res.text[:180]}). "
                            "Retrying with system prompt merged into user message and standard format..."
                        )
                        merged_content = f"### Instruction:\n{system_prompt}\n\n### Task Context:\n{prompt}" if system_prompt else prompt
                        fallback_payload: Dict[str, Any] = {
                            "model": model,
                            "messages": [{"role": "user", "content": merged_content}],
                            "temperature": temperature,
                            "max_tokens": min(max_tokens, 512),
                            **({"chat_template_kwargs": tkw} if tkw else {}),
                            "stream": False,
                        }
                        res = client.post(
                            f"{self.base_url}/chat/completions",
                            headers=self.headers,
                            json=fallback_payload,
                        )

                if res.status_code != 200:
                    raise RuntimeError(
                        f"LLM request failed with status {res.status_code}: {res.text}"
                    )
                return self._parse_response_content(res.json())
        except httpx.ConnectError:
            raise ConnectionError(
                f"Could not connect to local LLM backend at {self.base_url}. "
                "Ensure Ollama, LM Studio, or vLLM is running locally."
            )
        except Exception as e:
            logger.error(f"Error during LLM generation on {self.base_url} ({model}): {e}")
            raise

    def _parse_response_content(self, data: Dict[str, Any]) -> str:
        """Helper to extract content and reasoning tokens from OpenAI-compatible response."""
        from .adapters import ReasoningTraceAdapter
        return ReasoningTraceAdapter.parse_chat_response(data)

    async def async_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "qwen2.5:7b",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        json_mode: bool = False,
        chat_template_kwargs: Optional[Dict[str, Any]] = None,
        client: Optional[httpx.AsyncClient] = None,
    ) -> str:
        """Asynchronous execution of chat completion request (ideal for vLLM & concurrent batching)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        tkw = self._template_kwargs(model, chat_template_kwargs)
        if tkw:
            payload["chat_template_kwargs"] = tkw

        async def _do_post(c: httpx.AsyncClient) -> str:
            res = await c.post(
                f"{self.base_url}/chat/completions",
                headers=self.headers,
                json=payload,
            )

            if res.status_code == 400:
                err_lower = res.text.lower()
                is_context_err = any(k in err_lower for k in ("context length", "output tokens", "reduce the length", "max_tokens", "max_model_len"))
                if is_context_err:
                    reduced_tokens = 384 if max_tokens > 384 else max(128, max_tokens // 2)
                    logger.warning(
                        f"Async LLM request returned 400 context limit error. Retrying with reduced max_tokens={reduced_tokens}..."
                    )
                    retry_payload = dict(payload)
                    retry_payload["max_tokens"] = reduced_tokens
                    res = await c.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=retry_payload,
                    )
                elif system_prompt or json_mode:
                    merged_content = f"### Instruction:\n{system_prompt}\n\n### Task Context:\n{prompt}" if system_prompt else prompt
                    fallback_payload: Dict[str, Any] = {
                        "model": model,
                        "messages": [{"role": "user", "content": merged_content}],
                        "temperature": temperature,
                        "max_tokens": min(max_tokens, 512),
                        **({"chat_template_kwargs": tkw} if tkw else {}),
                        "stream": False,
                    }
                    res = await c.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=fallback_payload,
                    )

            if res.status_code != 200:
                raise RuntimeError(
                    f"LLM request failed with status {res.status_code}: {res.text}"
                )
            return self._parse_response_content(res.json())

        try:
            if client is not None:
                return await _do_post(client)
            else:
                async with httpx.AsyncClient(timeout=self.timeout) as own_client:
                    return await _do_post(own_client)
        except httpx.ConnectError:
            raise ConnectionError(
                f"Could not connect to local LLM backend at {self.base_url}. "
                "Ensure Ollama, LM Studio, or vLLM is running locally."
            )
        except Exception as e:
            logger.error(f"Error during async LLM generation on {self.base_url} ({model}): {e}")
            raise

    async def async_generate_batch(
        self,
        requests: List[Union[str, Dict[str, Any]]],
        concurrency: int = 4,
    ) -> List[str]:
        """
        Execute multiple generation requests concurrently with an asyncio Semaphore and pooled HTTP connections.
        Accepts either a list of prompt strings or a list of request dictionaries.
        Provides massive 5x-15x throughput on continuous-batching servers like vLLM.
        """
        if not requests:
            return []
        normalized_requests = [
            r if isinstance(r, dict) else {"prompt": str(r)}
            for r in requests
        ]
        sem = asyncio.Semaphore(max(1, concurrency))
        limits = httpx.Limits(max_connections=max(10, concurrency * 2), max_keepalive_connections=max(5, concurrency))

        async with httpx.AsyncClient(timeout=self.timeout, limits=limits) as shared_client:
            async def _bounded_worker(req: Dict[str, Any]) -> str:
                async with sem:
                    return await self.async_generate(
                        prompt=req["prompt"],
                        system_prompt=req.get("system_prompt"),
                        model=req.get("model", "qwen2.5:7b"),
                        temperature=req.get("temperature", 0.2),
                        max_tokens=req.get("max_tokens", 1024),
                        json_mode=req.get("json_mode", False),
                        chat_template_kwargs=req.get("chat_template_kwargs"),
                        client=shared_client,
                    )

            tasks = [_bounded_worker(r) for r in normalized_requests]
            return await asyncio.gather(*tasks)

    def generate_concurrent_batch(
        self,
        requests: List[Union[str, Dict[str, Any]]],
        concurrency: int = 4,
        max_workers: Optional[int] = None,
    ) -> List[str]:
        """
        Synchronous wrapper for concurrent batch generation using ThreadPoolExecutor.
        Accepts either a list of prompt strings or a list of request dictionaries.
        """
        if not requests:
            return []
        effective_workers = max_workers if max_workers is not None else concurrency
        normalized_requests = [
            r if isinstance(r, dict) else {"prompt": str(r)}
            for r in requests
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, effective_workers)) as executor:
            futures = [
                executor.submit(
                    self.generate,
                    prompt=req["prompt"],
                    system_prompt=req.get("system_prompt"),
                    model=req.get("model", "qwen2.5:7b"),
                    temperature=req.get("temperature", 0.2),
                    max_tokens=req.get("max_tokens", 1024),
                    json_mode=req.get("json_mode", False),
                    chat_template_kwargs=req.get("chat_template_kwargs"),
                )
                for req in normalized_requests
            ]
            return [f.result() for f in futures]

    # Alias for continuous-batching generation interfaces
    batch_generate = generate_concurrent_batch

    def embed(
        self,
        texts: Any,
        model: str = "text-embedding-nomic-embed-text-v1.5",
    ) -> List[List[float]]:
        """
        Execute embedding request against local engine (/embeddings).
        Accepts a single string or a list of strings, returning a list of float vectors.
        """
        is_single = isinstance(texts, str)
        input_payload = [texts] if is_single else texts

        payload: Dict[str, Any] = {
            "model": model,
            "input": input_payload,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                res = client.post(
                    f"{self.base_url}/embeddings",
                    headers=self.headers,
                    json=payload,
                )
                if res.status_code != 200:
                    raise RuntimeError(
                        f"Embedding request failed with status {res.status_code}: {res.text}"
                    )
                data = res.json()
                items = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
                return [item["embedding"] for item in items]
        except Exception as e:
            logger.error(f"Error during embedding on {self.base_url} ({model}): {e}")
            raise

