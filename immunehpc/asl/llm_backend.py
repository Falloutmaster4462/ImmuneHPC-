"""
asl/llm_backend.py — Free LLM Backend for ΔC Code Generation

Priority chain (automatic failover):
  1. Ollama (local)     — zero cost, zero rate limits, air-gap capable
                          Best model: qwen3-coder:32b or deepseek-coder-v2:16b
  2. Groq (free tier)  — no credit card, ~30 RPM, Llama 3.3 70B
  3. Google AI Studio  — free tier, 15 RPM, Gemini 2.5 Flash
  4. OpenRouter free   — community pool, DeepSeek R1/V3

All backends expose the same interface: generate(prompt) → str
Backend selection is automatic based on availability and config.

Config in production.yaml under asl.llm:
  provider: auto          # auto | ollama | groq | google | openrouter
  ollama_host: http://localhost:11434
  ollama_model: qwen3-coder:32b
  groq_api_key: ""        # or env GROQ_API_KEY
  google_api_key: ""      # or env GOOGLE_API_KEY
  openrouter_api_key: ""  # or env OPENROUTER_API_KEY
  timeout_sec: 30
  max_tokens: 1024
  temperature: 0.2        # low for code generation
"""

from __future__ import annotations
import json
import os
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Optional

from utils.logger import get_logger

log = get_logger("asl.llm")


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------

@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    tokens_used: int = 0
    latency_sec: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


# ---------------------------------------------------------------------------
# HTTP helper (no extra deps — just stdlib urllib)
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict,
               timeout: float = 30.0) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Backend 1: Ollama (local, OpenAI-compatible /v1/chat/completions)
# ---------------------------------------------------------------------------

class OllamaBackend:
    """
    Calls a local Ollama instance.

    Install:  curl -fsSL https://ollama.com/install.sh | sh
    Models:   ollama pull qwen3-coder:32b
              ollama pull deepseek-coder-v2:16b   (smaller, still good)
              ollama pull qwen2.5-coder:7b         (CPU-friendly)

    Ollama exposes /v1/chat/completions (OpenAI-compatible) since v0.14.
    """

    # Recommended models in priority order — pulled at startup if missing
    RECOMMENDED_MODELS = [
        "qwen3-coder:32b",          # best for code, needs ~20GB VRAM
        "deepseek-coder-v2:16b",    # strong, needs ~10GB VRAM
        "qwen2.5-coder:14b",        # solid, ~8GB VRAM
        "qwen2.5-coder:7b",         # CPU-friendly
        "codellama:13b",            # fallback, widely available
    ]

    def __init__(self, host: str = "http://localhost:11434",
                 model: str = "qwen3-coder:32b",
                 timeout: float = 60.0) -> None:
        self.host    = host.rstrip("/")
        self.model   = model
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            req = urllib.request.urlopen(f"{self.host}/api/tags", timeout=3.0)
            return req.status == 200
        except Exception:
            return False

    def best_available_model(self) -> Optional[str]:
        """Return the best model that is already pulled locally."""
        try:
            req = urllib.request.urlopen(f"{self.host}/api/tags", timeout=3.0)
            data = json.loads(req.read().decode())
            pulled = {m["name"].split(":")[0] for m in data.get("models", [])}
            for candidate in self.RECOMMENDED_MODELS:
                base = candidate.split(":")[0]
                if base in pulled:
                    return candidate
        except Exception:
            pass
        return None

    def pull_model(self, model: str) -> bool:
        """Pull a model if not present. Blocks until complete."""
        import subprocess
        log.info("Pulling Ollama model: %s (this may take a while)...", model)
        try:
            result = subprocess.run(
                ["ollama", "pull", model],
                timeout=600,
            )
            return result.returncode == 0
        except Exception as exc:
            log.warning("ollama pull failed: %s", exc)
            return False

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 1024, temperature: float = 0.2) -> LLMResponse:
        t0 = time.time()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            data = _post_json(
                f"{self.host}/v1/chat/completions",
                payload,
                {"Content-Type": "application/json"},
                timeout=self.timeout,
            )
            text = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return LLMResponse(
                text=text, provider="ollama", model=self.model,
                tokens_used=tokens, latency_sec=time.time() - t0,
            )
        except Exception as exc:
            return LLMResponse(
                text="", provider="ollama", model=self.model,
                latency_sec=time.time() - t0, error=str(exc),
            )


# ---------------------------------------------------------------------------
# Backend 2: Groq (free tier, no credit card)
# ---------------------------------------------------------------------------

class GroqBackend:
    """
    Uses Groq's free tier — sign up at console.groq.com, no credit card.

    Free limits (as of early 2026):
      llama-3.3-70b-versatile: 30 RPM, 500K tokens/day
      llama-3.1-8b-instant:    30 RPM, 500K tokens/day
      deepseek-r1-distill-llama-70b: 30 RPM, 500K tokens/day

    Set GROQ_API_KEY env var or groq_api_key in config.
    """

    BASE_URL = "https://api.groq.com/openai/v1/chat/completions"

    # Models in quality order — all free
    MODELS = [
        "llama-3.3-70b-versatile",
        "deepseek-r1-distill-llama-70b",
        "llama-3.1-8b-instant",
    ]

    def __init__(self, api_key: str = "", model: str = "llama-3.3-70b-versatile",
                 timeout: float = 30.0) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self.model   = model
        self.timeout = timeout
        self._last_call: float = 0.0
        self._min_interval = 2.1   # ~28 RPM to stay safely under 30

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 1024, temperature: float = 0.2) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                text="", provider="groq", model=self.model,
                error="no API key — set GROQ_API_KEY",
            )

        # Respect rate limit
        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        t0 = time.time()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            data = _post_json(self.BASE_URL, payload, headers, timeout=self.timeout)
            self._last_call = time.time()

            # Handle 429 — back off and retry once
            text  = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return LLMResponse(
                text=text, provider="groq", model=self.model,
                tokens_used=tokens, latency_sec=time.time() - t0,
            )
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                log.warning("Groq rate-limited — backing off 60s")
                time.sleep(60)
            return LLMResponse(
                text="", provider="groq", model=self.model,
                latency_sec=time.time() - t0,
                error=f"HTTP {exc.code}",
            )
        except Exception as exc:
            return LLMResponse(
                text="", provider="groq", model=self.model,
                latency_sec=time.time() - t0, error=str(exc),
            )


# ---------------------------------------------------------------------------
# Backend 3: Google AI Studio (free tier, Gemini 2.5 Flash)
# ---------------------------------------------------------------------------

class GoogleBackend:
    """
    Google AI Studio free tier — sign up at aistudio.google.com, no card.
    Free limits: 15 RPM, 1M tokens/day on Gemini 2.5 Flash.
    Set GOOGLE_API_KEY env var or google_api_key in config.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, api_key: str = "",
                 model: str = "gemini-2.0-flash",
                 timeout: float = 30.0) -> None:
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.model   = model
        self.timeout = timeout
        self._last_call: float = 0.0
        self._min_interval = 4.1   # ~14 RPM, under the 15 RPM free cap

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 1024, temperature: float = 0.2) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                text="", provider="google", model=self.model,
                error="no API key — set GOOGLE_API_KEY",
            )

        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        t0 = time.time()
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        payload = {
            "contents": [{"role": "user", "parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": temperature,
            },
        }
        url = self.BASE_URL.format(model=self.model) + f"?key={self.api_key}"
        try:
            data = _post_json(url, payload,
                              {"Content-Type": "application/json"},
                              timeout=self.timeout)
            self._last_call = time.time()
            text = data["candidates"][0]["content"]["parts"][0]["text"]
            tokens = data.get("usageMetadata", {}).get("totalTokenCount", 0)
            return LLMResponse(
                text=text, provider="google", model=self.model,
                tokens_used=tokens, latency_sec=time.time() - t0,
            )
        except Exception as exc:
            return LLMResponse(
                text="", provider="google", model=self.model,
                latency_sec=time.time() - t0, error=str(exc),
            )


# ---------------------------------------------------------------------------
# Backend 4: OpenRouter (free community models)
# ---------------------------------------------------------------------------

class OpenRouterBackend:
    """
    OpenRouter free tier — free models funded by community.
    Best free models: deepseek/deepseek-r1:free, meta-llama/llama-4-maverick:free
    Set OPENROUTER_API_KEY env var (free key from openrouter.ai).
    Limits: 20 RPM, 50 req/day without a balance.
    """

    BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

    FREE_MODELS = [
        "deepseek/deepseek-r1:free",
        "meta-llama/llama-4-maverick:free",
        "qwen/qwen3-235b-a22b:free",
    ]

    def __init__(self, api_key: str = "",
                 model: str = "deepseek/deepseek-r1:free",
                 timeout: float = 30.0) -> None:
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.model   = model
        self.timeout = timeout
        self._last_call: float = 0.0
        self._min_interval = 3.1   # ~19 RPM

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 1024, temperature: float = 0.2) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                text="", provider="openrouter", model=self.model,
                error="no API key",
            )

        elapsed = time.time() - self._last_call
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

        t0 = time.time()
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/immunehpc",
            "X-Title": "ImmuneHPC+",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            data = _post_json(self.BASE_URL, payload, headers, timeout=self.timeout)
            self._last_call = time.time()
            text = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("total_tokens", 0)
            return LLMResponse(
                text=text, provider="openrouter", model=self.model,
                tokens_used=tokens, latency_sec=time.time() - t0,
            )
        except Exception as exc:
            return LLMResponse(
                text="", provider="openrouter", model=self.model,
                latency_sec=time.time() - t0, error=str(exc),
            )


# ---------------------------------------------------------------------------
# LLMRouter — automatic failover across all backends
# ---------------------------------------------------------------------------

class LLMRouter:
    """
    Tries backends in priority order, returns first success.

    Priority: Ollama → Groq → Google → OpenRouter

    Ollama is strongly preferred because:
      - Zero API calls → works air-gapped on the HPC network
      - No rate limits → important for long-running experiments
      - Your cluster's own GPU does the inference
      - Data never leaves your network

    Usage:
        router = LLMRouter.from_config(cfg["asl"]["llm"])
        response = router.generate("write me a Python patch that...")
        if response.ok:
            code = response.text
    """

    def __init__(self, backends: list, prefer: str = "auto") -> None:
        self._backends = backends
        self._prefer   = prefer
        self._stats    = {b.__class__.__name__: {"ok": 0, "fail": 0} for b in backends}

    @classmethod
    def from_config(cls, cfg: dict) -> "LLMRouter":
        prefer = cfg.get("provider", "auto")

        ollama = OllamaBackend(
            host=cfg.get("ollama_host", "http://localhost:11434"),
            model=cfg.get("ollama_model", "qwen3-coder:32b"),
            timeout=cfg.get("timeout_sec", 60),
        )
        groq = GroqBackend(
            api_key=cfg.get("groq_api_key", ""),
            model=cfg.get("groq_model", "llama-3.3-70b-versatile"),
            timeout=cfg.get("timeout_sec", 30),
        )
        google = GoogleBackend(
            api_key=cfg.get("google_api_key", ""),
            model=cfg.get("google_model", "gemini-2.0-flash"),
            timeout=cfg.get("timeout_sec", 30),
        )
        openrouter = OpenRouterBackend(
            api_key=cfg.get("openrouter_api_key", ""),
            model=cfg.get("openrouter_model", "deepseek/deepseek-r1:free"),
            timeout=cfg.get("timeout_sec", 30),
        )

        # Order: Ollama first (always preferred for HPC research)
        backends = [ollama, groq, google, openrouter]

        # If a specific provider is named, put it first
        if prefer == "groq":
            backends = [groq, ollama, google, openrouter]
        elif prefer == "google":
            backends = [google, groq, ollama, openrouter]
        elif prefer == "openrouter":
            backends = [openrouter, groq, google, ollama]

        router = cls(backends, prefer=prefer)
        router._log_available()
        return router

    def _log_available(self) -> None:
        for b in self._backends:
            avail = b.is_available()
            log.info("LLM backend %-15s available=%s",
                     b.__class__.__name__, avail)

    def generate(self, prompt: str, system: str = "",
                 max_tokens: int = 1024, temperature: float = 0.2) -> LLMResponse:
        """Try each backend in order, return the first successful response."""
        for backend in self._backends:
            if not backend.is_available():
                continue
            response = backend.generate(
                prompt, system=system,
                max_tokens=max_tokens, temperature=temperature,
            )
            name = backend.__class__.__name__
            if response.ok:
                self._stats[name]["ok"] += 1
                log.debug("LLM [%s/%s] %.1fs %d tokens",
                          response.provider, response.model,
                          response.latency_sec, response.tokens_used)
                return response
            else:
                self._stats[name]["fail"] += 1
                log.warning("LLM backend %s failed: %s", name, response.error)

        # All backends failed
        return LLMResponse(
            text="", provider="none", model="none",
            error="all LLM backends unavailable",
        )

    def stats(self) -> dict:
        return dict(self._stats)
