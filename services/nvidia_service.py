import logging
import asyncio
import time
import httpx
from typing import Optional, Dict, Any, List
from enum import Enum
from dataclasses import dataclass, field

from config.settings import get_settings
from config.models_config import NVIDIA_MODELS

logger = logging.getLogger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------

class CircuitState(Enum):
    CLOSED = "closed"       # Normal — requests flow through
    OPEN = "open"           # Tripped — requests fail fast
    HALF_OPEN = "half_open" # Probing — one test request allowed


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int = 3       # failures before opening
    recovery_timeout: float = 60.0   # seconds before trying again
    success_threshold: int = 2       # successes in half-open to close

    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failures: int = field(default=0, init=False)
    _successes: int = field(default=0, init=False)
    _opened_at: float = field(default=0.0, init=False)

    @property
    def state(self) -> CircuitState:
        if self._state == CircuitState.OPEN:
            if time.monotonic() - self._opened_at >= self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._successes = 0
                logger.info(f"Circuit [{self.name}] → HALF_OPEN (probing)")
        return self._state

    def is_available(self) -> bool:
        return self.state != CircuitState.OPEN

    def record_success(self):
        if self._state == CircuitState.HALF_OPEN:
            self._successes += 1
            if self._successes >= self.success_threshold:
                self._state = CircuitState.CLOSED
                self._failures = 0
                logger.info(f"Circuit [{self.name}] → CLOSED (recovered)")
        else:
            self._failures = 0

    def record_failure(self):
        self._failures += 1
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(f"Circuit [{self.name}] → OPEN (failed in half-open)")
        elif self._failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            logger.warning(f"Circuit [{self.name}] → OPEN after {self._failures} failures")

    def status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failures": self._failures,
        }


# ---------------------------------------------------------------------------
# NVIDIAService
# ---------------------------------------------------------------------------

class NVIDIAService:

    # Fallback chains: if primary model fails, try these in order
    CHAT_FALLBACK_CHAIN = ["chat_1", "chat_2", "orchestrator"]
    ORCHESTRATOR_FALLBACK_CHAIN = ["orchestrator", "agent", "chat_1"]
    AGENT_FALLBACK_CHAIN = ["agent", "orchestrator", "chat_1"]

    # Retry config
    MAX_RETRIES = 3
    BASE_BACKOFF = 1.0   # seconds
    MAX_BACKOFF = 16.0

    def __init__(self):
        self.api_key = settings.nvidia_api_key_main
        # Correct NVIDIA NIM hosted API base URL
        self.base_url = "https://integrate.api.nvidia.com/v1"
        self.models = NVIDIA_MODELS

        # Separate HTTP clients with different timeouts per use case
        self._chat_client = httpx.AsyncClient(timeout=45.0)
        self._media_client = httpx.AsyncClient(timeout=60.0)
        self._health_client = httpx.AsyncClient(timeout=10.0)

        # One circuit breaker per model role
        self._circuits: Dict[str, CircuitBreaker] = {
            key: CircuitBreaker(name=key)
            for key in self.models
        }

        # Track last successful model per role for adaptive selection
        self._last_good: Dict[str, str] = {}

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    async def call_orchestrator_model(
        self,
        system_prompt: str,
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> str:
        return await self._call_with_fallback(
            chain=self.ORCHESTRATOR_FALLBACK_CHAIN,
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def call_agent_model(
        self,
        system_prompt: str,
        user_message: str,
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> str:
        return await self._call_with_fallback(
            chain=self.AGENT_FALLBACK_CHAIN,
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def call_chat_model(
        self,
        system_prompt: str,
        user_message: str,
        model_variant: str = "chat_1",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        # Put the requested variant first in the chain
        chain = [model_variant] + [m for m in self.CHAT_FALLBACK_CHAIN if m != model_variant]
        return await self._call_with_fallback(
            chain=chain,
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def call_tts_model(
        self,
        text: str,
        language: str = "en-US",
        voice: str = "male",
    ) -> bytes:
        model_config = self.models["tts"]
        api_key = settings.nvidia_api_key_tts

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "audio/wav",
            "Content-Type": "application/json",
        }
        payload = {"text": text, "language": language, "voice": voice}

        return await self._retry_request(
            circuit_key="tts",
            method="post",
            url=model_config["endpoint"],
            headers=headers,
            json=payload,
            extract=lambda r: r.content,
        )

    async def call_stt_model(
        self,
        audio_data: bytes,
        language: str = "en",
    ) -> str:
        model_config = self.models["stt"]
        api_key = settings.nvidia_api_key_stt

        headers = {"Authorization": f"Bearer {api_key}"}
        files = {
            "audio": ("audio.wav", audio_data, "audio/wav"),
            "language": (None, language),
        }

        return await self._retry_request(
            circuit_key="stt",
            method="post",
            url=model_config["endpoint"],
            headers=headers,
            files=files,
            extract=lambda r: r.json().get("text", ""),
            client=self._media_client,
        )

    async def call_image_gen_model(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
    ) -> List[bytes]:
        model_config = self.models["image_gen"]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_config["name"],
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
        }

        def extract_images(response):
            data = response.json()
            images = []
            if "images" in data:
                for img in data["images"]:
                    images.append(img.encode("utf-8") if isinstance(img, str) else img)
            elif "image" in data:
                img = data["image"]
                images.append(img.encode("utf-8") if isinstance(img, str) else img)
            return images

        return await self._retry_request(
            circuit_key="image_gen",
            method="post",
            url=model_config["endpoint"],
            headers=headers,
            json=payload,
            extract=extract_images,
            client=self._media_client,
        )

    async def health_check(self) -> bool:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            url = f"{self.base_url}/models"
            r = await self._health_client.get(url, headers=headers)
            return r.status_code == 200
        except Exception as e:
            logger.error(f"NVIDIA health check failed: {e}")
            return False

    def get_circuit_status(self) -> List[Dict[str, Any]]:
        return [cb.status() for cb in self._circuits.values()]

    async def close(self):
        await asyncio.gather(
            self._chat_client.aclose(),
            self._media_client.aclose(),
            self._health_client.aclose(),
        )

    # -----------------------------------------------------------------------
    # Internal: fallback chain
    # -----------------------------------------------------------------------

    async def _call_with_fallback(
        self,
        chain: List[str],
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        role = chain[0]  # used for adaptive last-good tracking

        # Try last known-good model first if it's in the chain and its circuit is closed
        last_good = self._last_good.get(role)
        if last_good and last_good in chain and last_good != chain[0]:
            ordered = [last_good] + [m for m in chain if m != last_good]
        else:
            ordered = chain

        last_error = None
        for model_key in ordered:
            circuit = self._circuits.get(model_key)
            if circuit and not circuit.is_available():
                logger.debug(f"Circuit [{model_key}] is OPEN, skipping")
                continue

            model_config = self.models.get(model_key)
            if not model_config:
                continue

            try:
                result = await self._call_chat_model_with_retry(
                    circuit_key=model_key,
                    model_name=model_config["name"],
                    system_prompt=system_prompt,
                    user_message=user_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                self._last_good[role] = model_key
                if model_key != ordered[0]:
                    logger.info(f"Fell back to [{model_key}] successfully for role [{role}]")
                return result

            except Exception as e:
                last_error = e
                logger.warning(f"Model [{model_key}] failed: {e} — trying next in chain")
                continue

        logger.error(f"All models in chain {ordered} exhausted for role [{role}]")
        raise RuntimeError(
            f"All models failed for role [{role}]. Last error: {last_error}"
        )

    # -----------------------------------------------------------------------
    # Internal: retry with exponential backoff + circuit recording
    # -----------------------------------------------------------------------

    async def _call_chat_model_with_retry(
        self,
        circuit_key: str,
        model_name: str,
        system_prompt: str,
        user_message: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": 0.7,
            "stop": ["<|endoftext|>"],
        }

        url = f"{self.base_url}/chat/completions"

        def extract(response):
            data = response.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
            raise ValueError(f"Unexpected response format: {data}")

        return await self._retry_request(
            circuit_key=circuit_key,
            method="post",
            url=url,
            headers=headers,
            json=payload,
            extract=extract,
        )

    async def _retry_request(
        self,
        circuit_key: str,
        method: str,
        url: str,
        extract,
        headers: Optional[Dict] = None,
        json: Optional[Dict] = None,
        files: Optional[Dict] = None,
        client: Optional[httpx.AsyncClient] = None,
    ):
        circuit = self._circuits.get(circuit_key)
        if circuit and not circuit.is_available():
            raise RuntimeError(f"Circuit [{circuit_key}] is OPEN — request blocked")

        http = client or self._chat_client
        backoff = self.BASE_BACKOFF
        last_error = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                kwargs: Dict[str, Any] = {"headers": headers or {}}
                if json is not None:
                    kwargs["json"] = json
                if files is not None:
                    kwargs["files"] = files

                if method == "post":
                    response = await http.post(url, **kwargs)
                else:
                    response = await http.get(url, **kwargs)

                # Treat 429 and 5xx as retryable
                if response.status_code == 429:
                    retry_after = float(response.headers.get("retry-after", backoff))
                    logger.warning(
                        f"[{circuit_key}] rate-limited (429), waiting {retry_after:.1f}s "
                        f"(attempt {attempt}/{self.MAX_RETRIES})"
                    )
                    await asyncio.sleep(retry_after)
                    backoff = min(backoff * 2, self.MAX_BACKOFF)
                    continue

                if response.status_code >= 500:
                    raise httpx.HTTPStatusError(
                        f"Server error {response.status_code}",
                        request=response.request,
                        response=response,
                    )

                response.raise_for_status()
                result = extract(response)

                if circuit:
                    circuit.record_success()
                return result

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                last_error = e
                if circuit:
                    circuit.record_failure()
                logger.warning(
                    f"[{circuit_key}] network error on attempt {attempt}/{self.MAX_RETRIES}: {e}"
                )

            except httpx.HTTPStatusError as e:
                last_error = e
                if circuit:
                    circuit.record_failure()
                # 4xx errors (except 429) are not retryable
                if e.response.status_code < 500:
                    raise

                logger.warning(
                    f"[{circuit_key}] server error on attempt {attempt}/{self.MAX_RETRIES}: {e}"
                )

            except Exception as e:
                last_error = e
                if circuit:
                    circuit.record_failure()
                logger.warning(
                    f"[{circuit_key}] unexpected error on attempt {attempt}/{self.MAX_RETRIES}: {e}"
                )

            if attempt < self.MAX_RETRIES:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self.MAX_BACKOFF)

        if circuit:
            circuit.record_failure()
        raise RuntimeError(
            f"[{circuit_key}] failed after {self.MAX_RETRIES} attempts. Last: {last_error}"
        )
