import litellm
import instructor
from pydantic import BaseModel
from typing import TypeVar, Type

T = TypeVar("T", bound=BaseModel)


class LLMBackend:
    """
    Drop-in replacement for guidance.models.OpenAI.
    Works with any LiteLLM-supported model string:
      - "gpt-4o", "claude-sonnet-4-20250514"
      - "gemini/gemini-2.0-flash", "ollama/llama3.2"
    """

    def __init__(self, model: str, **kwargs):
        self.model = model
        self.kwargs = kwargs
        self._client = instructor.from_litellm(litellm.completion)

    def complete_structured(
        self,
        prompt: str,
        response_model: Type[T],
        system: str = "",
        max_retries: int = 3,
    ) -> T:
        """Returns a validated Pydantic object. Retries on parse failure."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self._client.create(
            model=self.model,
            messages=messages,
            response_model=response_model,
            max_retries=max_retries,
            **self.kwargs,
        )

    def complete_text(self, prompt: str, system: str = "") -> str:
        """Plain text completion for critique/freeform outputs."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = litellm.completion(model=self.model, messages=messages, **self.kwargs)
        return resp.choices[0].message.content
